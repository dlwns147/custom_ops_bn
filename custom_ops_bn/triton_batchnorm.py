"""
Triton kernels for batch normalization.

This module provides pure Triton kernel implementations for batch normalization.
The kernels use Welford's online algorithm for numerically stable mean and
variance computation.

For PyTorch integration, use the custom operators registered in ops.py.

Note: For production use, the CUDA implementation is recommended for better
performance and stability.
"""

import triton
import triton.language as tl


@triton.jit
def batchnorm_forward_training_kernel(
    # Pointers
    input_ptr,          # Input tensor pointer
    gamma_ptr,          # Scale parameter pointer
    beta_ptr,           # Shift parameter pointer
    output_ptr,         # Output tensor pointer
    mean_ptr,           # Saved mean pointer
    invstd_ptr,         # Saved inverse std pointer
    running_mean_ptr,   # Running mean pointer (updated in-place)
    running_var_ptr,    # Running variance pointer (updated in-place)
    # Dimensions
    N, C, spatial_dim,  # Batch size, channels, spatial dimension (H*W)
    # Parameters
    momentum,           # Momentum for running stats
    eps,                # Epsilon for numerical stability
    # Strides
    stride_n, stride_c, stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    M = N * spatial_dim

    offsets = tl.arange(0, BLOCK_SIZE)

    # 1. Compute mean and variance for this channel
    sum_x = tl.full((), 0.0, tl.float32)
    sum_x2 = tl.full((), 0.0, tl.float32)

    for start in range(tl.cdiv(M, BLOCK_SIZE)):
        idx = start * BLOCK_SIZE + offsets
        mask = idx < M

        n = idx // spatial_dim
        s = idx % spatial_dim

        x = tl.load(
            input_ptr + n * stride_n + c * stride_c + s * stride_s,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        sum_x += tl.sum(x, 0)
        sum_x2 += tl.sum(x * x, 0)

    mean = sum_x / M
    var = tl.maximum(sum_x2 / M - mean * mean, 0.0)
    invstd = tl.rsqrt(var + eps)

    # Save statistics for backward
    tl.store(mean_ptr + c, mean)
    tl.store(invstd_ptr + c, invstd)

    # 2. Update running statistics
    old_mean = tl.load(running_mean_ptr + c).to(tl.float32)
    old_var = tl.load(running_var_ptr + c).to(tl.float32)

    # PyTorch uses unbiased variance for running_var
    unbiased_var = tl.where(
        M > 1,
        var * M / (M - 1),
        var,
    )

    new_mean = (1.0 - momentum) * old_mean + momentum * mean
    new_var = (1.0 - momentum) * old_var + momentum * unbiased_var

    tl.store(running_mean_ptr + c, new_mean)
    tl.store(running_var_ptr + c, new_var)

    # 3. Apply batch normalization
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)

    for start in range(tl.cdiv(M, BLOCK_SIZE)):
        idx = start * BLOCK_SIZE + offsets
        mask = idx < M

        n = idx // spatial_dim
        s = idx % spatial_dim

        ptr = input_ptr + n * stride_n + c * stride_c + s * stride_s

        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * invstd * gamma + beta

        tl.store(
            output_ptr + n * stride_n + c * stride_c + s * stride_s,
            y,
            mask=mask,
        )


@triton.jit
def batchnorm_forward_inference_kernel(
    # Pointers
    input_ptr,          # Input tensor pointer
    gamma_ptr,          # Scale parameter pointer
    beta_ptr,           # Shift parameter pointer
    mean_ptr,           # Running mean pointer
    var_ptr,            # Running variance pointer
    output_ptr,         # Output tensor pointer
    invstd_ptr,         # Saved inverse std pointer (for backward)
    # Dimensions
    N, C, spatial_dim,  # Batch size, channels, spatial dimension
    # Parameters
    eps,                # Epsilon for numerical stability
    # Strides
    stride_n, stride_c, stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    M = N * spatial_dim

    offsets = tl.arange(0, BLOCK_SIZE)

    # 1. Load running mean and variance
    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)

    # 2. Compute inverse standard deviation
    invstd = tl.rsqrt(var + eps)

    # Save inverse std for backward
    tl.store(invstd_ptr + c, invstd)

    # 3. Load scale and shift parameters
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)

    # 4. Normalize input and apply affine transformation
    for start in range(tl.cdiv(M, BLOCK_SIZE)):
        idx = start * BLOCK_SIZE + offsets
        mask = idx < M

        n = idx // spatial_dim
        s = idx % spatial_dim

        input_offset = (
            n * stride_n
            + c * stride_c
            + s * stride_s
        )

        x = tl.load(
            input_ptr + input_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        y = (x - mean) * invstd * gamma + beta

        tl.store(
            output_ptr + input_offset,
            y,
            mask=mask,
        )


# ops.py launches the backward with a fixed grid=(C,) and BLOCK_SIZE=1024, so
# each channel is reduced by a single program. With C close to the SM count
# that means one program per SM, and 4 warps (the default) cannot keep enough
# loads in flight to saturate DRAM. Only the warp count is tuned here; configs
# carry no meta-parameters so they never conflict with the caller's BLOCK_SIZE.
# The kernel only stores (no atomics), so re-running it while autotuning is safe.
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["N", "C", "spatial_dim"],
)
@triton.jit
def batchnorm_backward_kernel(
    # Pointers
    grad_output_ptr,    # Gradient from next layer
    input_ptr,          # Original input
    gamma_ptr,          # Scale parameter
    mean_ptr,           # Saved mean from forward
    invstd_ptr,         # Saved inverse std from forward
    grad_input_ptr,     # Gradient w.r.t. input (output)
    grad_gamma_ptr,     # Gradient w.r.t. gamma (output)
    grad_beta_ptr,      # Gradient w.r.t. beta (output)
    # Dimensions
    N, C, spatial_dim,
    # Strides
    stride_n, stride_c, stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per channel c. With M = N * spatial_dim and
    x_hat = (x - mean) * invstd:

        grad_beta  = sum(dy)
        grad_gamma = sum(dy * x_hat) = invstd * sum(dy * (x - mean))
        grad_input = gamma * invstd / M * (M * dy - sum(dy) - x_hat * sum(dy * x_hat))

    Pass 1 reduces the two sums, pass 2 writes grad_input.

    Compared to a flat loop over M with idx // spatial_dim:
      - The loop runs over (n, spatial block), so every block is a contiguous
        run of one (n, c) plane. No integer div/mod per element, and Triton can
        prove contiguity and issue vectorized 128-bit loads/stores.
      - Partial sums are kept per lane in a BLOCK_SIZE-wide register vector and
        reduced with tl.sum once at the end, instead of a cross-warp reduction
        (shared memory + barrier) on every iteration.
      - grad_input is folded into dx = k_dy * dy + k_x * x + k_0, two FMAs per
        element, with the per-channel constants computed once.
      - Pass 2 walks the channel in reverse, so it starts on the data pass 1
        touched last, which is the part most likely to still be in L2.
    """
    c = tl.program_id(0)
    M = N * spatial_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    num_s_blocks = tl.cdiv(spatial_dim, BLOCK_SIZE)
    c_offset = c.to(tl.int64) * stride_c

    # 1. Load saved statistics and scale parameter
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)

    # 2. First pass: per-lane partial sums of dy and dy * (x - mean)
    acc_dy = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc_dy_xmu = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for n in range(N):
        plane_offset = c_offset + n * stride_n
        for sb in range(num_s_blocks):
            s = sb * BLOCK_SIZE + offsets
            mask = s < spatial_dim
            offset = plane_offset + s * stride_s

            x = tl.load(input_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(grad_output_ptr + offset, mask=mask, other=0.0).to(tl.float32)

            acc_dy += dy
            # masked lanes have dy = 0, so (0 - mean) contributes nothing
            acc_dy_xmu += dy * (x - mean)

    sum_dy = tl.sum(acc_dy, 0)
    sum_dy_xhat = tl.sum(acc_dy_xmu, 0) * invstd

    tl.store(grad_beta_ptr + c, sum_dy)
    tl.store(grad_gamma_ptr + c, sum_dy_xhat)

    # 3. Per-channel coefficients for dx = k_dy * dy + k_x * x + k_0
    k_dy = gamma * invstd
    k_x = -k_dy * invstd * sum_dy_xhat / M
    k_0 = -k_dy * sum_dy / M - k_x * mean

    # 4. Second pass (reverse order): compute grad_input
    for i in range(N):
        n = N - 1 - i
        plane_offset = c_offset + n * stride_n
        for j in range(num_s_blocks):
            s = (num_s_blocks - 1 - j) * BLOCK_SIZE + offsets
            mask = s < spatial_dim
            offset = plane_offset + s * stride_s

            x = tl.load(input_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(grad_output_ptr + offset, mask=mask, other=0.0).to(tl.float32)

            dx = k_dy * dy + k_x * x + k_0

            tl.store(grad_input_ptr + offset, dx, mask=mask)
