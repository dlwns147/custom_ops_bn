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
    # TODO: Implement this kernel
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
    # TODO: Implement this kernel
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
    # TODO: Implement this kernel
        c = tl.program_id(0)
        M = N * spatial_dim

        offsets = tl.arange(0, BLOCK_SIZE)

        # 1. Load saved statistics and scale parameter
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        gamma = tl.load(gamma_ptr + c).to(tl.float32)

        # 2. First pass: compute grad_beta and grad_gamma
        sum_dy = tl.full((), 0.0, tl.float32)
        sum_dy_xhat = tl.full((), 0.0, tl.float32)

        for start in range(tl.cdiv(M, BLOCK_SIZE)):
            idx = start * BLOCK_SIZE + offsets
            mask = idx < M

            n = idx // spatial_dim
            s = idx % spatial_dim

            offset = (
                n * stride_n
                + c * stride_c
                + s * stride_s
            )

            x = tl.load(
                input_ptr + offset,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            dy = tl.load(
                grad_output_ptr + offset,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            xhat = (x - mean) * invstd

            sum_dy += tl.sum(dy, 0)
            sum_dy_xhat += tl.sum(dy * xhat, 0)

        tl.store(grad_beta_ptr + c, sum_dy)
        tl.store(grad_gamma_ptr + c, sum_dy_xhat)

        # 3. Second pass: compute grad_input
        for start in range(tl.cdiv(M, BLOCK_SIZE)):
            idx = start * BLOCK_SIZE + offsets
            mask = idx < M

            n = idx // spatial_dim
            s = idx % spatial_dim

            offset = (
                n * stride_n
                + c * stride_c
                + s * stride_s
            )

            x = tl.load(
                input_ptr + offset,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            dy = tl.load(
                grad_output_ptr + offset,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            xhat = (x - mean) * invstd

            dx = (
                gamma * invstd / M
                * (
                    M * dy
                    - sum_dy
                    - xhat * sum_dy_xhat
                )
            )

            tl.store(
                grad_input_ptr + offset,
                dx,
                mask=mask,
            )

