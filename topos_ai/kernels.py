import torch
import triton
import triton.language as tl

@triton.jit
def flash_topos_kernel_2d(
    q_ptr, k_ptr, out_ptr,
    M, N, D,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    batch_id = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    q_batch_base = q_ptr + batch_id * stride_qb
    k_batch_base = k_ptr + batch_id * stride_kb

    for d in range(D):
        q_ptrs = q_batch_base + offs_m * stride_qm + d * stride_qd
        k_ptrs = k_batch_base + offs_n * stride_kn + d * stride_kd

        q_val = tl.load(q_ptrs, mask=mask_m, other=0.0)
        k_val = tl.load(k_ptrs, mask=mask_n, other=0.0)

        impl = 1.0 - q_val[:, None] + k_val[None, :]
        impl = tl.minimum(impl, 1.0)
        acc += impl

    acc = acc / D
    out_ptrs = out_ptr + batch_id * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def flash_topos_attention(q, k):
    """O(N^2 * D) VRAM gereksinimini ortadan kaldıran, O(1) Memory kullanan donanım kerneli."""
    B, M, D = q.shape
    _, N, _ = k.shape
    out = torch.empty((B, M, N), device=q.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    flash_topos_kernel_2d[grid](
        q, k, out, M, N, D,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return out
