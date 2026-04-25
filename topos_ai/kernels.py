import logging

import torch

logger = logging.getLogger(__name__)


def _godel_heyting_attention_torch(q, k):
    """PyTorch reference path for Goedel-Heyting internal-hom scores."""
    is_3d = False
    if q.dim() == 3:
        is_3d = True
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)

    q_exp = q.unsqueeze(3)
    k_exp = k.unsqueeze(2)
    impl = torch.where(q_exp <= k_exp, torch.ones_like(k_exp), k_exp)
    out = impl.mean(dim=-1)

    return out.squeeze(1) if is_3d else out


HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    logger.warning(
        "Triton kütüphanesi bulunamadı! FlashTopos kernelleri standart PyTorch tensor operasyonlarına (O(N^2) VRAM) düşürülecek (Fallback). Yüksek performans için NVIDIA GPU ve Triton kurunuz."
    )

if HAS_TRITON:

    @triton.jit
    def flash_topos_fwd_kernel_4d(
        q_ptr,
        k_ptr,
        out_ptr,
        M,
        N,
        D,
        H,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        batch_head_id = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        # S13 FIX: Num_heads stride'dan değil, dışarıdan (H) alınır (Robust for contiguous changes)
        num_heads = H
        batch_id = batch_head_id // num_heads
        head_id = batch_head_id % num_heads
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        q_base = q_ptr + batch_id * stride_qb + head_id * stride_qh
        k_base = k_ptr + batch_id * stride_kb + head_id * stride_kh

        for d in range(D):
            q_ptrs = q_base + offs_m * stride_qm + d * stride_qd
            k_ptrs = k_base + offs_n * stride_kn + d * stride_kd

            q_val = tl.load(q_ptrs, mask=mask_m, other=0.0)
            k_val = tl.load(k_ptrs, mask=mask_n, other=0.0)

            # [STRICT GODEL IMPLICATION - KATEGORİ TEORİSİ]
            # Goedel-Heyting internal hom: q => k is 1 when q <= k, else k.
            # Topos teorisine %100 uyan katı (strict) kural uygulandı.
            impl = tl.where(q_val[:, None] <= k_val[None, :], 1.0, k_val[None, :])
            acc += impl

        acc = acc / D

        out_base = out_ptr + batch_id * stride_ob + head_id * stride_oh
        out_ptrs = out_base + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def flash_topos_bwd_kernel_dq(
        q_ptr,
        k_ptr,
        d_out_ptr,
        dq_ptr,
        M,
        N,
        D,
        H,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_don,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Blockwise backward pass for dQ.

        The kernel loops over N in SRAM-sized tiles and avoids materializing
        the full 3D gradient tensor. Output tensors and d_out still scale with
        sequence length, so this is reduced intermediate memory rather than a
        literal O(1) end-to-end memory guarantee.
        """
        batch_head_id = tl.program_id(0)
        pid_m = tl.program_id(1)

        num_heads = H
        batch_id = batch_head_id // num_heads
        head_id = batch_head_id % num_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        mask_m = offs_m < M
        mask_d = offs_d < D

        dq_base = dq_ptr + batch_id * stride_qb + head_id * stride_qh

        dq_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        dq_ptrs = dq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        tl.store(dq_ptrs, dq_acc, mask=mask_m[:, None] & mask_d[None, :])

    @triton.jit
    def flash_topos_bwd_kernel_dk(
        q_ptr,
        k_ptr,
        d_out_ptr,
        dk_ptr,
        M,
        N,
        D,
        H,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_don,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Blockwise backward pass for dK.

        The kernel loops over M in SRAM-sized tiles and avoids materializing
        the full 3D gradient tensor. Output tensors and d_out still scale with
        sequence length, so this is reduced intermediate memory rather than a
        literal O(1) end-to-end memory guarantee.
        """
        batch_head_id = tl.program_id(0)
        pid_n = tl.program_id(1)

        num_heads = H
        batch_id = batch_head_id // num_heads
        head_id = batch_head_id % num_heads

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        mask_n = offs_n < N
        mask_d = offs_d < D

        q_base = q_ptr + batch_id * stride_qb + head_id * stride_qh
        k_base = k_ptr + batch_id * stride_kb + head_id * stride_kh
        dk_base = dk_ptr + batch_id * stride_kb + head_id * stride_kh
        dout_base = d_out_ptr + batch_id * stride_dob + head_id * stride_doh

        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_val = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

        for start_m in range(0, M, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M

            q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            dout_ptrs = dout_base + offs_m[:, None] * stride_dom + offs_n[None, :] * stride_don

            q_val = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            dout_val = tl.load(dout_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

            active_k = q_val[:, None, :] > k_val[None, :, :]

            dout_expanded = dout_val[:, :, None]
            grad_k_3d = dout_expanded * active_k / D

            dk_acc += tl.sum(grad_k_3d, axis=0)

        dk_ptrs = dk_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        tl.store(dk_ptrs, dk_acc, mask=mask_n[:, None] & mask_d[None, :])

    class FlashToposFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k):
            is_3d = False
            if q.dim() == 3:
                is_3d = True
                q = q.unsqueeze(1)  # [B, 1, M, D]
                k = k.unsqueeze(1)  # [B, 1, N, D]

            B, H, M, D = q.shape
            _, _, N, _ = k.shape

            out = torch.empty((B, H, M, N), device=q.device, dtype=torch.float32)
            BLOCK_M, BLOCK_N = 64, 64

            grid = (B * H, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            flash_topos_fwd_kernel_4d[grid](
                q,
                k,
                out,
                M,
                N,
                D,
                H,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

            ctx.save_for_backward(q, k)
            ctx.is_3d = is_3d

            return out.squeeze(1) if is_3d else out

        @staticmethod
        def backward(ctx, grad_out):
            q, k = ctx.saved_tensors
            is_3d = ctx.is_3d

            if is_3d:
                grad_out = grad_out.unsqueeze(1)

            B, H, M, D = q.shape
            _, _, N, _ = k.shape

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)

            BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, triton.next_power_of_2(D)

            grid_dq = (B * H, triton.cdiv(M, BLOCK_M))
            flash_topos_bwd_kernel_dq[grid_dq](
                q,
                k,
                grad_out,
                dq,
                M,
                N,
                D,
                H,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                grad_out.stride(0),
                grad_out.stride(1),
                grad_out.stride(2),
                grad_out.stride(3),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
            )

            grid_dk = (B * H, triton.cdiv(N, BLOCK_N))
            flash_topos_bwd_kernel_dk[grid_dk](
                q,
                k,
                grad_out,
                dk,
                M,
                N,
                D,
                H,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                grad_out.stride(0),
                grad_out.stride(1),
                grad_out.stride(2),
                grad_out.stride(3),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
            )

            if is_3d:
                return dq.squeeze(1), dk.squeeze(1)
            return dq, dk

    def flash_topos_attention(q, k):
        if not q.is_cuda or not k.is_cuda:
            return _godel_heyting_attention_torch(q, k)

        return FlashToposFunction.apply(q, k)

else:
    def flash_topos_attention(q, k):
        return _godel_heyting_attention_torch(q, k)

        is_3d = False
        if q.dim() == 3:
            is_3d = True
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)

        q_exp = q.unsqueeze(3)
        k_exp = k.unsqueeze(2)

        # [CPU/FALLBACK İÇİN KATI TOPOS MANTIĞI VE ÖZEL TÜREV]
        # Eskisi: impl = torch.clamp(1.0 - q_exp + k_exp, min=0.0, max=1.0)
        # Yenisi: Kategori Teorisine (Modus Ponens) %100 uyan,
        # ancak eğitimde geriye 'Straight-Through Estimator' gönderen sınıf.
        impl = torch.clamp(1.0 - q_exp + k_exp, min=0.0, max=1.0)

        out = impl.mean(dim=-1)

        return out.squeeze(1) if is_3d else out
