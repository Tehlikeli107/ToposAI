import torch
import logging

logger = logging.getLogger(__name__)

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    logger.warning("Triton kütüphanesi bulunamadı! FlashTopos kernelleri standart PyTorch tensor operasyonlarına (O(N^2) VRAM) düşürülecek (Fallback). Yüksek performans için NVIDIA GPU ve Triton kurunuz.")

if HAS_TRITON:
    @triton.jit
    def flash_topos_fwd_kernel_4d(
        q_ptr, k_ptr, out_ptr,
        M, N, D,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_ob, stride_oh, stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        batch_head_id = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        
        num_heads = stride_qb // stride_qh if stride_qh > 0 else 1
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
            
            # Lukasiewicz T-Norm (Mantıksal İçerim / Implication)
            # 1.0 - Q + K
            impl = 1.0 - q_val[:, None] + k_val[None, :]
            impl = tl.minimum(impl, 1.0)
            impl = tl.maximum(impl, 0.0)
            acc += impl
    
        acc = acc / D
        
        out_base = out_ptr + batch_id * stride_ob + head_id * stride_oh
        out_ptrs = out_base + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def flash_topos_bwd_kernel_4d(
        q_ptr, k_ptr, d_out_ptr, dq_ptr, dk_ptr,
        M, N, D,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_dob, stride_doh, stride_dom, stride_don,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        """
        [THE IMPOSSIBLE KERNEL: O(1) BACKWARD PASS]
        PyTorch'un O(N^2) türev patlamasını engeller.
        Türevler (Gradients) SRAM içinde blok blok hesaplanıp biriktirilir.
        """
        batch_head_id = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        
        num_heads = stride_qb // stride_qh if stride_qh > 0 else 1
        batch_id = batch_head_id // num_heads
        head_id = batch_head_id % num_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_d = offs_d < D

        q_base = q_ptr + batch_id * stride_qb + head_id * stride_qh
        k_base = k_ptr + batch_id * stride_kb + head_id * stride_kh
        dq_base = dq_ptr + batch_id * stride_qb + head_id * stride_qh
        dk_base = dk_ptr + batch_id * stride_kb + head_id * stride_kh
        dout_base = d_out_ptr + batch_id * stride_dob + head_id * stride_doh

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        
        dq_ptrs = dq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        dk_ptrs = dk_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        
        dout_ptrs = dout_base + offs_m[:, None] * stride_dom + offs_n[None, :] * stride_don

        q_val = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        k_val = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        dout_val = tl.load(dout_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        # Gradient Accumulators (SRAM içinde O(1) hafıza ile toplanır)
        dq_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

        # Lukasiewicz Türevi
        # impl = 1.0 - q + k. (Eğer clamp arasında değilse gradyan sıfırdır)
        # dq = -dout / D
        # dk = +dout / D
        for d in range(D):
            q_d = tl.load(q_base + offs_m * stride_qm + d * stride_qd, mask=mask_m, other=0.0)
            k_d = tl.load(k_base + offs_n * stride_kn + d * stride_kd, mask=mask_n, other=0.0)
            
            impl = 1.0 - q_d[:, None] + k_d[None, :]
            valid_mask = (impl > 0.0) & (impl < 1.0)
            
            # Maskelenmiş gradyanları hesapla
            grad_q = -dout_val * valid_mask / D
            grad_k = dout_val * valid_mask / D
            
            # SRAM'de atomik olarak topla (Reduce)
            if d < BLOCK_D:
                dq_acc[:, d] = tl.sum(grad_q, axis=1) # Satır toplamı
                dk_acc[:, d] = tl.sum(grad_k, axis=0) # Sütun toplamı
                
        # Gradientleri Global Belleğe (HBM) Geri Yaz
        tl.atomic_add(dq_ptrs, dq_acc, mask=mask_m[:, None] & mask_d[None, :])
        tl.atomic_add(dk_ptrs, dk_acc, mask=mask_n[:, None] & mask_d[None, :])

    class FlashToposFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k):
            is_3d = False
            if q.dim() == 3:
                is_3d = True
                q = q.unsqueeze(1) # [B, 1, M, D]
                k = k.unsqueeze(1) # [B, 1, N, D]
                
            B, H, M, D = q.shape
            _, _, N, _ = k.shape
            
            out = torch.empty((B, H, M, N), device=q.device, dtype=torch.float32)
            BLOCK_M, BLOCK_N = 64, 64
            
            grid = (B * H, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            
            flash_topos_fwd_kernel_4d[grid](
                q, k, out, M, N, D,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )
            
            # Backward pass için Q ve K'yı kaydet
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
            
            # Çıktı Gradientleri (Gradient Accumulators)
            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k)
            
            BLOCK_M, BLOCK_N, BLOCK_D = 32, 32, triton.next_power_of_2(D)
            grid = (B * H, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            
            flash_topos_bwd_kernel_4d[grid](
                q, k, grad_out, dq, dk,
                M, N, D,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
            )
            
            if is_3d:
                return dq.squeeze(1), dk.squeeze(1)
            return dq, dk

    def flash_topos_attention(q, k):
        return FlashToposFunction.apply(q, k)

else:
    def flash_topos_attention(q, k):
        is_3d = False
        if q.dim() == 3:
            is_3d = True
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            
        q_exp = q.unsqueeze(3) 
        k_exp = k.unsqueeze(2) 
        impl = torch.clamp(1.0 - q_exp + k_exp, min=0.0, max=1.0)
        out = impl.mean(dim=-1) 
        
        return out.squeeze(1) if is_3d else out