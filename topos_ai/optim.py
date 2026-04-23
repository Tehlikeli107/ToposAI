import torch
import math
from torch.optim import Optimizer

# =====================================================================
# TOPOS ADAM (TOPOLOGICAL NATURAL GRADIENT DESCENT)
# Problem: AdamW, uzayın düz (Euclidean) olduğunu varsayar. Ancak ToposAI
# (Kategori Teorisi) [0, 1] olasılık sınırlarına sahip, bükülü (Curved) 
# bir Riemannian Manifold'udur (Information Geometry).
# Çözüm: ToposAdam, Fisher Bilgi Matrisini (Fisher Information Metric) 
# kullanarak klasik eğimleri (Gradients) Doğal Eğimlere (Natural Gradients)
# çevirir. [0, 1] uç noktalarındaki "Vanishing Gradient" (Ölü türev) 
# sorununu çözer ve Kategori oklarını çok daha hızlı optimize eder.
# =====================================================================

class ToposAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, topological_weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Geçersiz learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Geçersiz epsilon değeri: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Geçersiz beta parametresi: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Geçersiz beta parametresi: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, top_wd=topological_weight_decay)
        super(ToposAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Klasik Adam Momentumları
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # [TOPOLOGICAL FISHER INFORMATION SCALING]
                # Ağırlıkların (w) Topos uzayındaki karşılığı p = sigmoid(w)
                # Fisher Information Metric bağımsız Bernoulli'ler için: p * (1 - p)
                # Düzeltilmiş (Natural) Gradient, klasik gradientin Fisher Metriğine bölünmesidir.
                
                # Mevcut ağırlığın olasılık uzayındaki izdüşümü
                p_val = torch.sigmoid(p)
                
                # p*(1-p) değeri sınırlar (0 veya 1) yaklaştıkça küçülür (Gradyan ölür).
                # Biz bunu tersine çevirerek (Bölerek) o ölü noktaları canlandırıyoruz (Curved Geometry)
                fisher_metric = (p_val * (1.0 - p_val)).clamp(min=1e-4) # Sıfıra bölmeyi engelle
                
                # Natural Update
                step_size = group['lr'] / bias_correction1
                natural_update = (exp_avg / denom) / fisher_metric

                # Topological Weight Decay (Ağırlıkları yavaşça 0'a yani p=0.5 'Maksimum Belirsizliğe' çeker)
                if group['top_wd'] > 0.0:
                    p.mul_(1 - group['lr'] * group['top_wd'])

                # Doğal Eğimi (Natural Gradient) uygula
                p.add_(-step_size * natural_update)

        return loss
