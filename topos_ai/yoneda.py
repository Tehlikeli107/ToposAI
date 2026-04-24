import torch
import torch.nn as nn
from topos_ai.logic import StrictGodelImplication

class YonedaUniverse(nn.Module):
    """
    [GERÇEK YONEDA UZAYI (Strict Asymmetric Yoneda)]
    
    Bir nesneyi (Object) anlamak için, ondan çıkan (Covariant) ve 
    ona gelen (Contravariant) okların bütününe bakılır. Öklid uzaklığı kullanılamaz.
    Bunun yerine Yönlü İçsel Hom (StrictGodelImplication) kullanılmalıdır.
    """

    def __init__(self, num_probes, dim):
        super().__init__()
        # Yoneda Probları, "Evrensel Referans Nesneleri" (Representable Functors) gibi çalışır.
        self.probes = nn.Parameter(torch.rand(num_probes, dim))

    def get_morphisms(self, X):
        """
        Return the Contravariant Hom-functor values: Hom(Probe, X).
        Eğer Probe <= X ise (Modus Ponens), ok gücü 1.0 olur. Değilse X'in gücü.
        (Öklid uzaklığı silinmiş, yönlü Asimetri kurulmuştur)
        """
        X_exp = X.unsqueeze(1)               # [Batch, 1, Dim]
        probes_exp = self.probes.unsqueeze(0) # [1, Probes, Dim]
        
        # Probe'dan X'e doğru "Katı Çıkarım" (Implication) gücü
        implication = StrictGodelImplication.apply(probes_exp, X_exp)
        
        # Her probe için boyut ortalaması (veya supremum/infimum) alarak nihai Yoneda skorunu çıkar
        return implication.mean(dim=-1)


class YonedaReconstructor(nn.Module):
    """
    Reconstruct coordinates whose probe-distance vector matches a target.

    The optimization variable is the coordinate vector itself, so this is an
    inverse-distance reconstruction baseline rather than a categorical inverse.
    """

    def __init__(self, num_probes, dim):
        super().__init__()
        self.estimated_X = nn.Parameter(torch.zeros(1, dim))

    def forward(self, true_morphisms, universe: YonedaUniverse):
        estimated_morphisms = universe.get_morphisms(self.estimated_X)
        loss = torch.nn.functional.mse_loss(estimated_morphisms, true_morphisms)
        return loss, self.estimated_X
