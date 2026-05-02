"""
Módulo de Data Augmentation — ML de Vendas e Estoque
Técnicas implementadas:
  1. Ruído Gaussiano   — robustez do DemandForecaster
  2. SMOTE             — balanceamento do StockAlertClassifier
  3. Mixup             — generalização nas fronteiras de decisão
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


# ═══════════════════════════════════════════════
# 1. RUÍDO GAUSSIANO — para séries temporais
# ═══════════════════════════════════════════════

class GaussianNoiseAugmenter:
    """
    Adiciona perturbações gaussianas às séries de venda.
    Gera N cópias sintéticas de cada série com pequenas variações,
    tornando o DemandForecaster mais robusto a ruídos de mercado.

    Parâmetros:
        noise_std_pct: desvio padrão do ruído como % da média da série (default 5%)
        n_copies: quantas cópias ruidosas gerar por produto (default 2)
        seed: semente aleatória para reprodutibilidade
    """

    def __init__(self, noise_std_pct: float = 0.05, n_copies: int = 2, seed: int = 42):
        self.noise_std_pct = noise_std_pct
        self.n_copies      = n_copies
        self.rng           = np.random.default_rng(seed)

    def augment(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Recebe o DataFrame de fetch_sales_history() e retorna
        o original + cópias com ruído gaussiano.

        Returns:
            DataFrame com (1 + n_copies) vezes mais linhas.
        """
        original  = sales_df.copy()
        augmented = [original]

        for copy_idx in range(1, self.n_copies + 1):
            noisy = original.copy()

            # Aplica ruído proporcional à média de cada produto
            for sk, group in original.groupby("sk_produto"):
                mean_qty = group["TotalQty"].mean()
                if mean_qty <= 0:
                    continue
                std  = mean_qty * self.noise_std_pct
                idx  = noisy["sk_produto"] == sk
                noise = self.rng.normal(0, std, size=idx.sum())
                noisy.loc[idx, "TotalQty"] = (
                    noisy.loc[idx, "TotalQty"] + noise
                ).clip(lower=0).round()

                # Aplica ruído proporcional também na receita
                mean_rev = group["TotalRevenue"].mean()
                if mean_rev > 0:
                    std_rev = mean_rev * self.noise_std_pct
                    noise_rev = self.rng.normal(0, std_rev, size=idx.sum())
                    noisy.loc[idx, "TotalRevenue"] = (
                        noisy.loc[idx, "TotalRevenue"] + noise_rev
                    ).clip(lower=0)

            # Marca cópias para identificação (opcional)
            noisy["_augmented"] = copy_idx
            augmented.append(noisy)

        result = pd.concat(augmented, ignore_index=True)
        # Remove coluna auxiliar se existir no original
        if "_augmented" not in sales_df.columns:
            result = result.drop(columns=["_augmented"], errors="ignore")

        print(f"[GaussianNoise] {len(sales_df)} → {len(result)} linhas "
              f"({self.n_copies} cópias, std={self.noise_std_pct*100:.1f}%)")
        return result


# ═══════════════════════════════════════════════
# 2. SMOTE — para classificador de estoque
# ═══════════════════════════════════════════════

class SmoteAugmenter:
    """
    Aplica SMOTE para balancear as classes NORMAL / BAIXO / CRITICO
    antes de treinar o StockAlertClassifier.

    O dataset tem desbalanceamento severo:
        CRITICO: 266  NORMAL: 232  BAIXO: 6

    SMOTE cria amostras sintéticas da classe BAIXO interpolando
    entre vizinhos reais, sem duplicação simples.

    Parâmetros:
        k_neighbors: vizinhos para interpolação (default 3 — seguro para classes pequenas)
        seed: semente aleatória
    """

    def __init__(self, k_neighbors: int = 3, seed: int = 42):
        self.k_neighbors = k_neighbors
        self.seed        = seed

    def augment(self, X: np.ndarray, y: np.ndarray,
                label_enc: LabelEncoder) -> tuple[np.ndarray, np.ndarray]:
        """
        Recebe features e labels já codificados e retorna
        versão balanceada com SMOTE.

        Args:
            X: array de features (já escalado)
            y: array de labels codificados (int)
            label_enc: LabelEncoder para mostrar distribuição

        Returns:
            X_res, y_res — arrays balanceados
        """
        # Conta antes
        unique, counts = np.unique(y, return_counts=True)
        before = dict(zip(label_enc.inverse_transform(unique), counts))
        print(f"[SMOTE] Antes:  {before}")

        # k_neighbors deve ser < menor classe
        min_count = counts.min()
        k = min(self.k_neighbors, max(1, min_count - 1))

        smote = SMOTE(k_neighbors=k, random_state=self.seed)
        X_res, y_res = smote.fit_resample(X, y)

        # Conta depois
        unique2, counts2 = np.unique(y_res, return_counts=True)
        after = dict(zip(label_enc.inverse_transform(unique2), counts2))
        print(f"[SMOTE] Depois: {after}")
        print(f"[SMOTE] {len(X)} -> {len(X_res)} amostras")

        return X_res, y_res


# ═══════════════════════════════════════════════
# 3. MIXUP — para generalização nas fronteiras
# ═══════════════════════════════════════════════

class MixupAugmenter:
    """
    Mixup cria amostras sintéticas interpolando pares de exemplos reais:
        x_mix = λ·x_i + (1-λ)·x_j
        y_mix = λ·y_i + (1-λ)·y_j  (soft labels)

    Para o XGBoost (que não aceita soft labels nativamente),
    usamos hard labels arredondados — o benefício vem da
    diversidade das features interpoladas.

    Parâmetros:
        alpha: parâmetro da distribuição Beta (default 0.2 — mixup suave)
        n_samples: quantas amostras sintéticas gerar
        seed: semente aleatória
    """

    def __init__(self, alpha: float = 0.2, n_samples: int = 100, seed: int = 42):
        self.alpha     = alpha
        self.n_samples = n_samples
        self.rng       = np.random.default_rng(seed)

    def augment(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera amostras mixup e concatena com os dados originais.

        Returns:
            X_aug, y_aug — originais + sintéticos
        """
        n = len(X)
        X_mix_list = []
        y_mix_list = []

        for _ in range(self.n_samples):
            # Sorteia lambda da distribuição Beta
            lam = self.rng.beta(self.alpha, self.alpha)

            # Sorteia par de índices
            i = self.rng.integers(0, n)
            j = self.rng.integers(0, n)

            x_mix = lam * X[i] + (1 - lam) * X[j]
            # Hard label: classe do exemplo com maior peso
            y_mix = y[i] if lam >= 0.5 else y[j]

            X_mix_list.append(x_mix)
            y_mix_list.append(y_mix)

        X_aug = np.vstack([X, np.array(X_mix_list)])
        y_aug = np.concatenate([y, np.array(y_mix_list)])

        print(f"[Mixup] {n} -> {len(X_aug)} amostras (alpha={self.alpha})")
        return X_aug, y_aug


# ═══════════════════════════════════════════════
# PIPELINE COMPLETO — aplica todas as técnicas
# ═══════════════════════════════════════════════

def augment_sales_for_forecaster(sales_df: pd.DataFrame,
                                  noise_std_pct: float = 0.05,
                                  n_copies: int = 2) -> pd.DataFrame:
    """
    Pipeline de augmentation para o DemandForecaster.
    Aplica Ruído Gaussiano nas séries temporais de venda.

    Args:
        sales_df: retorno de fetch_sales_history()
        noise_std_pct: % de ruído (default 5%)
        n_copies: cópias por produto (default 2)

    Returns:
        DataFrame ampliado com séries ruidosas
    """
    aug = GaussianNoiseAugmenter(noise_std_pct=noise_std_pct, n_copies=n_copies)
    return aug.augment(sales_df)


def augment_features_for_classifier(X: np.ndarray, y: np.ndarray,
                                     label_enc: LabelEncoder,
                                     use_smote: bool = True,
                                     use_mixup: bool = True,
                                     mixup_samples: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """
    Pipeline de augmentation para o StockAlertClassifier.
    Aplica SMOTE + Mixup nas features de produto.

    Args:
        X: features escaladas
        y: labels codificados
        label_enc: encoder para log
        use_smote: aplicar SMOTE (default True)
        use_mixup: aplicar Mixup (default True)
        mixup_samples: amostras mixup a gerar (default 150)

    Returns:
        X_aug, y_aug — prontos para treinar o XGBClassifier
    """
    X_aug, y_aug = X.copy(), y.copy()

    if use_smote:
        smote = SmoteAugmenter(k_neighbors=3)
        X_aug, y_aug = smote.augment(X_aug, y_aug, label_enc)

    if use_mixup:
        mixup = MixupAugmenter(alpha=0.2, n_samples=mixup_samples)
        X_aug, y_aug = mixup.augment(X_aug, y_aug)

    return X_aug, y_aug
