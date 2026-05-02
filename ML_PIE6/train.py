"""
Pipeline de treinamento — ELSONS_DW
Augmentation:
  - DemandForecaster : sem augmentation (36 meses reais por produto são suficientes)
  - StockAlertClassifier : SMOTE + Mixup (balanceamento de classes)
"""

import time
import numpy as np
from data.db_connector import fetch_sales_history, fetch_product_features
from models.ml_models import DemandForecaster, StockAlertClassifier, SalesPatternAnalyzer
from data_augmentation import augment_features_for_classifier

print("=" * 55)
print("  TREINAMENTO — ELSONS_DW")
print("=" * 55)

# ── Busca dados do DW ────────────────────────────────────
print("\nBuscando dados do DW...")
t0 = time.time()
sales_df    = fetch_sales_history(months=36)
features_df = fetch_product_features()
print(f"Vendas   : {len(sales_df):,} linhas")
print(f"Produtos : {len(features_df):,} registros")
print(f"Tempo carga: {round(time.time() - t0, 1)} s")


# ════════════════════════════════════════════════════════
# 1. DEMANDFORECASTER
# ════════════════════════════════════════════════════════
print("\n--- DemandForecaster ---")
print("(sem augmentation — dados reais suficientes com 36 meses)")
t = time.time()

f = DemandForecaster()
m = f.train(sales_df)
print(f"Produtos treinados: {len(m):,} em {round(time.time() - t, 1)} s")


# ════════════════════════════════════════════════════════
# 2. STOCKALERTCLASSIFIER + SMOTE + MIXUP
# ════════════════════════════════════════════════════════
print("\n--- StockAlertClassifier + SMOTE + Mixup ---")
t = time.time()

s      = StockAlertClassifier()
df_eng = s._engineer(features_df).dropna(subset=s.FEAT_COLS)
y_raw  = s._create_labels(df_eng)

dist = y_raw.value_counts()
print(f"Classes originais — NORMAL: {dist.get('NORMAL', 0)} "
      f"| BAIXO: {dist.get('BAIXO', 0)} "
      f"| CRITICO: {dist.get('CRITICO', 0)}")

X_scaled = s.scaler.fit_transform(df_eng[s.FEAT_COLS])
y_enc    = s.label_enc.fit_transform(y_raw)

X_aug, y_aug = augment_features_for_classifier(
    X_scaled, y_enc, s.label_enc,
    use_smote=True,
    use_mixup=True,
    mixup_samples=300,  # aumentado proporcionalmente ao dataset maior
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, os

X_tr, X_te, y_tr, y_te = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42,
    stratify=y_aug if len(np.unique(y_aug)) > 1 else None
)

s.model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

report = classification_report(
    y_te, s.model.predict(X_te),
    target_names=s.label_enc.classes_,
    output_dict=True
)

joblib.dump(
    {"model": s.model, "scaler": s.scaler, "encoder": s.label_enc},
    os.path.join("models", "saved", "stock_classifier.pkl")
)

acc = report["accuracy"]
print(f"Acuracia: {round(acc * 100, 2)} %  em {round(time.time() - t, 1)} s")
print("Resultado por classe:")
for classe in ["CRITICO", "BAIXO", "NORMAL"]:
    if classe in report:
        prec = round(report[classe]["precision"] * 100, 1)
        rec  = round(report[classe]["recall"] * 100, 1)
        sup  = int(report[classe]["support"])
        print(f"   {classe:<8} — precision: {prec}%  recall: {rec}%  support: {sup}")


# ════════════════════════════════════════════════════════
# 3. SALESPATTERANALYZER — 6 clusters para 3k+ produtos
# ════════════════════════════════════════════════════════
print("\n--- SalesPatternAnalyzer ---")
t = time.time()

p   = SalesPatternAnalyzer(n_clusters=6)
seg = p.train(features_df)
for c in p.get_cluster_summary(seg):
    nome  = c["pattern"]
    qtd   = c["product_count"]
    media = c["avg_monthly_qty"]
    print(f"   {nome:<25} : {qtd:>5} produtos | média {media:.1f} un/mês")
print(f"Tempo: {round(time.time() - t, 1)} s")

print("\n" + "=" * 55)
print("  CONCLUIDO — modelos salvos em models/saved/")
print("=" * 55)
