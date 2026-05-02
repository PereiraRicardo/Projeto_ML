"""
Retreina apenas StockAlertClassifier + SalesPatternAnalyzer.
O DemandForecaster já salvo (demand_models.pkl) é preservado.
Use quando quiser ajustar classificação/segmentação sem retreinar previsão de demanda.
"""

import time
import numpy as np
from data.db_connector import fetch_product_features
from models.ml_models import StockAlertClassifier, SalesPatternAnalyzer
from data_augmentation import augment_features_for_classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, os

print("=" * 55)
print("  TREINAMENTO — Classificador + Segmentacao")
print("=" * 55)

print("\nBuscando features do DW...")
t0 = time.time()
features_df = fetch_product_features()
print(f"Produtos : {len(features_df):,} registros em {round(time.time()-t0, 1)} s")


# ════════════════════════════════════════════════════════
# 1. STOCKALERTCLASSIFIER + SMOTE + MIXUP
# ════════════════════════════════════════════════════════
print("\n--- StockAlertClassifier + SMOTE + Mixup ---")
t = time.time()

s      = StockAlertClassifier()
df_eng = s._engineer(features_df).dropna(subset=s.FEAT_COLS)
y_raw  = s._create_labels(df_eng)

dist = y_raw.value_counts()
print(f"Classes originais — NORMAL: {dist.get('NORMAL', 0)} "
      f"| BAIXO: {dist.get('BAIXO', 0)} "
      f"| CRITICO: {dist.get('CRITICO', 0)} "
      f"| INATIVO: {dist.get('INATIVO', 0)}")

X_scaled = s.scaler.fit_transform(df_eng[s.FEAT_COLS])
y_enc    = s.label_enc.fit_transform(y_raw)

X_aug, y_aug = augment_features_for_classifier(
    X_scaled, y_enc, s.label_enc,
    use_smote=True,
    use_mixup=True,
    mixup_samples=300,
)

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
print(f"Acuracia: {round(acc * 100, 2)} %  em {round(time.time()-t, 1)} s")
print("Resultado por classe:")
for classe in ["CRITICO", "BAIXO", "NORMAL", "INATIVO"]:
    if classe in report:
        prec = round(report[classe]["precision"] * 100, 1)
        rec  = round(report[classe]["recall"] * 100, 1)
        sup  = int(report[classe]["support"])
        print(f"   {classe:<8} — precision: {prec}%  recall: {rec}%  support: {sup}")


# ════════════════════════════════════════════════════════
# 2. SALESPATTERANALYZER — 6 clusters
# ════════════════════════════════════════════════════════
print("\n--- SalesPatternAnalyzer ---")
t   = time.time()
p   = SalesPatternAnalyzer(n_clusters=6)
seg = p.train(features_df)
for c in p.get_cluster_summary(seg):
    print(f"   {c['pattern']:<30} : {c['product_count']:>5} produtos | media {c['avg_monthly_qty']:.1f} un/mes")
print(f"Tempo: {round(time.time()-t, 1)} s")

print("\n" + "=" * 55)
print("  CONCLUIDO — stock_classifier.pkl + pattern_model.pkl salvos")
print("=" * 55)
