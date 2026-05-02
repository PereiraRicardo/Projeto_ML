from data.db_connector import fetch_product_features
from models.ml_models import StockAlertClassifier, SalesPatternAnalyzer
import os, joblib

# Remove modelos antigos do disco para forçar re-treino limpo
for f in ["stock_classifier.pkl"]:
    path = os.path.join("models", "saved", f)
    if os.path.exists(path):
        os.remove(path)
        print("Removido:", path)

features_df = fetch_product_features()

print("\n--- Distribuicao de TotalQty12m ---")
print(features_df["TotalQty12m"].describe())
print("Zeros:", (features_df["TotalQty12m"] == 0).sum(), "de", len(features_df))

print("\n--- Treinando StockAlertClassifier ---")
s = StockAlertClassifier()
r = s.train(features_df)
acc = r["accuracy"]
print("Acuracia:", round(acc * 100, 2), "%")

print("\n--- Resultado por classe ---")
for classe in ["CRITICO", "BAIXO", "NORMAL"]:
    if classe in r:
        prec = round(r[classe]["precision"] * 100, 1)
        rec  = round(r[classe]["recall"] * 100, 1)
        sup  = int(r[classe]["support"])
        print(f"  {classe}: precision={prec}% recall={rec}% support={sup}")

print("\n--- Predicao final ---")
alertas  = s.predict(features_df)
criticos = alertas[alertas["StockStatus"] == "CRITICO"]
baixos   = alertas[alertas["StockStatus"] == "BAIXO"]
normais  = alertas[alertas["StockStatus"] == "NORMAL"]
print("CRITICO:", len(criticos))
print("BAIXO:  ", len(baixos))
print("NORMAL: ", len(normais))

print("\n--- Top 5 CRITICOS ---")
print(criticos[["ProductName", "AvgMonthlyQty", "TrendPct", "Confidence"]].head(5).to_string())

print("\n--- Top 5 NORMAIS ---")
print(normais[["ProductName", "AvgMonthlyQty", "TrendPct", "Confidence"]].head(5).to_string())
