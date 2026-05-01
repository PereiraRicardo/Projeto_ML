from models.ml_models import DemandForecaster, StockAlertClassifier
from data.db_connector import fetch_product_features

f = DemandForecaster()
f.load()

produtos = list(f.models.keys())[:3]
for sk in produtos:
    preds = f.predict(sk)
    print("sk_" + str(sk), "->", preds)

print()
s = StockAlertClassifier()
s.load()
feat = fetch_product_features()
alertas = s.predict(feat)
criticos = alertas[alertas["StockStatus"] == "CRITICO"]
print("Produtos CRITICOS:", len(criticos))
print(criticos[["ProductName", "StockStatus", "Confidence"]].head(5).to_string())
