"""
Modelos de Machine Learning — adventureworks_dw
GPU: NVIDIA RTX 3060 (CUDA) via XGBoost
Correções:
  - DemandForecaster: projeta a partir do último mês real do DW (jun/2014)
  - StockAlertClassifier: labels baseados no histórico completo do DW
"""

import numpy as np
import pandas as pd
import joblib
import os
import torch

import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
XGB_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
print(f"[ML] Usando: {DEVICE.upper()} — "
      f"{torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════
# 1. PREVISÃO DE DEMANDA — XGBoost GPU
# ═══════════════════════════════════════════════

class DemandForecaster:
    """
    Prevê TotalQty dos próximos N meses por produto.
    A projeção parte do último mês real dos dados (ex: jun/2014),
    não da data atual do sistema.
    """

    FEAT_COLS = ["month", "quarter", "year",
                 "lag_1", "lag_2", "lag_3",
                 "roll_3", "roll_6", "trend"]

    def __init__(self, forecast_months: int = 3):
        self.forecast_months = forecast_months
        self.models: dict    = {}
        # Último mês real dos dados — definido durante o treino
        self.last_data_month: pd.Timestamp = None

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["SaleMonth"] = pd.to_datetime(df["SaleMonth"])
        df = df.sort_values("SaleMonth")
        df["month"]   = df["SaleMonth"].dt.month
        df["quarter"] = df["SaleMonth"].dt.quarter
        df["year"]    = df["SaleMonth"].dt.year
        df["lag_1"]   = df["TotalQty"].shift(1)
        df["lag_2"]   = df["TotalQty"].shift(2)
        df["lag_3"]   = df["TotalQty"].shift(3)
        df["roll_3"]  = df["TotalQty"].rolling(3).mean()
        df["roll_6"]  = df["TotalQty"].rolling(6).mean()
        df["trend"]   = range(len(df))
        return df.dropna()

    def _make_xgb(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators     = 300,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            device           = XGB_DEVICE,
            tree_method      = "hist",
            random_state     = 42,
            verbosity        = 0,
        )

    def train(self, sales_df: pd.DataFrame) -> dict:
        # ── Detecta o último mês real dos dados ──────────────
        self.last_data_month = pd.to_datetime(
            sales_df["SaleMonth"]
        ).max().replace(day=1)
        print(f"  Último mês dos dados: {self.last_data_month.strftime('%Y-%m')}")

        metrics  = {}
        produtos = sales_df["sk_produto"].unique()
        total    = len(produtos)

        for i, sk in enumerate(produtos, 1):
            prod_df = sales_df[sales_df["sk_produto"] == sk].copy()
            if len(prod_df) < 6:
                continue

            if PROPHET_AVAILABLE and len(prod_df) >= 12:
                pdf = prod_df[["SaleMonth", "TotalQty"]].rename(
                    columns={"SaleMonth": "ds", "TotalQty": "y"})
                m = Prophet(yearly_seasonality=True,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            seasonality_mode="multiplicative")
                m.fit(pdf)
                future = m.make_future_dataframe(periods=1, freq="MS")
                fc     = m.predict(future)
                mae    = mean_absolute_error(pdf["y"].tail(3), fc["yhat"].tail(3))
                self.models[sk] = ("prophet", m)
            else:
                feat_df = self._add_features(prod_df)
                X = feat_df[self.FEAT_COLS]
                y = feat_df["TotalQty"]
                split = max(1, int(len(X) * 0.8))
                X_tr, X_te = X.iloc[:split], X.iloc[split:]
                y_tr, y_te = y.iloc[:split], y.iloc[split:]
                m = self._make_xgb()
                m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
                mae = mean_absolute_error(y_te, m.predict(X_te)) if len(X_te) > 0 else 0
                # Salva última linha de features + último mês real do produto
                last_row = feat_df.iloc[-1].copy()
                last_row["_last_month"] = prod_df["SaleMonth"].max()
                self.models[sk] = ("xgb", m, last_row)

            metrics[int(sk)] = round(mae, 2)
            if i % 50 == 0:
                print(f"  [{i}/{total}] produtos treinados...")

        joblib.dump(
            {"models": self.models, "last_data_month": self.last_data_month},
            os.path.join(MODEL_DIR, "demand_models.pkl")
        )
        print(f"[DemandForecaster] {len(metrics)} modelos salvos.")
        return metrics

    def predict(self, sk_produto: int) -> list[dict]:
        if sk_produto not in self.models:
            return []
        entry = self.models[sk_produto]

        if entry[0] == "prophet":
            m = entry[1]
            future = m.make_future_dataframe(periods=self.forecast_months, freq="MS")
            fc = m.predict(future).tail(self.forecast_months)
            return [{"month":    str(r["ds"])[:7],
                     "forecast": max(0, round(r["yhat"])),
                     "lower":    max(0, round(r["yhat_lower"])),
                     "upper":    max(0, round(r["yhat_upper"]))}
                    for _, r in fc.iterrows()]

        m, last = entry[1], entry[2]

        # ── Base = último mês REAL do produto, não hoje ───────
        last_month = pd.to_datetime(last.get("_last_month", self.last_data_month))
        base = last_month.replace(day=1)

        results = []
        prev  = float(last["TotalQty"])
        lag2  = float(last.get("lag_2", prev))
        lag3  = float(last.get("lag_3", prev))
        roll3 = float(last["roll_3"])
        roll6 = float(last["roll_6"])
        trend = float(last["trend"]) + 1

        for i in range(1, self.forecast_months + 1):
            d = base + pd.DateOffset(months=i)
            X = pd.DataFrame([{
                "month":   d.month,
                "quarter": d.quarter,
                "year":    d.year,
                "lag_1":   prev,
                "lag_2":   lag2,
                "lag_3":   lag3,
                "roll_3":  roll3,
                "roll_6":  roll6,
                "trend":   trend + i,
            }])
            pred = max(0, round(m.predict(X)[0]))
            results.append({
                "month":    d.strftime("%Y-%m"),
                "forecast": pred,
                "lower":    max(0, round(pred * 0.82)),
                "upper":    round(pred * 1.18),
            })
            lag3, lag2, prev = lag2, prev, float(pred)
            roll3 = round((roll3 * 2 + pred) / 3, 1)
        return results

    def load(self):
        path = os.path.join(MODEL_DIR, "demand_models.pkl")
        if os.path.exists(path):
            data = joblib.load(path)
            if isinstance(data, dict) and "models" in data:
                self.models          = data["models"]
                self.last_data_month = data.get("last_data_month")
            else:
                self.models = data  # compatibilidade com versão antiga
            print(f"[DemandForecaster] {len(self.models)} modelos carregados.")


# ═══════════════════════════════════════════════
# 2. CLASSIFICADOR DE ESTOQUE — XGBoost GPU
# ═══════════════════════════════════════════════

class StockAlertClassifier:
    """
    Classifica produtos em NORMAL / BAIXO / CRITICO.
    Labels baseados no histórico COMPLETO do DW (não em janela recente).
    """

    FEAT_COLS = [
        "ListPrice", "StandardCost",
        "TotalQty12m", "TotalRevenue12m", "TotalProfit12m",
        "AvgMonthlyQty", "StdDevQty", "AvgMargin",
        "MonthsWithSales", "OrderCount",
        "TotalQty3m", "AvgMonthlyQty3m", "TrendPct",
        "CoverageMonths", "SalesConsistency",
    ]

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators     = 300,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            device           = XGB_DEVICE,
            tree_method      = "hist",
            eval_metric      = "mlogloss",
            random_state     = 42,
            verbosity        = 0,
        )
        self.scaler    = StandardScaler()
        self.label_enc = LabelEncoder()

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # CoverageMonths: quantos meses de demanda o estoque cobre
        df["CoverageMonths"] = df.apply(
            lambda r: min(r["TotalQty12m"] / (r["AvgMonthlyQty"] * 12), 24)
            if r["AvgMonthlyQty"] > 0 else 24, axis=1)
        # SalesConsistency: % dos meses com venda
        df["SalesConsistency"] = df["MonthsWithSales"] / 12.0
        return df

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Regras baseadas no volume total e tendência do DW.
        Usa percentis do próprio dataset para definir os thresholds,
        evitando que todos caiam na mesma classe.
        """
        p33 = df["TotalQty12m"].quantile(0.33)
        p66 = df["TotalQty12m"].quantile(0.66)

        def label(r):
            # Sem nenhuma venda no período → crítico
            if r["TotalQty12m"] == 0:
                return "CRITICO"
            # Queda forte na tendência recente → crítico
            if r["TrendPct"] < -40:
                return "CRITICO"
            # Volume baixo (tercil inferior) ou queda moderada → baixo
            if r["TotalQty12m"] <= p33 or r["TrendPct"] < -15:
                return "BAIXO"
            # Volume alto (tercil superior) → normal
            return "NORMAL"

        return df.apply(label, axis=1)

    def train(self, features_df: pd.DataFrame) -> dict:
        df    = self._engineer(features_df).dropna(subset=self.FEAT_COLS)
        y     = self._create_labels(df)

        # Log da distribuição de labels
        dist = y.value_counts()
        print(f"  Labels: NORMAL={dist.get('NORMAL',0)} | BAIXO={dist.get('BAIXO',0)} | CRITICO={dist.get('CRITICO',0)}")

        X     = self.scaler.fit_transform(df[self.FEAT_COLS])
        y_enc = self.label_enc.fit_transform(y)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_enc, test_size=0.2, random_state=42,
            stratify=y_enc if len(np.unique(y_enc)) > 1 else None)

        self.model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        report = classification_report(
            y_te, self.model.predict(X_te),
            target_names=self.label_enc.classes_,
            output_dict=True)

        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "encoder": self.label_enc},
            os.path.join(MODEL_DIR, "stock_classifier.pkl"))
        print("[StockAlertClassifier] Treinado com GPU e salvo.")
        return report

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df     = self._engineer(features_df).dropna(subset=self.FEAT_COLS)
        X      = self.scaler.transform(df[self.FEAT_COLS])
        labels = self.label_enc.inverse_transform(self.model.predict(X))
        probs  = self.model.predict_proba(X).max(axis=1)

        out = df[["sk_produto", "ProductID", "ProductName",
                  "Category", "AvgMonthlyQty", "TrendPct",
                  "SalesConsistency"]].copy()
        out["StockStatus"] = labels
        out["Confidence"]  = np.round(probs * 100, 1)
        priority = {"CRITICO": 0, "BAIXO": 1, "NORMAL": 2}
        out["_sort"] = out["StockStatus"].map(priority)
        return out.sort_values(["_sort", "TrendPct"]).drop("_sort", axis=1).reset_index(drop=True)

    def load(self):
        path = os.path.join(MODEL_DIR, "stock_classifier.pkl")
        if os.path.exists(path):
            d = joblib.load(path)
            self.model     = d["model"]
            self.scaler    = d["scaler"]
            self.label_enc = d["encoder"]
            print("[StockAlertClassifier] Modelos carregados.")


# ═══════════════════════════════════════════════
# 3. PADRÕES DE VENDA — KMeans CPU
# ═══════════════════════════════════════════════

class SalesPatternAnalyzer:
    """
    Segmenta produtos em 4 perfis via K-Means.
    504 produtos — KMeans CPU é instantâneo.
    """

    FEAT_COLS = [
        "TotalQty12m", "AvgMonthlyQty", "StdDevQty",
        "MonthsWithSales", "ListPrice", "TrendPct",
        "TotalRevenue12m", "AvgMargin",
    ]

    def __init__(self, n_clusters: int = 4):
        self.n_clusters  = n_clusters
        self.model       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler      = StandardScaler()
        self._label_map: dict = {}

    def train(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.dropna(subset=self.FEAT_COLS).copy()
        X  = self.scaler.fit_transform(df[self.FEAT_COLS])
        df["Cluster"] = self.model.fit_predict(X)

        cluster_vol = df.groupby("Cluster")["AvgMonthlyQty"].median().sort_values(ascending=False)
        labels_list = ["Alta Rotatividade", "Venda Sazonal", "Venda Estável", "Baixa Rotatividade"]
        self._label_map = {cid: labels_list[i] for i, cid in enumerate(cluster_vol.index)}
        df["Pattern"] = df["Cluster"].map(self._label_map)

        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "label_map": self._label_map},
            os.path.join(MODEL_DIR, "pattern_model.pkl"))
        print(f"[SalesPatternAnalyzer] {len(df)} produtos segmentados.")
        return df[["sk_produto", "ProductID", "ProductName", "Category",
                   "Subcategory", "AvgMonthlyQty", "StdDevQty",
                   "MonthsWithSales", "TrendPct", "AvgMargin",
                   "Cluster", "Pattern"]].reset_index(drop=True)

    def get_cluster_summary(self, segmented_df: pd.DataFrame) -> list[dict]:
        summary = []
        for pattern, g in segmented_df.groupby("Pattern"):
            summary.append({
                "pattern":         pattern,
                "product_count":   int(len(g)),
                "avg_monthly_qty": round(float(g["AvgMonthlyQty"].mean()), 1),
                "avg_variation":   round(float(g["StdDevQty"].mean()), 1),
                "avg_margin":      round(float(g["AvgMargin"].mean()), 2),
                "avg_trend_pct":   round(float(g["TrendPct"].mean()), 2),
            })
        return sorted(summary, key=lambda x: x["avg_monthly_qty"], reverse=True)

    def load(self):
        path = os.path.join(MODEL_DIR, "pattern_model.pkl")
        if os.path.exists(path):
            d = joblib.load(path)
            self.model      = d["model"]
            self.scaler     = d["scaler"]
            self._label_map = d["label_map"]
            print("[SalesPatternAnalyzer] Modelo carregado.")
