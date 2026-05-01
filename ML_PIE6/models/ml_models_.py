"""
Modelos de Machine Learning — adventureworks_dw
Usa as colunas exatas das tabelas: fato_vendas, dim_produto, dim_tempo, etc.
Requer: pip install scikit-learn pandas numpy joblib
Opcional: pip install prophet
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════
# 1. PREVISÃO DE DEMANDA
# ═══════════════════════════════════════════════

class DemandForecaster:
    """
    Prevê TotalQty dos próximos N meses por produto (sk_produto).
    Input: DataFrame de fetch_sales_history()
    """

    def __init__(self, forecast_months: int = 3):
        self.forecast_months = forecast_months
        self.models: dict = {}   # { sk_produto: ("prophet"|"gbr", model, ...) }

    # ── Helpers ──────────────────────────────────

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features temporais para o fallback GradientBoosting."""
        df = df.copy()
        df["SaleMonth"] = pd.to_datetime(df["SaleMonth"])
        df = df.sort_values("SaleMonth")
        df["month"]     = df["SaleMonth"].dt.month
        df["quarter"]   = df["SaleMonth"].dt.quarter
        df["year"]      = df["SaleMonth"].dt.year
        df["lag_1"]     = df["TotalQty"].shift(1)
        df["lag_2"]     = df["TotalQty"].shift(2)
        df["lag_3"]     = df["TotalQty"].shift(3)
        df["roll_3"]    = df["TotalQty"].rolling(3).mean()
        df["roll_6"]    = df["TotalQty"].rolling(6).mean()
        return df.dropna()

    FEAT_COLS_GBR = ["month", "quarter", "year", "lag_1", "lag_2", "lag_3", "roll_3", "roll_6"]

    # ── Treinamento ───────────────────────────────

    def train(self, sales_df: pd.DataFrame) -> dict:
        """
        Treina um modelo por produto. Exige >= 6 meses de histórico.

        Args:
            sales_df: retorno de fetch_sales_history()
        Returns:
            dict {sk_produto: MAE}
        """
        metrics = {}

        for sk_prod, group in sales_df.groupby("sk_produto"):
            prod_df = group.sort_values("SaleMonth").copy()
            if len(prod_df) < 6:
                continue

            if PROPHET_AVAILABLE:
                prophet_df = prod_df[["SaleMonth", "TotalQty"]].rename(
                    columns={"SaleMonth": "ds", "TotalQty": "y"}
                )
                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="multiplicative",
                )
                m.fit(prophet_df)
                future   = m.make_future_dataframe(periods=1, freq="MS")
                forecast = m.predict(future)
                mae = mean_absolute_error(
                    prophet_df["y"].tail(3),
                    forecast["yhat"].tail(3),
                )
                self.models[sk_prod] = ("prophet", m)

            else:
                feat_df = self._add_time_features(prod_df)
                X = feat_df[self.FEAT_COLS_GBR]
                y = feat_df["TotalQty"]
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, shuffle=False
                )
                m = GradientBoostingRegressor(n_estimators=150, random_state=42)
                m.fit(X_tr, y_tr)
                mae = mean_absolute_error(y_te, m.predict(X_te))
                # Guarda últimas features para inferência iterativa
                self.models[sk_prod] = ("gbr", m, feat_df.iloc[-1].copy())

            metrics[int(sk_prod)] = round(mae, 2)

        joblib.dump(self.models, os.path.join(MODEL_DIR, "demand_models.pkl"))
        print(f"[DemandForecaster] {len(metrics)} produtos treinados.")
        return metrics

    # ── Inferência ────────────────────────────────

    def predict(self, sk_produto: int) -> list[dict]:
        """
        Retorna previsão mensal para os próximos self.forecast_months meses.

        Returns:
            [{month, forecast, lower, upper}]
        """
        if sk_produto not in self.models:
            return []

        entry = self.models[sk_produto]

        if entry[0] == "prophet":
            m = entry[1]
            future = m.make_future_dataframe(
                periods=self.forecast_months, freq="MS"
            )
            fc = m.predict(future).tail(self.forecast_months)
            return [
                {
                    "month":    str(r["ds"])[:7],
                    "forecast": max(0, round(r["yhat"])),
                    "lower":    max(0, round(r["yhat_lower"])),
                    "upper":    max(0, round(r["yhat_upper"])),
                }
                for _, r in fc.iterrows()
            ]

        else:  # GBR
            m, last = entry[1], entry[2]
            results = []
            base = pd.Timestamp.now().replace(day=1)
            prev_qty  = float(last["TotalQty"])
            roll_3    = float(last["roll_3"])
            roll_6    = float(last["roll_6"])
            lag2      = float(last["lag_2"])
            lag3      = float(last["lag_3"])

            for i in range(1, self.forecast_months + 1):
                d = base + pd.DateOffset(months=i)
                X = pd.DataFrame([{
                    "month":   d.month,
                    "quarter": d.quarter,
                    "year":    d.year,
                    "lag_1":   prev_qty,
                    "lag_2":   lag2,
                    "lag_3":   lag3,
                    "roll_3":  roll_3,
                    "roll_6":  roll_6,
                }])
                pred = max(0, round(m.predict(X)[0]))
                results.append({
                    "month":    d.strftime("%Y-%m"),
                    "forecast": pred,
                    "lower":    max(0, round(pred * 0.82)),
                    "upper":    round(pred * 1.18),
                })
                # desloca lags para próxima iteração
                lag3, lag2, prev_qty = lag2, prev_qty, float(pred)
                roll_3 = round((roll_3 * 2 + pred) / 3, 1)

            return results

    def load(self):
        path = os.path.join(MODEL_DIR, "demand_models.pkl")
        if os.path.exists(path):
            self.models = joblib.load(path)
            print(f"[DemandForecaster] {len(self.models)} modelos carregados.")


# ═══════════════════════════════════════════════
# 2. CLASSIFICADOR DE ESTOQUE CRÍTICO
# ═══════════════════════════════════════════════

class StockAlertClassifier:
    """
    Classifica cada produto em: NORMAL / BAIXO / CRITICO.
    Usa RandomForest com features de venda do DW.
    Obs.: o DW não tem ReorderPoint diretamente, então usamos
    thresholds derivados do comportamento de venda.
    """

    FEAT_COLS = [
        "ListPrice", "StandardCost",
        "TotalQty12m", "TotalRevenue12m", "TotalProfit12m",
        "AvgMonthlyQty", "StdDevQty", "AvgMargin",
        "MonthsWithSales", "OrderCount",
        "TotalQty3m", "AvgMonthlyQty3m", "TrendPct",
        "CoverageMonths",    # calculado abaixo
        "SalesConsistency",  # calculado abaixo
    ]

    def __init__(self):
        self.model     = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42,
        )
        self.scaler    = StandardScaler()
        self.label_enc = LabelEncoder()

    # ── Feature engineering ──────────────────────

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # CoverageMonths: quantos meses de estoque restam
        # (baseado na média mensal vendida — sem dado real de estoque no DW)
        df["CoverageMonths"] = df.apply(
            lambda r: min(r["TotalQty12m"] / (r["AvgMonthlyQty"] * 12), 24)
            if r["AvgMonthlyQty"] > 0 else 24,
            axis=1,
        )
        # SalesConsistency: regularidade das vendas (0-1)
        df["SalesConsistency"] = df["MonthsWithSales"] / 12.0
        return df

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Rótulos derivados da tendência e consistência de vendas.
        CRITICO  → venda caindo muito (TrendPct < -30%) ou quase sem vendas
        BAIXO    → tendência negativa moderada ou baixa consistência
        NORMAL   → estável ou crescendo
        """
        def label(r):
            if r["TotalQty12m"] == 0:
                return "CRITICO"
            if r["TrendPct"] < -30 or r["SalesConsistency"] < 0.25:
                return "CRITICO"
            if r["TrendPct"] < -10 or r["SalesConsistency"] < 0.5:
                return "BAIXO"
            return "NORMAL"
        return df.apply(label, axis=1)

    # ── Treinamento ───────────────────────────────

    def train(self, features_df: pd.DataFrame) -> dict:
        df = self._engineer(features_df).dropna(subset=self.FEAT_COLS)
        y  = self._create_labels(df)

        X       = df[self.FEAT_COLS]
        X_sc    = self.scaler.fit_transform(X)
        y_enc   = self.label_enc.fit_transform(y)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sc, y_enc, test_size=0.2, random_state=42,
            stratify=y_enc if len(np.unique(y_enc)) > 1 else None,
        )
        self.model.fit(X_tr, y_tr)
        report = classification_report(
            y_te, self.model.predict(X_te),
            target_names=self.label_enc.classes_,
            output_dict=True,
        )

        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "encoder": self.label_enc},
            os.path.join(MODEL_DIR, "stock_classifier.pkl"),
        )
        print("[StockAlertClassifier] Treinado com sucesso.")
        return report

    # ── Inferência ────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df  = self._engineer(features_df).dropna(subset=self.FEAT_COLS)
        X   = self.scaler.transform(df[self.FEAT_COLS])

        labels   = self.label_enc.inverse_transform(self.model.predict(X))
        probs    = self.model.predict_proba(X).max(axis=1)

        out = df[["sk_produto", "ProductID", "ProductName",
                  "Category", "AvgMonthlyQty", "TrendPct",
                  "SalesConsistency"]].copy()
        out["StockStatus"]  = labels
        out["Confidence"]   = np.round(probs * 100, 1)

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
# 3. PADRÕES DE VENDA (CLUSTERING)
# ═══════════════════════════════════════════════

class SalesPatternAnalyzer:
    """
    Segmenta produtos em 4 perfis de comportamento usando K-Means.
    Perfis: Alta Rotatividade / Venda Sazonal / Venda Estável / Baixa Rotatividade
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

        # Nomear clusters pela mediana de volume (maior vol = maior rotatividade)
        cluster_vol = df.groupby("Cluster")["AvgMonthlyQty"].median().sort_values(ascending=False)
        labels_list = ["Alta Rotatividade", "Venda Sazonal", "Venda Estável", "Baixa Rotatividade"]
        self._label_map = {cid: labels_list[i] for i, cid in enumerate(cluster_vol.index)}
        df["Pattern"] = df["Cluster"].map(self._label_map)

        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "label_map": self._label_map},
            os.path.join(MODEL_DIR, "pattern_model.pkl"),
        )
        print(f"[SalesPatternAnalyzer] {len(df)} produtos segmentados.")
        return df[["sk_produto", "ProductID", "ProductName", "Category",
                   "Subcategory", "AvgMonthlyQty", "StdDevQty",
                   "MonthsWithSales", "TrendPct", "AvgMargin",
                   "Cluster", "Pattern"]].reset_index(drop=True)

    def get_cluster_summary(self, segmented_df: pd.DataFrame) -> list[dict]:
        summary = []
        for pattern, g in segmented_df.groupby("Pattern"):
            summary.append({
                "pattern":        pattern,
                "product_count":  int(len(g)),
                "avg_monthly_qty": round(float(g["AvgMonthlyQty"].mean()), 1),
                "avg_variation":  round(float(g["StdDevQty"].mean()), 1),
                "avg_margin":     round(float(g["AvgMargin"].mean()), 2),
                "avg_trend_pct":  round(float(g["TrendPct"].mean()), 2),
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
