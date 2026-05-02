"""
Rotas Flask — ML sobre ELSONS_DW
Registre no app.py com: app.register_blueprint(ml_bp)

Endpoints:
  GET  /api/ml/status
  POST /api/ml/train/demand
  POST /api/ml/train/stock
  POST /api/ml/train/patterns
  GET  /api/ml/forecast/<sk_produto>
  GET  /api/ml/forecast/all
  GET  /api/ml/stock/alerts
  GET  /api/ml/patterns
  GET  /api/ml/sellers
  GET  /api/ml/territories
  GET  /api/ml/promotions
  GET  /api/ml/dashboard
"""

from flask import Blueprint, jsonify, request
import traceback, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.db_connector import (
    test_connection,
    fetch_sales_history,
    fetch_product_features,
    fetch_sales_by_territory,
    fetch_seller_performance,
    fetch_promotion_impact,
    fetch_dashboard_kpis,
)
from models.ml_models import (
    DemandForecaster,
    StockAlertClassifier,
    SalesPatternAnalyzer,
)

ml_bp = Blueprint("ml", __name__, url_prefix="/api/ml")

# ── Instâncias globais (carregam modelos do disco ao iniciar) ────────────────
forecaster  = DemandForecaster(forecast_months=3)
stock_clf   = StockAlertClassifier()
pattern_ana = SalesPatternAnalyzer()

forecaster.load()
stock_clf.load()
pattern_ana.load()

# ── Cache de features (evita query de ~58s por request) ─────────────────────
_features_cache = None

def _get_features():
    global _features_cache
    if _features_cache is None or _features_cache.empty:
        _features_cache = fetch_product_features()
    return _features_cache


# ══════════════════════════════════════════
# STATUS
# ══════════════════════════════════════════

@ml_bp.route("/status", methods=["GET"])
def status():
    """Verifica conexão com ELSONS_DW e estado dos modelos."""
    db = test_connection()
    return jsonify({
        "database":       db,
        "demand_model":   "loaded" if forecaster.models            else "not_trained",
        "stock_model":    "loaded" if hasattr(stock_clf.model, "n_estimators") else "not_trained",
        "pattern_model":  "loaded" if hasattr(pattern_ana.model, "cluster_centers_") else "not_trained",
    })


# ══════════════════════════════════════════
# TREINAMENTO
# ══════════════════════════════════════════

@ml_bp.route("/train/demand", methods=["POST"])
def train_demand():
    """
    Treina previsão de demanda para todos os produtos.
    Body JSON opcional: { "months": 36 }
    """
    try:
        months   = (request.json or {}).get("months", 36)
        sales_df = fetch_sales_history(months=months)
        if sales_df.empty:
            return jsonify({"error": "fato_vendas sem dados. Execute o ETL primeiro."}), 404
        metrics = forecaster.train(sales_df)
        return jsonify({
            "success":           True,
            "products_trained":  len(metrics),
            "sample_mae":        dict(list(metrics.items())[:5]),
            "prophet_available": forecaster.__class__.__module__ != "__main__",
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@ml_bp.route("/train/stock", methods=["POST"])
def train_stock():
    """Treina classificador de alerta de estoque."""
    try:
        features_df = fetch_product_features()
        if features_df.empty:
            return jsonify({"error": "dim_produto sem registros ativos."}), 404
        report = stock_clf.train(features_df)
        return jsonify({"success": True, "classification_report": report})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@ml_bp.route("/train/patterns", methods=["POST"])
def train_patterns():
    """Treina clustering de padrões de venda."""
    try:
        features_df = fetch_product_features()
        if features_df.empty:
            return jsonify({"error": "dim_produto sem registros ativos."}), 404
        segmented = pattern_ana.train(features_df)
        summary   = pattern_ana.get_cluster_summary(segmented)
        return jsonify({
            "success":        True,
            "total_products": len(segmented),
            "clusters":       summary,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════
# PREVISÃO DE DEMANDA
# ══════════════════════════════════════════

@ml_bp.route("/forecast/<int:sk_produto>", methods=["GET"])
def forecast_product(sk_produto: int):
    try:
        months = min(int(request.args.get("months", 3)), 12)
        forecaster.forecast_months = months
        prediction = forecaster.predict(sk_produto)
        if not prediction:
            return jsonify({
                "error": f"sk_produto {sk_produto} não treinado ou sem histórico suficiente."
            }), 404
        return jsonify({"sk_produto": sk_produto, "forecast": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_bp.route("/forecast/products/list", methods=["GET"])
def forecast_products_list():
    """Lista todos os produtos treinados com nome — sem calcular previsão."""
    try:
        feat_df  = _get_features()
        name_map = dict(zip(feat_df["sk_produto"], feat_df["ProductName"]))

        products = []
        for sk in sorted(forecaster.models.keys()):
            products.append({
                "sk_produto":   int(sk),
                "ProductName":  name_map.get(sk, f"Produto sk_{sk}"),
            })
        return jsonify({"products": products, "total": len(products)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_bp.route("/forecast/all", methods=["GET"])
def forecast_all():
    """
    Previsão do próximo mês para todos os produtos treinados.
    Limita a 100 produtos para performance.
    """
    try:
        results = []
        for sk in list(forecaster.models.keys())[:100]:
            preds = forecaster.predict(sk)
            if preds:
                results.append({"sk_produto": int(sk), "next_month": preds[0]})
        return jsonify({"forecasts": results, "total": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════
# ALERTAS DE ESTOQUE
# ══════════════════════════════════════════

@ml_bp.route("/stock/alerts", methods=["GET"])
def stock_alerts():
    """
    Produtos classificados por risco de estoque.
    Query param: status=CRITICO|BAIXO|NORMAL|ALL (default: ALL)
    """
    try:
        status_filter = request.args.get("status", "ALL").upper()
        features_df   = _get_features()
        if features_df.empty:
            return jsonify({"error": "Nenhum produto encontrado."}), 404

        result = stock_clf.predict(features_df)
        if status_filter != "ALL":
            result = result[result["StockStatus"] == status_filter]

        alerts = result.to_dict(orient="records")
        for a in alerts:
            a["AvgMonthlyQty"] = round(float(a.get("AvgMonthlyQty", 0)), 1)
            a["TrendPct"]      = round(float(a.get("TrendPct", 0)), 2)
            a["Confidence"]    = float(a.get("Confidence", 0))

        return jsonify({
            "alerts":   alerts,
            "total":    len(alerts),
            "critical": sum(1 for a in alerts if a["StockStatus"] == "CRITICO"),
            "low":      sum(1 for a in alerts if a["StockStatus"] == "BAIXO"),
            "normal":   sum(1 for a in alerts if a["StockStatus"] == "NORMAL"),
            "inactive": sum(1 for a in alerts if a["StockStatus"] == "INATIVO"),
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════
# PADRÕES DE VENDA
# ══════════════════════════════════════════

@ml_bp.route("/patterns", methods=["GET"])
def sales_patterns():
    """
    Segmentação de produtos por padrão de venda.
    Query param: pattern=<nome do padrão> (opcional)
    """
    try:
        features_df    = _get_features()
        segmented      = pattern_ana.predict(features_df)
        summary        = pattern_ana.get_cluster_summary(segmented)

        pattern_filter = request.args.get("pattern")
        if pattern_filter:
            segmented = segmented[segmented["Pattern"] == pattern_filter]

        products = segmented[[
            "sk_produto", "ProductID", "ProductName", "Category",
            "Subcategory", "AvgMonthlyQty", "TrendPct", "Pattern"
        ]].to_dict(orient="records")

        return jsonify({"summary": summary, "products": products})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════
# VENDEDORES
# ══════════════════════════════════════════

@ml_bp.route("/sellers", methods=["GET"])
def sellers():
    """Performance dos vendedores com % de atingimento de cota."""
    try:
        months = int(request.args.get("months", 12))
        df     = fetch_seller_performance(months=months)
        data   = df.to_dict(orient="records")
        for d in data:
            for k, v in d.items():
                if hasattr(v, "item"):
                    d[k] = v.item()   # converte numpy types
        return jsonify({"sellers": data, "total": len(data)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════
# TERRITÓRIOS
# ══════════════════════════════════════════

@ml_bp.route("/territories", methods=["GET"])
def territories():
    """Vendas mensais por território."""
    try:
        months = int(request.args.get("months", 12))
        df     = fetch_sales_by_territory(months=months)
        return jsonify({
            "territories": df.to_dict(orient="records"),
            "total_rows":  len(df),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════
# PROMOÇÕES
# ══════════════════════════════════════════

@ml_bp.route("/promotions", methods=["GET"])
def promotions():
    """Impacto de cada promoção nas vendas."""
    try:
        df = fetch_promotion_impact()
        return jsonify({"promotions": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════
# DASHBOARD UNIFICADO
# ══════════════════════════════════════════

@ml_bp.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Retorna todos os dados necessários para o painel executivo em uma chamada.
    Inclui: KPIs, alertas críticos, padrões de venda, top vendedores.
    """
    try:
        # KPIs do DW
        kpis = fetch_dashboard_kpis()

        # Alertas de estoque
        features_df   = _get_features()
        stock_result  = stock_clf.predict(features_df) if not features_df.empty else None

        critical_products = []
        stock_summary     = {"critical": 0, "low": 0, "normal": 0, "inactive": 0}

        if stock_result is not None and not stock_result.empty:
            key_map = {"CRITICO": "critical", "BAIXO": "low", "NORMAL": "normal", "INATIVO": "inactive"}
            for status_key, summary_key in key_map.items():
                stock_summary[summary_key] = int((stock_result["StockStatus"] == status_key).sum())

            top5 = (
                stock_result[stock_result["StockStatus"] == "CRITICO"]
                .head(5)[["ProductID", "ProductName", "AvgMonthlyQty", "TrendPct"]]
                .to_dict(orient="records")
            )
            critical_products = [{
                k: (round(float(v), 2) if isinstance(v, float) else v)
                for k, v in row.items()
            } for row in top5]

        # Padrões de venda
        patterns_summary = []
        if not features_df.empty:
            segmented        = pattern_ana.predict(features_df)
            patterns_summary = pattern_ana.get_cluster_summary(segmented)

        # Top 5 vendedores — histórico completo do DW
        sellers_df  = fetch_seller_performance(months=999)
        top_sellers = sellers_df.head(5)[[
            "SellerName", "Territory", "TotalRevenue", "QuotaAttainmentPct"
        ]].to_dict(orient="records")
        for s in top_sellers:
            s["TotalRevenue"]       = round(float(s.get("TotalRevenue", 0) or 0), 2)
            s["QuotaAttainmentPct"] = round(float(s.get("QuotaAttainmentPct", 0) or 0), 2)

        return jsonify({
            "kpis":              kpis,
            "stock":             {**stock_summary, "top_critical": critical_products},
            "patterns":          patterns_summary,
            "top_sellers":       top_sellers,
            "total_products":    int(len(features_df)),
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
