# ════════════════════════════════════════════════════════
# COMO INTEGRAR AO SEU app.py EXISTENTE
# ════════════════════════════════════════════════════════

# Adicione APENAS estas 2 linhas ao seu app.py:

from routes.ml_routes import ml_bp       # linha 1 — no bloco de imports
app.register_blueprint(ml_bp)            # linha 2 — logo após criar o app Flask


# ════════════════════════════════════════════════════════
# ESTRUTURA DE PASTAS FINAL
# ════════════════════════════════════════════════════════
"""
seu_projeto/
├── app.py                        ← seu arquivo (adiciona 2 linhas)
├── static/
│   └── ...                       ← arquivos existentes do YOLO
├── data/
│   └── db_connector.py           ← conexão com adventureworks_dw
├── models/
│   ├── ml_models.py              ← DemandForecaster, StockAlertClassifier, SalesPatternAnalyzer
│   └── saved/                    ← modelos treinados (.pkl) — criados automaticamente
├── routes/
│   └── ml_routes.py              ← endpoints /api/ml/...
└── requirements_ml.txt
"""

# ════════════════════════════════════════════════════════
# SEQUÊNCIA DE USO (via Postman, curl ou frontend)
# ════════════════════════════════════════════════════════
"""
1. Verificar conexão com o banco:
   GET  /api/ml/status

2. Treinar os modelos (fazer uma vez, ou re-treinar periodicamente):
   POST /api/ml/train/demand      → previsão de demanda por produto
   POST /api/ml/train/stock       → classificação de risco de estoque
   POST /api/ml/train/patterns    → clustering de padrões de venda

3. Usar os resultados:
   GET  /api/ml/dashboard             → painel executivo completo
   GET  /api/ml/forecast/<sk_produto> → previsão para 1 produto (?months=3)
   GET  /api/ml/forecast/all          → previsão geral (todos os produtos)
   GET  /api/ml/stock/alerts          → alertas (?status=CRITICO|BAIXO|NORMAL|ALL)
   GET  /api/ml/patterns              → segmentação (?pattern=Alta Rotatividade)
   GET  /api/ml/sellers               → performance dos vendedores (?months=12)
   GET  /api/ml/territories           → vendas por território (?months=12)
   GET  /api/ml/promotions            → impacto das promoções
"""

# ════════════════════════════════════════════════════════
# MAPEAMENTO: tabelas DW → modelos ML
# ════════════════════════════════════════════════════════
"""
fato_vendas + dim_tempo + dim_produto  →  DemandForecaster (previsão mensal por produto)
dim_produto + fato_vendas (features)   →  StockAlertClassifier (CRITICO/BAIXO/NORMAL)
dim_produto + fato_vendas (features)   →  SalesPatternAnalyzer (clustering K-Means)
dim_vendedor + fato_vendas             →  /api/ml/sellers (performance/cota)
dim_territorio + fato_vendas           →  /api/ml/territories (análise regional)
dim_promocao + fato_vendas             →  /api/ml/promotions (efetividade)
"""
