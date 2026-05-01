"""
Módulo de conexão e extração de dados - adventureworks_dw
Banco: SQL Server local | Usuário: sa | DB: adventureworks_dw
Tabelas: fato_vendas, dim_produto, dim_cliente, dim_vendedor,
         dim_territorio, dim_promocao, dim_tempo
"""

import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus


# ─────────────────────────────────────────
# CONFIGURAÇÃO — mesma do recreate_dw_tables.py
# ─────────────────────────────────────────
DB_CONFIG = {
    "server":   "localhost",
    "database": "adventureworks_dw",
    "username": "sa",
    "password": "123456",
    "driver":   "ODBC Driver 17 for SQL Server",
}


def get_connection_string() -> str:
    return (
        f"DRIVER={{{DB_CONFIG['driver']}}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['username']};"
        f"PWD={DB_CONFIG['password']};"
        "TrustServerCertificate=yes;"
    )


def get_engine():
    """Engine SQLAlchemy (usado pelo pandas para queries grandes)."""
    conn_str = quote_plus(get_connection_string())
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={conn_str}",
        fast_executemany=True,
    )


def get_conn():
    """Conexão pyodbc direta (usada para KPIs e queries simples)."""
    return pyodbc.connect(get_connection_string())


def test_connection() -> dict:
    """Testa a conexão e retorna status com contagem de linhas."""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fato_vendas")
        row_count = cursor.fetchone()[0]
        conn.close()
        return {"ok": True, "fato_vendas_rows": row_count}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════
# QUERIES DE EXTRAÇÃO
# ═══════════════════════════════════════════════════════

def fetch_sales_history(months: int = 36) -> pd.DataFrame:
    """
    Histórico mensal de vendas por produto.
    Usa o período real do DW (detectado automaticamente via MAX/MIN da dim_tempo).
    Usado em: DemandForecaster.train()
    """
    query = f"""
    SELECT
        fv.sk_produto,
        dp.nk_produto                               AS ProductID,
        dp.nome_produto                             AS ProductName,
        dp.categoria                                AS Category,
        dp.subcategoria                             AS Subcategory,
        CAST(DATEFROMPARTS(dt.ano, dt.mes, 1) AS DATE) AS SaleMonth,
        SUM(fv.quantidade)                          AS TotalQty,
        SUM(fv.receita_liquida)                     AS TotalRevenue,
        SUM(fv.lucro_bruto)                         AS TotalProfit,
        AVG(fv.preco_unitario)                      AS AvgUnitPrice,
        AVG(fv.margem_percentual)                   AS AvgMargin,
        COUNT(DISTINCT fv.nk_pedido)                AS OrderCount
    FROM fato_vendas fv
    JOIN dim_tempo   dt ON fv.sk_tempo   = dt.sk_tempo
    JOIN dim_produto dp ON fv.sk_produto = dp.sk_produto
    WHERE dp.registro_atual = 1
      AND DATEFROMPARTS(dt.ano, dt.mes, 1) >=
          DATEADD(MONTH, -{months}, (SELECT CAST(DATEFROMPARTS(MAX(ano), MAX(mes), 1) AS DATE) FROM dim_tempo))
    GROUP BY
        fv.sk_produto, dp.nk_produto, dp.nome_produto,
        dp.categoria, dp.subcategoria, dt.ano, dt.mes
    ORDER BY SaleMonth, dp.nk_produto
    """
    return pd.read_sql(query, get_engine())


def fetch_product_features() -> pd.DataFrame:
    """
    Features por produto para os modelos de ML.
    Usa o histórico COMPLETO do DW dividido em dois períodos:
      - Período total: todos os dados disponíveis (2011-2014)
      - Período recente: segunda metade do período total (tendência)
    Isso evita zeros artificiais causados por janelas fixas de 12/3 meses.
    """
    query = """
    DECLARE @dt_min DATE = (SELECT CAST(DATEFROMPARTS(MIN(ano), MIN(mes), 1) AS DATE) FROM dim_tempo)
    DECLARE @dt_max DATE = (SELECT CAST(DATEFROMPARTS(MAX(ano), MAX(mes), 1) AS DATE) FROM dim_tempo)
    DECLARE @dt_mid DATE = DATEADD(MONTH, DATEDIFF(MONTH, @dt_min, @dt_max) / 2, @dt_min)

    SELECT
        dp.sk_produto,
        dp.nk_produto                               AS ProductID,
        dp.nome_produto                             AS ProductName,
        dp.categoria                                AS Category,
        dp.subcategoria                             AS Subcategory,
        ISNULL(dp.preco_lista,   0)                 AS ListPrice,
        ISNULL(dp.custo_padrao,  0)                 AS StandardCost,
        ISNULL(dp.classe,        '')                AS ProductClass,
        ISNULL(dp.linha_produto, '')                AS ProductLine,

        -- Período TOTAL (histórico completo)
        ISNULL(stot.TotalQtyAll,      0)            AS TotalQty12m,
        ISNULL(stot.TotalRevenueAll,  0)            AS TotalRevenue12m,
        ISNULL(stot.TotalProfitAll,   0)            AS TotalProfit12m,
        ISNULL(stot.AvgMonthlyQty,    0)            AS AvgMonthlyQty,
        ISNULL(stot.StdDevQty,        0)            AS StdDevQty,
        ISNULL(stot.AvgMargin,        0)            AS AvgMargin,
        ISNULL(stot.MonthsWithSales,  0)            AS MonthsWithSales,
        ISNULL(stot.OrderCount,       0)            AS OrderCount,

        -- Período RECENTE (segunda metade — tendência)
        ISNULL(srec.TotalQtyRec,      0)            AS TotalQty3m,
        ISNULL(srec.AvgMonthlyQtyRec, 0)            AS AvgMonthlyQty3m,

        -- Tendência: período recente vs média histórica
        CASE
            WHEN ISNULL(stot.AvgMonthlyQty, 0) > 0
            THEN ROUND(
                (ISNULL(srec.AvgMonthlyQtyRec, 0) - stot.AvgMonthlyQty)
                / stot.AvgMonthlyQty * 100, 2)
            ELSE 0
        END                                         AS TrendPct

    FROM dim_produto dp

    -- Subquery período TOTAL
    LEFT JOIN (
        SELECT
            fv.sk_produto,
            SUM(fv.quantidade)                          AS TotalQtyAll,
            SUM(fv.receita_liquida)                     AS TotalRevenueAll,
            SUM(fv.lucro_bruto)                         AS TotalProfitAll,
            AVG(CAST(fv.quantidade AS FLOAT))           AS AvgMonthlyQty,
            ISNULL(STDEV(CAST(fv.quantidade AS FLOAT)), 0) AS StdDevQty,
            AVG(fv.margem_percentual)                   AS AvgMargin,
            COUNT(DISTINCT
                CAST(dt.ano AS VARCHAR(4)) + '-'
                + RIGHT('0' + CAST(dt.mes AS VARCHAR(2)), 2)
            )                                           AS MonthsWithSales,
            COUNT(DISTINCT fv.nk_pedido)                AS OrderCount
        FROM fato_vendas fv
        JOIN dim_tempo dt ON fv.sk_tempo = dt.sk_tempo
        GROUP BY fv.sk_produto
    ) stot ON dp.sk_produto = stot.sk_produto

    -- Subquery período RECENTE (segunda metade do DW)
    LEFT JOIN (
        SELECT
            fv.sk_produto,
            SUM(fv.quantidade)                          AS TotalQtyRec,
            AVG(CAST(fv.quantidade AS FLOAT))           AS AvgMonthlyQtyRec
        FROM fato_vendas fv
        JOIN dim_tempo dt ON fv.sk_tempo = dt.sk_tempo
        WHERE DATEFROMPARTS(dt.ano, dt.mes, 1) >= @dt_mid
        GROUP BY fv.sk_produto
    ) srec ON dp.sk_produto = srec.sk_produto

    WHERE dp.registro_atual = 1
    """
    return pd.read_sql(query, get_engine())


def fetch_sales_by_territory(months: int = 12) -> pd.DataFrame:
    """Vendas mensais por território. Usado para análise regional."""
    query = f"""
    SELECT
        dter.nome_territorio                        AS Territory,
        dter.pais                                   AS Country,
        dter.grupo                                  AS Region,
        dt.ano                                      AS Year,
        dt.mes                                      AS Month,
        dt.trimestre                                AS Quarter,
        dt.nome_mes                                 AS MonthName,
        SUM(fv.quantidade)                          AS TotalQty,
        SUM(fv.receita_liquida)                     AS TotalRevenue,
        SUM(fv.lucro_bruto)                         AS TotalProfit,
        COUNT(DISTINCT fv.nk_pedido)                AS OrderCount,
        COUNT(DISTINCT fv.sk_cliente)               AS UniqueCustomers
    FROM fato_vendas fv
    JOIN dim_tempo      dt   ON fv.sk_tempo      = dt.sk_tempo
    JOIN dim_territorio dter ON fv.sk_territorio = dter.sk_territorio
    WHERE DATEFROMPARTS(dt.ano, dt.mes, 1) >=
          CAST(DATEADD(MONTH, -{months}, DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)) AS DATE)
    GROUP BY
        dter.nome_territorio, dter.pais, dter.grupo,
        dt.ano, dt.mes, dt.trimestre, dt.nome_mes
    ORDER BY Year, Month, Territory
    """
    return pd.read_sql(query, get_engine())


def fetch_seller_performance(months: int = 12) -> pd.DataFrame:
    """Performance e atingimento de cota dos vendedores."""
    query = f"""
    SELECT
        dv.nk_vendedor                              AS SellerID,
        dv.nome_completo                            AS SellerName,
        dv.territorio                               AS Territory,
        ISNULL(dv.cota_anual, 0)                    AS AnnualQuota,
        ISNULL(dv.comissao_pct, 0)                  AS CommissionPct,
        SUM(fv.receita_liquida)                     AS TotalRevenue,
        SUM(fv.lucro_bruto)                         AS TotalProfit,
        SUM(fv.quantidade)                          AS TotalQty,
        COUNT(DISTINCT fv.nk_pedido)                AS OrderCount,
        COUNT(DISTINCT fv.sk_cliente)               AS UniqueCustomers,
        AVG(fv.margem_percentual)                   AS AvgMargin,
        CASE
            WHEN ISNULL(dv.cota_anual, 0) > 0
            THEN ROUND(SUM(fv.receita_liquida) / dv.cota_anual * 100, 2)
            ELSE NULL
        END                                         AS QuotaAttainmentPct
    FROM fato_vendas fv
    JOIN dim_tempo    dt ON fv.sk_tempo    = dt.sk_tempo
    JOIN dim_vendedor dv ON fv.sk_vendedor = dv.sk_vendedor
    WHERE dv.registro_atual = 1
      AND DATEFROMPARTS(dt.ano, dt.mes, 1) >=
          CAST(DATEADD(MONTH, -{months}, DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1)) AS DATE)
    GROUP BY
        dv.nk_vendedor, dv.nome_completo, dv.territorio,
        dv.cota_anual, dv.comissao_pct
    ORDER BY TotalRevenue DESC
    """
    return pd.read_sql(query, get_engine())


def fetch_promotion_impact() -> pd.DataFrame:
    """Impacto de cada promoção nas vendas e margem."""
    query = """
    SELECT
        dpr.descricao                               AS PromotionName,
        dpr.tipo_desconto                           AS DiscountType,
        dpr.categoria                               AS PromotionCategory,
        ISNULL(dpr.percentual_desconto, 0)          AS DiscountPct,
        COUNT(DISTINCT fv.nk_pedido)                AS OrderCount,
        SUM(fv.quantidade)                          AS TotalQty,
        SUM(fv.receita_bruta)                       AS GrossRevenue,
        SUM(fv.receita_liquida)                     AS NetRevenue,
        SUM(fv.lucro_bruto)                         AS TotalProfit,
        SUM(fv.desconto_unitario * fv.quantidade)   AS TotalDiscount,
        AVG(fv.margem_percentual)                   AS AvgMargin
    FROM fato_vendas fv
    JOIN dim_promocao dpr ON fv.sk_promocao = dpr.sk_promocao
    GROUP BY
        dpr.descricao, dpr.tipo_desconto,
        dpr.categoria, dpr.percentual_desconto
    ORDER BY NetRevenue DESC
    """
    return pd.read_sql(query, get_engine())


def fetch_dashboard_kpis() -> dict:
    """
    KPIs consolidados do DW para o painel executivo.
    Retorna dict com receita, lucro, margem, variações MoM e YoY.
    """
    query = """
    SELECT
        SUM(fv.receita_liquida)                             AS total_revenue,
        SUM(fv.lucro_bruto)                                 AS total_profit,
        SUM(fv.quantidade)                                  AS total_qty,
        COUNT(DISTINCT fv.nk_pedido)                        AS total_orders,
        COUNT(DISTINCT fv.sk_cliente)                       AS unique_customers,
        AVG(fv.margem_percentual)                           AS avg_margin,

        SUM(CASE WHEN dt.mes = MONTH(GETDATE())
                  AND dt.ano = YEAR(GETDATE())
             THEN fv.receita_liquida ELSE 0 END)            AS current_month_revenue,

        SUM(CASE WHEN dt.mes = MONTH(DATEADD(MONTH,-1,GETDATE()))
                  AND dt.ano = YEAR(DATEADD(MONTH,-1,GETDATE()))
             THEN fv.receita_liquida ELSE 0 END)            AS prev_month_revenue,

        SUM(CASE WHEN dt.ano = YEAR(GETDATE())
             THEN fv.receita_liquida ELSE 0 END)            AS ytd_revenue,

        SUM(CASE WHEN dt.ano = YEAR(GETDATE()) - 1
             THEN fv.receita_liquida ELSE 0 END)            AS prev_year_revenue
    FROM fato_vendas fv
    JOIN dim_tempo dt ON fv.sk_tempo = dt.sk_tempo
    """
    with get_engine().connect() as conn:
        row = dict(conn.execute(text(query)).fetchone()._mapping)

    def pct(cur, prev):
        if prev and prev > 0:
            return round((cur - prev) / prev * 100, 2)
        return None

    return {
        "total_revenue":         round(float(row["total_revenue"] or 0), 2),
        "total_profit":          round(float(row["total_profit"] or 0), 2),
        "total_qty":             int(row["total_qty"] or 0),
        "total_orders":          int(row["total_orders"] or 0),
        "unique_customers":      int(row["unique_customers"] or 0),
        "avg_margin_pct":        round(float(row["avg_margin"] or 0), 2),
        "current_month_revenue": round(float(row["current_month_revenue"] or 0), 2),
        "prev_month_revenue":    round(float(row["prev_month_revenue"] or 0), 2),
        "mom_change_pct":        pct(row["current_month_revenue"], row["prev_month_revenue"]),
        "ytd_revenue":           round(float(row["ytd_revenue"] or 0), 2),
        "prev_year_revenue":     round(float(row["prev_year_revenue"] or 0), 2),
        "yoy_change_pct":        pct(row["ytd_revenue"], row["prev_year_revenue"]),
    }
