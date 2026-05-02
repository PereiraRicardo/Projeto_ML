"""
Módulo de conexão e extração de dados — ELSONS_DW
Banco: SQL Server local | DB: ELSONS_DW
Fato:  FT_VENDAS
Dims:  DIM_PRODUTO, DIM_CLIENTE, DIM_VENDEDOR, DIM_TEMPO

IMPORTANTE: FT_VENDAS.IDProduto é FK para DIM_PRODUTO.IDSK (surrogate),
            não para DIM_PRODUTO.IDProduto (natural key).
            Agrupamentos usam dp.IDProduto (chave natural) como sk_produto.
            Produtos com IDProduto IS NULL (~10%% das vendas) são excluídos.
"""

import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus


DB_CONFIG = {
    "server":   "HomeOffice\\SQLEXPRESS",
    "database": "ELSONS_DW",
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
    conn_str = quote_plus(get_connection_string())
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={conn_str}",
        fast_executemany=True,
    )


def get_conn():
    return pyodbc.connect(get_connection_string())


def test_connection() -> dict:
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM FT_VENDAS")
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
    Join: FT_VENDAS.IDProduto = DIM_PRODUTO.IDSK (surrogate).
    Agrupamento por DIM_PRODUTO.IDProduto (natural key).
    Usado em: DemandForecaster.train()
    """
    query = f"""
    ;WITH refDate AS (
        SELECT DATEFROMPARTS(YEAR(MAX(dt.Data)), MONTH(MAX(dt.Data)), 1) AS maxMonth
        FROM FT_VENDAS fv
        JOIN DIM_TEMPO dt ON fv.IDTempo = dt.IDSK
        WHERE fv.Quant > 0 AND fv.TotLiq > 0
    )
    SELECT
        dp.IDProduto                                        AS sk_produto,
        MAX(dp.Cod)                                         AS ProductID,
        MAX(dp.Descri)                                      AS ProductName,
        MAX(dp.Grupo)                                       AS Category,
        MAX(dp.SubGrp)                                      AS Subcategory,
        DATEFROMPARTS(YEAR(dt.Data), MONTH(dt.Data), 1)    AS SaleMonth,
        SUM(fv.Quant)                                       AS TotalQty,
        SUM(fv.TotLiq)                                      AS TotalRevenue,
        SUM(fv.TotLiq * ISNULL(dp.Margem, 0) / 100)        AS TotalProfit,
        AVG(fv.PrVenda)                                     AS AvgUnitPrice,
        AVG(ISNULL(dp.Margem, 0))                           AS AvgMargin,
        COUNT(DISTINCT fv.IDPed)                            AS OrderCount
    FROM FT_VENDAS fv
    JOIN DIM_TEMPO   dt ON fv.IDTempo   = dt.IDSK
    JOIN DIM_PRODUTO dp ON fv.IDProduto = dp.IDSK
    CROSS JOIN refDate rd
    WHERE fv.Quant > 0
      AND fv.TotLiq > 0
      AND dp.IDProduto IS NOT NULL
      AND DATEFROMPARTS(YEAR(dt.Data), MONTH(dt.Data), 1) >=
          DATEADD(MONTH, -{months}, rd.maxMonth)
    GROUP BY
        dp.IDProduto,
        YEAR(dt.Data), MONTH(dt.Data)
    ORDER BY SaleMonth, dp.IDProduto
    """
    return pd.read_sql(query, get_engine())


def fetch_product_features() -> pd.DataFrame:
    """
    Features por produto para os modelos de ML.
    Join: FT_VENDAS.IDProduto = DIM_PRODUTO.IDSK (surrogate).
    Outer select usa versão mais recente (MAX IDSK por IDProduto).
    Usado em: StockAlertClassifier.train(), SalesPatternAnalyzer.train()
    """
    query = """
    DECLARE @dt_min DATE = (
        SELECT MIN(dt.Data) FROM FT_VENDAS fv
        JOIN DIM_TEMPO dt ON fv.IDTempo = dt.IDSK
        WHERE fv.Quant > 0 AND fv.TotLiq > 0
    )
    DECLARE @dt_max DATE = (
        SELECT MAX(dt.Data) FROM FT_VENDAS fv
        JOIN DIM_TEMPO dt ON fv.IDTempo = dt.IDSK
        WHERE fv.Quant > 0 AND fv.TotLiq > 0
    )
    DECLARE @dt_mid DATE = DATEADD(MONTH, DATEDIFF(MONTH, @dt_min, @dt_max) / 2, @dt_min)

    ;WITH dpLatest AS (
        SELECT IDProduto, MAX(IDSK) AS IDSK
        FROM DIM_PRODUTO
        WHERE IDProduto IS NOT NULL
        GROUP BY IDProduto
    ),
    stot AS (
        SELECT
            dp.IDProduto,
            SUM(fv.Quant)                                       AS TotalQtyAll,
            SUM(fv.TotLiq)                                      AS TotalRevenueAll,
            SUM(fv.TotLiq * ISNULL(dp.Margem, 0) / 100)        AS TotalProfitAll,
            AVG(CAST(fv.Quant AS FLOAT))                        AS AvgMonthlyQty,
            ISNULL(STDEV(CAST(fv.Quant AS FLOAT)), 0)           AS StdDevQty,
            COUNT(DISTINCT
                CAST(YEAR(dt.Data) AS VARCHAR(4)) + '-'
                + RIGHT('0' + CAST(MONTH(dt.Data) AS VARCHAR(2)), 2)
            )                                                   AS MonthsWithSales,
            COUNT(DISTINCT fv.IDPed)                            AS OrderCount
        FROM FT_VENDAS fv
        JOIN DIM_TEMPO   dt ON fv.IDTempo   = dt.IDSK
        JOIN DIM_PRODUTO dp ON fv.IDProduto = dp.IDSK
        WHERE fv.Quant > 0 AND fv.TotLiq > 0
          AND dp.IDProduto IS NOT NULL
        GROUP BY dp.IDProduto
    ),
    srec AS (
        SELECT
            dp.IDProduto,
            SUM(fv.Quant)                   AS TotalQtyRec,
            AVG(CAST(fv.Quant AS FLOAT))    AS AvgMonthlyQtyRec
        FROM FT_VENDAS fv
        JOIN DIM_TEMPO   dt ON fv.IDTempo   = dt.IDSK
        JOIN DIM_PRODUTO dp ON fv.IDProduto = dp.IDSK
        WHERE fv.Quant > 0 AND fv.TotLiq > 0
          AND dp.IDProduto IS NOT NULL
          AND dt.Data >= @dt_mid
        GROUP BY dp.IDProduto
    )
    SELECT
        dp.IDProduto                                            AS sk_produto,
        dp.Cod                                                  AS ProductID,
        dp.Descri                                               AS ProductName,
        dp.Grupo                                                AS Category,
        dp.SubGrp                                               AS Subcategory,
        ISNULL(dp.Uprc, 0)                                      AS ListPrice,
        ISNULL(dp.Uprc * (1 - ISNULL(dp.Margem, 0) / 100), 0)  AS StandardCost,
        ''                                                       AS ProductClass,
        ''                                                       AS ProductLine,

        ISNULL(stot.TotalQtyAll,      0)                        AS TotalQty12m,
        ISNULL(stot.TotalRevenueAll,  0)                        AS TotalRevenue12m,
        ISNULL(stot.TotalProfitAll,   0)                        AS TotalProfit12m,
        ISNULL(stot.AvgMonthlyQty,    0)                        AS AvgMonthlyQty,
        ISNULL(stot.StdDevQty,        0)                        AS StdDevQty,
        ISNULL(dp.Margem,             0)                        AS AvgMargin,
        ISNULL(stot.MonthsWithSales,  0)                        AS MonthsWithSales,
        ISNULL(stot.OrderCount,       0)                        AS OrderCount,

        ISNULL(srec.TotalQtyRec,      0)                        AS TotalQty3m,
        ISNULL(srec.AvgMonthlyQtyRec, 0)                        AS AvgMonthlyQty3m,

        CASE
            WHEN ISNULL(stot.AvgMonthlyQty, 0) > 0
            THEN ROUND(
                (ISNULL(srec.AvgMonthlyQtyRec, 0) - stot.AvgMonthlyQty)
                / stot.AvgMonthlyQty * 100, 2)
            ELSE 0
        END                                                     AS TrendPct

    FROM dpLatest dpl
    JOIN DIM_PRODUTO dp  ON dp.IDSK      = dpl.IDSK
    LEFT JOIN stot       ON dp.IDProduto = stot.IDProduto
    LEFT JOIN srec       ON dp.IDProduto = srec.IDProduto
    """
    return pd.read_sql(query, get_engine())


def fetch_sales_by_territory(months: int = 12) -> pd.DataFrame:
    """Vendas mensais agrupadas por estado do cliente."""
    query = f"""
    ;WITH dcLatest AS (
        SELECT IDCliente, MAX(IDSK) AS IDSK
        FROM DIM_CLIENTE
        GROUP BY IDCliente
    ),
    refDate AS (
        SELECT DATEFROMPARTS(YEAR(MAX(dt.Data)), MONTH(MAX(dt.Data)), 1) AS maxMonth
        FROM FT_VENDAS fv
        JOIN DIM_TEMPO dt ON fv.IDTempo = dt.IDSK
        WHERE fv.Quant > 0 AND fv.TotLiq > 0
    )
    SELECT
        ISNULL(dc.Est, 'N/I')                               AS Territory,
        ISNULL(dc.Est, 'N/I')                               AS Country,
        'Brasil'                                             AS Region,
        YEAR(dt.Data)                                        AS Year,
        MONTH(dt.Data)                                       AS Month,
        DATEPART(QUARTER, dt.Data)                           AS Quarter,
        dt.NomeMes                                           AS MonthName,
        SUM(fv.Quant)                                        AS TotalQty,
        SUM(fv.TotLiq)                                       AS TotalRevenue,
        SUM(fv.TotLiq * ISNULL(dp.Margem, 0) / 100)         AS TotalProfit,
        COUNT(DISTINCT fv.IDPed)                             AS OrderCount,
        COUNT(DISTINCT fv.IDCliente)                         AS UniqueCustomers
    FROM FT_VENDAS fv
    JOIN DIM_TEMPO   dt  ON fv.IDTempo   = dt.IDSK
    JOIN DIM_PRODUTO dp  ON fv.IDProduto = dp.IDSK
    JOIN dcLatest    dcl ON fv.IDCliente = dcl.IDCliente
    JOIN DIM_CLIENTE dc  ON dc.IDSK      = dcl.IDSK
    CROSS JOIN refDate rd
    WHERE fv.Quant > 0
      AND fv.TotLiq > 0
      AND DATEFROMPARTS(YEAR(dt.Data), MONTH(dt.Data), 1) >=
          DATEADD(MONTH, -{months}, rd.maxMonth)
    GROUP BY
        dc.Est,
        YEAR(dt.Data), MONTH(dt.Data),
        DATEPART(QUARTER, dt.Data), dt.NomeMes
    ORDER BY Year, Month, Territory
    """
    return pd.read_sql(query, get_engine())


def fetch_seller_performance(months: int = 12) -> pd.DataFrame:
    """Performance dos vendedores. Cota e comissão retornam 0 (não disponíveis)."""
    query = """
    ;WITH dvLatest AS (
        SELECT IDVendedor, MAX(IDSK) AS IDSK
        FROM DIM_VENDEDOR
        GROUP BY IDVendedor
    )
    SELECT
        dv.IDVendedor                                        AS SellerID,
        dv.Nome                                              AS SellerName,
        ISNULL(dv.Equipe, dv.Mun)                           AS Territory,
        0                                                    AS AnnualQuota,
        0                                                    AS CommissionPct,
        SUM(fv.TotLiq)                                       AS TotalRevenue,
        SUM(fv.TotLiq * ISNULL(dp.Margem, 0) / 100)         AS TotalProfit,
        SUM(fv.Quant)                                        AS TotalQty,
        COUNT(DISTINCT fv.IDPed)                             AS OrderCount,
        COUNT(DISTINCT fv.IDCliente)                         AS UniqueCustomers,
        AVG(ISNULL(dp.Margem, 0))                            AS AvgMargin,
        NULL                                                 AS QuotaAttainmentPct
    FROM FT_VENDAS fv
    JOIN DIM_PRODUTO  dp  ON fv.IDProduto  = dp.IDSK
    JOIN dvLatest     dvl ON fv.IDVendedor = dvl.IDVendedor
    JOIN DIM_VENDEDOR dv  ON dv.IDSK       = dvl.IDSK
    WHERE fv.Quant > 0 AND fv.TotLiq > 0
    GROUP BY dv.IDVendedor, dv.Nome, dv.Equipe, dv.Mun
    ORDER BY TotalRevenue DESC
    """
    return pd.read_sql(query, get_engine())


def fetch_promotion_impact() -> pd.DataFrame:
    """Sem dim_promocao no ELSONS_DW. Retorna DataFrame vazio compatível."""
    return pd.DataFrame(columns=[
        "PromotionName", "DiscountType", "PromotionCategory",
        "DiscountPct", "OrderCount", "TotalQty",
        "GrossRevenue", "NetRevenue", "TotalProfit",
        "TotalDiscount", "AvgMargin",
    ])


def fetch_dashboard_kpis() -> dict:
    """KPIs consolidados para o painel executivo."""
    engine = get_engine()

    with engine.connect() as conn:
        r = conn.execute(text("""
            SELECT YEAR(MAX(dt.Data)) AS max_ano, MONTH(MAX(dt.Data)) AS max_mes
            FROM FT_VENDAS fv
            JOIN DIM_TEMPO dt ON fv.IDTempo = dt.IDSK
            WHERE fv.Quant > 0 AND fv.TotLiq > 0
        """)).fetchone()
        max_ano = int(r[0])
        max_mes = int(r[1])

    query = f"""
    SELECT
        SUM(fv.TotLiq)                                                  AS total_revenue,
        SUM(fv.TotLiq * ISNULL(dp.Margem, 0) / 100)                    AS total_profit,
        SUM(fv.Quant)                                                   AS total_qty,
        COUNT(DISTINCT fv.IDPed)                                        AS total_orders,
        COUNT(DISTINCT fv.IDCliente)                                    AS unique_customers,
        AVG(ISNULL(dp.Margem, 0))                                       AS avg_margin,

        SUM(CASE WHEN YEAR(dt.Data) = {max_ano} AND MONTH(dt.Data) = {max_mes}
             THEN fv.TotLiq ELSE 0 END)                                 AS current_month_revenue,

        SUM(CASE WHEN DATEFROMPARTS(YEAR(dt.Data), MONTH(dt.Data), 1) =
                      DATEADD(MONTH, -1, DATEFROMPARTS({max_ano}, {max_mes}, 1))
             THEN fv.TotLiq ELSE 0 END)                                 AS prev_month_revenue,

        SUM(CASE WHEN YEAR(dt.Data) = {max_ano}
             THEN fv.TotLiq ELSE 0 END)                                 AS ytd_revenue,

        SUM(CASE WHEN YEAR(dt.Data) = {max_ano - 1}
             THEN fv.TotLiq ELSE 0 END)                                 AS prev_year_revenue
    FROM FT_VENDAS fv
    JOIN DIM_TEMPO   dt ON fv.IDTempo   = dt.IDSK
    JOIN DIM_PRODUTO dp ON fv.IDProduto = dp.IDSK
    WHERE fv.Quant > 0 AND fv.TotLiq > 0
    """
    with engine.connect() as conn:
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
        "reference_month":       f"{max_ano}-{str(max_mes).zfill(2)}",
        "reference_year":        max_ano,
    }
