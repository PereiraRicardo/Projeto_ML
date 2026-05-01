"""
Módulo de conexão e extração de dados - AdventureWorks
Requer: pip install pyodbc pandas sqlalchemy
"""

import pyodbc
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


# ─────────────────────────────────────────
# CONFIGURAÇÃO - altere apenas aqui
# ─────────────────────────────────────────
DB_CONFIG = {
    "server":   "localhost",          # ou nome\\instancia
    "database": "AdventureWorks2019",
    "driver":   "ODBC Driver 17 for SQL Server",
    # Autenticação Windows (recomendado em ambiente local):
    "trusted_connection": True,
    # Descomente abaixo para usar usuário/senha SQL:
    # "username": "sa",
    # "password": "sua_senha",
}


def get_connection_string() -> str:
    """Monta a connection string para pyodbc."""
    if DB_CONFIG.get("trusted_connection"):
        return (
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            "Trusted_Connection=yes;"
        )
    return (
        f"DRIVER={{{DB_CONFIG['driver']}}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['username']};"
        f"PWD={DB_CONFIG['password']};"
    )


def get_engine():
    """Retorna engine SQLAlchemy (mais eficiente para pandas)."""
    conn_str = quote_plus(get_connection_string())
    return create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str}")


def test_connection() -> bool:
    """Testa a conexão com o banco."""
    try:
        conn = pyodbc.connect(get_connection_string(), timeout=5)
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] Erro de conexão: {e}")
        return False


# ─────────────────────────────────────────
# QUERIES DE EXTRAÇÃO
# ─────────────────────────────────────────

def fetch_sales_history(months: int = 36) -> pd.DataFrame:
    """
    Histórico de vendas por produto e mês.
    Fonte: Sales.SalesOrderHeader + Sales.SalesOrderDetail
    """
    query = f"""
    SELECT
        sod.ProductID,
        p.Name                          AS ProductName,
        pc.Name                         AS Category,
        ps.Name                         AS Subcategory,
        CAST(FORMAT(soh.OrderDate, 'yyyy-MM-01') AS DATE) AS SaleMonth,
        SUM(sod.OrderQty)               AS TotalQty,
        SUM(sod.LineTotal)              AS TotalRevenue,
        AVG(sod.UnitPrice)              AS AvgUnitPrice,
        COUNT(DISTINCT soh.SalesOrderID) AS OrderCount
    FROM Sales.SalesOrderDetail sod
    JOIN Sales.SalesOrderHeader soh  ON sod.SalesOrderID = soh.SalesOrderID
    JOIN Production.Product p        ON sod.ProductID    = p.ProductID
    LEFT JOIN Production.ProductSubcategory ps
        ON p.ProductSubcategoryID = ps.ProductSubcategoryID
    LEFT JOIN Production.ProductCategory pc
        ON ps.ProductCategoryID   = pc.ProductCategoryID
    WHERE soh.OrderDate >= DATEADD(MONTH, -{months}, GETDATE())
      AND soh.Status = 5  -- apenas pedidos concluídos
    GROUP BY
        sod.ProductID, p.Name, pc.Name, ps.Name,
        FORMAT(soh.OrderDate, 'yyyy-MM-01')
    ORDER BY SaleMonth, sod.ProductID
    """
    return pd.read_sql(query, get_engine())


def fetch_inventory_status() -> pd.DataFrame:
    """
    Situação atual de estoque por produto.
    Fonte: Production.ProductInventory + Production.WorkOrder
    """
    query = """
    SELECT
        p.ProductID,
        p.Name                          AS ProductName,
        p.ReorderPoint,
        p.SafetyStockLevel,
        p.StandardCost,
        p.ListPrice,
        SUM(pi.Quantity)                AS CurrentStock,
        ISNULL(SUM(wo.OrderQty - wo.ScrappedQty), 0) AS InProduction,
        CASE
            WHEN SUM(pi.Quantity) <= p.ReorderPoint THEN 'CRITICO'
            WHEN SUM(pi.Quantity) <= p.SafetyStockLevel THEN 'BAIXO'
            ELSE 'NORMAL'
        END                             AS StockStatus
    FROM Production.Product p
    LEFT JOIN Production.ProductInventory pi ON p.ProductID = pi.ProductID
    LEFT JOIN Production.WorkOrder wo
        ON p.ProductID = wo.ProductID AND wo.EndDate IS NULL
    WHERE p.FinishedGoodsFlag = 1
    GROUP BY
        p.ProductID, p.Name, p.ReorderPoint,
        p.SafetyStockLevel, p.StandardCost, p.ListPrice
    ORDER BY CurrentStock ASC
    """
    return pd.read_sql(query, get_engine())


def fetch_product_features() -> pd.DataFrame:
    """
    Features agregadas por produto para o modelo de ML.
    Combina vendas, estoque e características do produto.
    """
    query = """
    SELECT
        p.ProductID,
        p.Name                              AS ProductName,
        p.StandardCost,
        p.ListPrice,
        p.ReorderPoint,
        p.SafetyStockLevel,
        p.DaysToManufacture,
        pc.Name                             AS Category,
        -- Métricas de venda dos últimos 12 meses
        ISNULL(s.TotalQty12m, 0)            AS TotalQty12m,
        ISNULL(s.AvgMonthlyQty, 0)          AS AvgMonthlyQty,
        ISNULL(s.StdDevQty, 0)              AS StdDevQty,
        ISNULL(s.MonthsWithSales, 0)        AS MonthsWithSales,
        -- Estoque atual
        ISNULL(inv.CurrentStock, 0)         AS CurrentStock
    FROM Production.Product p
    LEFT JOIN Production.ProductSubcategory ps
        ON p.ProductSubcategoryID = ps.ProductSubcategoryID
    LEFT JOIN Production.ProductCategory pc
        ON ps.ProductCategoryID = pc.ProductCategoryID
    LEFT JOIN (
        SELECT
            sod.ProductID,
            SUM(sod.OrderQty)               AS TotalQty12m,
            AVG(CAST(sod.OrderQty AS FLOAT)) AS AvgMonthlyQty,
            STDEV(sod.OrderQty)             AS StdDevQty,
            COUNT(DISTINCT FORMAT(soh.OrderDate,'yyyy-MM')) AS MonthsWithSales
        FROM Sales.SalesOrderDetail sod
        JOIN Sales.SalesOrderHeader soh ON sod.SalesOrderID = soh.SalesOrderID
        WHERE soh.OrderDate >= DATEADD(MONTH, -12, GETDATE())
        GROUP BY sod.ProductID
    ) s ON p.ProductID = s.ProductID
    LEFT JOIN (
        SELECT ProductID, SUM(Quantity) AS CurrentStock
        FROM Production.ProductInventory
        GROUP BY ProductID
    ) inv ON p.ProductID = inv.ProductID
    WHERE p.FinishedGoodsFlag = 1
    """
    return pd.read_sql(query, get_engine())