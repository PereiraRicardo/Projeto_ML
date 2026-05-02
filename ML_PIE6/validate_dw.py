"""
Validação do ELSONS_DW antes de treinar o ML.
Execute: python validate_dw.py
"""
import pyodbc
from datetime import datetime

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=HomeOffice\\SQLEXPRESS;"
    "DATABASE=ELSONS_DW;"
    "UID=sa;"
    "PWD=123456;"
    "TrustServerCertificate=yes;"
)


def run():
    conn = pyodbc.connect(CONN_STR)
    cur  = conn.cursor()

    checks = [
        # (descrição, query, mínimo esperado)
        ("FT_VENDAS - total de linhas",
         "SELECT COUNT(*) FROM FT_VENDAS", 1000),

        ("DIM_PRODUTO - total de produtos",
         "SELECT COUNT(DISTINCT IDProduto) FROM DIM_PRODUTO", 1),

        ("DIM_TEMPO - datas disponíveis",
         "SELECT COUNT(*) FROM DIM_TEMPO", 365),

        ("DIM_CLIENTE - clientes cadastrados",
         "SELECT COUNT(DISTINCT IDCliente) FROM DIM_CLIENTE", 1),

        ("DIM_VENDEDOR - vendedores cadastrados",
         "SELECT COUNT(DISTINCT IDVendedor) FROM DIM_VENDEDOR", 1),

        ("FT_VENDAS - FKs nulas (IDTempo)",
         "SELECT COUNT(*) FROM FT_VENDAS WHERE IDTempo IS NULL", -1),

        ("FT_VENDAS - FKs nulas (IDProduto)",
         "SELECT COUNT(*) FROM FT_VENDAS WHERE IDProduto IS NULL", -1),

        ("FT_VENDAS - FKs nulas (IDCliente)",
         "SELECT COUNT(*) FROM FT_VENDAS WHERE IDCliente IS NULL", -1),

        ("FT_VENDAS - FKs nulas (IDVendedor)",
         "SELECT COUNT(*) FROM FT_VENDAS WHERE IDVendedor IS NULL", -1),

        ("FT_VENDAS - vendas sem quantidade (Quant <= 0)",
         "SELECT COUNT(*) FROM FT_VENDAS WHERE Quant <= 0", -1),

        ("FT_VENDAS - vendas sem valor (TotLiq <= 0)",
         "SELECT COUNT(*) FROM FT_VENDAS WHERE TotLiq <= 0", -1),

        ("DIM_PRODUTO - produtos sem Uprc (preço nulo)",
         "SELECT COUNT(*) FROM DIM_PRODUTO WHERE Uprc IS NULL OR Uprc = 0", -1),
    ]

    print("\n" + "=" * 60)
    print("  VALIDACAO DO DATA WAREHOUSE — ELSONS_DW")
    print("=" * 60)

    all_ok = True
    for desc, query, minimo in checks:
        cur.execute(query)
        val = cur.fetchone()[0]
        if minimo == -1:
            ok = val == 0
            tag = "OK" if ok else "ATENCAO"
        else:
            ok = val >= minimo
            tag = "OK" if ok else "ERRO"
        print(f"  [{tag:^7}]  {desc}: {val:,}")
        if not ok:
            all_ok = False

    # Período real dos dados
    cur.execute("""
        SELECT MIN(dt.Data), MAX(dt.Data),
               DATEDIFF(MONTH, MIN(dt.Data), MAX(dt.Data)) AS TotalMeses
        FROM FT_VENDAS fv
        JOIN DIM_TEMPO dt ON fv.IDTempo = dt.IDSK
    """)
    row = cur.fetchone()
    if row[0]:
        print(f"\n  Periodo em FT_VENDAS : {row[0]}  ->  {row[1]}  ({row[2]} meses)")

    # Receita e lucro aproximado
    cur.execute("""
        SELECT
            SUM(fv.TotLiq)                              AS Receita,
            SUM(fv.TotLiq * ISNULL(dp.Margem, 0) / 100) AS LucroAprox
        FROM FT_VENDAS fv
        JOIN (
            SELECT IDProduto, MAX(IDSK) AS IDSK
            FROM DIM_PRODUTO GROUP BY IDProduto
        ) dpl ON fv.IDProduto = dpl.IDProduto
        JOIN DIM_PRODUTO dp ON dp.IDSK = dpl.IDSK
    """)
    row = cur.fetchone()
    if row[0]:
        print(f"  Receita liquida total: R$ {row[0]:>15,.2f}")
        print(f"  Lucro aprox. total   : R$ {row[1]:>15,.2f}  (via Margem em DIM_PRODUTO)")

    # Produtos aptos para ML (>= 6 meses de histórico)
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT fv.IDProduto
            FROM FT_VENDAS fv
            JOIN DIM_TEMPO dt ON fv.IDTempo = dt.IDSK
            GROUP BY fv.IDProduto
            HAVING COUNT(DISTINCT YEAR(dt.Data) * 100 + MONTH(dt.Data)) >= 6
        ) t
    """)
    aptos = cur.fetchone()[0]
    print(f"  Produtos >= 6 meses (aptos ML)  : {aptos:,}")

    # Produtos com >= 12 meses (Prophet)
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT fv.IDProduto
            FROM FT_VENDAS fv
            JOIN DIM_TEMPO dt ON fv.IDTempo = dt.IDSK
            GROUP BY fv.IDProduto
            HAVING COUNT(DISTINCT YEAR(dt.Data) * 100 + MONTH(dt.Data)) >= 12
        ) t
    """)
    prophet = cur.fetchone()[0]
    print(f"  Produtos >= 12 meses (Prophet)  : {prophet:,}")

    # Top 5 grupos por volume
    cur.execute("""
        SELECT TOP 5 dp.Grupo, SUM(fv.Quant) AS TotalQty, SUM(fv.TotLiq) AS TotalRec
        FROM FT_VENDAS fv
        JOIN (SELECT IDProduto, MAX(IDSK) AS IDSK FROM DIM_PRODUTO GROUP BY IDProduto) dpl
            ON fv.IDProduto = dpl.IDProduto
        JOIN DIM_PRODUTO dp ON dp.IDSK = dpl.IDSK
        GROUP BY dp.Grupo
        ORDER BY TotalRec DESC
    """)
    grupos = cur.fetchall()
    if grupos:
        print("\n  Top 5 grupos por receita:")
        for g in grupos:
            print(f"    {g[0]:<40}  Qty: {g[1]:>10,.0f}  Rec: R$ {g[2]:>12,.2f}")

    # FT_TROCAS (devoluções)
    cur.execute("SELECT COUNT(*), ISNULL(SUM(TotLiq), 0) FROM FT_TROCAS")
    row = cur.fetchone()
    print(f"\n  FT_TROCAS (devolucoes): {row[0]:,} registros | R$ {row[1]:,.2f}")

    print("\n" + "=" * 60)
    if all_ok:
        print("  [OK] DW validado — pode treinar os modelos ML!")
    else:
        print("  [ATENCAO] Verifique os itens marcados acima antes de treinar.")
    print("=" * 60 + "\n")

    conn.close()


if __name__ == "__main__":
    run()
