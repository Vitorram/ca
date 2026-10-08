import pandas as pd

# ------------------------------------------
# 1. Carregar a base da etapa 04
# ------------------------------------------

df = pd.read_parquet(
    r"salvos\lending_club_correlacao.parquet"
)

print("Shape inicial:")
print(df.shape)


# ------------------------------------------
# 2. Variáveis que possuem tratamento
#    de outliers na dissertação
# ------------------------------------------

variaveis_outliers = [
    "annual_inc",
    "revol_util",
    "dti"
]


# ------------------------------------------
# 3. Calcular percentis
# ------------------------------------------

print("\nPercentis antes do tratamento:")

for coluna in variaveis_outliers:

    if coluna in df.columns:

        p05 = df[coluna].quantile(0.05)
        p95 = df[coluna].quantile(0.95)

        print(f"\n{coluna}")
        print(f"  P05: {p05:.4f}")
        print(f"  P95: {p95:.4f}")


# ------------------------------------------
# 4. Tratamento de annual_inc
# ------------------------------------------

if "annual_inc" in df.columns:

    p95_annual = df["annual_inc"].quantile(0.95)
    max_annual = df["annual_inc"].max()

    df.loc[
        df["annual_inc"] == max_annual,
        "annual_inc"
    ] = p95_annual

    print(
        f"\nannual_inc: valor máximo ({max_annual:.4f}) "
        f"substituído pelo P95 ({p95_annual:.4f})."
    )


# ------------------------------------------
# 5. Tratamento de revol_util
# ------------------------------------------

if "revol_util" in df.columns:

    p95_revol = df["revol_util"].quantile(0.95)
    max_revol = df["revol_util"].max()

    df.loc[
        df["revol_util"] == max_revol,
        "revol_util"
    ] = p95_revol

    print(
        f"revol_util: valor máximo ({max_revol:.4f}) "
        f"substituído pelo P95 ({p95_revol:.4f})."
    )


# ------------------------------------------
# 6. Tratamento de dti
# ------------------------------------------

if "dti" in df.columns:

    p05_dti = df["dti"].quantile(0.05)
    min_dti = df["dti"].min()

    df.loc[
        df["dti"] == min_dti,
        "dti"
    ] = p05_dti

    print(
        f"dti: valor mínimo ({min_dti:.4f}) "
        f"substituído pelo P05 ({p05_dti:.4f})."
    )


# ------------------------------------------
# 7. Verificação depois do tratamento
# ------------------------------------------

print("\nValores depois do tratamento:")

for coluna in variaveis_outliers:

    if coluna in df.columns:

        print(f"\n{coluna}")

        print(
            f"  Mínimo: {df[coluna].min():.4f}"
        )

        print(
            f"  P05:    {df[coluna].quantile(0.05):.4f}"
        )

        print(
            f"  Mediana:{df[coluna].median():.4f}"
        )

        print(
            f"  P95:    {df[coluna].quantile(0.95):.4f}"
        )

        print(
            f"  Máximo: {df[coluna].max():.4f}"
        )


# ------------------------------------------
# 8. Verificar missing
# ------------------------------------------

print("\nTotal de valores ausentes:")
print(df.isnull().sum().sum())


# ------------------------------------------
# 9. Shape final
# ------------------------------------------

print("\nShape final:")
print(df.shape)


# ------------------------------------------
# 10. Salvar
# ------------------------------------------

df.to_parquet(
    r"salvos\lending_club_outliers.parquet",
    index=False
)

print("\nArquivo salvo:")
print(
    r"salvos\lending_club_outliers.parquet"
)
