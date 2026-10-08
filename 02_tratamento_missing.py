import pandas as pd

# ------------------------------------------
# 1. Carregar a base preparada no arquivo 01
# ------------------------------------------

df = pd.read_parquet("salvos/lending_club_macro.parquet")
print("Shape inicial:")
print(df.shape)


# ------------------------------------------
# 2. Calcular percentual de valores ausentes
# ------------------------------------------

missing_percentual = df.isnull().mean() * 100
print("\nColunas com valores ausentes:")
print(
    missing_percentual[missing_percentual > 0]
    .sort_values(ascending=False)
)


# ------------------------------------------
# 3. Remover colunas com mais de 40% missing
# ------------------------------------------

limite_missing = 40

colunas_remover = missing_percentual[
    missing_percentual > limite_missing
].index

print("\nQuantidade de colunas removidas:")
print(len(colunas_remover))

print("\nColunas removidas:")
for coluna in colunas_remover:
    print(f"- {coluna}")

df = df.drop(columns=colunas_remover)


# ------------------------------------------
# 4. Separar variáveis numéricas e categóricas
# ------------------------------------------

colunas_numericas = df.select_dtypes(
    include=["number"]
).columns

colunas_categoricas = df.select_dtypes(
    exclude=["number"]
).columns

print("\nQuantidade de variáveis numéricas:")
print(len(colunas_numericas))

print("\nQuantidade de variáveis categóricas:")
print(len(colunas_categoricas))


# ------------------------------------------
# 5. Preencher variáveis numéricas com a média
# ------------------------------------------

for coluna in colunas_numericas:

    if df[coluna].isnull().any():

        media = df[coluna].mean()

        df[coluna] = df[coluna].fillna(media)

        print(
            f"Numérica: {coluna} → "
            f"missing preenchido com média = {media:.4f}"
        )


# ------------------------------------------
# 6. Preencher variáveis categóricas com a moda
# ------------------------------------------

for coluna in colunas_categoricas:

    if df[coluna].isnull().any():

        moda = df[coluna].mode()

        if len(moda) > 0:

            df[coluna] = df[coluna].fillna(
                moda.iloc[0]
            )

            print(
                f"Categórica: {coluna} → "
                f"missing preenchido com moda = {moda.iloc[0]}"
            )


# ------------------------------------------
# 7. Verificar se ainda existem valores ausentes
# ------------------------------------------

total_missing = df.isnull().sum().sum()

print("\nTotal de valores ausentes restantes:")
print(total_missing)


# ------------------------------------------
# 8. Resultado final
# ------------------------------------------

print("\nShape final:")
print(df.shape)


# ------------------------------------------
# 9. Salvar resultado da etapa
# ------------------------------------------

df.to_parquet(
    "lending_club_missing.parquet",
    index=False
)

print("\nArquivo salvo:")
print("lending_club_missing.parquet")