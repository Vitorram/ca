import pandas as pd

# ------------------------------------------
# 1. Carregar a base da etapa 03
# ------------------------------------------

df = pd.read_parquet(
    r"salvos\lending_club_selecao.parquet"
)

print("Shape inicial:")
print(df.shape)


# ------------------------------------------
# 2. Selecionar variáveis numéricas
# ------------------------------------------

df_numerico = df.select_dtypes(
    include=["number"]
)

print("\nQuantidade de variáveis numéricas:")
print(df_numerico.shape[1])


# ------------------------------------------
# 3. Calcular matriz de correlação
# ------------------------------------------

correlacao = df_numerico.corr()


# ------------------------------------------
# 4. Mostrar correlações relevantes
# ------------------------------------------

pares_importantes = [
    ("fico_range_low", "fico_range_high"),
    ("loan_amnt", "installment"),
    ("open_acc", "total_acc")
]

print("\nCorrelações analisadas pela dissertação:")

for var1, var2 in pares_importantes:

    if var1 in correlacao.columns and var2 in correlacao.columns:

        valor = correlacao.loc[var1, var2]

        print(
            f"{var1} × {var2}: {valor:.3f}"
        )

    else:

        print(
            f"{var1} × {var2}: "
            "uma ou ambas não existem na base"
        )


# ------------------------------------------
# 5. Remover variáveis conforme a dissertação
# ------------------------------------------

colunas_remover = [
    "fico_range_low",
    "installment"
]

colunas_existentes = [
    coluna
    for coluna in colunas_remover
    if coluna in df.columns
]

print("\nVariáveis removidas:")

for coluna in colunas_existentes:
    print(f"- {coluna}")

df = df.drop(
    columns=colunas_existentes
)


# ------------------------------------------
# 6. Verificar open_acc e total_acc
# ------------------------------------------

print("\nVerificação de open_acc e total_acc:")

if "open_acc" in df.columns:
    print("open_acc → mantida")

if "total_acc" in df.columns:
    print("total_acc → mantida")


# ------------------------------------------
# 7. Resultado
# ------------------------------------------

print("\nShape final:")
print(df.shape)


# ------------------------------------------
# 8. Salvar
# ------------------------------------------

df.to_parquet(
    r"salvos\lending_club_correlacao.parquet",
    index=False
)

print("\nArquivo salvo:")
print(
    r"salvos\lending_club_correlacao.parquet"
)