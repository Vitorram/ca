import pandas as pd

# ------------------------------------------
# 1. Carregar a base da etapa 02
# ------------------------------------------

df = pd.read_parquet(
    r"salvos\lending_club_missing.parquet"
)

print("Shape inicial:")
print(df.shape)


# ------------------------------------------
# 2. Variáveis removidas segundo a dissertação
# ------------------------------------------

colunas_remover = [
    "pymnt_plan",
    "policy_code",
    "url",
    "id",
    "address_state",
    "purpose",
    "title",
    "zip_code",
    "emp_title"
]

# Verificar quais realmente existem na base
colunas_existentes = [
    coluna
    for coluna in colunas_remover
    if coluna in df.columns
]

colunas_inexistentes = [
    coluna
    for coluna in colunas_remover
    if coluna not in df.columns
]

print("\nColunas que serão removidas:")

for coluna in colunas_existentes:
    print(f"- {coluna}")

if colunas_inexistentes:
    print("\nColunas da lista que não existem na base:")

    for coluna in colunas_inexistentes:
        print(f"- {coluna}")


df = df.drop(
    columns=colunas_existentes
)


# ------------------------------------------
# 3. Transformar variáveis categóricas
#    em categoria
# ------------------------------------------

colunas_categoricas = [
    "grade",
    "home_ownership",
    "term"
]

for coluna in colunas_categoricas:

    if coluna in df.columns:
        df[coluna] = df[coluna].astype("category")


# ------------------------------------------
# 4. Converter strings numéricas
# ------------------------------------------

# int_rate
if "int_rate" in df.columns:

    df["int_rate"] = (
        df["int_rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
    )

    df["int_rate"] = pd.to_numeric(
        df["int_rate"],
        errors="coerce"
    )


# dti
if "dti" in df.columns:

    df["dti"] = pd.to_numeric(
        df["dti"],
        errors="coerce"
    )


# funded_amnt_inv
if "funded_amnt_inv" in df.columns:

    df["funded_amnt_inv"] = pd.to_numeric(
        df["funded_amnt_inv"],
        errors="coerce"
    )


# ------------------------------------------
# 5. Converter datas
# ------------------------------------------

colunas_datas = [
    "issue_d",
    "earliest_cr_line",
    "last_pymnt_d",
    "last_credit_pull_d"
]

for coluna in colunas_datas:

    if coluna in df.columns:

        df[coluna] = pd.to_datetime(
            df[coluna],
            errors="coerce"
        )


# ------------------------------------------
# 6. Verificar o resultado
# ------------------------------------------

print("\nShape depois da seleção inicial:")
print(df.shape)

print("\nTipos das principais variáveis:")

colunas_verificar = [
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "emp_length",
    "dti",
    "fico_range_high",
    "home_ownership",
    "annual_inc",
    "issue_d",
    "total_acc",
    "GDP",
    "IR",
    "EX",
    "NS",
    "HD"
]

colunas_verificar = [
    coluna
    for coluna in colunas_verificar
    if coluna in df.columns
]

print(
    df[colunas_verificar].dtypes
)


# ------------------------------------------
# 7. Verificar novamente valores ausentes
# ------------------------------------------

print("\nTotal de valores ausentes:")
print(
    df.isnull().sum().sum()
)


# ------------------------------------------
# 8. Salvar resultado
# ------------------------------------------

df.to_parquet(
    r"salvos\lending_club_selecao.parquet",
    index=False
)

print("\nArquivo salvo:")
print(
    r"salvos\lending_club_selecao.parquet"
)