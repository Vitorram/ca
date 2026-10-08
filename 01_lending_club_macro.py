import pandas as pd
from pathlib import Path

PASTA_MACRO = Path("macro")

# ==========================================
# 1. CARREGAR MACRO
# ==========================================

macro = pd.read_csv(
    PASTA_MACRO / "macro.csv"
)

print("\nTabela macro:")
print(macro)

# ==========================================
# 2. CARREGAR LENDING CLUB
# ==========================================

CAMINHO_LENDING = Path("accepted_2007_to_2018Q4.csv")

df = pd.read_csv(
    CAMINHO_LENDING,
    low_memory=False
)

print("\nLending Club original:")
print(df.shape)

# ==========================================
# 3. EXTRAIR ANO
# ==========================================

df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%b-%Y",
    errors="coerce"
)

# ==========================================
# 4. FILTRAR ANO - 2008 2018
# ==========================================

df["ano"] = df["issue_d"].dt.year

df = df[
    df["ano"].between(2008, 2018)
].copy()

print("\nLending Club 2008–2018:")
print(df.shape)

print("\nAnos encontrados:")
print(
    df["ano"]
    .value_counts()
    .sort_index()
)

# ==========================================
# 5. MERGE COM MACRO
# ==========================================

df = df.merge(
    macro,
    on="ano",
    how="left"
)

print("\nDepois do merge:")
print(df.shape)

# ==========================================
# 6. VERIFICAR MACRO
# ==========================================

colunas_macro = [
    "GDP",
    "IR",
    "EX",
    "NS",
    "HD"
]

print("\nValores ausentes nas variáveis macro:")
print(
    df[colunas_macro].isna().sum()
)

print("\nExemplo:")
print(
    df[
        ["ano", "GDP", "IR", "EX", "NS", "HD"]
    ].head(10)
)
# Salvar resultado da etapa 01
df.to_parquet(
    "lending_club_macro.parquet",
    index=False
)

print("\nArquivo salvo com sucesso:")
print("lending_club_macro.parquet")