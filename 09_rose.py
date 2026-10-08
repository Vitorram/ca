import pandas as pd
import numpy as np


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ============================================================
# 1. CARREGAR DADOS
# ============================================================

print("==========================================")
print("CARREGANDO DADOS")
print("==========================================")

X_train = pd.read_parquet(
    r"salvos\X_dt_train.parquet"
)

y_train = pd.read_parquet(
    r"salvos\y_dt_train.parquet"
)["loan_default"]


print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")


# ============================================================
# 2. DISTRIBUIÇÃO ORIGINAL
# ============================================================

print("\n==========================================")
print("DISTRIBUIÇÃO ORIGINAL")
print("==========================================")

print(y_train.value_counts())

print("\nPercentual:")
print(
    y_train
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 3. SEPARAR CLASSES
# ============================================================

print("\n==========================================")
print("SEPARANDO CLASSES")
print("==========================================")

# Junta X e y temporariamente para facilitar a seleção
dados = X_train.copy()
dados["loan_default"] = y_train.values


dados_0 = dados[dados["loan_default"] == 0].copy()
dados_1 = dados[dados["loan_default"] == 1].copy()


print(f"Classe 0: {len(dados_0)}")
print(f"Classe 1: {len(dados_1)}")


# ============================================================
# 4. QUANTIDADE DE DADOS SINTÉTICOS
# ============================================================

quantidade_classe_0 = len(dados_0)
quantidade_classe_1 = len(dados_1)

quantidade_sintetica = quantidade_classe_0 - quantidade_classe_1


print("\n==========================================")
print("QUANTIDADE SINTÉTICA")
print("==========================================")

print(
    f"Precisamos gerar {quantidade_sintetica:,} "
    "observações sintéticas."
)


# ============================================================
# 5. VARIÁVEIS
# ============================================================

variaveis_numericas = [
    coluna
    for coluna in X_train.columns
    if pd.api.types.is_numeric_dtype(X_train[coluna])
]


variaveis_categoricas = [
    coluna
    for coluna in X_train.columns
    if coluna not in variaveis_numericas
]


print("\n==========================================")
print("VARIÁVEIS")
print("==========================================")

print("\nNuméricas:")
print(variaveis_numericas)

print("\nCategóricas/textuais:")
print(variaveis_categoricas)


# ============================================================
# 6. PREPARAR A CLASSE MINORITÁRIA
# ============================================================

X_minority = dados_1.drop(
    columns=["loan_default"]
).reset_index(drop=True)


# ============================================================
# 7. GERAR ÍNDICES DAS OBSERVAÇÕES-BASE
# ============================================================

print("\n==========================================")
print("GERANDO OBSERVAÇÕES-BASE")
print("==========================================")


indices_base = np.random.choice(
    len(X_minority),
    size=quantidade_sintetica,
    replace=True
)


dados_sinteticos = X_minority.iloc[
    indices_base
].copy().reset_index(drop=True)


# ============================================================
# 8. GERAR DADOS SINTÉTICOS NUMÉRICOS
# ============================================================

print("\n==========================================")
print("GERANDO DADOS SINTÉTICOS")
print("==========================================")


for coluna in variaveis_numericas:

    valores = X_minority[coluna].astype(float)

    desvio = valores.std()

    # Caso uma variável tenha desvio zero,
    # não adicionamos ruído.
    if pd.isna(desvio) or desvio == 0:
        continue

    # Banda de suavização.
    #
    # O objetivo é criar valores próximos às
    # observações reais, mas não simplesmente
    # copiar seus valores.
    bandwidth = 0.10 * desvio

    ruido = np.random.normal(
        loc=0,
        scale=bandwidth,
        size=quantidade_sintetica
    )

    dados_sinteticos[coluna] = (
        dados_sinteticos[coluna].astype(float)
        + ruido
    )


# ============================================================
# 9. LIMITAR VALORES AO INTERVALO OBSERVADO
# ============================================================

print("\n==========================================")
print("VALIDANDO LIMITES")
print("==========================================")


for coluna in variaveis_numericas:

    minimo = X_minority[coluna].min()
    maximo = X_minority[coluna].max()

    dados_sinteticos[coluna] = (
        dados_sinteticos[coluna]
        .clip(
            lower=minimo,
            upper=maximo
        )
    )


# ============================================================
# 10. VARIÁVEIS CATEGÓRICAS
# ============================================================

print("\n==========================================")
print("TRATANDO VARIÁVEIS CATEGÓRICAS")
print("==========================================")


for coluna in variaveis_categoricas:

    # A categoria da observação-base é mantida.
    #
    # Isso evita criar valores inválidos como:
    # grade = 4.37
    # term = 48.2
    # home_ownership = 1.53

    dados_sinteticos[coluna] = (
        dados_sinteticos[coluna]
        .astype(X_minority[coluna].dtype)
    )


# ============================================================
# 11. GARANTIR A MESMA ORDEM DAS COLUNAS
# ============================================================

dados_sinteticos = dados_sinteticos[
    X_train.columns
]


# ============================================================
# 12. CRIAR TARGET SINTÉTICO
# ============================================================

y_sintetico = pd.Series(
    np.ones(
        quantidade_sintetica,
        dtype=int
    ),
    name="loan_default"
)


# ============================================================
# 13. JUNTAR COM A CLASSE MINORITÁRIA ORIGINAL
# ============================================================

X_classe_1 = X_minority.copy()

y_classe_1 = pd.Series(
    np.ones(
        len(X_classe_1),
        dtype=int
    ),
    name="loan_default"
)


X_classe_1_rose = pd.concat(
    [
        X_classe_1,
        dados_sinteticos
    ],
    ignore_index=True
)


y_classe_1_rose = pd.concat(
    [
        y_classe_1,
        y_sintetico
    ],
    ignore_index=True
)


# ============================================================
# 14. JUNTAR COM A CLASSE MAJORITÁRIA
# ============================================================

X_classe_0 = dados_0.drop(
    columns=["loan_default"]
).reset_index(drop=True)


y_classe_0 = pd.Series(
    np.zeros(
        len(X_classe_0),
        dtype=int
    ),
    name="loan_default"
)


X_rose = pd.concat(
    [
        X_classe_0,
        X_classe_1_rose
    ],
    ignore_index=True
)


y_rose = pd.concat(
    [
        y_classe_0,
        y_classe_1_rose
    ],
    ignore_index=True
)


# ============================================================
# 15. EMBARALHAR
# ============================================================

indices = np.random.permutation(
    len(X_rose)
)


X_rose = X_rose.iloc[
    indices
].reset_index(drop=True)


y_rose = y_rose.iloc[
    indices
].reset_index(drop=True)


# ============================================================
# 16. RESULTADO FINAL
# ============================================================

print("\n==========================================")
print("RESULTADO ROSE")
print("==========================================")

print("\nShape X_rose:")
print(X_rose.shape)

print("\nShape y_rose:")
print(y_rose.shape)


print("\nDistribuição:")
print(
    y_rose.value_counts()
)


print("\nPercentual:")
print(
    y_rose
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 17. VERIFICAR DUPLICATAS
# ============================================================

print("\n==========================================")
print("VERIFICAÇÃO")
print("==========================================")

print(
    f"Duplicatas em X_rose: "
    f"{X_rose.duplicated().sum():,}"
)


# ============================================================
# 18. SALVAR
# ============================================================

print("\n==========================================")
print("SALVANDO ARQUIVOS")
print("==========================================")


X_rose.to_parquet(
    r"salvos\X_dt_rose.parquet",
    index=False
)
X_rose.to_parquet(
    r"salvos\X_rose.parquet",
    index=False
)


y_rose.to_frame().to_parquet(
    r"salvos\y_dt_rose.parquet",
    index=False
)
y_rose.to_frame().to_parquet(
    r"salvos\y_rose.parquet",
    index=False
)


print("\nArquivos salvos:")

print(
    r"salvos\X_dt_rose.parquet"
)

print(
    r"salvos\y_dt_rose.parquet"
)


print("\n==========================================")
print("ROSE FINALIZADO")
print("==========================================")
