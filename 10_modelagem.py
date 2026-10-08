import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CARREGAR CONJUNTO DE TESTE
# ============================================================

print("==========================================")
print("CARREGANDO CONJUNTO DE TESTE")
print("==========================================")

X_test = pd.read_parquet(
    r"salvos\X_test_model.parquet"
)

y_test = pd.read_parquet(
    r"salvos\y_test.parquet"
)["loan_default"]


# ============================================================
# 2. CARREGAR BASES DE TREINAMENTO
# ============================================================

X_imb = pd.read_parquet(
    r"salvos\X_imb.parquet"
)

y_imb = pd.read_parquet(
    r"salvos\y_imb.parquet"
)["loan_default"]


X_us = pd.read_parquet(
    r"salvos\X_us.parquet"
)

y_us = pd.read_parquet(
    r"salvos\y_us.parquet"
)["loan_default"]


X_os = pd.read_parquet(
    r"salvos\X_os.parquet"
)

y_os = pd.read_parquet(
    r"salvos\y_os.parquet"
)["loan_default"]


X_rose = pd.read_parquet(
    r"salvos\X_rose.parquet"
)

y_rose = pd.read_parquet(
    r"salvos\y_rose.parquet"
)["loan_default"]


# ============================================================
# 3. IDENTIFICAR VARIÁVEIS
# ============================================================

variaveis_categoricas = [
    "term",
    "grade",
    "home_ownership",
    "emp_length"
]


variaveis_numericas = [
    coluna
    for coluna in X_imb.columns
    if coluna not in variaveis_categoricas
]


print("\n==========================================")
print("VARIÁVEIS")
print("==========================================")

print(f"Numéricas: {len(variaveis_numericas)}")
print(variaveis_numericas)

print(f"\nCategóricas: {len(variaveis_categoricas)}")
print(variaveis_categoricas)


# ============================================================
# 4. ONE-HOT ENCODER
# ============================================================

print("\n==========================================")
print("PREPARANDO ONE-HOT ENCODER")
print("==========================================")


encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# IMPORTANTE:
# O encoder é ajustado somente no treinamento original.
encoder.fit(
    X_imb[variaveis_categoricas]
)


# ============================================================
# 5. FUNÇÃO DE TRANSFORMAÇÃO
# ============================================================

def preparar_dados(X):

    # Parte numérica
    X_numerico = X[
        variaveis_numericas
    ].reset_index(drop=True)


    # Parte categórica
    X_categorico = encoder.transform(
        X[
            variaveis_categoricas
        ]
    )


    # Nomes das novas colunas
    nomes_categoricos = encoder.get_feature_names_out(
        variaveis_categoricas
    )


    X_categorico = pd.DataFrame(
        X_categorico,
        columns=nomes_categoricos
    )


    # Junta tudo
    X_final = pd.concat(
        [
            X_numerico,
            X_categorico
        ],
        axis=1
    )


    return X_final


# ============================================================
# 6. TRANSFORMAR AS BASES
# ============================================================

print("\n==========================================")
print("TRANSFORMANDO BASES")
print("==========================================")


X_imb_model = preparar_dados(X_imb)

print(
    "IMB:",
    X_imb_model.shape
)


X_us_model = preparar_dados(X_us)

print(
    "US:",
    X_us_model.shape
)


X_os_model = preparar_dados(X_os)

print(
    "OS:",
    X_os_model.shape
)


X_rose_model = preparar_dados(X_rose)

print(
    "ROSE:",
    X_rose_model.shape
)


X_test_model = preparar_dados(X_test)

print(
    "TEST:",
    X_test_model.shape
)


# ============================================================
# 7. VERIFICAR COLUNAS
# ============================================================

print("\n==========================================")
print("VERIFICANDO COLUNAS")
print("==========================================")


print(
    "IMB == TEST:",
    list(X_imb_model.columns) == list(X_test_model.columns)
)

print(
    "US == TEST:",
    list(X_us_model.columns) == list(X_test_model.columns)
)

print(
    "OS == TEST:",
    list(X_os_model.columns) == list(X_test_model.columns)
)

print(
    "ROSE == TEST:",
    list(X_rose_model.columns) == list(X_test_model.columns)
)


# ============================================================
# 8. VERIFICAR VALORES AUSENTES
# ============================================================

print("\n==========================================")
print("VERIFICANDO MISSING")
print("==========================================")


print(
    "IMB:",
    X_imb_model.isna().sum().sum()
)

print(
    "US:",
    X_us_model.isna().sum().sum()
)

print(
    "OS:",
    X_os_model.isna().sum().sum()
)

print(
    "ROSE:",
    X_rose_model.isna().sum().sum()
)

print(
    "TEST:",
    X_test_model.isna().sum().sum()
)


# ============================================================
# 9. RESUMO
# ============================================================

print("\n==========================================")
print("PREPARAÇÃO FINALIZADA")
print("==========================================")

print(
    f"IMB  : {X_imb_model.shape}"
)

print(
    f"US   : {X_us_model.shape}"
)

print(
    f"OS   : {X_os_model.shape}"
)

print(
    f"ROSE : {X_rose_model.shape}"
)

print(
    f"TEST : {X_test_model.shape}"
)