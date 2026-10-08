import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

modelo_lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)
# ============================================================
# 1. CARREGAMENTO
# ============================================================

arquivo = "accepted_2007_to_2018Q4.csv"

df = pd.read_csv(arquivo)

print("Dataset carregado:")
print(df.shape)


# ============================================================
# 2. DEFINIÇÃO DO TARGET
# ============================================================

status_default = [
    "Charged Off",
    "Default",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)"
]

df["loan_default"] = df["loan_status"].map({
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "In Grace Period": 1,
    "Late (16-30 days)": 1,
    "Late (31-120 days)": 1
})


# ============================================================
# 3. FILTRAR OS STATUS UTILIZADOS
# ============================================================

df = df[
    df["loan_status"].isin(
        ["Fully Paid"] + status_default
    )
].copy()


# ============================================================
# 4. VARIÁVEIS SELECIONADAS
# ============================================================

variaveis_modelo = [
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "emp_length",
    "dti",
    "delinq_2yrs",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "revol_util",
    "home_ownership",
    "annual_inc",
    "issue_d",
    "total_acc"
]


# ============================================================
# 5. REMOÇÃO DE COLUNAS COM MAIS DE 40% DE MISSING
# ============================================================

missing_percent = (
    df.isnull().sum() / len(df)
) * 100

colunas_missing = missing_percent[
    missing_percent > 40
].index.tolist()

df = df.drop(
    columns=colunas_missing
)


# ============================================================
# 6. TRATAMENTO DOS VALORES AUSENTES
# ============================================================

colunas_numericas = df.select_dtypes(
    include=["number"]
).columns

colunas_categoricas = df.select_dtypes(
    include=["object", "str"]
).columns


# Numéricas → média
for coluna in colunas_numericas:

    if df[coluna].isnull().any():

        df[coluna] = df[coluna].fillna(
            df[coluna].mean()
        )


# Categóricas → moda
for coluna in colunas_categoricas:

    if df[coluna].isnull().any():

        moda = df[coluna].mode()

        if not moda.empty:

            df[coluna] = df[coluna].fillna(
                moda.iloc[0]
            )


# ============================================================
# 7. FEATURE SELECTION
# ============================================================

variaveis_remover = [
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

df = df.drop(
    columns=variaveis_remover,
    errors="ignore"
)


# ============================================================
# 8. REMOÇÃO DAS VARIÁVEIS CORRELACIONADAS
# ============================================================

variaveis_correlacionadas = [
    "fico_range_low",
    "installment"
]

df = df.drop(
    columns=variaveis_correlacionadas,
    errors="ignore"
)


# ============================================================
# 9. TRATAMENTO DOS OUTLIERS DO DTI
# ============================================================

percentil_5 = df["dti"].quantile(0.05)
percentil_95 = df["dti"].quantile(0.95)

df.loc[
    df["dti"] < percentil_5,
    "dti"
] = percentil_5

df.loc[
    df["dti"] > percentil_95,
    "dti"
] = percentil_95


# ============================================================
# 10. CODIFICAÇÃO DAS VARIÁVEIS CATEGÓRICAS
# ============================================================

variaveis_categoricas = [
    "term",
    "grade",
    "home_ownership"
]

df = pd.get_dummies(
    df,
    columns=variaveis_categoricas,
    dtype=int
)


# ============================================================
# 11. CONVERSÃO DE EMP_LENGTH
# ============================================================

mapa_emp_length = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10
}

df["emp_length"] = df[
    "emp_length"
].map(mapa_emp_length)


# ============================================================
# 12. TRATAMENTO DA DATA
# ============================================================

df["ano"] = df[
    "issue_d"
].str[-4:]

df["ano"] = pd.to_numeric(
    df["ano"],
    errors="coerce"
)

df = df.drop(
    columns=["issue_d"]
)


# ============================================================
# 13. SEPARAÇÃO DAS FEATURES E TARGET
# ============================================================

# ============================================================
# 13. SELEÇÃO FINAL DAS FEATURES
# ============================================================

colunas_features = [
    "loan_amnt",
    "int_rate",
    "emp_length",
    "dti",
    "delinq_2yrs",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "revol_util",
    "annual_inc",
    "total_acc",
    "ano"
]

# Adicionar as colunas criadas pelo one-hot encoding
colunas_dummy = [
    coluna
    for coluna in df.columns
    if coluna.startswith("term_")
    or coluna.startswith("grade_")
    or coluna.startswith("home_ownership_")
]

colunas_features.extend(colunas_dummy)

X = df[colunas_features].copy()

y = df["loan_default"].copy()




# ============================================================
# 14. DIVISÃO TREINO / TESTE
# ============================================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


print("\n========================================")
print("DIVISÃO TREINO / TESTE")
print("========================================")

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")


# ============================================================
# 15. TREINAMENTO — REGRESSÃO LOGÍSTICA
#     CENÁRIO IMB (SEM BALANCEAMENTO)
# ============================================================



print("\n========================================")
print("TREINANDO REGRESSÃO LOGÍSTICA")
print("CENÁRIO: IMB")
print("========================================")

modelo_lr.fit(
    X_train,
    y_train
)


print("\nTreinamento concluído!")


# ============================================================
# 16. PREDIÇÕES
# ============================================================

y_pred = modelo_lr.predict(
    X_test
)

y_prob = modelo_lr.predict_proba(
    X_test
)[:, 1]


print("\nPrimeiras 10 classes previstas:")
print(y_pred[:10])

print("\nPrimeiras 10 probabilidades de default:")
print(y_prob[:10])

# ============================================================
# 17. MATRIZ DE CONFUSÃO — REGRESSÃO LOGÍSTICA IMB
# ============================================================

from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(
    y_test,
    y_pred
)

print("\n========================================")
print("MATRIZ DE CONFUSÃO")
print("========================================")

print(matriz)