import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# 1. CARREGAR DADOS
# ==========================================

arquivo = r"salvos\lending_club_outliers.parquet"

df = pd.read_parquet(arquivo)

print("Shape inicial:")
print(df.shape)


# ==========================================
# 2. CRIAR A VARIÁVEL TARGET
# ==========================================

mapa_default = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "In Grace Period": 1,
    "Late (16-30 days)": 1,
    "Late (31-120 days)": 1
}

df["loan_default"] = df["loan_status"].map(mapa_default)


# ==========================================
# 3. REMOVER STATUS NÃO UTILIZADOS
# ==========================================

antes = len(df)

df = df.dropna(subset=["loan_default"])

depois = len(df)

print("\nRegistros removidos:")
print(antes - depois)


# ==========================================
# 4. CONVERTER TARGET PARA INTEIRO
# ==========================================

df["loan_default"] = df["loan_default"].astype(int)


# ==========================================
# 5. DISTRIBUIÇÃO DO TARGET
# ==========================================

print("\nDistribuição de loan_default:")

print(df["loan_default"].value_counts())

print("\nDistribuição percentual:")

print(
    df["loan_default"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==========================================
# 6. SEPARAR X E Y
# ==========================================

X = df.drop(columns=["loan_default", "loan_status"])

y = df["loan_default"]


# ==========================================
# 7. DIVISÃO 70/30
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. RESULTADOS
# ==========================================

print("\nTreinamento:")
print(X_train.shape)

print("\nTeste:")
print(X_test.shape)

print("\nTarget treinamento:")
print(y_train.value_counts())

print("\nTarget teste:")
print(y_test.value_counts())


# ==========================================
# 9. SALVAR
# ==========================================

X_train.to_parquet(r"salvos\X_train.parquet")
X_test.to_parquet(r"salvos\X_test.parquet")

y_train.to_frame().to_parquet(r"salvos\y_train.parquet")
y_test.to_frame().to_parquet(r"salvos\y_test.parquet")

print("\nArquivos salvos:")
print("salvos\\X_train.parquet")
print("salvos\\X_test.parquet")
print("salvos\\y_train.parquet")
print("salvos\\y_test.parquet")