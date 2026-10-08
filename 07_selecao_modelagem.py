import pandas as pd


# ==========================================
# 1. CARREGAR TREINO E TESTE
# ==========================================

X_train = pd.read_parquet(r"salvos\X_train.parquet")
X_test = pd.read_parquet(r"salvos\X_test.parquet")

print("Shape inicial do treino:")
print(X_train.shape)

print("\nShape inicial do teste:")
print(X_test.shape)


# ==========================================
# 2. VARIÁVEIS DA MODELAGEM
# ==========================================

variaveis_modelagem = [
    "int_rate",
    "fico_range_high",
    "inq_last_6mths",
    "dti",
    "loan_amnt",
    "IR",
    "HD",
    "delinq_2yrs",
    "GDP",
    "open_acc",
    "revol_util",
    "EX",
    "total_acc",
    "NS",
    "term",
    "grade",
    "home_ownership",
    "annual_inc",
    "emp_length"
]


# ==========================================
# 3. VERIFICAR SE TODAS EXISTEM
# ==========================================

faltantes_train = [
    coluna for coluna in variaveis_modelagem
    if coluna not in X_train.columns
]

faltantes_test = [
    coluna for coluna in variaveis_modelagem
    if coluna not in X_test.columns
]


if faltantes_train:
    print("\nERRO — variáveis ausentes no treino:")
    print(faltantes_train)
    raise ValueError("Existem variáveis da modelagem ausentes no treino.")

if faltantes_test:
    print("\nERRO — variáveis ausentes no teste:")
    print(faltantes_test)
    raise ValueError("Existem variáveis da modelagem ausentes no teste.")


# ==========================================
# 4. SELECIONAR AS 19 VARIÁVEIS
# ==========================================

X_train_model = X_train[variaveis_modelagem].copy()
X_test_model = X_test[variaveis_modelagem].copy()


# ==========================================
# 5. CONFERÊNCIA
# ==========================================

print("\nVariáveis selecionadas:")
for i, coluna in enumerate(X_train_model.columns, start=1):
    print(f"{i:02d}. {coluna}")


print("\nShape final do treino:")
print(X_train_model.shape)

print("\nShape final do teste:")
print(X_test_model.shape)


# ==========================================
# 6. TIPOS DAS VARIÁVEIS
# ==========================================

print("\nTipos das variáveis:")
print(X_train_model.dtypes)


# ==========================================
# 7. VALORES AUSENTES
# ==========================================

print("\nValores ausentes no treino:")
print(X_train_model.isna().sum().sum())

print("\nValores ausentes no teste:")
print(X_test_model.isna().sum().sum())


# ==========================================
# 8. SALVAR
# ==========================================

X_train_model.to_parquet(
    r"salvos\X_train_model.parquet"
)

X_test_model.to_parquet(
    r"salvos\X_test_model.parquet"
)


print("\nArquivos salvos:")
print("salvos\\X_train_model.parquet")
print("salvos\\X_test_model.parquet")