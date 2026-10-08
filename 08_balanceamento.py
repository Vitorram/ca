import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# ==========================================
# 1. CARREGAR TREINO
# ==========================================

X_train = pd.read_parquet(
    r"salvos\X_train_model.parquet"
)

y_train = pd.read_parquet(
    r"salvos\y_train.parquet"
)["loan_default"]


print("X_train:")
print(X_train.shape)

print("\ny_train:")
print(y_train.shape)


# ==========================================
# 2. DISTRIBUIÇÃO ORIGINAL
# ==========================================

print("\nDistribuição original:")

print(y_train.value_counts())

X_original = X_train.copy()
y_original = y_train.copy()

# Separar a validação antes de qualquer balanceamento, para que cópias
# criadas por over-sampling/ROSE não apareçam nos dois conjuntos.
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

X_train.reset_index(drop=True).to_parquet(
    r"salvos\X_dt_train.parquet",
    index=False
)
y_train.reset_index(drop=True).to_frame().to_parquet(
    r"salvos\y_dt_train.parquet",
    index=False
)
X_valid.reset_index(drop=True).to_parquet(
    r"salvos\X_dt_valid.parquet",
    index=False
)
y_valid.reset_index(drop=True).to_frame().to_parquet(
    r"salvos\y_dt_valid.parquet",
    index=False
)

# Bases balanceadas exclusivas da Decision Tree. A validação fica de fora.
X_dt_imb, y_dt_imb = X_train.copy(), y_train.copy()

dt_us = RandomUnderSampler(sampling_strategy="auto", random_state=42)
X_dt_us, y_dt_us = dt_us.fit_resample(X_train, y_train)

dt_os = RandomOverSampler(sampling_strategy="auto", random_state=42)
X_dt_os, y_dt_os = dt_os.fit_resample(X_train, y_train)

for nome, X_dt, y_dt in [
    ("imb", X_dt_imb, y_dt_imb),
    ("us", X_dt_us, y_dt_us),
    ("os", X_dt_os, y_dt_os),
]:
    X_dt.to_parquet(rf"salvos\X_dt_{nome}.parquet", index=False)
    y_dt.to_frame().to_parquet(rf"salvos\y_dt_{nome}.parquet", index=False)

# Mantém o comportamento anterior das bases compartilhadas pelos outros modelos.
X_train, y_train = X_original, y_original

print("\nPercentual original:")

print(
    y_train
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==========================================
# 3. IMB — SEM BALANCEAMENTO
# ==========================================

X_imb = X_train.copy()
y_imb = y_train.copy()

print("\n==========================================")
print("IMB")
print("==========================================")

print("Shape:", X_imb.shape)
print(y_imb.value_counts())


# ==========================================
# 4. UNDER-SAMPLING
# ==========================================

us = RandomUnderSampler(
    sampling_strategy="auto",
    random_state=42
)

X_us, y_us = us.fit_resample(
    X_train,
    y_train
)

print("\n==========================================")
print("UNDER-SAMPLING")
print("==========================================")

print("Shape:", X_us.shape)
print(y_us.value_counts())


# ==========================================
# 5. OVER-SAMPLING
# ==========================================

os = RandomOverSampler(
    sampling_strategy="auto",
    random_state=42
)

X_os, y_os = os.fit_resample(
    X_train,
    y_train
)

print("\n==========================================")
print("OVER-SAMPLING")
print("==========================================")

print("Shape:", X_os.shape)
print(y_os.value_counts())


# ==========================================
# 6. SALVAR IMB
# ==========================================

X_imb.to_parquet(
    r"salvos\X_imb.parquet"
)

y_imb.to_frame().to_parquet(
    r"salvos\y_imb.parquet"
)


# ==========================================
# 7. SALVAR UNDER-SAMPLING
# ==========================================

X_us.to_parquet(
    r"salvos\X_us.parquet"
)

y_us.to_frame().to_parquet(
    r"salvos\y_us.parquet"
)


# ==========================================
# 8. SALVAR OVER-SAMPLING
# ==========================================

X_os.to_parquet(
    r"salvos\X_os.parquet"
)

y_os.to_frame().to_parquet(
    r"salvos\y_os.parquet"
)


print("\n==========================================")
print("ARQUIVOS SALVOS")
print("==========================================")

print("salvos\\X_imb.parquet")
print("salvos\\y_imb.parquet")

print("salvos\\X_us.parquet")
print("salvos\\y_us.parquet")

print("salvos\\X_os.parquet")
print("salvos\\y_os.parquet")
