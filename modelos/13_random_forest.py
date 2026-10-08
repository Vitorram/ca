import pandas as pd

from sklearn.tree import DecisionTreeClassifier


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RANDOM_STATE = 42


# ============================================================
# 1. CARREGAR CONJUNTO DE TESTE
# ============================================================

print("==========================================")
print("DECISION TREE")
print("==========================================")

print("\nCarregando conjunto de teste...")

X_test = pd.read_parquet(
    r"..\salvos\X_test_model.parquet"
)

y_test = pd.read_parquet(
    r"..\salvos\y_test.parquet"
)["loan_default"]


print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")


# ============================================================
# 2. PREPARAR VARIÁVEIS CATEGÓRICAS
# ============================================================

print("\n==========================================")
print("PREPARANDO VARIÁVEIS")
print("==========================================")


variaveis_categoricas = [
    "term",
    "grade",
    "home_ownership",
    "emp_length"
]


variaveis_numericas = [
    coluna
    for coluna in X_test.columns
    if coluna not in variaveis_categoricas
]


# ------------------------------------------------------------
# Como os arquivos dos modelos estão dentro de /modelos,
# precisamos transformar as categorias para valores numéricos.
#
# Vamos usar o mesmo mapeamento para todas as bases.
# ------------------------------------------------------------

def preparar_categoricas(
    X_train,
    X_test
):

    X_train = X_train.copy()
    X_test = X_test.copy()

    for coluna in variaveis_categoricas:

        categorias = sorted(
            X_train[coluna]
            .astype(str)
            .unique()
        )

        mapa = {
            categoria: indice
            for indice, categoria
            in enumerate(categorias)
        }

        X_train[coluna] = (
            X_train[coluna]
            .astype(str)
            .map(mapa)
        )

        X_test[coluna] = (
            X_test[coluna]
            .astype(str)
            .map(mapa)
        )

    return X_train, X_test


# ============================================================
# 3. FUNÇÃO PARA TREINAR A ÁRVORE
# ============================================================

def treinar_decision_tree(
    nome_base,
    caminho_x,
    caminho_y
):

    print("\n==========================================")
    print(f"DT + {nome_base}")
    print("==========================================")

    # --------------------------------------------------------
    # Carregar treinamento
    # --------------------------------------------------------

    X_train = pd.read_parquet(
        caminho_x
    )

    y_train = pd.read_parquet(
        caminho_y
    )["loan_default"]


    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")


    print("\nDistribuição:")
    print(y_train.value_counts())


    # --------------------------------------------------------
    # Preparar categorias
    # --------------------------------------------------------

    X_train, X_test_preparado = preparar_categoricas(
        X_train,
        X_test
    )


    # --------------------------------------------------------
    # Garantir que todas as colunas sejam numéricas
    # --------------------------------------------------------

    X_train = X_train.astype(float)
    X_test_preparado = X_test_preparado.astype(float)


    # --------------------------------------------------------
    # Criar modelo CART
    # --------------------------------------------------------

    modelo = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )


    # --------------------------------------------------------
    # Treinamento
    # --------------------------------------------------------

    print("\nTreinando árvore...")

    modelo.fit(
        X_train,
        y_train
    )


    print("Treinamento concluído.")


    # --------------------------------------------------------
    # Predições
    # --------------------------------------------------------

    print("\nGerando predições...")

    y_pred = modelo.predict(
        X_test_preparado
    )


    y_prob = modelo.predict_proba(
        X_test_preparado
    )[:, 1]


    # --------------------------------------------------------
    # Informações da árvore
    # --------------------------------------------------------

    print("\n==========================================")
    print("INFORMAÇÕES DA ÁRVORE")
    print("==========================================")

    print(
        f"Profundidade: {modelo.get_depth()}"
    )

    print(
        f"Número de nós: {modelo.tree_.node_count}"
    )

    print(
        f"Número de folhas: {modelo.get_n_leaves()}"
    )


    # --------------------------------------------------------
    # Retornar resultados
    # --------------------------------------------------------

    return {
        "modelo": modelo,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


# ============================================================
# 4. DECISION TREE + IMB
# ============================================================

resultado_imb = treinar_decision_tree(
    "IMB",
    r"..\salvos\X_imb.parquet",
    r"..\salvos\y_imb.parquet"
)


# ============================================================
# 5. DECISION TREE + US
# ============================================================

resultado_us = treinar_decision_tree(
    "US",
    r"..\salvos\X_us.parquet",
    r"..\salvos\y_us.parquet"
)


# ============================================================
# 6. DECISION TREE + OS
# ============================================================

resultado_os = treinar_decision_tree(
    "OS",
    r"..\salvos\X_os.parquet",
    r"..\salvos\y_os.parquet"
)


# ============================================================
# 7. DECISION TREE + ROSE
# ============================================================

resultado_rose = treinar_decision_tree(
    "ROSE",
    r"..\salvos\X_rose.parquet",
    r"..\salvos\y_rose.parquet"
)


# ============================================================
# 8. RESUMO
# ============================================================

print("\n==========================================")
print("DECISION TREE FINALIZADA")
print("==========================================")

print("\nModelos treinados:")

print("DT + IMB  ✓")
print("DT + US   ✓")
print("DT + OS   ✓")
print("DT + ROSE ✓")

print("\nAs predições foram geradas")
print("sobre o mesmo conjunto de teste.")