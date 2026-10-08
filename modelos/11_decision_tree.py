import pandas as pd
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RANDOM_STATE = 42
MAX_ALPHAS = 15
MAX_TREE_SEARCH_ROWS = 100_000
MAX_VALIDATION_ROWS = 50_000


# ============================================================
# 1. CARREGAR TESTE
# ============================================================

print("==========================================")
print("DECISION TREE - CART")
print("==========================================")

print("\nCarregando teste...")

X_test = pd.read_parquet(
    r"..\salvos\X_test_model.parquet"
)

y_test = pd.read_parquet(
    r"..\salvos\y_test.parquet"
)["loan_default"]


print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 2. VARIÁVEIS CATEGÓRICAS
# ============================================================

variaveis_categoricas = [
    "term",
    "grade",
    "home_ownership",
    "emp_length"
]


def preparar_categoricas(X_train, X_valid, X_test):

    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()

    for coluna in variaveis_categoricas:

        # categorias conhecidas pelo treinamento
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

        X_valid[coluna] = (
            X_valid[coluna]
            .astype(str)
            .map(mapa)
        )

        X_test[coluna] = (
            X_test[coluna]
            .astype(str)
            .map(mapa)
        )

    return (
        X_train.astype(float),
        X_valid.astype(float),
        X_test.astype(float)
    )


# ============================================================
# 3. FUNÇÃO DE PODA
# ============================================================

def encontrar_melhor_alpha(
    X_train,
    y_train,
    X_valid,
    y_valid
):

    # A busca de alpha usa uma amostra estratificada para evitar construir
    # uma árvore enorme repetidamente sobre toda a base balanceada.
    tamanho_busca = min(MAX_TREE_SEARCH_ROWS, len(X_train))
    if tamanho_busca < len(X_train):
        X_busca, _, y_busca, _ = train_test_split(
            X_train,
            y_train,
            train_size=tamanho_busca,
            random_state=RANDOM_STATE,
            stratify=y_train
        )
    else:
        X_busca, y_busca = X_train, y_train

    tamanho_validacao = min(MAX_VALIDATION_ROWS, len(X_valid))
    if tamanho_validacao < len(X_valid):
        X_valid_busca, _, y_valid_busca, _ = train_test_split(
            X_valid,
            y_valid,
            train_size=tamanho_validacao,
            random_state=RANDOM_STATE,
            stratify=y_valid
        )
    else:
        X_valid_busca, y_valid_busca = X_valid, y_valid

    print(
        f"\nAmostra para busca: {len(X_busca):,} linhas; "
        f"validação: {len(X_valid_busca):,} linhas."
    )
    print("Calculando caminho de poda...")


    # --------------------------------------------------------
    # Caminho de Cost-Complexity Pruning
    # --------------------------------------------------------

    caminho = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ).cost_complexity_pruning_path(
        X_busca,
        y_busca
    )

    ccp_alphas = caminho.ccp_alphas

    # Avalia poucos alphas distribuídos pelo caminho, incluindo os extremos.
    if len(ccp_alphas) > MAX_ALPHAS:
        indices = np.linspace(
            0,
            len(ccp_alphas) - 1,
            num=MAX_ALPHAS,
            dtype=int
        )
        ccp_alphas = ccp_alphas[indices]


    print(
        "\nQuantidade de ccp_alpha:",
        len(ccp_alphas)
    )


    # --------------------------------------------------------
    # Avaliar cada alpha na validação
    # --------------------------------------------------------

    resultados = []


    for i, alpha in enumerate(ccp_alphas, start=1):

        print(f"Avaliando alpha {i}/{len(ccp_alphas)}...", flush=True)

        modelo = DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            ccp_alpha=alpha
        )

        modelo.fit(
            X_busca,
            y_busca
        )

        pred = modelo.predict(
            X_valid_busca
        )

        accuracy = accuracy_score(
            y_valid_busca,
            pred
        )

        erro = 1 - accuracy


        resultados.append(
            {
                "ccp_alpha": alpha,
                "accuracy": accuracy,
                "erro": erro,
                "profundidade": modelo.get_depth(),
                "folhas": modelo.get_n_leaves()
            }
        )


    resultados = pd.DataFrame(
        resultados
    )


    # --------------------------------------------------------
    # Melhor alpha
    # --------------------------------------------------------

    melhor = resultados.loc[
        resultados["erro"].idxmin()
    ]


    print("\n==========================================")
    print("MELHOR PODA")
    print("==========================================")

    print(
        f"ccp_alpha: {melhor['ccp_alpha']}"
    )

    print(
        f"Accuracy validação: "
        f"{melhor['accuracy']:.6f}"
    )

    print(
        f"Erro validação: "
        f"{melhor['erro']:.6f}"
    )

    print(
        f"Profundidade: "
        f"{int(melhor['profundidade'])}"
    )

    print(
        f"Folhas: "
        f"{int(melhor['folhas'])}"
    )


    return (
        melhor["ccp_alpha"],
        resultados
    )


# ============================================================
# 4. TREINAR MODELO FINAL
# ============================================================
def treinar_decision_tree(
    nome,
    caminho_x,
    caminho_y
):

    print("\n\n==========================================")
    print(f"DT + {nome}")
    print("==========================================")


    # --------------------------------------------------------
    # Carregar treinamento
    # --------------------------------------------------------

    X = pd.read_parquet(caminho_x)

    y = pd.read_parquet(caminho_y)["loan_default"]


    print(
        "Treinamento:",
        X.shape
    )

    print("\nDistribuição:")
    print(y.value_counts())


    # --------------------------------------------------------
    # Divisão interna 80/20
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )


    print("\nDivisão interna:")

    print(
        "Train:",
        X_train.shape
    )

    print(
        "Validation:",
        X_valid.shape
    )


    # --------------------------------------------------------
    # Preparar categorias
    # --------------------------------------------------------

    (
        X_train,
        X_valid,
        X_test_preparado
    ) = preparar_categoricas(
        X_train,
        X_valid,
        X_test
    )


    # --------------------------------------------------------
    # Encontrar melhor poda
    # --------------------------------------------------------

    melhor_alpha, tabela_poda = (
        encontrar_melhor_alpha(
            X_train,
            y_train,
            X_valid,
            y_valid
        )
    )


    # --------------------------------------------------------
    # Treinar modelo final
    # --------------------------------------------------------

    print(
        "\nTreinando modelo final..."
    )


    X_completo = X.copy()

    (
        X_completo,
        _,
        X_test_final
    ) = preparar_categoricas(
        X_completo,
        X_completo,
        X_test
    )


    modelo_final = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        ccp_alpha=melhor_alpha
    )


    modelo_final.fit(
        X_completo,
        y
    )


    # --------------------------------------------------------
    # Predições no teste
    # --------------------------------------------------------

    y_pred = modelo_final.predict(
        X_test_final
    )

    y_prob = modelo_final.predict_proba(
        X_test_final
    )[:, 1]


    # --------------------------------------------------------
    # Informações finais
    # --------------------------------------------------------

    print("\n==========================================")
    print(f"MODELO FINAL - DT + {nome}")
    print("==========================================")


    print(
        f"ccp_alpha: {melhor_alpha}"
    )

    print(
        f"Profundidade: "
        f"{modelo_final.get_depth()}"
    )

    print(
        f"Nós: "
        f"{modelo_final.tree_.node_count}"
    )

    print(
        f"Folhas: "
        f"{modelo_final.get_n_leaves()}"
    )


    # --------------------------------------------------------
    # Salvar previsões
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "y_real": y_test.values,
            "y_pred": y_pred,
            "y_prob": y_prob
        }
    ).to_parquet(
        rf"..\salvos\dt_{nome.lower()}_predicoes.parquet",
        index=False
    )


    # --------------------------------------------------------
    # Salvar tabela de poda
    # --------------------------------------------------------

    tabela_poda.to_parquet(
        rf"..\salvos\dt_{nome.lower()}_poda.parquet",
        index=False
    )


    return modelo_final


# ============================================================
# 5. IMB
# ============================================================

dt_imb = treinar_decision_tree(
    "IMB",
    r"..\salvos\X_dt_imb.parquet",
    r"..\salvos\y_dt_imb.parquet"
)


# ============================================================
# 6. US
# ============================================================

dt_us = treinar_decision_tree(
    "US",
    r"..\salvos\X_dt_us.parquet",
    r"..\salvos\y_dt_us.parquet"
)


# ============================================================
# 7. OS
# ============================================================

dt_os = treinar_decision_tree(
    "OS",
    r"..\salvos\X_dt_os.parquet",
    r"..\salvos\y_dt_os.parquet"
)


# ============================================================
# 8. ROSE
# ============================================================

dt_rose = treinar_decision_tree(
    "ROSE",
    r"..\salvos\X_dt_rose.parquet",
    r"..\salvos\y_dt_rose.parquet"
)


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print("DECISION TREE FINALIZADA")
print("==========================================")

print("DT + IMB  ✓")
print("DT + US   ✓")
print("DT + OS   ✓")
print("DT + ROSE ✓")

print("\nPredições salvas em salvos\\")
