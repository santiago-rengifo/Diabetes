# -*- coding: utf-8 -*-
"""
App de Streamlit — Diabetes (v9) — Pipeline de ML + EDA (versión fiel e interactiva)

Cambios solicitados:
- ✅ **EDA completo** (distribuciones, faltantes, correlaciones, balance de clases, info por tipo).
- ✅ **Todo en español** (textos de UI y mensajes).
- ✅ **Carga de datos solo desde UCI Repo (`ucimlrepo`, id=296)**. Si no está instalado, se muestra un error claro (no hay *fallback* a otros datasets ni carga por CSV).
- ✅ **Misma lógica de tu `diabetes_v9.py`** con controles interactivos para *tuning*, modelos y reportes.

Cómo ejecutar:
1) `pip install streamlit ucimlrepo scikit-learn imbalanced-learn matplotlib pandas numpy seaborn`
2) `streamlit run app_diabetes_v9.py`

Autor: Adaptado para Santiago Rengifo (2025-08-30)
"""

import os
import time
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, SMOTE

RANDOM_STATE_DEFAULT = 42

# ==========================
# CONFIGURACIÓN DE PÁGINA
# ==========================
st.set_page_config(
    page_title="Diabetes v9 — Pipeline ML + EDA",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩺 Diabetes — Pipeline de ML (v9) + EDA")
st.caption("Versión fiel e interactiva del código, con EDA y carga **exclusiva** desde UCI Repo (id=296).")

# ==========================
# FUNCIONES AUXILIARES (compatibilidad y EDA)
# ==========================

def make_ohe():
    """Devuelve OneHotEncoder compatible con scikit-learn nuevo/antiguo."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

@st.cache_data(show_spinner=False)
def cargar_uci() -> Tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset desde UCI Repo (id=296). **Sin alternativas**."""
    ds = fetch_ucirepo(id=296)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    if hasattr(y, "shape") and len(getattr(y, "shape", [])) == 2 and y.shape[1] == 1:
        y = y.iloc[:, 0]
    y = y.astype(str)
    return X, y

@st.cache_data(show_spinner=False)
def inferir_tipos(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols

# ==========================
# BARRA LATERAL — PARÁMETROS
# ==========================
with st.sidebar:
    st.header("1) Datos")
    st.write("Fuente: **UCI Repo (id=296)** — sin opción de CSV.")

    st.divider()
    st.header("2) Partición y validación")
    test_size = st.slider("Tamaño de test", 0.1, 0.4, 0.25, 0.05)
    random_state = st.number_input("Semilla aleatoria", value=RANDOM_STATE_DEFAULT, step=1)
    n_splits = st.slider("Folds (StratifiedKFold)", 2, 10, 5, 1)

    st.divider()
    st.header("3) Modelos a evaluar")
    modelos_disponibles = [
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "LogisticRegression",
        "SVM_Linear",
    ]
    modelos_seleccionados = st.multiselect("Modelos", modelos_disponibles, default=modelos_disponibles)

    st.divider()
    st.header("4) Opciones avanzadas")
    n_iter = st.slider("Iteraciones de RandomizedSearchCV", 5, 60, 20, 5)
    usar_selector = st.checkbox("Usar SelectFromModel(ExtraTrees)", value=True)
    permitir_smote_lineales = st.checkbox("Permitir SMOTE/SMOTENC en modelos lineales (LR/SVM)", value=True)
    graficar_pr = st.checkbox("Graficar curvas Precision–Recall (OvR)", value=False)

    st.divider()
    st.header("5) Controles de EDA")
    max_cols_hist = st.slider("Máx. variables numéricas para histogramas", 1, 20, 8, 1)
    corr_top_k = st.slider("Top-k variables para matriz de correlación (por varianza)", 3, 40, 12, 1)

# ==========================
# CARGA DE DATOS (UCI **obligatorio**)
# ==========================
try:
    X_full, y_full = cargar_uci()
except Exception as e:
    st.error("No se pudo cargar desde **UCI Repo**. Asegúrate de instalar `ucimlrepo` y tener conexión.")
    st.stop()

num_cols, cat_cols = inferir_tipos(X_full)

with st.expander("🔎 EDA — Vista general", expanded=True):
    c1, c2, c3, c4 = st.columns([1.2,1,1,1])
    with c1:
        st.write("**Dimensiones**: X = ", X_full.shape, " | y = ", y_full.shape)
        st.write("**Clases**:")
        st.write(Counter(y_full))
    with c2:
        st.write("**Nº variables numéricas**:", len(num_cols))
        st.write("**Nº variables categóricas**:", len(cat_cols))
    with c3:
        pct_null = X_full.isna().mean().mean()*100
        st.write(f"**% faltantes promedio en X**: {pct_null:.2f}%")
    with c4:
        st.write("**Ejemplo (primeras filas)**:")
        st.dataframe(X_full.head(5))

with st.expander("📉 EDA — Faltantes por columna", expanded=False):
    na_pct = X_full.isna().mean().sort_values(ascending=False)
    df_na = na_pct.to_frame("porcentaje_faltantes").reset_index(names=["columna"])
    st.dataframe(df_na, use_container_width=True)
    fig_na, ax_na = plt.subplots(figsize=(8,3))
    ax_na.barh(df_na["columna"], df_na["porcentaje_faltantes"])  
    ax_na.set_xlabel("Porcentaje de faltantes")
    ax_na.set_title("Faltantes por columna")
    plt.tight_layout()
    st.pyplot(fig_na)

with st.expander("📊 EDA — Histogramas (variables numéricas)", expanded=False):
    if len(num_cols) == 0:
        st.info("No hay variables numéricas.")
    else:
        # Selecciona las de mayor varianza para no sobrecargar
        var_order = X_full[num_cols].var().sort_values(ascending=False).index.tolist()
        cols_plot = var_order[:max_cols_hist]
        ncols = 2
        nrows = int(np.ceil(len(cols_plot)/ncols))
        fig_h, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3*nrows))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(cols_plot):
            ax = axes[i]
            ax.hist(X_full[col].dropna(), bins=30)
            ax.set_title(col)
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        st.pyplot(fig_h)

with st.expander("📈 EDA — Boxplots (variables numéricas)", expanded=False):
    if len(num_cols) == 0:
        st.info("No hay variables numéricas.")
    else:
        cols_plot = num_cols[:min(8, len(num_cols))]
        fig_b, ax_b = plt.subplots(figsize=(10,4))
        sns.boxplot(data=X_full[cols_plot], ax=ax_b)
        ax_b.set_title("Boxplots (muestra de variables)")
        plt.tight_layout()
        st.pyplot(fig_b)

with st.expander("🔗 EDA — Matriz de correlación (Top-k por varianza)", expanded=False):
    if len(num_cols) < 2:
        st.info("Se requieren al menos 2 variables numéricas para correlación.")
    else:
        var_order = X_full[num_cols].var().sort_values(ascending=False).index.tolist()
        sel = var_order[:min(corr_top_k, len(var_order))]
        corr = X_full[sel].corr(numeric_only=True)
        fig_c, ax_c = plt.subplots(figsize=(1.0*len(sel)+2, 1.0*len(sel)+2))
        sns.heatmap(corr, annot=False, cmap="vlag", center=0, ax=ax_c)
        ax_c.set_title("Correlación (Top-k por varianza)")
        plt.tight_layout()
        st.pyplot(fig_c)

with st.expander("⚖️ EDA — Balance de clases", expanded=False):
    vc = y_full.value_counts()
    fig_cls, ax_cls = plt.subplots(figsize=(6,3))
    ax_cls.bar(vc.index.astype(str), vc.values)
    ax_cls.set_xlabel("Clase")
    ax_cls.set_ylabel("Conteo")
    ax_cls.set_title("Distribución de clases (y)")
    plt.tight_layout()
    st.pyplot(fig_cls)

# ==========================
# SPLIT + ENCODING DE ETIQUETAS
# ==========================
le = LabelEncoder()
y_enc = le.fit_transform(y_full)
class_names = list(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_enc, test_size=test_size, stratify=y_enc, random_state=random_state
)

st.info(f"Train: {X_train.shape} — Test: {X_test.shape} — Clases: {class_names}")

# ==========================
# PIPELINES DE PREPROCESAMIENTO
# ==========================
num_pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler(with_mean=True)),
    ("pca", "passthrough"),  # alternable en la búsqueda
])

cat_pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", make_ohe()),
    ("svd", "passthrough"),   # alternable (TruncatedSVD ~ MCA)
])

pre = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

selector = SelectFromModel(
    ExtraTreesClassifier(random_state=random_state, n_estimators=200), threshold="median"
)

# Selección de balanceador
cat_idx = [X_full.columns.get_loc(c) for c in cat_cols]
if len(cat_idx) == 0:
    smote_like = SMOTE(random_state=random_state)
else:
    smote_like = SMOTENC(categorical_features=cat_idx, random_state=random_state)

# Modelos base
modelos = {
    "RandomForest": RandomForestClassifier(random_state=random_state),
    "ExtraTrees": ExtraTreesClassifier(random_state=random_state),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=random_state, solver="liblinear"),
    "SVM_Linear": SVC(kernel="linear", probability=True, max_iter=2000, random_state=random_state),
}

pipes = {}
for nombre, clf in modelos.items():
    if nombre == "HistGradientBoosting":
        def _to_dense(X):
            return X.toarray() if hasattr(X, "toarray") else X
        densify = FunctionTransformer(_to_dense, accept_sparse=True)
        pipe = ImbPipeline(steps=[
            ("bal", "passthrough"),
            ("pre", pre),
            ("densify", densify),
            ("selector", selector),
            ("clf", clf),
        ])
    else:
        pipe = ImbPipeline(steps=[
            ("bal", "passthrough"),
            ("pre", pre),
            ("selector", selector),
            ("clf", clf),
        ])
    pipes[nombre] = pipe

# Espacios de búsqueda
comunes = {
    "pre__num__pca": ["passthrough", PCA(n_components=0.9, random_state=random_state)],
    "pre__cat__svd": ["passthrough", TruncatedSVD(n_components=50, random_state=random_state)],
    "selector": [selector if usar_selector else "passthrough"],
    "selector__threshold": ["median", "1.5*median"] if usar_selector else ["median"],
    "bal": ["passthrough", smote_like],
}

param_grids = {
    "RandomForest": {
        **comunes,
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 15, 30],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", "log2"],
        "clf__class_weight": [None, "balanced"],
    },
    "ExtraTrees": {
        **comunes,
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 15, 30],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", "log2"],
        "clf__class_weight": [None, "balanced"],
    },
    "HistGradientBoosting": {
        **{k: v for k, v in comunes.items() if k != "clf__class_weight"},
        "clf__max_iter": [100, 200],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [20, 50],
        "clf__l2_regularization": [0.0, 0.1],
    },
    "LogisticRegression": {
        **comunes,
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
        "clf__class_weight": [None, "balanced"],
    },
    "SVM_Linear": {
        **comunes,
        "clf__C": [0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    },
}

# Ajustes: desactivar balanceo para árboles (memoria) y PCA num en árboles
for nombre in list(param_grids.keys()):
    grid = param_grids[nombre]
    if nombre in ("RandomForest", "ExtraTrees", "HistGradientBoosting"):
        grid["bal"] = ["passthrough"]
        if "pre__num__pca" in grid:
            grid["pre__num__pca"] = ["passthrough"]
    if not usar_selector:
        grid = {k: v for k, v in grid.items() if not k.startswith("selector__")}
        grid["selector"] = ["passthrough"]
        param_grids[nombre] = grid
    if nombre in ("LogisticRegression", "SVM_Linear") and not permitir_smote_lineales:
        grid["bal"] = ["passthrough"]
        param_grids[nombre] = grid

# ==========================
# ENTRENAMIENTO Y EVALUACIÓN
# ==========================
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
resultados = []
mejores = {}

if st.button("🚀 Entrenar y evaluar"):
    st.write("### Entrenamiento")
    barra = st.progress(0.0, text="Iniciando...")
    total = max(1, len(modelos_seleccionados))

    for i, nombre in enumerate(modelos_seleccionados, start=1):
        t0 = time.time()
        pipe = pipes[nombre]
        grid = param_grids[nombre]

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=grid if len(grid) > 0 else {"clf__random_state": [random_state]},
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            pre_dispatch="2*n_jobs",
            random_state=random_state,
            verbose=1,
            error_score=np.nan,
        )

        with st.spinner(f"Ajustando {nombre}..."):
            search.fit(X_train, y_train)

        best_est = search.best_estimator_
        mejores[nombre] = best_est
        f1_cv = search.best_score_
        elapsed = time.time() - t0

        y_pred = best_est.predict(X_test)
        f1_t = f1_score(y_test, y_pred, average="macro")

        resultados.append({
            "Modelo": nombre,
            "CV_F1_macro": f1_cv,
            "Test_F1_macro": f1_t,
            "Tiempo_s": elapsed,
            "Best_Params": str(search.best_params_),
        })

        st.subheader(f"Resultados — {nombre}")
        st.code(classification_report(y_test, y_pred, target_names=class_names))

        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", values_format='d', ax=ax_cm)
        ax_cm.set_title(f"Matriz de Confusión — {nombre}")
        st.pyplot(fig_cm)

        # ROC/PR OvR
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

        y_score = None
        if hasattr(best_est, "predict_proba"):
            try:
                y_score = best_est.predict_proba(X_test)
            except Exception:
                y_score = None
        if y_score is None and hasattr(best_est, "decision_function"):
            try:
                y_score = best_est.decision_function(X_test)
            except Exception:
                y_score = None

        if y_score is not None:
            try:
                est_classes = best_est.named_steps["clf"].classes_
            except Exception:
                try:
                    est_classes = best_est.classes_
                except Exception:
                    est_classes = np.arange(y_score.shape[1])
            order_idx = np.array([np.where(est_classes == k)[0][0] for k in range(len(class_names))])
            y_score = y_score[:, order_idx]

            fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
            plotted = 0
            for j in range(len(class_names)):
                if np.unique(y_test_bin[:, j]).size < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_test_bin[:, j], y_score[:, j])
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, lw=2, label=f"{class_names[j]} (AUC={roc_auc:.3f})")
                plotted += 1
            if plotted > 0:
                ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=1)
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel("Tasa de falsos positivos")
                ax_roc.set_ylabel("Tasa de verdaderos positivos")
                ax_roc.set_title(f"Curvas ROC (OvR) — {nombre}")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)

            if graficar_pr:
                fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
                plotted_pr = 0
                for j in range(len(class_names)):
                    if np.unique(y_test_bin[:, j]).size < 2:
                        continue
                    precision, recall, _ = precision_recall_curve(y_test_bin[:, j], y_score[:, j])
                    ap = average_precision_score(y_test_bin[:, j], y_score[:, j])
                    ax_pr.plot(recall, precision, lw=2, label=f"{class_names[j]} (AP={ap:.3f})")
                    plotted_pr += 1
                if plotted_pr > 0:
                    ax_pr.set_xlim([0.0, 1.0])
                    ax_pr.set_ylim([0.0, 1.05])
                    ax_pr.set_xlabel("Recall")
                    ax_pr.set_ylabel("Precision")
                    ax_pr.set_title(f"Curvas Precision–Recall (OvR) — {nombre}")
                    ax_pr.legend(loc="lower left")
                    st.pyplot(fig_pr)
        else:
            st.warning("No fue posible trazar ROC/PR: el modelo no expone probabilidades ni decision_function.")

        barra.progress(i/total, text=f"Completado {i}/{total}")

    if resultados:
        df_res = pd.DataFrame(resultados).sort_values("Test_F1_macro", ascending=False)
        st.success("Entrenamiento finalizado")
        st.write("### Resumen de modelos (ordenado por F1_macro en Test)")
        st.dataframe(df_res, use_container_width=True)
        top = df_res.iloc[0]
        st.info(
            f"**Mejor modelo**: {top['Modelo']} — F1_macro Test = {top['Test_F1_macro']:.4f} (CV={top['CV_F1_macro']:.4f})\n\n"
            f"**Hiperparámetros**: {top['Best_Params']}"
        )

        # ==========================
        # PRUEBAS ADICIONALES (sanidad post-entrenamiento)
        # ==========================
        try:
            assert not df_res["Test_F1_macro"].isna().any(), "F1_macro en test no debe contener NaN."
            assert df_res["Test_F1_macro"].between(0, 1).all(), "F1_macro fuera de rango [0,1]."
            assert len(df_res) == len(modelos_seleccionados), "Faltan filas en el resumen para algún modelo seleccionado."
        except AssertionError as e:
            st.warning(f"Self-test resultados: {e}")
else:
    st.warning("Configura opciones en la barra lateral y pulsa **Entrenar y evaluar**.")

# ==========================
# PRUEBAS BÁSICAS (sanidad previas)
# ==========================
try:
    assert len(class_names) >= 2, "Se esperaban ≥2 clases. Verifica la columna objetivo del UCI Repo."
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Split vacío: revisa el tamaño de test."
    # Chequeo ligero del preprocesamiento: transformar 10 filas no debe fallar
    _ = pre.fit(X_train, y_train).transform(X_train.iloc[:10])
except AssertionError as e:
    st.warning(f"Self-test: {e}")
except Exception as e:
    st.warning(f"Self-test preprocesamiento: {e}")
