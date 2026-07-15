"""Wrappers de explicabilidad para el notebook 08 (otras tecnicas de XAI).

CONVENCION: funciones PURAS (numpy/pandas/sklearn/lime), sin I/O de ficheros salvo que se
indique explicitamente (el notebook hace joblib.load/pd.read_parquet y llama a estas funciones
con los objetos ya cargados, igual que 05 hace con src.cost_utils).

Agrupa:
  - Traduccion de valores estandarizados a escala original (mismo patron que 05, celda 21),
    reutilizable aqui y por cualquier notebook futuro que necesite leer un umbral/valor del
    pipeline en euros/anios/ratios en vez de en desviaciones tipicas.
  - Importancia por permutacion (agnostica al modelo, XAI.pdf p.166) y el contraste a 3 bandas
    (permutacion vs XGBoost nativo vs subrogado de 05).
  - PDP/ICE de bajo nivel (grid + valores; el notebook decide como plotear).
  - LIME local con medicion de INESTABILIDAD entre reejecuciones (contraste critico frente a
    SHAP, XAI.pdf pp.185-186 y 199).
  - Subgrupos por tramo de edad (mini-auditoria de sesgo).

IMPORTANTE (05, hallazgo de fidelidad): estas tecnicas se aplican sobre las DECISIONES/SCORES
del modelo de caja negra (XGBoost), igual que el subrogado; no sustituyen a SHAP ni al
subrogado, son auditorias COMPLEMENTARIAS e independientes (notebooks 06/07 no se importan
aqui).

NOTA sobre el wrapper `ModeloBoosted` (src/modeling.py): NO es compatible con
sklearn.inspection.PartialDependenceDisplay/permutation_importance en su forma "nativa" para
PDP (falta __sklearn_tags__/BaseEstimator). Verificado: `estimador_` (el XGBClassifier nativo
dentro del wrapper) da predict_proba IDENTICO al wrapper (max|dif|=0.0) cuando X se pasa en el
orden canonico de columnas, asi que las funciones de PDP de este modulo reciben el estimador
NATIVO (`bundle["modelo"].estimador_`), no el wrapper. La importancia por permutacion SI acepta
el wrapper directamente (no exige tags de sklearn si se pasa un `scoring` callable explicito).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import cost_utils as cu

# ---------------------------------------------------------------------- #
# Nombres legibles y tipos de feature (mismo mapeo que 05, duplicado a    #
# proposito: cada notebook de auditoria es autocontenido; si diverge hay #
# que sincronizar a mano con notebooks/05_auditoria_subrogado.ipynb).     #
# ---------------------------------------------------------------------- #
NOMBRE_LEGIBLE = {
    "RevolvingUtilizationOfUnsecuredLines": "Utilizacion de credito revolving",
    "age": "Edad",
    "NumberOfTime30-59DaysPastDueNotWorse": "N. moras 30-59 dias",
    "DebtRatio": "Ratio de deuda",
    "MonthlyIncome": "Ingreso mensual",
    "NumberOfOpenCreditLinesAndLoans": "N. lineas de credito abiertas",
    "NumberOfTimes90DaysLate": "N. impagos 90+ dias",
    "NumberRealEstateLoansOrLines": "N. hipotecas/lineas inmob.",
    "NumberOfTime60-89DaysPastDueNotWorse": "N. moras 60-89 dias",
    "NumberOfDependents": "N. dependientes",
    "is_missing_monthlyincome": "Ingreso ausente (flag)",
    "es_centinela_pastdue": "Centinela mora (flag)",
}

FEATURES_CONTEO = {
    "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines", "NumberOfDependents",
}


# ---------------------------------------------------------------------- #
# Traduccion escala estandarizada -> escala original (patron de 05)      #
# ---------------------------------------------------------------------- #
def columna_a_original(pipeline, feature: str, valores_std) -> np.ndarray:
    """Convierte una columna COMPLETA de valores estandarizados a escala original.

    Usa pipeline.inverse_transform sobre una matriz de ceros con la columna `feature`
    rellena con `valores_std` (mismo patron que 05, celda 21, generalizado a vector).
    Valido porque StandardScaler/log1p/winsor se invierten columna a columna, sin
    acoplamiento cruzado en el inverse_transform (la unica dependencia cruzada del
    pipeline, D-1.2, esta en el FORWARD de DebtRatio via el tramo de edad, no en el
    inverse de una columna concreta).
    """
    cols = list(pipeline.columnas_salida_)
    valores = np.asarray(valores_std, dtype="float64").ravel()
    n = valores.shape[0]
    df_zeros = pd.DataFrame(np.zeros((n, len(cols))), columns=cols)
    df_zeros[feature] = valores
    orig = pipeline.inverse_transform(df_zeros)
    return np.asarray(orig[feature], dtype="float64")


def valor_a_original(pipeline, feature: str, valor_std: float) -> float:
    """Version escalar de `columna_a_original` (un unico valor, p.ej. un umbral de PDP)."""
    return float(columna_a_original(pipeline, feature, [valor_std])[0])


# ---------------------------------------------------------------------- #
# Importancia por permutacion (agnostica, XAI.pdf p.166)                 #
# ---------------------------------------------------------------------- #
def importancia_permutacion(modelo, X, y, c_fp: float, c_fn: float,
                            n_repeats: int = 10, random_state: int = 42,
                            muestra: int | None = None) -> pd.DataFrame:
    """Importancia por permutacion usando el COSTE ESPERADO (negado) como score.

    `modelo` es cualquier objeto con `.predict(X) -> {0,1}` (el wrapper ModeloBoosted vale:
    permutation_importance no exige tags de sklearn si se le pasa un `scoring` callable).
    Si `muestra` esta fijado y es menor que len(X), se usa una submuestra estratificada por
    indice aleatorio (D-8.2); se deja constancia del tamano en el resultado para no recortar
    en silencio.
    """
    from sklearn.inspection import permutation_importance

    if muestra is not None and muestra < len(X):
        idx = X.sample(muestra, random_state=random_state).index
        Xs, ys = X.loc[idx], y.loc[idx] if hasattr(y, "loc") else y[X.index.get_indexer(idx)]
    else:
        Xs, ys = X, y

    def _scorer(estimator, X_, y_):
        pred = estimator.predict(X_)
        return -cu.coste_esperado(y_, pred, c_fp, c_fn)   # negado: permutation_importance maximiza

    r = permutation_importance(modelo, Xs, ys, scoring=_scorer,
                               n_repeats=n_repeats, random_state=random_state)
    return pd.DataFrame({
        "variable": list(Xs.columns),
        "importancia_permutacion_media": r.importances_mean,
        "importancia_permutacion_std": r.importances_std,
        "n_filas_usadas": len(Xs),
    }).sort_values("importancia_permutacion_media", ascending=False, ignore_index=True)


def importancia_nativa_xgboost(estimador_nativo, feature_names) -> pd.DataFrame:
    """Importancia nativa (ganancia de impureza) del XGBClassifier, para el contraste 3-bandas."""
    imp = np.asarray(estimador_nativo.feature_importances_, dtype="float64")
    return pd.DataFrame({
        "variable": list(feature_names),
        "importancia_xgboost_nativa": imp,
    }).sort_values("importancia_xgboost_nativa", ascending=False, ignore_index=True)


def tabla_importancia_3bandas(perm_df: pd.DataFrame, nativa_df: pd.DataFrame,
                              subrogado_df: pd.DataFrame | None,
                              columna_subrogado: str,
                              nombre_legible: dict = NOMBRE_LEGIBLE) -> pd.DataFrame:
    """Combina permutacion + XGBoost nativo + importancia del subrogado de 05.

    `subrogado_df` es el CSV `results/tables/sub_05_importancia_variables.csv` ya leido (usa
    nombres LEGIBLES en espanol como clave `variable`); se remapea a nombre TECNICO antes del
    merge. Si `subrogado_df` es None, esa columna queda a NaN (no bloquea el resto).
    """
    tabla = perm_df.merge(nativa_df, on="variable", how="outer")
    if subrogado_df is not None:
        legible_a_tecnico = {v: k for k, v in nombre_legible.items()}
        sub = subrogado_df.copy()
        sub["variable"] = sub["variable"].map(legible_a_tecnico)
        sub = sub[["variable", columna_subrogado]].rename(
            columns={columna_subrogado: "importancia_subrogado_05"})
        tabla = tabla.merge(sub, on="variable", how="left")
    else:
        tabla["importancia_subrogado_05"] = np.nan
    return tabla.sort_values("importancia_permutacion_media", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------- #
# PDP / ICE de bajo nivel (XAI.pdf pp.147-165)                           #
# ---------------------------------------------------------------------- #
def pdp_1d(estimador_nativo, X, feature: str, grid_resolution: int = 30,
          kind: str = "average"):
    """PDP (kind='average') o ICE (kind='individual'/'both') de 1 variable.

    Se pasa el estimador NATIVO (bundle["modelo"].estimador_), no el wrapper (ver docstring
    de modulo). Devuelve el objeto crudo de sklearn.inspection.partial_dependence (dict con
    'grid_values' y 'average'/'individual' segun `kind`), para que el notebook decida si
    tabula, plotea o ambas cosas.
    """
    from sklearn.inspection import partial_dependence
    return partial_dependence(estimador_nativo, X, features=[feature],
                              grid_resolution=grid_resolution, kind=kind)


def pdp_2d(estimador_nativo, X, par_features: tuple[str, str], grid_resolution: int = 20):
    """PDP 2D de interaccion entre dos variables (p.ej. age x contador de impago).

    Pieza central del notebook 08: si el hallazgo de 05 (las 'dos personalidades de riesgo'
    segun el umbral) es real, el PDP 2D deberia mostrar que el efecto de `age` en la
    prediccion depende del nivel del contador de impagos (superficie no aditiva).
    """
    from sklearn.inspection import partial_dependence
    return partial_dependence(estimador_nativo, X, features=list(par_features),
                              grid_resolution=grid_resolution, kind="average")


# ---------------------------------------------------------------------- #
# LIME local + inestabilidad (contraste critico frente a SHAP)           #
# ---------------------------------------------------------------------- #
def construir_lime_explainer(X_train, feature_names, class_names=("conceder", "denegar"),
                             random_state: int = 42):
    """LimeTabularExplainer sobre las features en orden canonico (discretize_continuous=True,
    el modo por defecto que usa el propio material del profesor para tabular, XAI.pdf p.182)."""
    from lime.lime_tabular import LimeTabularExplainer
    Xtr = np.asarray(X_train, dtype="float64")
    return LimeTabularExplainer(
        Xtr, feature_names=list(feature_names), class_names=list(class_names),
        discretize_continuous=True, random_state=random_state)


def lime_explicar_caso(modelo, x, X_train, feature_names, class_names=("conceder", "denegar"),
                       num_features: int = 8, random_state: int = 42):
    """Una unica explicacion LIME (semilla fija) para el caso `x` (array 1D, orden canonico)."""
    explainer = construir_lime_explainer(X_train, feature_names, class_names, random_state)
    x = np.asarray(x, dtype="float64")
    return explainer.explain_instance(x, modelo.predict_proba, num_features=num_features)


def lime_inestabilidad(modelo, x, X_train, feature_names, class_names=("conceder", "denegar"),
                       num_features: int | None = None, k_reejecuciones: int = 5,
                       semilla_base: int = 42, clase_objetivo: int = 1):
    """Reejecuta LIME K veces sobre el MISMO caso x, con un explainer RECONSTRUIDO cada vez
    con una semilla de muestreo distinta (semilla_base, semilla_base+1, ...), para medir
    cuanto varia el peso de cada variable entre ejecuciones.

    Es la version empirica de la advertencia del profesor: "el analisis LIME es inestable.
    Si se vuelven a crear ejemplos generados, puede dar una respuesta diferente" (XAI.pdf
    p.199). Usa `exp.local_exp[clase_objetivo]` (pares indice-feature/peso), NO
    `exp.as_list()`: el texto de `as_list()` para variables discretizadas puede tener el
    formato "a < feature <= b" (feature en MEDIO de la cadena), lo que rompe cualquier
    parsing por `str.startswith` (verificado con casos reales del dataset).

    Devuelve (lista_de_explicaciones_lime, DataFrame de inestabilidad: variable, peso_medio,
    peso_std, peso_min, peso_max — ordenado por peso_std descendente, la variable que MAS
    baila primero).
    """
    if num_features is None:
        num_features = len(feature_names)
    x = np.asarray(x, dtype="float64")

    explicaciones = []
    pesos = {f: [] for f in feature_names}
    for k in range(k_reejecuciones):
        exp = lime_explicar_caso(modelo, x, X_train, feature_names, class_names,
                                 num_features=num_features, random_state=semilla_base + k)
        explicaciones.append(exp)
        pesos_esta_ejecucion = dict.fromkeys(feature_names, 0.0)
        for idx_f, w in exp.local_exp[clase_objetivo]:
            pesos_esta_ejecucion[feature_names[idx_f]] = float(w)
        for f in feature_names:
            pesos[f].append(pesos_esta_ejecucion[f])

    filas = []
    for f in feature_names:
        arr = np.asarray(pesos[f], dtype="float64")
        filas.append({
            "variable": f,
            "peso_medio": float(arr.mean()),
            "peso_std": float(arr.std()),
            "peso_min": float(arr.min()),
            "peso_max": float(arr.max()),
            "k_reejecuciones": k_reejecuciones,
        })
    tabla = pd.DataFrame(filas).sort_values("peso_std", ascending=False, ignore_index=True)
    return explicaciones, tabla


# ---------------------------------------------------------------------- #
# Subgrupos por tramo de edad (mini-auditoria de sesgo)                  #
# ---------------------------------------------------------------------- #
ORDEN_TRAMOS_EDAD = ("<=30", "31-40", "41-50", "51-60", "61-70", ">70")


def tramo_edad(edad_original) -> np.ndarray:
    """Clasifica edades REALES (no estandarizadas) en los 6 tramos decadales de D-1.1.

    Mismos cortes que la imputacion de MonthlyIncome (02_preprocesado.ipynb): <=30, 31-40,
    41-50, 51-60, 61-70, >70.
    """
    edad = np.asarray(edad_original, dtype="float64")
    tramos = np.full(edad.shape, "", dtype=object)
    cortes = (
        (edad <= 30, "<=30"),
        ((edad > 30) & (edad <= 40), "31-40"),
        ((edad > 40) & (edad <= 50), "41-50"),
        ((edad > 50) & (edad <= 60), "51-60"),
        ((edad > 60) & (edad <= 70), "61-70"),
        (edad > 70, ">70"),
    )
    for mask, etiqueta in cortes:
        tramos[mask] = etiqueta
    return tramos


def subgrupos_por_edad(edad_original, y_true, y_pred, c_fp: float, c_fn: float) -> pd.DataFrame:
    """Tasa de denegacion, FP-rate, FN-rate, tasa real de impago y coste medio por tramo de edad.

    `edad_original` debe venir en escala REAL (usar `columna_a_original` antes de llamar, no
    la columna 'age' estandarizada de test.parquet). Mini-auditoria de sesgo (no es un analisis
    de fairness formal, ver notebook 08 seccion 5): contrasta si el trato del modelo por edad es
    proporcional al riesgo real (target) o desproporcionado, en la linea de que el credito es
    sistema de "alto riesgo" segun el reglamento europeo de IA (XAI.pdf p.232).
    """
    tramos = tramo_edad(edad_original)
    y = np.asarray(y_true, dtype="int64").ravel()
    p = np.asarray(y_pred, dtype="int64").ravel()

    filas = []
    for t in ORDEN_TRAMOS_EDAD:
        m = tramos == t
        n = int(m.sum())
        if n == 0:
            continue
        yt, pt = y[m], p[m]
        n_buenos = int((yt == 0).sum())
        n_morosos = int((yt == 1).sum())
        fp_rate = float(((pt == 1) & (yt == 0)).sum() / n_buenos) if n_buenos else np.nan
        fn_rate = float(((pt == 0) & (yt == 1)).sum() / n_morosos) if n_morosos else np.nan
        filas.append({
            "tramo_edad": t,
            "n": n,
            "tasa_denegacion": float((pt == 1).mean()),
            "tasa_impago_real": float(yt.mean()),
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
            "coste_medio": cu.coste_esperado(yt, pt, c_fp, c_fn),
        })
    return pd.DataFrame(filas)
