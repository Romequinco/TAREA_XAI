"""Coste esperado y umbral optimo de decision para el taller B4-T2.

CONVENCION DE COSTE (credito). NO se hereda la nomenclatura cruzada del profesor.
  Positivo = clase 1 = IMPAGO (SeriousDlqin2yrs==1), el suceso de interes.
  accion / prediccion:  1 = DENEGAR (predice impago) ; 0 = CONCEDER (predice no-impago).
  Errores y coste:
    FN = (pred==0) & (y==1)  -> conceder a un moroso -> coste C_FN   (el CARO)
    FP = (pred==1) & (y==0)  -> denegar a uno bueno   -> coste C_FP
    Aciertos (TP, TN) -> coste 0.
  Coste esperado empirico = mean( ((pred==1)&(y==0))*C_FP + ((pred==0)&(y==1))*C_FN ).
  Escenarios: esc.1 C_FP=C_FN=1 (simetrico, nb 03) ; esc.2 C_FP=1, C_FN=10 (asimetrico, nb 04).

Modulo puro (numpy/pandas). El umbral se obtiene por ARGMIN empirico del coste sobre
las probas (algoritmo O(N log N) exacto, ver `barrido_umbral`), NO por la formula
analitica (`umbral_analitico` es solo referencia/verificacion).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------- #
# Constantes de contrato                                                 #
# ---------------------------------------------------------------------- #
TARGET = "SeriousDlqin2yrs"
ETIQUETAS = (0, 1)
COSTES_ESCENARIO_1 = (1.0, 1.0)    # (c_fp, c_fn)  -> nb 03 (simetrico)
COSTES_ESCENARIO_2 = (1.0, 10.0)   # (c_fp, c_fn)  -> nb 04 (asimetrico)


# ---------------------------------------------------------------------- #
# Matriz de coste                                                        #
# ---------------------------------------------------------------------- #
def matriz_coste(c_fp: float, c_fn: float) -> np.ndarray:
    """Matriz de coste 2x2 con convencion M[real, pred].

    M[0,0]=TN=0, M[0,1]=FP=c_fp, M[1,0]=FN=c_fn, M[1,1]=TP=0.
    """
    return np.array([[0.0, float(c_fp)], [float(c_fn), 0.0]], dtype="float64")


def costes_desde_matriz(matriz) -> tuple[float, float]:
    """Recupera (c_fp, c_fn) de una matriz M[real, pred] = (M[0,1], M[1,0])."""
    M = np.asarray(matriz, dtype="float64")
    return float(M[0, 1]), float(M[1, 0])


def umbral_analitico(c_fp: float, c_fn: float) -> float:
    """Umbral bayesiano optimo teorico = c_fp / (c_fp + c_fn).

    Simetrico (1,1) -> 0.5 ; asimetrico (1,10) -> 1/11 = 0.090909.
    SOLO referencia / verificacion de sanidad; NO se usa para decidir (se decide
    por argmin empirico sobre las probas OOF, ver `umbral_optimo`).
    """
    c_fp = float(c_fp)
    c_fn = float(c_fn)
    return c_fp / (c_fp + c_fn)


# ---------------------------------------------------------------------- #
# Coercion / validacion de etiquetas duras                              #
# ---------------------------------------------------------------------- #
def _a_np_binario(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Coacciona `y_true` e `y_pred` a np.ndarray int {0,1}.

    Valida igualdad de longitud y dominio {0,1}. Ignora el indice de pd.Series
    (usa solo los valores). Rechaza NaN.
    """
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    if y.shape[0] != p.shape[0]:
        raise ValueError(
            f"y_true e y_pred tienen longitudes distintas: {y.shape[0]} vs {p.shape[0]}")
    if np.isnan(y.astype("float64")).any() or np.isnan(p.astype("float64")).any():
        raise ValueError("y_true/y_pred contienen NaN; no son etiquetas duras validas.")
    y = y.astype("int64")
    p = p.astype("int64")
    if not np.isin(y, ETIQUETAS).all():
        raise ValueError("y_true fuera del dominio {0,1}.")
    if not np.isin(p, ETIQUETAS).all():
        raise ValueError("y_pred fuera del dominio {0,1}.")
    return y, p


# ---------------------------------------------------------------------- #
# Coste esperado empirico                                                #
# ---------------------------------------------------------------------- #
def coste_esperado(y_true, y_pred, c_fp: float, c_fn: float) -> float:
    """Coste medio empirico de decisiones DURAS y_pred vs realidad y_true.

    mean( ((pred==1)&(y==0))*c_fp + ((pred==0)&(y==1))*c_fn ).
    """
    y, p = _a_np_binario(y_true, y_pred)
    fp = ((p == 1) & (y == 0)) * float(c_fp)
    fn = ((p == 0) & (y == 1)) * float(c_fn)
    return float(np.mean(fp + fn))


# ---------------------------------------------------------------------- #
# Barrido de umbral O(N log N) (algoritmo A.1.1)                         #
# ---------------------------------------------------------------------- #
def barrido_umbral(y_true, proba, c_fp: float, c_fn: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Coste esperado para cada umbral candidato, en O(N log N).

    Regla de decision: pred = 1 sii proba > umbral (operador ESTRICTO, patron del
    profesor). Candidatos = `np.unique(proba)` (ascendente). Para cada umbral u,
    con `searchsorted(side='right')` se cuenta k = #{p <= u}:
        FP = #{p >  u & y==0}   (denegados buenos)
        FN = #{p <= u & y==1}   (concedidos morosos)
        coste(u) = (c_fp*FP + c_fn*FN) / N
    Borde: el umbral maximo (= max(proba)) equivale a conceder a todos; "denegar a
    todos" NO es alcanzable por el barrido (para eso `evaluar_baselines`). NaN en
    proba -> error explicito. El doble bucle ingenuo es el oraculo de test (coincide
    a 1e-12).
    """
    p = np.asarray(proba, dtype="float64").ravel()
    y = np.asarray(y_true).ravel()
    if p.shape[0] != y.shape[0]:
        raise ValueError(
            f"proba e y_true tienen longitudes distintas: {p.shape[0]} vs {y.shape[0]}")
    if np.isnan(p).any():
        raise ValueError("proba contiene NaN; no se puede barrer el umbral.")
    y = y.astype("int64")
    if not np.isin(y, ETIQUETAS).all():
        raise ValueError("y_true fuera del dominio {0,1}.")

    n = p.shape[0]
    orden = np.argsort(p, kind="mergesort")
    p_s = p[orden]
    y_s = y[orden]
    umbrales = np.unique(p)                                   # ascendente

    cum0 = np.concatenate([[0], np.cumsum(y_s == 0)])         # #{p<=u & y==0} acumulado
    cum1 = np.concatenate([[0], np.cumsum(y_s == 1)])         # #{p<=u & y==1} acumulado
    k = np.searchsorted(p_s, umbrales, side="right")          # #{p <= u}

    total_buenos = int((y_s == 0).sum())
    FP = total_buenos - cum0[k]                               # p>u & y==0
    FN = cum1[k]                                              # p<=u & y==1
    costes = (float(c_fp) * FP + float(c_fn) * FN) / n
    return umbrales, costes.astype("float64")


def umbral_optimo(y_true, proba, c_fp: float, c_fn: float) -> tuple[float, float]:
    """Umbral que minimiza el coste esperado (argmin sobre `barrido_umbral`).

    `umbrales[np.argmin(costes)]`: `np.argmin` devuelve el PRIMER minimo, luego en
    caso de empate gana el umbral MAS BAJO (mas conservador: deniega mas). Este es
    el argmin empirico que el profesor no proporciona.
    """
    umbrales, costes = barrido_umbral(y_true, proba, c_fp, c_fn)
    i = int(np.argmin(costes))
    return float(umbrales[i]), float(costes[i])


# ---------------------------------------------------------------------- #
# Baselines triviales                                                    #
# ---------------------------------------------------------------------- #
def evaluar_baselines(y_true, c_fp: float, c_fn: float) -> dict:
    """Coste de las dos politicas triviales.

    conceder_todo (pred=0 -> paga solo FN) y denegar_todo (pred=1 -> paga solo FP).
    """
    y = np.asarray(y_true).ravel().astype("int64")
    n = y.shape[0]
    ceros = np.zeros(n, dtype="int64")
    unos = np.ones(n, dtype="int64")
    return {
        "conceder_todo": coste_esperado(y, ceros, c_fp, c_fn),
        "denegar_todo": coste_esperado(y, unos, c_fp, c_fn),
    }


# ---------------------------------------------------------------------- #
# Matriz de confusion (real x pred)                                      #
# ---------------------------------------------------------------------- #
def _matriz_confusion(y, p) -> np.ndarray:
    """Matriz 2x2 con filas=real, cols=pred (entradas int)."""
    cm = np.zeros((2, 2), dtype="int64")
    for r in (0, 1):
        for c in (0, 1):
            cm[r, c] = int(np.sum((y == r) & (p == c)))
    return cm


# ---------------------------------------------------------------------- #
# Evaluacion de decisiones duras (bandit / accion directa)              #
# ---------------------------------------------------------------------- #
def evaluar_predicciones(y_true, y_pred, c_fp: float, c_fn: float) -> dict:
    """Evalua decisiones DURAS (accion 0/1). Para el bandit (greedy) o cualquier
    politica que emite accion directa sin umbralizar una proba.

    Devuelve coste, baselines, matriz de confusion (2x2, real x pred) y tasa de
    denegacion (mean(pred==1)).
    """
    y, p = _a_np_binario(y_true, y_pred)
    return {
        "coste": coste_esperado(y, p, c_fp, c_fn),
        "coste_baselines": evaluar_baselines(y, c_fp, c_fn),
        "matriz_confusion": _matriz_confusion(y, p),
        "tasa_denegacion": float(np.mean(p == 1)),
    }


# ---------------------------------------------------------------------- #
# Aplicar umbral                                                         #
# ---------------------------------------------------------------------- #
def aplicar_umbral(proba, umbral: float) -> np.ndarray:
    """Decision dura desde proba: (proba > umbral).astype(int). Operador ESTRICTO."""
    p = np.asarray(proba, dtype="float64").ravel()
    return (p > float(umbral)).astype("int64")


# ---------------------------------------------------------------------- #
# Evaluacion de un modelo supervisado                                    #
# ---------------------------------------------------------------------- #
def evaluar_modelo(modelo, X, y_true, c_fp: float, c_fn: float,
                   umbral: float | None = None) -> dict:
    """Evalua un modelo supervisado (con predict_proba) por coste.

    proba = modelo.predict_proba(X)[:, 1] = P(impago/denegar).
    - `umbral is None`  -> se calcula `umbral_optimo` sobre ESTAS probas (uso INTERNO,
      p.ej. inspeccion; NO usar sobre test porque seria fuga).
    - `umbral` dado     -> se aplica TAL CUAL (uso en TEST, ANTI-FUGA: el umbral vino
      de la CV en train).
    Devuelve {umbral_optimo, coste, coste_baselines, matriz_confusion,
    tasa_denegacion, proba}. `umbral_optimo` guarda el umbral REALMENTE aplicado.
    """
    proba = np.asarray(modelo.predict_proba(X))[:, 1].astype("float64")
    y = np.asarray(y_true).ravel().astype("int64")
    if umbral is None:
        umbral_aplicado, _ = umbral_optimo(y, proba, c_fp, c_fn)
    else:
        umbral_aplicado = float(umbral)
    pred = aplicar_umbral(proba, umbral_aplicado)
    return {
        "umbral_optimo": float(umbral_aplicado),
        "coste": coste_esperado(y, pred, c_fp, c_fn),
        "coste_baselines": evaluar_baselines(y, c_fp, c_fn),
        "matriz_confusion": _matriz_confusion(y, pred),
        "tasa_denegacion": float(np.mean(pred == 1)),
        "proba": proba,
    }
