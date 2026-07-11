"""Familias de modelos de concesion de credito e orquestacion CV anti-fuga (B4-T2).

Interfaz comun `ModeloCredito` (Protocol) + `BaseModeloCredito`. Familias:
  - `ModeloLineal`      (familia='lineal',   LogisticRegression)
  - `ModeloBoosted`     (familia='boosted',  XGBoost, SIN early_stopping)
  - `ModeloBoostedLGBM` (opcional, solo si lightgbm instalado)
  - `RedNeuronalCredito`(familia='neuronal', Keras 3, 'pequena'/'profunda', joblib-safe)
  - `BanditLinUCB`      (familia='bandit',   LinUCB contextual numpy puro; VIVE AQUI)

Seleccion de umbral Y familia por CV(5) estratificado OOF sobre TRAIN (anti-fuga):
`comparar_familias_cv`. El umbral se elige por argmin del coste sobre las probas OOF
CONCATENADAS (pooled), no por fold. El bandit se evalua por accion (greedy), no proba.

Convencion de coste: ver `src.cost_utils` (positivo=impago=1; pred1=denegar; FN caro).
Espanol sin tildes en comentarios/docstrings.
"""

from __future__ import annotations

import copy
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import cost_utils as cu
from .preprocessing import PreprocesadorCredito

# ---------------------------------------------------------------------- #
# Contrato de features (12 = 10 continuas + 2 flags), en orden canonico   #
# ---------------------------------------------------------------------- #
FEATURES_MODELO = (tuple(PreprocesadorCredito.COLUMNAS_FEATURES)
                   + tuple(PreprocesadorCredito.FLAGS))     # 12 columnas

# LightGBM es opcional: se anade al orquestador solo si esta instalado.
try:  # pragma: no cover
    import lightgbm  # noqa: F401
    _TIENE_LGBM = True
except Exception:
    _TIENE_LGBM = False


# ---------------------------------------------------------------------- #
# Reproducibilidad                                                       #
# ---------------------------------------------------------------------- #
def fijar_semillas(seed: int = 42) -> None:
    """Fija TODAS las fuentes de aleatoriedad para reproducibilidad.

    os.environ PYTHONHASHSEED=str(seed), TF_DETERMINISTIC_OPS='1'; random.seed;
    np.random.seed; si keras/tf disponibles: keras.utils.set_random_seed(seed) +
    tf.config.experimental.enable_op_determinism(). Llamar al inicio de cada
    notebook y de cada fit neuronal.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    try:  # keras/tf pueden no estar cargados en todos los flujos
        import keras
        import tensorflow as tf
        keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ---------------------------------------------------------------------- #
# Helpers de matriz de features                                          #
# ---------------------------------------------------------------------- #
def _matriz_features(X) -> np.ndarray:
    """Devuelve X como np.ndarray float64. Si es DataFrame con las 12 columnas de
    FEATURES_MODELO, las selecciona en el ORDEN canonico; si no, usa X tal cual."""
    if hasattr(X, "columns") and hasattr(X, "loc"):
        cols = list(X.columns)
        if all(f in cols for f in FEATURES_MODELO):
            X = X.loc[:, list(FEATURES_MODELO)]
        return X.to_numpy(dtype="float64")
    return np.asarray(X, dtype="float64")


def _vector_y(y) -> np.ndarray:
    """y -> np.ndarray int64 1D (ignora indice de pd.Series)."""
    return np.asarray(y).ravel().astype("int64")


def _es_keras_model(obj) -> bool:
    try:
        import keras
        return isinstance(obj, keras.Model)
    except Exception:
        return False


# ---------------------------------------------------------------------- #
# Protocolo e interfaz base                                              #
# ---------------------------------------------------------------------- #
@runtime_checkable
class ModeloCredito(Protocol):
    """Interfaz comun de todas las familias (supervisadas y bandit)."""
    familia: str            # 'lineal'|'boosted'|'neuronal'|'bandit'
    umbral_: float          # supervisado: umbral de decision; bandit: 0.5 (no se usa)
    estimador_: object      # modelo nativo (para SHAP/inspeccion en 07)

    def fit(self, X, y) -> "ModeloCredito": ...
    def predict_proba(self, X) -> np.ndarray: ...   # (n,2), col 1 = P(impago/denegar)
    def predict(self, X) -> np.ndarray: ...         # (n,), {0,1} = accion


class BaseModeloCredito:
    """Base con umbral, predict comun y clonado (para CV) por deepcopy."""
    familia = "base"

    def __init__(self, umbral: float = 0.5, random_state: int = 42) -> None:
        self.umbral_ = float(umbral)
        self.random_state = int(random_state)
        self.estimador_ = None
        self.nombres_features_ = None

    def fit(self, X, y):  # pragma: no cover - lo implementa cada familia
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        if self.estimador_ is None:
            raise RuntimeError("Modelo no ajustado (estimador_ is None).")
        return np.asarray(self.estimador_.predict_proba(_matriz_features(X)))

    def predict(self, X) -> np.ndarray:
        """Decision dura comun: (P(denegar) > umbral_). Operador ESTRICTO."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > self.umbral_).astype("int64")

    def set_umbral(self, umbral: float) -> "BaseModeloCredito":
        """Fija el umbral de decision y devuelve self (encadenable)."""
        self.umbral_ = float(umbral)
        return self

    def _registrar_features(self, X) -> None:
        if hasattr(X, "columns"):
            self.nombres_features_ = list(X.columns)
        else:
            self.nombres_features_ = list(FEATURES_MODELO)


# ---------------------------------------------------------------------- #
# Familia lineal                                                         #
# ---------------------------------------------------------------------- #
class ModeloLineal(BaseModeloCredito):
    """Regresion logistica. class_weight='balanced' es variante opt-in (M-4)."""
    familia = "lineal"

    def __init__(self, C: float = 1.0, class_weight=None,
                 umbral: float = 0.5, random_state: int = 42) -> None:
        super().__init__(umbral=umbral, random_state=random_state)
        self.C = float(C)
        self.class_weight = class_weight

    def fit(self, X, y):
        self._registrar_features(X)
        Xn = _matriz_features(X)
        yn = _vector_y(y)
        self.estimador_ = LogisticRegression(
            C=self.C, max_iter=1000, solver="lbfgs",
            class_weight=self.class_weight, random_state=self.random_state)
        self.estimador_.fit(Xn, yn)
        return self


# ---------------------------------------------------------------------- #
# Familia boosted (XGBoost, SIN early_stopping)                          #
# ---------------------------------------------------------------------- #
class ModeloBoosted(BaseModeloCredito):
    """XGBoost con hiperparametros del spec. SIN early_stopping (evita eval_set y su
    riesgo de fuga). scale_pos_weight~=14 es variante opt-in."""
    familia = "boosted"

    def __init__(self, scale_pos_weight: float = 1.0,
                 umbral: float = 0.5, random_state: int = 42) -> None:
        super().__init__(umbral=umbral, random_state=random_state)
        self.scale_pos_weight = float(scale_pos_weight)

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self._registrar_features(X)
        Xn = _matriz_features(X)
        yn = _vector_y(y)
        self.estimador_ = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=self.scale_pos_weight, eval_metric="logloss",
            tree_method="hist", n_jobs=1, random_state=self.random_state)
        self.estimador_.fit(Xn, yn)          # sin eval_set / early_stopping
        return self


class ModeloBoostedLGBM(BaseModeloCredito):
    """LightGBM (opcional). Identico contrato; solo si lightgbm esta instalado."""
    familia = "boosted"

    def __init__(self, scale_pos_weight: float = 1.0,
                 umbral: float = 0.5, random_state: int = 42) -> None:
        super().__init__(umbral=umbral, random_state=random_state)
        self.scale_pos_weight = float(scale_pos_weight)

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        self._registrar_features(X)
        Xn = _matriz_features(X)
        yn = _vector_y(y)
        self.estimador_ = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=self.scale_pos_weight, n_jobs=1,
            random_state=self.random_state, verbose=-1)
        self.estimador_.fit(Xn, yn)
        return self


# ---------------------------------------------------------------------- #
# Familia neuronal (Keras 3, joblib-safe, pesos congelados)              #
# ---------------------------------------------------------------------- #
class RedNeuronalCredito(BaseModeloCredito):
    """MLP Keras 3 con dos arquitecturas candidatas.

    'pequena' : Dense(32,relu)->Dense(16,relu)->Dense(1,sigmoid).
    'profunda': Dense(64,relu)->Drop(.3)->Dense(32,relu)->Drop(.3)->Dense(16,relu)->Dense(1,sigmoid).
    Serializacion joblib-safe: en __getstate__ los pesos entrenados se embeben como
    bytes .keras; en __setstate__ se reconstruye el modelo. Se guarda el objeto
    ENTRENADO (pesos congelados), nunca una receta de reentreno.
    """
    familia = "neuronal"

    def __init__(self, arquitectura: str = "pequena", epochs: int = 50,
                 batch_size: int = 256, umbral: float = 0.5,
                 random_state: int = 42) -> None:
        super().__init__(umbral=umbral, random_state=random_state)
        if arquitectura not in ("pequena", "profunda"):
            raise ValueError("arquitectura debe ser 'pequena' o 'profunda'.")
        self.arquitectura = arquitectura
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self._keras_bytes = None

    def _construir_red(self, n_features: int):
        import keras
        from keras import layers
        if self.arquitectura == "pequena":
            capas = [
                layers.Input(shape=(n_features,)),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        else:  # profunda
            capas = [
                layers.Input(shape=(n_features,)),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        modelo = keras.Sequential(capas)
        modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                       loss="binary_crossentropy",
                       metrics=[keras.metrics.AUC(name="auc")])
        return modelo

    def fit(self, X, y):
        fijar_semillas(self.random_state)
        import keras  # noqa: F401  (asegura backend cargado)
        from keras.callbacks import EarlyStopping
        self._registrar_features(X)
        Xn = _matriz_features(X).astype("float32")
        yn = _vector_y(y).astype("float32")
        self.estimador_ = self._construir_red(Xn.shape[1])
        cb = EarlyStopping(monitor="val_loss", patience=6,
                           restore_best_weights=True)
        self.estimador_.fit(
            Xn, yn, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.1, callbacks=[cb], verbose=0, class_weight=None)
        self._keras_bytes = None
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.estimador_ is None:
            raise RuntimeError("Modelo neuronal no ajustado.")
        Xn = _matriz_features(X).astype("float32")
        p1 = np.asarray(self.estimador_.predict(Xn, verbose=0)).ravel()
        return np.column_stack([1.0 - p1, p1])

    # -- Serializacion joblib-safe (embebe bytes .keras) ---------------- #
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        est = state.get("estimador_", None)
        if est is not None and _es_keras_model(est):
            import keras
            d = tempfile.mkdtemp()
            try:
                ruta = os.path.join(d, "m.keras")
                keras.saving.save_model(est, ruta)
                with open(ruta, "rb") as fh:
                    state["_keras_bytes"] = fh.read()
            finally:
                shutil.rmtree(d, ignore_errors=True)
            state["estimador_"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        kb = state.get("_keras_bytes", None)
        self.__dict__.update(state)
        if kb is not None:
            import keras
            d = tempfile.mkdtemp()
            try:
                ruta = os.path.join(d, "m.keras")
                with open(ruta, "wb") as fh:
                    fh.write(kb)
                self.estimador_ = keras.saving.load_model(ruta, compile=True)
            finally:
                shutil.rmtree(d, ignore_errors=True)
            self._keras_bytes = None


# ---------------------------------------------------------------------- #
# Bandit LinUCB contextual (numpy puro) — VIVE EN ESTE MODULO            #
# ---------------------------------------------------------------------- #
def _coste_accion(accion: int, y: int, c_fp: float, c_fn: float) -> float:
    """Coste de una accion individual (para el reward=-coste del bandit online)."""
    if accion == 0 and y == 1:
        return c_fn
    if accion == 1 and y == 0:
        return c_fp
    return 0.0


class BanditLinUCB(BaseModeloCredito):
    """LinUCB contextual disjoint (2 acciones), reward = -coste. numpy puro.

    accion 0 = conceder, accion 1 = denegar. A_a=I, b_a=0 iniciales; A_inv por accion
    mantenido incremental via Sherman-Morrison (O(d^2)/paso). Entreno ONLINE con
    exploracion; evaluacion GREEDY vectorizada sin exploracion ni update. Se estandariza
    con mu/sd de train (el parquet ya viene z-scored -> casi identidad, pero autocontenido).
    """
    familia = "bandit"

    def __init__(self, c_fp: float, c_fn: float, alpha: float = 1.0,
                 passes: int = 1, umbral: float = 0.5,
                 random_state: int = 42) -> None:
        super().__init__(umbral=umbral, random_state=random_state)
        self.c_fp = float(c_fp)
        self.c_fn = float(c_fn)
        self.alpha = float(alpha)
        self.passes = int(passes)
        self.mu_ = None
        self.sd_ = None
        self.A_inv_ = None
        self.b_ = None
        self.theta_ = None

    # -- estandarizacion (mu/sd de train, una sola vez) ----------------- #
    def _estandarizar_fit(self, X: np.ndarray) -> None:
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_ = np.where(self.sd_ == 0.0, 1.0, self.sd_)

    def _estandarizar(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) / self.sd_

    # -- decision/actualizacion por fila (fase online) ------------------ #
    def _predict_una(self, x_aug: np.ndarray, explore: bool = True) -> int:
        best, best_score = 0, -np.inf
        for a in range(2):
            mean = float(self.theta_[a] @ x_aug)
            if explore:
                bonus = self.alpha * np.sqrt(
                    max(0.0, float(x_aug @ self.A_inv_[a] @ x_aug)))
            else:
                bonus = 0.0
            score = mean + bonus
            if score > best_score:
                best_score, best = score, a
        return best

    def _update(self, x_aug: np.ndarray, accion: int, reward: float) -> None:
        Ainv = self.A_inv_[accion]
        Ax = Ainv @ x_aug
        denom = 1.0 + float(x_aug @ Ax)
        self.A_inv_[accion] = Ainv - np.outer(Ax, Ax) / denom     # Sherman-Morrison
        self.b_[accion] = self.b_[accion] + reward * x_aug
        self.theta_[accion] = self.A_inv_[accion] @ self.b_[accion]

    def fit(self, X, y):
        self._registrar_features(X)
        Xn = _matriz_features(X)
        yn = _vector_y(y)
        self._estandarizar_fit(Xn)
        Xs = self._estandarizar(Xn)
        d = Xn.shape[1] + 1                       # + intercepto
        self.A_inv_ = [np.eye(d), np.eye(d)]
        self.b_ = [np.zeros(d), np.zeros(d)]
        self.theta_ = [np.zeros(d), np.zeros(d)]
        rng = np.random.RandomState(self.random_state)
        n = len(Xs)
        for _ in range(self.passes):
            for i in rng.permutation(n):          # online, orden aleatorio por pasada
                x = np.append(Xs[i], 1.0)
                a = self._predict_una(x, explore=True)
                r = -_coste_accion(a, int(yn[i]), self.c_fp, self.c_fn)
                self._update(x, a, r)
        self.estimador_ = self                    # modelo nativo = el propio LinUCB
        return self

    def _scores(self, X) -> tuple[np.ndarray, np.ndarray]:
        Xn = _matriz_features(X)
        Xs = self._estandarizar(Xn)
        Xa = np.column_stack([Xs, np.ones(len(Xs))])
        return Xa @ self.theta_[0], Xa @ self.theta_[1]

    def predict(self, X) -> np.ndarray:
        """Greedy VECTORIZADO: pred=1 sii score_1 > score_0 (empate -> conceder)."""
        s0, s1 = self._scores(X)
        return (s1 > s0).astype("int64")

    def predict_proba(self, X) -> np.ndarray:
        """Pseudo-proba sintetica (NO calibrada) = softmax([score_0, score_1]).

        Solo para que 05/07 lo traten como caja negra; la COMPARACION usa `predict`
        (accion), nunca umbraliza esta pseudo-proba.
        """
        s0, s1 = self._scores(X)
        S = np.column_stack([s0, s1])
        S = S - S.max(axis=1, keepdims=True)
        e = np.exp(S)
        P = e / e.sum(axis=1, keepdims=True)
        return P                                   # col 1 = P(denegar)


# ---------------------------------------------------------------------- #
# Orquestacion CV(5) OOF anti-fuga                                       #
# ---------------------------------------------------------------------- #
@dataclass
class ResultadoComparacion:
    tabla: pd.DataFrame               # una fila por candidato + baselines, coste_cv asc
    modelos: dict = field(default_factory=dict)   # nombre -> ModeloCredito refit TRAIN
    umbrales: dict = field(default_factory=dict)  # nombre -> umbral OOF (None bandit)
    oof_proba: dict = field(default_factory=dict) # nombre -> np.ndarray OOF (None bandit)
    ganador: str = ""                 # argmin(coste_cv) entre modelos (no baselines)
    ganador_supervisado: str = ""     # mejor familia supervisada (para D-0.2 en 04)


def construir_candidatos(c_fp: float, c_fn: float, random_state: int = 42) -> dict:
    """Diccionario de candidatos (instancias sin ajustar) en orden canonico.

    logistica, xgboost, (xgboost_lgbm si _TIENE_LGBM), mlp_pequena, mlp_profunda, linucb.
    """
    cands: dict = {
        "logistica": ModeloLineal(random_state=random_state),
        "xgboost": ModeloBoosted(random_state=random_state),
    }
    if _TIENE_LGBM:
        cands["xgboost_lgbm"] = ModeloBoostedLGBM(random_state=random_state)
    cands["mlp_pequena"] = RedNeuronalCredito("pequena", random_state=random_state)
    cands["mlp_profunda"] = RedNeuronalCredito("profunda", random_state=random_state)
    cands["linucb"] = BanditLinUCB(c_fp, c_fn, alpha=1.0, passes=1,
                                   random_state=random_state)
    return cands


def comparar_familias_cv(X_train, y_train, c_fp: float, c_fn: float,
                         n_splits: int = 5, random_state: int = 42,
                         candidatos: dict | None = None) -> ResultadoComparacion:
    """Protocolo CANONICO de seleccion umbral+familia (D-2.1), anti-fuga.

    StratifiedKFold(n_splits, shuffle=True, random_state). Para cada candidato se
    recorren los folds sobre CLONES frescos (deepcopy) y se ensambla la prediccion
    OOF alineada al orden de train:
      SUPERVISADAS: OOF proba; umbral*, coste_cv = umbral_optimo(y, oof_proba, c_fp, c_fn)
        (argmin sobre las probas OOF CONCATENADAS, pooled, no la mediana de folds).
        AUC de contexto sobre la OOF proba. Refit final en TRAIN COMPLETO + set_umbral.
      BANDIT: OOF accion greedy; coste_cv = coste_esperado(y, oof_accion, c_fp, c_fn);
        umbral=None. Refit final en TRAIN COMPLETO.
    Baselines conceder/denegar-todo en la misma tabla (sanidad). Devuelve tabla ordenada
    por coste_cv asc; ganador = argmin coste_cv entre modelos (no baselines).
    """
    fijar_semillas(random_state)
    X = _matriz_features(X_train)
    y = _vector_y(y_train)
    n = len(y)
    if candidatos is None:
        candidatos = construir_candidatos(c_fp, c_fn, random_state=random_state)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, y))                      # mismos folds para todos

    baselines = cu.evaluar_baselines(y, c_fp, c_fn)
    coste_conceder = baselines["conceder_todo"]

    filas = []
    modelos: dict = {}
    umbrales: dict = {}
    oof_proba_dict: dict = {}

    for nombre, plantilla in candidatos.items():
        es_bandit = getattr(plantilla, "familia", "") == "bandit"

        if es_bandit:
            oof_accion = np.zeros(n, dtype="int64")
            for tr_idx, va_idx in splits:
                m = copy.deepcopy(plantilla)            # clon fresco sin ajustar
                m.fit(X[tr_idx], y[tr_idx])
                oof_accion[va_idx] = m.predict(X[va_idx])
            coste_cv = cu.coste_esperado(y, oof_accion, c_fp, c_fn)
            umbral_star = None
            auc = float("nan")
            tasa_deny = float(np.mean(oof_accion == 1))
            oof_proba_dict[nombre] = None
            notas = "greedy accion; LinUCB es LINEAL"
            plantilla.fit(X, y)                          # refit train completo
        else:
            oof = np.zeros(n, dtype="float64")
            for tr_idx, va_idx in splits:
                m = copy.deepcopy(plantilla)
                m.fit(X[tr_idx], y[tr_idx])
                oof[va_idx] = np.asarray(m.predict_proba(X[va_idx]))[:, 1]
            umbral_star, coste_cv = cu.umbral_optimo(y, oof, c_fp, c_fn)
            pred_cv = cu.aplicar_umbral(oof, umbral_star)
            tasa_deny = float(np.mean(pred_cv == 1))
            auc = float(roc_auc_score(y, oof))
            oof_proba_dict[nombre] = oof
            notas = ""
            plantilla.fit(X, y).set_umbral(umbral_star)  # refit + umbral CV

        modelos[nombre] = plantilla
        umbrales[nombre] = (None if umbral_star is None else float(umbral_star))
        mejora = (coste_conceder - coste_cv) / coste_conceder if coste_conceder else 0.0
        filas.append({
            "familia": plantilla.familia,
            "modelo": nombre,
            "coste_cv": float(coste_cv),
            "umbral": (np.nan if umbral_star is None else float(umbral_star)),
            "auc_contexto": auc,
            "tasa_denegacion_cv": tasa_deny,
            "mejora_vs_conceder_todo": float(mejora),
            "notas": notas,
        })

    # Filas baseline (sanidad; no compiten por ganador)
    for nombre_bl, coste_bl, deny_bl in (
            ("conceder_todo", baselines["conceder_todo"], 0.0),
            ("denegar_todo", baselines["denegar_todo"], 1.0)):
        filas.append({
            "familia": "baseline",
            "modelo": nombre_bl,
            "coste_cv": float(coste_bl),
            "umbral": np.nan,
            "auc_contexto": float("nan"),
            "tasa_denegacion_cv": deny_bl,
            "mejora_vs_conceder_todo": (
                (coste_conceder - coste_bl) / coste_conceder if coste_conceder else 0.0),
            "notas": "baseline trivial",
        })

    tabla = pd.DataFrame(filas).sort_values(
        "coste_cv", kind="mergesort").reset_index(drop=True)

    nombres_modelos = [nom for nom in candidatos]
    ganador = min(nombres_modelos, key=lambda nm: modelos_coste(filas, nm))
    supervisados = [nom for nom in nombres_modelos
                    if getattr(candidatos[nom], "familia", "") != "bandit"]
    ganador_sup = min(supervisados, key=lambda nm: modelos_coste(filas, nm)) \
        if supervisados else ""

    return ResultadoComparacion(
        tabla=tabla, modelos=modelos, umbrales=umbrales,
        oof_proba=oof_proba_dict, ganador=ganador,
        ganador_supervisado=ganador_sup)


def modelos_coste(filas: list, nombre: str) -> float:
    """coste_cv de una fila-candidato por nombre (helper de argmin)."""
    for f in filas:
        if f["modelo"] == nombre:
            return f["coste_cv"]
    return float("inf")


# ---------------------------------------------------------------------- #
# Fallback holdout (NO canonico; solo maquina lenta con MLP)             #
# ---------------------------------------------------------------------- #
def dividir_validacion(X, y, test_size: float = 0.2, random_state: int = 42):
    """Holdout 80/20 estratificado. FALLBACK documentado; la via canonica es
    `comparar_familias_cv` (CV(5) OOF). Devuelve (X_fit, X_val, y_fit, y_val)."""
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=random_state)


# ---------------------------------------------------------------------- #
# Serializacion (bundle DICT joblib)                                     #
# ---------------------------------------------------------------------- #
def guardar_modelo(bundle: dict, ruta: str) -> None:
    """Serializa el bundle dict con joblib. Crea el dir padre si falta.

    El wrapper Keras (si esta dentro) embebe sus bytes .keras via __getstate__ ->
    UNA sola .joblib autocontenida. Round-trip recomendado (predict identico).
    """
    import joblib
    carpeta = os.path.dirname(os.path.abspath(ruta))
    os.makedirs(carpeta, exist_ok=True)
    joblib.dump(bundle, ruta)


def cargar_modelo(ruta: str) -> dict:
    """Deserializa el bundle dict con joblib."""
    import joblib
    return joblib.load(ruta)
