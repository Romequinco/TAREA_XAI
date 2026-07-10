"""Pipeline de preprocesado (fit/transform) serializable para el taller B4-T2.

Se ajusta UNA sola vez sobre el train-split y se aplica de forma identica a
train, test y produccion, garantizando el mismo tratamiento en los tres flujos
y sin fuga de datos (todo parametro se aprende en `fit`, nada se hardcodea).

Contrato:
    - Entrada: DataFrame con las 10 columnas de `COLUMNAS_FEATURES` (el target y
      cualquier id se ignoran/descartan: el modulo NO los toca).
    - Salida: DataFrame de 12 columnas (10 features float64 + 2 flags int8), con
      el indice de entrada preservado y sin dropear ni reordenar filas.

Decisiones canonicas (ver spec_final.md):
    - `income_no_usable = MonthlyIncome.isna() | (MonthlyIncome <= 1.0)` es la
      UNICA mascara para el flag `is_missing_monthlyincome`, la imputacion de
      income y el recalculo de DebtRatio.
    - Winsor POR COLUMNA (Revolving p99, DebtRatio p99.9), lower=0, aprendido en fit.
    - Recalculo de DebtRatio activado por defecto (hipotesis del numerador crudo).
    - log1p solo en {DebtRatio, MonthlyIncome, RevolvingUtilization} (NO age) y
      StandardScaler sobre las 10 features, ambos condicionados por `escalar`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class PreprocesadorCredito(BaseEstimator, TransformerMixin):
    """Preprocesador monolitico DataFrame->DataFrame, serializable en un joblib.

    Ejecuta los pasos P1..P11 en orden. Los pasos que aprenden parametros lo
    hacen SOLO en `fit` (sobre el train-split) y los guardan en atributos con
    sufijo `_`. `transform` los reaplica sin volver a aprender.
    """

    # ------------------------------------------------------------------ #
    # A.1 Constantes de clase (contrato de datos)                        #
    # ------------------------------------------------------------------ #
    TARGET = "SeriousDlqin2yrs"
    COLUMNAS_FEATURES = [
        "RevolvingUtilizationOfUnsecuredLines", "age",
        "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]
    COLS_PASTDUE = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
                    "NumberOfTime60-89DaysPastDueNotWorse"]
    COLS_LOG = ["DebtRatio", "MonthlyIncome", "RevolvingUtilizationOfUnsecuredLines"]  # NO age
    COLS_WINSOR = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]
    # Columnas SIN paso de imputacion propio (age/MonthlyIncome/NumberOfDependents SI lo
    # tienen). Con los datos actuales nunca traen NaN, pero un patron de nulos futuro se
    # colaria hasta la salida: la guarda P7b las rellena con la mediana de train.
    COLS_RESTO_NAN = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio",
                      "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
                      "NumberOfTime60-89DaysPastDueNotWorse",
                      "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]
    WINSOR_QUANTILES = {"RevolvingUtilizationOfUnsecuredLines": 0.99, "DebtRatio": 0.999}  # POR COLUMNA
    FLAGS = ["is_missing_monthlyincome", "es_centinela_pastdue"]  # solo 2 flags
    VALORES_CENTINELA = (96, 98)
    BINS_EDAD = [-np.inf, 30, 40, 50, 60, 70, np.inf]  # bordes fijos, NO se aprenden
    UMBRAL_INCOME_NO_USABLE = 1.0  # income <= 1 es no usable
    EPS_INCOME = 1.0  # guarda division recalc (no se activa: income imputado >> 1)

    # ------------------------------------------------------------------ #
    # A.2 __init__ — flags con default DECIDIDO                          #
    # ------------------------------------------------------------------ #
    def __init__(self, escalar: bool = True,
                 recalcular_debtratio: bool = True,
                 winsorizar: bool = True) -> None:
        self.escalar = escalar
        self.recalcular_debtratio = recalcular_debtratio
        self.winsorizar = winsorizar

    # ------------------------------------------------------------------ #
    # A.3 API publica                                                    #
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y=None) -> "PreprocesadorCredito":
        """Aprende TODOS los parametros sobre `X` (el train-split) y devuelve self.

        Recorre P1..P11 con `en_fit=True`: cada paso que aprende guarda su
        `atributo_` y transforma X para que el siguiente vea la salida correcta.
        `y` se ignora (compatibilidad sklearn); nada se aprende del target.
        """
        Xw = self._validar_y_copiar(X)                 # P1
        self.columnas_entrada_ = list(Xw.columns)
        self.n_features_in_ = len(self.COLUMNAS_FEATURES)
        self._ejecutar_pasos(Xw, en_fit=True)
        self.columnas_salida_ = list(self.COLUMNAS_FEATURES) + list(self.FLAGS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica el pipeline ya ajustado. Preserva indice, nunca dropea filas."""
        check_is_fitted(self, "columnas_salida_")
        indice_entrada = X.index
        Xw = self._validar_y_copiar(X)                 # P1
        out = self._ejecutar_pasos(Xw, en_fit=False)
        # Salvaguarda de alineacion (conflicto #6): mismas filas, mismo orden.
        assert out.index.equals(indice_entrada), "transform altero el indice de entrada"
        assert len(out) == len(X), "transform cambio el numero de filas"
        return out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse PARCIAL: recupera la ESCALA (euros/anos/ratio), no la procedencia.

        - `escalar=True`: revierte StandardScaler (`z*scale_+mean_`) de las 10
          features y aplica `expm1` SOLO en COLS_LOG. age y conteos quedan
          des-escalados en unidades reales (sin expm1).
        - `escalar=False`: identidad para las 10 features (ya estan en unidades reales).
        - Flags, target e id: passthrough (no se tocan).
        NO revierte imputacion, centinela, recalculo ni winsor (lossy).
        """
        check_is_fitted(self, "columnas_salida_")
        X = X.copy()
        if self.escalar and self.scaler_ is not None:
            for i, col in enumerate(self.COLUMNAS_FEATURES):
                if col in X.columns:
                    X[col] = X[col] * self.scaler_.scale_[i] + self.scaler_.mean_[i]
                    if col in self.COLS_LOG:
                        X[col] = np.expm1(X[col])
        # escalar=False -> identidad para las continuas; flags/target/id passthrough.
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nombres de las 12 columnas de salida, en el orden del contrato C.1."""
        check_is_fitted(self, "columnas_salida_")
        return np.asarray(self.columnas_salida_, dtype=object)

    def diagnostico_debtratio(self) -> dict:
        """Diagnostico falsable del recalculo de DebtRatio (KS antes/despues, etc.)."""
        check_is_fitted(self, "debtratio_recalculo_diagnostico_")
        return dict(self.debtratio_recalculo_diagnostico_)

    # ------------------------------------------------------------------ #
    # Orquestador interno de pasos P2..P11                               #
    # ------------------------------------------------------------------ #
    def _ejecutar_pasos(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        X = self._flags_missingness(X, en_fit)            # P2
        X = self._tratar_centinelas_pastdue(X, en_fit)    # P3
        X = self._limpiar_age(X, en_fit)                  # P4  [arista dura -> P5]
        X = self._imputar_income_por_tramo(X, en_fit)     # P5
        X = self._imputar_dependents(X, en_fit)           # P6
        X = self._recalcular_debtratio(X, en_fit)         # P7
        X = self._guarda_nan_resto(X, en_fit)             # P7b [red de seguridad NaN]
        X = self._winsorizar(X, en_fit)                   # P8
        X = self._log1p(X, en_fit)                        # P9
        X = self._escalar(X, en_fit)                      # P10
        X = self._ensamblar_salida(X, en_fit)             # P11
        return X

    # ------------------------------------------------------------------ #
    # P1 _validar_y_copiar                                               #
    # ------------------------------------------------------------------ #
    def _validar_y_copiar(self, X: pd.DataFrame) -> pd.DataFrame:
        """Copia defensiva; exige las 10 features; reindexa al orden canonico;
        descarta target/id/columnas extra; castea las 10 features a float64."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PreprocesadorCredito espera un pandas.DataFrame.")
        faltantes = [c for c in self.COLUMNAS_FEATURES if c not in X.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas de features requeridas: {faltantes}")
        # Reindexa al orden canonico -> descarta target, id_fila_original y extras.
        Xw = X.loc[:, self.COLUMNAS_FEATURES].copy()
        Xw = Xw.astype({c: "float64" for c in self.COLUMNAS_FEATURES})
        return Xw

    # ------------------------------------------------------------------ #
    # P2 _flags_missingness (determinista, no aprende)                   #
    # ------------------------------------------------------------------ #
    def _flags_missingness(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """is_missing_monthlyincome = income_no_usable (isna | income<=1), int8.

        Se calcula desde el income CRUDO, antes de imputar. NO crea flag de
        dependents (p=0.114677 al controlar por edad -> no significativo).
        """
        income = X["MonthlyIncome"]
        income_no_usable = income.isna() | (income <= self.UMBRAL_INCOME_NO_USABLE)
        X["is_missing_monthlyincome"] = income_no_usable.astype("int8")
        return X

    # ------------------------------------------------------------------ #
    # P3 _tratar_centinelas_pastdue                                      #
    # ------------------------------------------------------------------ #
    def _tratar_centinelas_pastdue(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """Flag centinela ANTES de sobreescribir; 96/98 -> mediana train (=0)."""
        mask_centinela = pd.Series(False, index=X.index)
        for col in self.COLS_PASTDUE:
            mask_centinela = mask_centinela | X[col].isin(self.VALORES_CENTINELA)
        X["es_centinela_pastdue"] = mask_centinela.astype("int8")

        if en_fit:
            self.medianas_pastdue_ = {}
            for col in self.COLS_PASTDUE:
                sin_centinela = X[col].where(~X[col].isin(self.VALORES_CENTINELA))
                self.medianas_pastdue_[col] = float(sin_centinela.median())

        for col in self.COLS_PASTDUE:
            mask_col = X[col].isin(self.VALORES_CENTINELA)
            if mask_col.any():
                X.loc[mask_col, col] = self.medianas_pastdue_[col]
        return X

    # ------------------------------------------------------------------ #
    # P4 _limpiar_age  [ARISTA DURA -> P5]                               #
    # ------------------------------------------------------------------ #
    def _limpiar_age(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """age==0 (1 fila, idx 89320 en train) y age NaN -> mediana_age_ (train)."""
        if en_fit:
            age_valida = X["age"].where(X["age"] > 0)
            self.mediana_age_ = float(age_valida.median())
        mask_age = X["age"].isna() | (X["age"] == 0)
        if mask_age.any():
            X.loc[mask_age, "age"] = self.mediana_age_
        return X

    # ------------------------------------------------------------------ #
    # P5 _imputar_income_por_tramo (necesita age ya limpia)              #
    # ------------------------------------------------------------------ #
    def _imputar_income_por_tramo(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """Filas income_no_usable -> NaN -> mediana del tramo de edad.

        Las medianas por tramo se aprenden SOLO sobre filas usables (income>1),
        excluyendo income_no_usable. Fallback: mediana_income_global_ (income>1).
        """
        mask_income = X["is_missing_monthlyincome"] == 1
        # Marca como NaN los no-usables (incluye income<=1, que son numeradores).
        X.loc[mask_income, "MonthlyIncome"] = np.nan

        n_bins = len(self.BINS_EDAD) - 1
        if en_fit:
            usables = ~mask_income
            codigos_usables = pd.cut(X.loc[usables, "age"], bins=self.BINS_EDAD).cat.codes
            medianas = X.loc[usables, "MonthlyIncome"].groupby(
                codigos_usables, observed=False).median()
            self.medianas_income_por_tramo_ = {
                int(k): float(v) for k, v in medianas.items() if pd.notna(v)
            }
            self.mediana_income_global_ = float(X.loc[usables, "MonthlyIncome"].median())

        # LUT alineada a los codigos de tramo (0..n_bins-1); fallback global.
        lut = np.array(
            [self.medianas_income_por_tramo_.get(i, self.mediana_income_global_)
             for i in range(n_bins)],
            dtype="float64",
        )
        codigos = pd.cut(X["age"], bins=self.BINS_EDAD).cat.codes.to_numpy()
        valores_tramo = lut[codigos]
        mask_np = mask_income.to_numpy()
        X.loc[mask_income, "MonthlyIncome"] = valores_tramo[mask_np]
        return X

    # ------------------------------------------------------------------ #
    # P6 _imputar_dependents (sin flag)                                  #
    # ------------------------------------------------------------------ #
    def _imputar_dependents(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """NaN -> moda_dependents_ (=0 en train). Sin flag (redundante con income)."""
        if en_fit:
            moda = X["NumberOfDependents"].mode(dropna=True)
            self.moda_dependents_ = float(moda.iloc[0]) if len(moda) else 0.0
        X["NumberOfDependents"] = X["NumberOfDependents"].fillna(self.moda_dependents_)
        return X

    # ------------------------------------------------------------------ #
    # P7 _recalcular_debtratio (aportacion analitica principal)          #
    # ------------------------------------------------------------------ #
    def _recalcular_debtratio(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """En filas income_no_usable: DebtRatio = DebtRatio_orig / income_imputado.

        NUNCA uniforme (solo esas filas), NUNCA /0 (income imputado >= min tramo).
        El diagnostico falsable se calcula SIEMPRE en fit (mida o no se aplique),
        para poder mostrar la figura F1 y justificar la activacion.
        """
        mask = X["is_missing_monthlyincome"] == 1
        dr_original = X["DebtRatio"].copy()
        income_imputado = X["MonthlyIncome"]

        dr_recalc = dr_original.copy()
        if mask.any():
            dr_recalc.loc[mask] = dr_original.loc[mask] / income_imputado.loc[mask]

        if en_fit:
            usables = ~mask
            dr_usables = dr_original.loc[usables].dropna()
            dr_no_usables_antes = dr_original.loc[mask].dropna()
            dr_no_usables_despues = dr_recalc.loc[mask].dropna()

            def _ks(a, b):
                if len(a) and len(b):
                    return float(ks_2samp(a, b).statistic)
                return float("nan")

            # Rango plausible = [0, p99.9 de la distribucion sana (usable)].
            if len(dr_usables):
                cota_plausible = float(dr_usables.quantile(0.999))
            else:
                cota_plausible = float("inf")
            if len(dr_no_usables_despues):
                pct_dentro = float(
                    dr_no_usables_despues.between(0.0, cota_plausible).mean() * 100.0)
                mediana_recalc = float(dr_no_usables_despues.median())
            else:
                pct_dentro = float("nan")
                mediana_recalc = float("nan")

            self.debtratio_recalculo_diagnostico_ = {
                "ks_antes": _ks(dr_no_usables_antes, dr_usables),
                "ks_despues": _ks(dr_no_usables_despues, dr_usables),
                "mediana_recalc_no_usables": mediana_recalc,
                "mediana_original_usables": (
                    float(dr_usables.median()) if len(dr_usables) else float("nan")),
                "pct_dentro_rango_plausible": pct_dentro,
                "cota_rango_plausible": cota_plausible,
                "n_filas_recalculadas": int(mask.sum()),
            }

        if self.recalcular_debtratio:
            X["DebtRatio"] = dr_recalc
        return X

    # ------------------------------------------------------------------ #
    # P7b _guarda_nan_resto (red de seguridad NaN, antes de winsor)      #
    # ------------------------------------------------------------------ #
    def _guarda_nan_resto(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """Rellena NaN residual en las columnas SIN imputacion propia con la mediana train.

        Cubre Revolving, DebtRatio, las 3 PastDue, Open y RealEstate (COLS_RESTO_NAN):
        columnas que hoy no tienen un paso de imputacion y que, ante un patron de nulos
        futuro, propagarian NaN por winsor/log1p/scale hasta la salida. Con los datos
        actuales estas columnas NO traen NaN, luego el `fillna` no toca ningun valor y la
        salida es BIT-IDENTICA (verificado); es una guarda defensiva, no un cambio de logica.
        `medianas_resto_` se aprende SOLO en fit (anti-fuga), tras el recalculo de DebtRatio.
        """
        if en_fit:
            self.medianas_resto_ = {
                col: float(X[col].median()) for col in self.COLS_RESTO_NAN
            }
        for col in self.COLS_RESTO_NAN:
            if X[col].isna().any():
                X[col] = X[col].fillna(self.medianas_resto_[col])
        return X

    # ------------------------------------------------------------------ #
    # P8 _winsorizar (por columna, DESPUES de P7)                        #
    # ------------------------------------------------------------------ #
    def _winsorizar(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """clip por columna a limites_winsor_ (lower=0); caps aprendidos en train.

        Revolving upper=p99 train (~1.10); DebtRatio upper=p99.9 train. El cap se
        aprende sobre el DebtRatio ya corregido por P7. clip -> robusto a no-vistos.
        Los limites se aprenden SIEMPRE en fit (aunque winsorizar=False no se aplique).
        """
        if en_fit:
            self.limites_winsor_ = {}
            for col in self.COLS_WINSOR:
                high = float(X[col].quantile(self.WINSOR_QUANTILES[col]))
                self.limites_winsor_[col] = (0.0, high)

        if self.winsorizar:
            for col in self.COLS_WINSOR:
                low, high = self.limites_winsor_[col]
                X[col] = X[col].clip(lower=low, upper=high)
        return X

    # ------------------------------------------------------------------ #
    # P9 _log1p (condicionado por escalar)                               #
    # ------------------------------------------------------------------ #
    def _log1p(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """np.log1p sobre COLS_LOG (NO age). Entrada >= 0 garantizada por P5/P8."""
        if self.escalar:
            for col in self.COLS_LOG:
                X[col] = np.log1p(X[col])
        return X

    # ------------------------------------------------------------------ #
    # P10 _escalar (condicionado por escalar)                            #
    # ------------------------------------------------------------------ #
    def _escalar(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """StandardScaler sobre las 10 features (NO flags). scaler_=None si escalar=False."""
        if self.escalar:
            if en_fit:
                self.scaler_ = StandardScaler()
                self.scaler_.fit(X[self.COLUMNAS_FEATURES])
            # StandardScaler.transform exige >=1 fila; en el caso borde vacio
            # las columnas ya estan vacias y con el dtype correcto (nada que escalar).
            if len(X) > 0:
                X[self.COLUMNAS_FEATURES] = self.scaler_.transform(X[self.COLUMNAS_FEATURES])
        elif en_fit:
            self.scaler_ = None
        return X

    # ------------------------------------------------------------------ #
    # P11 _ensamblar_salida                                              #
    # ------------------------------------------------------------------ #
    def _ensamblar_salida(self, X: pd.DataFrame, en_fit: bool) -> pd.DataFrame:
        """Ordena a las 12 columnas del contrato: 10 features float64 + 2 flags int8."""
        orden = list(self.COLUMNAS_FEATURES) + list(self.FLAGS)
        X = X.loc[:, orden].copy()
        X[self.COLUMNAS_FEATURES] = X[self.COLUMNAS_FEATURES].astype("float64")
        X[list(self.FLAGS)] = X[list(self.FLAGS)].astype("int8")
        return X


# ---------------------------------------------------------------------- #
# A.6 Factory y split (puntos de entrada del modulo)                     #
# ---------------------------------------------------------------------- #
def construir_preprocesador(escalar: bool = True,
                            recalcular_debtratio: bool = True,
                            winsorizar: bool = True) -> PreprocesadorCredito:
    """Construye el preprocesador con los defaults decididos en la spec."""
    return PreprocesadorCredito(escalar=escalar,
                                recalcular_debtratio=recalcular_debtratio,
                                winsorizar=winsorizar)


def dividir_train_test(df: pd.DataFrame, test_size: float = 0.2,
                       random_state: int = 42):
    """Split estratificado por TARGET. Devuelve (df_train, df_test)."""
    return train_test_split(
        df, test_size=test_size,
        stratify=df[PreprocesadorCredito.TARGET],
        random_state=random_state,
    )
