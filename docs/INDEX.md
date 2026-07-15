# Índice de documentación

Documentación de soporte del Taller B4-T2 (XAI aplicado a concesión de crédito).

## Decisiones de proyecto

- [DECISIONES.md](DECISIONES.md) — registro de decisiones tomadas durante el desarrollo del taller.

## Fuentes

- `_fuentes/` — material de clase, enunciado y notebook de partida del profesor (`taller_XAI.pdf`,
  `XAI.pdf`, `AR_Multiarmed_Bandits.pdf`, `cs_construccion.csv`, `cs_produccion.csv`,
  `DataDictionary.csv`, notebooks de ejercicio del profesor incluido `ejercicio3_clustering.ipynb`
  (XAI en clustering, añadido después del inventario inicial, pendiente de revisión profunda),
  etc. — ver `_fuentes/INVENTARIO.md` para el detalle completo). **Esta carpeta NO se versiona**
  (material bruto privado del profesor); no hace falta para ejecutar los notebooks.

## Repositorio plug-and-play

El repositorio versiona los datos de trabajo (`data/*.csv`), los intermedios del preprocesado
(`data/processed/*.parquet`), los modelos (`results/models/*.joblib`) y las predicciones, tablas y
figuras de `results/`. Así, tras `pip install -r requirements.txt`, cualquiera puede clonar y
ejecutar cualquier notebook sin regenerar la cadena anterior. Lo único fuera del control de
versiones es `docs/_fuentes/` (privado). Ver la sección "Repositorio plug-and-play" del `README.md`.

## Teoría

Notas de teoría de soporte para la auditoría del modelo. Ya redactadas:

- [teoria/shap.md](teoria/shap.md) — SHAP (Shapley Additive exPlanations), valores globales y locales.
- [teoria/contrafactuals.md](teoria/contrafactuals.md) — análisis de contrafactuales para explicaciones de decisiones individuales.
- [teoria/subrogados.md](teoria/subrogados.md) — modelos subrogados interpretables (reglas) para aproximar el modelo principal.
- [teoria/cost_sensitive.md](teoria/cost_sensitive.md) — aprendizaje sensible al coste, matrices de coste y optimización de umbral.
- [teoria/bandits.md](teoria/bandits.md) — Multiarmed Bandits como alternativa al modelo supervisado.
- [teoria/notebook_partida.md](teoria/notebook_partida.md) — estructura de los datos y patrón de trabajo heredado del notebook/ejercicios de partida del profesor.

## Notebooks

Pipeline de trabajo en `notebooks/`, ejecutado de forma incremental (01 → 07) y consolidado en un
único notebook de entrega (`99_ENTREGA.ipynb`). Cada notebook documenta en su propia celda
"Conectores" qué recibe del notebook anterior y qué entrega al siguiente.

- `notebooks/01_EDA.ipynb` — análisis exploratorio de `cs_construccion.csv`/`cs_produccion.csv`
  (target, nulos, outliers, distribuciones) para informar el preprocesado.
- `notebooks/02_preprocesado.ipynb` — pipeline de preprocesado (split estratificado, imputación,
  recálculo de DebtRatio, winsorización y escalado) ajustado solo sobre train y aplicado a
  train/test/producción; serializa el pipeline y deja los `.parquet` de salida.
- `notebooks/03_modelo_coste1.ipynb` — compara familias de modelos (lineal/logística,
  boosted/XGBoost, neuronal/MLP Keras y bandit/LinUCB) por coste esperado bajo el escenario
  simétrico (C_FP = C_FN = 1), eligiendo familia y umbral por validación cruzada anti-fuga; genera
  el entregable `cs_produccion1.csv`.
- `notebooks/04_modelo_coste10.ipynb` — mismo esquema de comparación por coste y umbral por CV
  anti-fuga bajo el escenario asimétrico (C_FN = 10·C_FP, confirmado por el profesor como la lectura
  correcta del escenario 2; ver D-3.2); genera el entregable `cs_produccion2.csv`.
- `notebooks/05_auditoria_subrogado.ipynb` — auditoría con modelo subrogado (árbol de decisión) que
  aproxima cada modelo de caja negra y extrae reglas legibles.
- `notebooks/06_contrafactuals.ipynb` — auditoría mediante análisis de contrafactuales sobre
  ejemplos de clase real 0 y 1, en ambos escenarios de coste.
- `notebooks/07_shap.ipynb` — auditoría mediante análisis SHAP global y local para ambos modelos de
  coste.
- `notebooks/08_otras_tecnicas.ipynb` — notebook adicional, no exigido por el enunciado (que invita
  explícitamente a "otras técnicas que los estudiantes consideren oportunas"): importancia por
  permutación (contrastada a 3 bandas contra XGBoost nativo y el subrogado de `05`), PDP/ICE + PDP
  2D de interacción, LIME local con medición de inestabilidad (contraste crítico frente a SHAP), y
  subgrupos por tramo de edad (mini-auditoría de sesgo). Autocontenido e independiente de `06`/`07`;
  ver `src/xai_utils.py` para las funciones reutilizables y `docs/DECISIONES.md` D-8.1/D-8.2.
- `notebooks/99_ENTREGA.ipynb` — notebook único de entrega: consolida el trabajo de los notebooks
  anteriores (portada, objetivo, datos/EDA, preprocesado, ambos modelos, tabla comparativa de
  resultados, las tres auditorías obligatorias, otras técnicas del `08`, decisiones de diseño y
  reflexión final).

**Estado actual de los notebooks**: `notebooks/01_EDA.ipynb`, `notebooks/02_preprocesado.ipynb`,
`notebooks/03_modelo_coste1.ipynb`, `notebooks/04_modelo_coste10.ipynb`,
`notebooks/05_auditoria_subrogado.ipynb` y `notebooks/08_otras_tecnicas.ipynb` ya NO son esqueletos:
todos están implementados y ejecutados de principio a fin sin errores.

- `01_EDA.ipynb` tiene el EDA completo (carga y validación de datos, EDA básico,
  centinelas/outliers de codificación, EDA avanzado con comparación train vs producción,
  correlaciones, PCA lineal, reducción de dimensionalidad no lineal, recomendación de tratamiento
  de nulos, recomendación de normalización/escalado y conclusiones con la tabla resumen
  `eda_resumen` guardada), con resultados reales en `results/figures/` y `results/tables/`.
- `02_preprocesado.ipynb` construye el pipeline de preprocesado: split estratificado train/test,
  imputación por tramo de edad más flags de missingness, recálculo de DebtRatio, winsorización,
  `log1p` + `StandardScaler` parametrizable por flag, y verificación anti-fuga mediante `assert`
  (el pipeline se ajusta solo sobre train). Serializa `results/models/preprocessing_pipeline.joblib`
  y deja `data/processed/train.parquet`, `test.parquet` y `produccion.parquet` (esta última con
  `id_fila_original` y el target vacío), más 7 tablas `prep_*.csv` y 4 figuras `prep_*.png`.
- `03_modelo_coste1.ipynb` y `04_modelo_coste10.ipynb` comparan por validación cruzada anti-fuga
  varias familias de modelos (logística, XGBoost, dos MLP Keras y un bandit LinUCB propio) y eligen
  el umbral por coste esperado sobre las probabilidades OOF: XGBoost gana en ambos escenarios. El
  `03` (simétrico, C_FP = C_FN = 1) deja un coste medio en test de 0.063 denegando el 2.4% (mejora
  ~5% frente a conceder a todos: escenario casi trivial); el `04` (asimétrico, C_FN = 10·C_FP) deja
  coste 0.328 denegando el 18.7% (mejora ~51%: aquí está el valor). Generan los dos entregables
  `cs_produccion1.csv` / `cs_produccion2.csv` (más `cs_produccion2_literal.csv`), los modelos
  `modelo_coste1.joblib` / `modelo_coste10.joblib`, la tabla `umbrales_coste.csv` (filas `coste1` y
  `coste10`) y las tablas/figuras `mod_03_*` / `mod_04_*`.
- `05_auditoria_subrogado.ipynb` audita ambos modelos con un árbol de decisión subrogado: barrido de
  profundidad y peso de clase (con el hallazgo de que el subrogado degenerado "conceder siempre"
  bate en fidelidad global bruta a cualquier árbol real, por lo que la fidelidad se mide también
  por clase), reglas de denegación traducidas a escala original (euros/años/ratios) en
  `reglas_subrogado_coste1.md` / `_coste10.md`, y una tabla de fidelidad consolidada. Hallazgo
  central: el modelo tiene "dos personalidades de riesgo" según qué umbral (escenario de coste) se
  aplique.
- `08_otras_tecnicas.ipynb` (no exigido): (1) importancia por permutación, con divergencias reales
  entre permutación/XGBoost nativo/subrogado que se documentan como hallazgo, no se ocultan; (2) PDP
  2D que confirma cuantitativamente (ratio ~2x) la interacción Edad × historial de impago de `05`,
  matizando que la superficie de probabilidad es idéntica en ambos escenarios (D-0.2) y lo que
  cambia es solo el umbral; (3) LIME reejecutado 6 veces por caso, con la variable
  `es_centinela_pastdue` como la más inestable y 9 de 72 combinaciones cambiando de signo entre
  ejecuciones; (4) subgrupos por edad: la denegación sigue la dirección del riesgo real pero no su
  magnitud (ratio hasta ~3.5x en los más jóvenes bajo coste asimétrico). Ver D-8.1/D-8.2 en
  `docs/DECISIONES.md`.

Los notebooks `06`, `07` y `99_ENTREGA.ipynb` siguen siendo esqueletos con celdas de código
marcadas `# TODO` (estructura, conectores y objetivos ya definidos, pero sin código ejecutado ni
resultados reales todavía) — salvo la sección 11 de `99_ENTREGA.ipynb`, que ya consolida los
resultados reales del `08`; `99_ENTREGA.ipynb` contiene además placeholders pendientes de
sustituir (nombres/emails reales del grupo, tablas de resultados, reflexión final) antes de la
entrega del 20 de julio de 2026.
