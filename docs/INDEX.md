# Índice de documentación

Documentación de soporte del Taller B4-T2 (XAI aplicado a concesión de crédito).

## Decisiones de proyecto

- [DECISIONES.md](DECISIONES.md) — registro de decisiones tomadas durante el desarrollo del taller.

## Fuentes

- `_fuentes/` — material de clase, enunciado y notebook de partida del profesor (`taller_XAI.pdf`,
  `XAI.pdf`, `AR_Multiarmed_Bandits.pdf`, `cs_construccion.csv`, `cs_produccion.csv`,
  `DataDictionary.csv`, notebooks de ejercicio del profesor incluido `ejercicio3_clustering.ipynb`
  (XAI en clustering, añadido después del inventario inicial, pendiente de revisión profunda),
  etc. — ver `_fuentes/INVENTARIO.md` para el detalle completo).

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
- `notebooks/03_modelo_coste1.ipynb` — entrenamiento/optimización del modelo bajo el escenario de
  coste FP = FN = 1; genera `cs_produccion1.csv`.
- `notebooks/04_modelo_coste10.ipynb` — entrenamiento/optimización del modelo bajo el escenario de
  coste FP = FN = 10; genera `cs_produccion2.csv`.
- `notebooks/05_auditoria_subrogado.ipynb` — auditoría con modelo subrogado (árbol de decisión) que
  aproxima cada modelo de caja negra y extrae reglas legibles.
- `notebooks/06_contrafactuals.ipynb` — auditoría mediante análisis de contrafactuales sobre
  ejemplos de clase real 0 y 1, en ambos escenarios de coste.
- `notebooks/07_shap.ipynb` — auditoría mediante análisis SHAP global y local para ambos modelos de
  coste.
- `notebooks/99_ENTREGA.ipynb` — notebook único de entrega: consolida el trabajo de los 7 notebooks
  anteriores (portada, objetivo, datos/EDA, preprocesado, ambos modelos, tabla comparativa de
  resultados, las tres auditorías, decisiones de diseño y reflexión final).

**Estado actual de los notebooks**: `notebooks/01_EDA.ipynb` y `notebooks/02_preprocesado.ipynb`
ya NO son esqueletos: ambos están implementados y ejecutados de principio a fin sin errores.

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

Los notebooks `03`–`07` y `99_ENTREGA.ipynb` siguen siendo esqueletos con celdas de código
marcadas `# TODO` (estructura, conectores y objetivos ya definidos, pero sin código ejecutado ni
resultados reales todavía); `99_ENTREGA.ipynb` contiene además placeholders pendientes de
sustituir (nombres/emails reales del grupo, tablas de resultados, reflexión final) antes de la
entrega del 20 de julio de 2026.
