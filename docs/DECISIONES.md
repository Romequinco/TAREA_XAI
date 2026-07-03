# Registro de decisiones del proyecto

Registro de las decisiones técnicas y de diseño tomadas a lo largo del taller, siguiendo un
formato ligero tipo ADR (Architecture Decision Record).

## Formato

Cada decisión se identifica con un ID `D-X.n`, donde `X` es el bloque temático (0 = decisiones
generales/transversales, 1 = preprocesado, 2 = modelado, 3 = coste, 4 = auditoría/XAI, ...) y `n`
es un correlativo dentro de ese bloque.

| Campo | Descripción |
|---|---|
| **ID** | Identificador único `D-X.n` |
| **Fecha** | Fecha en la que se registra o se cierra la decisión |
| **Decisión** | Pregunta o disyuntiva a resolver |
| **Alternativas consideradas** | Opciones evaluadas |
| **Justificación** | Motivo de la elección, con evidencia empírica cuando la haya |
| **Estado** | `ABIERTA` (aún por decidir) o `CERRADA` (decidida, con justificación final) |

---

## Decisiones

### D-0.1 — ¿Modelo supervisado o Multiarmed Bandit?

- **Fecha:** 2026-07-03
- **Alternativas consideradas:**
  - Modelo supervisado (clasificación binaria clásica sobre datos históricos etiquetados).
  - Multiarmed Bandit (aprendizaje online con exploración/explotación sobre las decisiones de concesión).
- **Justificación:** pendiente de evidencia empírica; el enunciado admite ambas opciones (a) modelo
  supervisado, b) Multiarmed Bandit).
- **Estado:** ABIERTA

### D-0.2 — ¿Un solo modelo con dos umbrales/políticas de decisión, o dos modelos distintos para coste 1 y coste 10?

- **Fecha:** 2026-07-03
- **Alternativas consideradas:**
  - Un único modelo de scoring (probabilidades) con dos umbrales de decisión distintos, uno
    optimizado para cada matriz de coste (FP=FN=1 y FP=FN=10).
  - Dos modelos entrenados de forma independiente, uno por escenario de coste.
- **Justificación:** pendiente de evidencia empírica sobre si el óptimo de umbral por escenario es
  suficiente o si conviene reentrenar con sensibilidad al coste en cada caso.
- **Estado:** ABIERTA

### D-0.3 — Elección de familia de modelo (árboles boosted vs red neuronal vs lineal)

- **Fecha:** 2026-07-03
- **Alternativas consideradas:**
  - Árboles boosted (XGBoost / LightGBM): buen rendimiento tabular, compatibles con SHAP TreeExplainer.
  - Red neuronal: mayor flexibilidad pero menor explicabilidad nativa.
  - Modelo lineal (regresión logística): máxima explicabilidad, posible menor rendimiento.
- **Justificación:** pendiente de evidencia empírica sobre el trade-off rendimiento/explicabilidad
  en este dataset concreto.
- **Estado:** ABIERTA
