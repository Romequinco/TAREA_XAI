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

**Evidencia encontrada (2026-07-03):** `docs/teoria/bandits.md` (análisis de
`AR_Multiarmed_Bandits.pdf` y de los dos notebooks de ejercicio del profesor) recoge argumentos
concretos, con cita de fuente, en ambos sentidos, pero concluye explícitamente que el material
teórico **no zanja la decisión** (no hay comparación empírica de coste medio en producción entre
ambos enfoques). En contra del bandit: el dataset del taller (`cs_construccion.csv`) es histórico
y ya está completamente etiquetado, por lo que la premisa central del bandit ("la recompensa se
desconoce y se aprende por interacción") aporta menos valor aquí; cualquier algoritmo de bandit
paga un coste de exploración/regret no nulo (teorema de Lai-Robbins, demostrado en el propio
material) que un supervisado entrenado una vez no paga; el problema de feedback contrafactual
propio de la concesión de crédito real (no se observa el resultado de un crédito denegado) no
está resuelto ni siquiera simulado de forma realista en el patrón de los notebooks de ejercicio
(el "entorno" simulado conoce siempre la etiqueta real de ambas acciones); y, en la práctica del
propio notebook del profesor, el bandit acaba auditándose con el mismo paso de modelo subrogado
que un supervisado, sin ventaja de explicabilidad evidente. A favor del bandit: encaje conceptual
limpio contexto→acción→recompensa con las variables del solicitante, e incorporación directa del
coste asimétrico como recompensa en vez de como umbral post-hoc. Esta evidencia no cierra D-0.1
por sí sola (sigue exigiendo la comparación empírica de coste en producción entre ambos enfoques),
pero aporta argumentos de peso, ya documentados, a favor del modelo supervisado.

### D-0.2 — ¿Un solo modelo con dos umbrales/políticas de decisión, o dos modelos distintos para coste 1 y coste 10?

- **Fecha:** 2026-07-03
- **Alternativas consideradas:**
  - Un único modelo de scoring (probabilidades) con dos umbrales de decisión distintos, uno
    optimizado para cada matriz de coste (FP=FN=1 y FP=FN=10).
  - Dos modelos entrenados de forma independiente, uno por escenario de coste.
- **Justificación:** pendiente de evidencia empírica sobre si el óptimo de umbral por escenario es
  suficiente o si conviene reentrenar con sensibilidad al coste en cada caso.
- **Estado:** ABIERTA

**Evidencia encontrada (2026-07-03):** `docs/teoria/cost_sensitive.md` aporta dos elementos de
evidencia real hacia la opción "un único modelo con dos umbrales". (1) El patrón de código de los
dos notebooks de partida del profesor (`ejercicio1_AR.ipynb`, `ejercicio2_proyecto_XAI.ipynb`)
entrena **un único** clasificador sin `class_weight`/`sample_weight` (comprobado por búsqueda
textual: no aparecen ni una vez en ninguno de los dos ficheros) y resuelve la sensibilidad al
coste enteramente ajustando el umbral de decisión después de entrenar, mediante barrido
exhaustivo sobre los valores únicos de `predict_proba` (sección 2.4). (2) Dado que los dos
escenarios oficiales del taller son ambos simétricos dentro de sí mismos (C_FP=C_FN=1 y,
separadamente, C_FP=C_FN=10), la teoría de decisión indica que el umbral óptimo no debería
desplazarse entre escenarios, solo reescalarse el coste esperado; `docs/teoria/cost_sensitive.md`
(sección 3.4) documenta además una comprobación numérica propia sobre el dataset real de este
taller (`cs_construccion.csv`) en la que, entrenando un clasificador y barriendo el umbral por
separado con C_FP=C_FN=1 y C_FP=C_FN=10, el umbral óptimo resultó ser exactamente el mismo en
ambos casos y el coste medio del segundo escenario fue, con precisión numérica, 10.00 veces el
del primero. Esto es evidencia concreta (no solo patrón observado, sino verificación numérica
sobre los datos reales) de que un único modelo con dos umbrales es suficiente para cubrir ambos
escenarios de coste del taller, sin necesidad de reentrenar un modelo distinto por escenario.

### D-0.3 — Elección de familia de modelo (árboles boosted vs red neuronal vs lineal)

- **Fecha:** 2026-07-03
- **Alternativas consideradas:**
  - Árboles boosted (XGBoost / LightGBM): buen rendimiento tabular, compatibles con SHAP TreeExplainer.
  - Red neuronal: mayor flexibilidad pero menor explicabilidad nativa.
  - Modelo lineal (regresión logística): máxima explicabilidad, posible menor rendimiento.
- **Justificación:** pendiente de evidencia empírica sobre el trade-off rendimiento/explicabilidad
  en este dataset concreto.
- **Estado:** ABIERTA
