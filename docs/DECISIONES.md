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

**Evidencia encontrada (2026-07-04):** `notebooks/01_EDA.ipynb` (secciones 5-7) aporta varios
indicios, ninguno concluyente por sí solo, de que las relaciones en este dataset no son puramente
lineales. (1) Los tres pares de columnas `NumberOfTime*PastDue*` tienen correlación de Pearson muy
alta (0.98-0.99) pero de Spearman mucho menor (0.25-0.32); el propio notebook razona que esa
divergencia está probablemente inflada por ~188 filas con valores centinela compartidos (96/98),
no por una relación monótona limpia en todo el rango — verificado repitiendo PCA/t-SNE/KernelPCA
excluyendo esas filas (sección 7.4): la componente principal PC1 baja de ~30% a ~19% de varianza
explicada y cambia de composición al excluirlas. (2) La sección 5.6 (información mutua) y la
comparación PCA (lineal) vs t-SNE/KernelPCA (no lineal) de la sección 7 no muestran una separación
de clases sustancialmente mejor bajo proyecciones no lineales que bajo PCA lineal — ambas muestran
solapamiento considerable entre clases. Lectura tentativa: no hay evidencia fuerte de que una
familia no lineal (árboles/red neuronal) capture estructura que un modelo lineal no pueda, pero
tampoco hay evidencia de que el problema sea "limpiamente lineal" dado el efecto de los centinelas
sin tratar y la multicolinealidad detectada por VIF (sección 5.5) entre las columnas `PastDue`.
Esta evidencia es indirecta (viene de EDA, no de comparar modelos entrenados) y no cierra D-0.3;
la decisión final debería apoyarse en el rendimiento real de cada familia una vez implementado
`03_modelo_coste1.ipynb`/`04_modelo_coste10.ipynb`.

**Evidencia añadida (2026-07-10):** el preprocesado (`02_preprocesado.ipynb`) se diseñó
**parametrizable** (flag `escalar`, ver D-1.5) precisamente para servir a cualquier familia de
modelo sin prejuzgarla: con `escalar=True` produce features log1p + estandarizadas para los
modelos lineales/bandit y con `escalar=False` las deja crudas para los árboles. Además, el
tratamiento de los centinelas 96/98 (ver D-1.2) resolvió la multicolinealidad severa que el EDA
había señalado sin cerrar: el VIF de las tres `PastDue` cae de 40.9/71.7/89.6 a ~1.18/1.15/1.17 y
el VIF máximo global baja a 1.91. Esto elimina uno de los argumentos que había a favor de modelos
robustos a la colinealidad (la colinealidad era un artefacto de los centinelas, no estructural).
La decisión **sigue ABIERTA**: la elección final debe apoyarse en el rendimiento real de cada
familia sobre los datos ya preprocesados.

### D-1.1 — Estrategia de imputación de nulos (`MonthlyIncome`, `NumberOfDependents`)

- **Fecha:** 2026-07-04
- **Alternativas consideradas:**
  - Imputación simple por mediana global (una única mediana para toda la columna).
  - Imputación por mediana agrupada (por ejemplo, por tramo de edad para `MonthlyIncome`).
  - Añadir una variable indicadora binaria de "era nulo" además de imputar.
  - Eliminar las filas con nulos (descartado de entrada: afectaría al 19.8% del dataset por
    `MonthlyIncome`).
- **Justificación:** `notebooks/01_EDA.ipynb` (sección 8) aporta evidencia empírica real, no solo
  supuestos: (1) la tasa de positivos del target difiere entre clientes con y sin `MonthlyIncome`
  nulo, y un test estadístico formal (sección 8.6, regresión logística controlando por edad)
  confirma que el patrón de nulos no es puramente aleatorio (MCAR) respecto al target una vez
  controlado por edad — es decir, la ausencia de dato es en sí misma parcialmente informativa.
  (2) La mediana de `MonthlyIncome` varía con la edad (sección 8.4), lo que apoya imputar por
  grupo de edad en vez de con una única mediana global. (3) `DebtRatio` muestra valores atípicos
  cuando `MonthlyIncome` es nulo (sección 8.2), coherente con la hipótesis de que en esos casos
  `DebtRatio` pueda estar mal definido o capturar otra magnitud. (4) El patrón de nulos de
  `NumberOfDependents` (2.6%) se comportó de forma más cercana a aleatoria en el cruce con edad y
  target (sección 8.3), por lo que una imputación simple (mediana o 0) parece suficiente ahí.
  La tabla `results/tables/08_recomendacion_imputacion_nulos.csv` (y la fila correspondiente de
  `results/tables/eda_resumen.csv`) recoge la recomendación consolidada por columna.
- **Estado:** CERRADA (2026-07-10). La evidencia del EDA quedó confirmada y afinada al implementar
  `02_preprocesado.ipynb`; los parámetros de imputación se aprenden solo sobre el train-split.

**Decisión final (2026-07-10, cerrada en `02_preprocesado.ipynb`):**

- `MonthlyIncome`: se imputa por la **mediana del tramo de edad** (6 tramos decadales: ≤30, 31-40,
  41-50, 51-60, 61-70, >70), con las medianas aprendidas únicamente sobre el income usable del
  train-split, en lugar de una mediana global. Se conserva un flag `is_missing_monthlyincome`.
- Definición del flag: marca `income nulo O income ≤ 1` (no solo nulo). Justificación: los valores
  income∈{0,1} comparten el MECANISMO de los nulos (son el numerador corrupto de `DebtRatio`,
  comparten la misma máscara de imputación y el mismo recálculo); estadísticamente income≤1 no es
  idéntico a nulo (chi² sobre el target, p=0.008758), pero comparte dirección y mecanismo. El flag
  se conserva porque la señal de ausencia es MNAR (EDA: logit controlando por edad, p=0.012248,
  OR 0.9150). El nombre `is_missing_monthlyincome` es una etiqueta de conveniencia; su definición
  exacta queda documentada aquí.
- `NumberOfDependents`: imputación simple a 0, SIN flag. Al controlar por edad su missingness deja
  de ser significativo (p=0.114677) y, además, sus nulos son un subconjunto exacto de los de
  `MonthlyIncome`.
- Centinelas 96/98 de las tres columnas `PastDue`: se tratan como missing y se imputan con la
  mediana de train (= 0), conservando un flag `es_centinela_pastdue` (tasa de target 53.7% frente
  al 6.6% general → señal fuerte que se preserva).

### D-1.2 — Tratamiento de outliers y de la cola de `DebtRatio`

- **Fecha:** 2026-07-10
- **Alternativas consideradas:**
  - Winsorizar a ciegas la cola de `DebtRatio` (recorte por percentil sin diagnosticar su origen).
  - Recalcular `DebtRatio` en las filas con ingreso corrupto, a partir del numerador crudo y del
    income imputado.
  - No tratar los valores extremos.
- **Justificación:** el hallazgo analítico principal es que `DebtRatio` guarda el **numerador
  crudo** (los pagos), no el ratio, cuando el ingreso falta o es ≤1. Evidencia: el 100% de la cola
  `DebtRatio > p97.5` corresponde a income no usable; la mediana de `DebtRatio` en esas filas es
  1152 frente a 0.29 en las filas sanas. Por eso se decide **RECALCULAR**
  `DebtRatio = DebtRatio_crudo / MonthlyIncome_imputado` en esas filas (17900 en el train-split), en
  vez de winsorizar a ciegas. Validación falsable: el estadístico KS (cola contaminada vs sana)
  baja de 0.9284 a 0.1671; la mediana recalculada (0.205) se solapa con la de las filas sanas
  (0.290); y la señal predictiva MEJORA (AUC univariante `DebtRatio → target` de 0.5212 a 0.5531).
  Efecto colateral declarado con honestidad: dividir por la mediana de income del tramo (que es
  función de la edad) inyecta un acoplamiento débil `DebtRatio ↔ age` (Spearman de +0.03 a −0.10;
  Pearson −0.006, despreciable); donde se divide por el income real observado no se crea correlación
  artificial. Winsorización por columna del residuo, aprendida en `fit`: `RevolvingUtilization`
  cap a p99 de train (≈1.10; a p99.9 el skew tras log1p seguiría en ~10) y `DebtRatio` cap a p99.9
  de train.
- **Estado:** CERRADA (2026-07-10, implementada en `02_preprocesado.ipynb`).

**BONUS (2026-07-10):** tratar los centinelas 96/98 (ver D-1.1) COLAPSA la multicolinealidad severa
que el EDA (`01_EDA.ipynb`) dejó señalada sin resolver: el VIF de las tres `PastDue` cae de
40.9/71.7/89.6 a ~1.18/1.15/1.17 y el VIF máximo de todas las features baja a 1.91. Es decir, la
multicolinealidad era un artefacto de los centinelas, no estructural → los modelos lineales/bandit
(D-0.3) no necesitan consolidar las tres `PastDue` en una sola variable.

### D-1.3 — Tratamiento de `age == 0`

- **Fecha:** 2026-07-10
- **Alternativas consideradas:**
  - Filtrar (eliminar) la fila con `age == 0`.
  - Tratar `age == 0` como valor inválido/missing e imputar con la mediana de edad de train.
- **Justificación:** solo hay 1 fila afectada en train (idx 89320) y 0 en producción. Se trata como
  valor inválido/missing y se imputa con la mediana de edad de train. No se filtra ninguna fila
  (producción conserva sus 45000 filas íntegras: no se puede eliminar ninguna solicitud a predecir).
- **Estado:** CERRADA (2026-07-10, implementada en `02_preprocesado.ipynb`).

### D-1.4 — División train/test

- **Fecha:** 2026-07-10
- **Alternativas consideradas:**
  - Aprender los parámetros de preprocesado sobre todo el dataset y luego dividir.
  - Dividir primero (hold-out estratificado) y aprender los parámetros solo sobre train.
- **Justificación:** `train_test_split(test_size=0.2, stratify=SeriousDlqin2yrs, random_state=42)`
  → 84000 train / 21000 test; las tasas de clase resultan 6.6833% / 6.6857% (diferencia < 0.01 pp),
  lo que confirma la estratificación. El split va **PRIMERO**, antes de aprender cualquier parámetro
  (medianas, percentiles de winsorización, escalador), para evitar fuga de información (data
  leakage). Nota informativa: 107 filas de test (0.51%) tienen una fila gemela exacta en train (como
  consecuencia de la decisión de conservar duplicados, ver D-1.6); su impacto en la métrica de test
  es ~0 y se declara aquí para tenerlo presente en `03`.
- **Estado:** CERRADA (2026-07-10, implementada en `02_preprocesado.ipynb`).

### D-1.5 — Escalado y transformaciones

- **Fecha:** 2026-07-10
- **Alternativas consideradas:**
  - Escalar/transformar de forma fija (una única salida) para todos los modelos.
  - Pipeline parametrizable que produzca la representación adecuada según la familia de modelo.
- **Justificación:** se aplica `log1p` a {`DebtRatio`, `MonthlyIncome`, `RevolvingUtilization`}
  (NO a `age`: su skew es 0.19 y `log1p` lo empeoraría a −0.46) seguido de `StandardScaler`. El
  pipeline es **PARAMETRIZABLE** con un flag `escalar`: `True` (log1p + StandardScaler, para modelos
  lineales/bandit) o `False` (sin escalar, para árboles). El artefacto serializado se ajusta con
  `escalar=True`. Esta parametrización se adopta expresamente para NO bloquear D-0.3 (familia de
  modelo), que sigue abierta. Dato: el `log1p` del income imputado deja un skew de −0.62 (frente a
  −4.37 si se aplicara sobre el income crudo con sus ceros/unos), lo que confirma que tratar
  income≤1 como no usable (D-1.1) también sanea la transformación del income.
- **Estado:** CERRADA (2026-07-10, implementada en `02_preprocesado.ipynb`).

### D-1.6 — Filas duplicadas

- **Fecha:** 2026-07-10
- **Alternativas consideradas:**
  - Eliminar los duplicados exactos.
  - Conservarlos.
- **Justificación:** se **CONSERVAN** los duplicados exactos (331 en train, 101 en producción). Sin
  una columna identificadora no se puede demostrar que sean errores de captura frente a colisiones
  legítimas: con 11 columnas y 105000 filas, dos clientes distintos con el mismo perfil son
  estadísticamente esperables. Además, en producción deben conservarse siempre (hay que devolver
  45000 predicciones, una por fila). Decisión tomada por el usuario.
- **Estado:** CERRADA (2026-07-10, decisión del usuario, aplicada en `02_preprocesado.ipynb`).
