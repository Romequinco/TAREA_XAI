# Notebook de partida: estructura de los datos y patrón de trabajo heredado

> Documento de teoría/inventario técnico del Taller B4-T2 (XAI aplicado a
> concesión de crédito, dataset "Give Me Some Credit"). Es **solo
> documentación**: describe qué contienen los datos y qué hacen ya los
> notebooks de ejercicio del profesor, pero no implementa ni decide nada — las
> decisiones de diseño se registran en `docs/DECISIONES.md` y la
> implementación corresponde a `src/` y al notebook final del grupo.
>
> El enunciado oficial (`docs/_fuentes/taller_XAI.pdf`) remite el "dataset y
> notebook de partida" a una URL externa de Google Drive
> (`https://drive.google.com/drive/folders/1G39FiV3R7v8c5sDtB5NRbwz8d40BuTu7`)
> que no se ha descargado a este repositorio. De todo el material disponible
> localmente en `docs/_fuentes/`, `ejercicio2_proyecto_XAI.ipynb` es el único
> notebook que efectivamente carga `cs_construccion.csv` y `DataDictionary.csv`
> (el dataset real del taller), por lo que se toma aquí como **candidato
> principal a notebook de partida**. Su notebook hermano, `ejercicio1_AR.ipynb`,
> resuelve el mismo patrón metodológico sobre un dataset de calentamiento
> (`breast_cancer` de scikit-learn) y, como se detalla en la sección 2, resulta
> que es literalmente el contenido que `ejercicio2_proyecto_XAI.ipynb` reutiliza
> sin adaptar a partir de su celda 15.

## 1. Estructura de los datos

### 1.1 Diccionario de columnas

Fuente: `docs/_fuentes/DataDictionary.csv` (11 filas, columnas `Variable Name`,
`Description`, `Type`, separadas por `;`). Se reproduce tal cual, incluyendo
erratas del original (p. ej. "monthy" en la fila de `DebtRatio`):

| # | Variable | Descripción | Tipo |
|---|----------|--------------|------|
| 0 | `SeriousDlqin2yrs` | Person experienced 90 days past due delinquency or worse | Y/N — **variable objetivo** |
| 1 | `RevolvingUtilizationOfUnsecuredLines` | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits | percentage |
| 2 | `age` | Age of borrower in years | integer |
| 3 | `NumberOfTime30-59DaysPastDueNotWorse` | Number of times borrower has been 30-59 days past due but no worse in the last 2 years. | integer |
| 4 | `DebtRatio` | Monthly debt payments, alimony,living costs divided by monthy gross income | percentage |
| 5 | `MonthlyIncome` | Monthly income | real |
| 6 | `NumberOfOpenCreditLinesAndLoans` | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards) | integer |
| 7 | `NumberOfTimes90DaysLate` | Number of times borrower has been 90 days or more past due. | integer |
| 8 | `NumberRealEstateLoansOrLines` | Number of mortgage and real estate loans including home equity lines of credit | integer |
| 9 | `NumberOfTime60-89DaysPastDueNotWorse` | Number of times borrower has been 60-89 days past due but no worse in the last 2 years. | integer |
| 10 | `NumberOfDependents` | Number of dependents in family excluding themselves (spouse, children etc.) | integer |

### 1.2 Variable objetivo

`SeriousDlqin2yrs` (fuente: `DataDictionary.csv`, fila 0, tipo declarado
`Y/N`). En `cs_construccion.csv` está codificada como entero `0`/`1` (así se
ve en la salida de `df` en la celda `cell-2` de
`ejercicio2_proyecto_XAI.ipynb`, p. ej. filas con valor `0`). Es la única
columna que **no** aparece en la lista de nulos de la celda `cell-7` de ese
mismo notebook, es decir, en `cs_construccion.csv` el target no tiene huecos.

En `cs_produccion.csv` la columna `SeriousDlqin2yrs` está vacía en todas las
filas inspeccionadas (comprobado leyendo directamente las primeras ~40 filas
del fichero: el primer campo, antes de la primera coma, queda en blanco en
todas ellas) — es la variable a predecir para la entrega, no un dato de
entrenamiento.

### 1.3 Tamaño de construcción y producción

- `cs_construccion.csv`: **105 000 filas × 11 columnas** (cabecera +
  105 000 líneas de datos, verificado con conteo de líneas del fichero;
  coincide con el `df` de 105000 filas mostrado en la celda `cell-2` de
  `ejercicio2_proyecto_XAI.ipynb` y con `count=105000` en varias columnas de
  `df.describe()` en la celda `cell-5`). Incluye el target y sirve para
  entrenar/validar.
- `cs_produccion.csv`: **45 000 filas × 11 columnas** (cabecera + 45 000
  líneas de datos, verificado con conteo de líneas del fichero). Mismas 11
  columnas que `cs_construccion.csv`, pero con `SeriousDlqin2yrs` vacío (ver
  1.2) — es el fichero sobre el que hay que generar las predicciones finales
  que pide el enunciado (`docs/_fuentes/taller_XAI.pdf`, apartado "1. Objetivo
  de la práctica": los entregables son `cs_produccion1.csv`, con coste
  FP=FN=1, y `cs_produccion2.csv`, con coste FP=FN=10 — **no** son ficheros ya
  presentes en `docs/_fuentes/`, hay que generarlos).

### 1.4 Nulos conocidos

Calculados explícitamente en `ejercicio2_proyecto_XAI.ipynb`, celda `cell-7`
(`df.isna().sum()` sobre `cs_construccion.csv`, expresado en % del total de
filas):

- `MonthlyIncome`: **19.80 %** de nulos (salida exacta: `19.804762`).
- `NumberOfDependents`: **2.61 %** de nulos (salida exacta: `2.607619`).
- El resto de columnas (incluida `SeriousDlqin2yrs`) no aparecen en esa salida,
  es decir, no tienen nulos en `cs_construccion.csv`.

Esto es consistente con `df.describe()` en la celda `cell-5`: `count` de
`MonthlyIncome` es `84205.0` sobre 105000 filas (105000−84205 = 20795 nulos,
≈19.8 %) y `count` de `NumberOfDependents` es `102262.0` (105000−102262 = 2738
nulos, ≈2.6 %); el resto de columnas tiene `count = 105000.0`.

No se ha comprobado aquí el % de nulos de `cs_produccion.csv` (el enunciado de
la tarea solo pedía mirar cabecera/estructura de ese fichero, no leerlo
entero); queda como EDA pendiente (sección 4).

### 1.5 Otras observaciones de la EDA ya presente en el notebook

De la misma tabla `df.describe().T[["count","min","max","mean","std"]]`
(`ejercicio2_proyecto_XAI.ipynb`, celda `cell-5`):

- **Desbalance de clases**: `SeriousDlqin2yrs` tiene media `0.066838`, es decir
  solo ≈6.7 % de casos positivos (impago grave) en `cs_construccion.csv`.
- Varios valores extremos llaman la atención en los `max`: `DebtRatio` llega a
  `329664.0` y `RevolvingUtilizationOfUnsecuredLines` a `22198.0`, pese a que
  el diccionario de datos los describe como `percentage`; `age` tiene un
  `min` de `0.0` (edad de prestatario igual a cero). El notebook se limita a
  mostrar estos estadísticos (celda `cell-5`) y un histograma de `age` (celda
  `cell-6`, salida omitida por tamaño), sin comentar ni tratar estos posibles
  problemas de calidad de dato.
- No hay ninguna otra celda de EDA sobre el dataset real más allá de las
  celdas `cell-2` a `cell-7` (carga, `describe()`, gráfico de `age`, % de
  nulos); las celdas `cell-8` a `cell-12` están vacías y `cell-13`/`cell-14`
  son celdas markdown también vacías — huecos que sugieren EDA planeada pero
  no redactada.

## 2. Qué hace ya `ejercicio2_proyecto_XAI.ipynb`

Leído completo con `Read` (69 celdas identificadas por el tool con ids
`cell-0` a `cell-69`).

### 2.1 Preámbulo con el dataset real (celdas `cell-0` a `cell-14`)

- `cell-0`: título markdown genérico "XAI en Aprendizaje por Refuerzo /
  Multiarmed Bandits Contextuales" (idéntico al de `ejercicio1_AR.ipynb`).
- `cell-1`: bajo un flag `COLAB = True`, descarga con `wget` desde dos URLs de
  Google Drive `cs_construccion.csv` y `DataDictionary.csv` (salida cacheada
  del `wget` incluida, confirma tamaños 4 922 656 bytes y 1 220 bytes
  respectivamente, que coinciden con los ficheros ya presentes en
  `docs/_fuentes/`).
- `cell-2`: `df = pd.read_csv("cs_construccion.csv")` y muestra el `df`
  (105000 filas × 11 columnas, columnas = las de la sección 1.1).
- `cell-3`: `!head DataDictionary.csv` (inspección rápida por shell).
- `cell-4`: carga `DataDictionary.csv` con `pd.read_csv(sep=";")`, quita la
  columna `Unnamed: 0` y la indexa por `Variable Name`.
- `cell-5`: `df.describe().T[["count","min","max","mean","std"]]` (estadísticos
  usados en la sección 1.5).
- `cell-6`: `df["age"].value_counts().plot(kind="bar")` (salida omitida por
  tamaño en la lectura).
- `cell-7`: cálculo de % de nulos por columna (usado en la sección 1.4).
- `cell-8` a `cell-12`: **celdas de código vacías**.
- `cell-13`, `cell-14`: **celdas markdown vacías**.

Es decir: la única parte del notebook que trabaja sobre el dataset real de
crédito es la carga + un EDA mínimo (`describe()`, un histograma de `age`, y
el cálculo de % de nulos). No hay ninguna celda que divida
`cs_construccion.csv` en train/test, entrene un modelo, defina una matriz de
coste o toque `cs_produccion.csv`.

### 2.2 A partir de `cell-15`: plantilla sin adaptar, operando sobre `breast_cancer`

`cell-15` es una celda markdown, `## Modelo supervisado clásico`. A partir de
ahí, y hasta el final del notebook (`cell-69`), el contenido —código y salidas
cacheadas incluidas— es textualmente idéntico, celda a celda, al de
`ejercicio1_AR.ipynb` a partir de su celda `cell-7` (mismo título markdown
`## Modelo supervisado clásico`), con un desfase constante de **+8** en la
numeración (`cell-15`↔`cell-7`, `cell-16`↔`cell-8`, …, `cell-59`↔`cell-51`,
`cell-69`↔`cell-61`, verificado en varios puntos: el `df` mostrado en
`cell-16` son las 569 filas × 31 columnas del dataset `breast_cancer` de
scikit-learn —columnas `mean radius`, `mean texture`, …, `target`—, idéntico
al `df` de `cell-8` de `ejercicio1_AR.ipynb`; el `RandomForestClassifier` de
`cell-25` da exactamente los mismos scores `(0.9572864…, 0.9181286…)` que el
de `cell-17` de `ejercicio1_AR.ipynb`; las 11 reglas del subrogado en
`cell-63` son idénticas letra por letra a las de `cell-55` de
`ejercicio1_AR.ipynb`).

En otras palabras: **`ejercicio2_proyecto_XAI.ipynb` nunca llega a construir
`X_train`/`X_test`/matriz de coste/bandit/subrogado sobre el dataset de
crédito real** — esa parte del notebook (más de 50 celdas, la mayoría del
fichero) sigue siendo, sin modificar, el ejercicio de calentamiento sobre
tumores de mama que resuelve `ejercicio1_AR.ipynb`. Es la evidencia directa
de que el notebook es una plantilla/work-in-progress: el dataset real solo se
"tocó" para el EDA de la sección 2.1, y el resto del pipeline (modelado,
coste, RL, subrogado) queda pendiente de reescribirse desde cero sobre
`cs_construccion.csv`/`cs_produccion.csv`.

## 3. Patrón de trabajo en `ejercicio1_AR.ipynb`

Leído completo con `Read` (62 celdas, ids `cell-0` a `cell-61`). Es el
notebook que sí ejecuta el patrón entero de principio a fin (aunque sobre el
dataset de calentamiento `breast_cancer`), y por tanto la referencia de qué
pasos hay que reproducir/adaptar. Cita explícitamente su propio origen: "Toma
el notebook `6-credit_scoring.ipynb` de la carpeta `AR/2-contextuales` y
tómalo como base para resolver el siguiente problema" (`cell-2`) — ese
notebook original no está presente en `docs/_fuentes/`.

### 3.1 Carga de datos y modelo supervisado baseline

- `cell-3`: carga `sklearn.datasets.load_breast_cancer()` como dataset de
  calentamiento; `cell-4` imprime su `DESCR` (sin nulos, 569 casos, target
  binario 212 maligno / 357 benigno).
- `cell-5`, `cell-6`: definición markdown explícita de la semántica de clase
  (`TARGET=1`: benigno, `TARGET=0`: maligno) y de las cuatro combinaciones
  VP/VN/FP/FN en términos del dominio (p. ej. "FALSO POSITIVO: predecir tumor
  maligno siendo realmente benigno").
- `cell-9`: reduce el problema a solo dos columnas explicativas
  (`worst perimeter`, `worst concavity`) más `target` — simplificación del
  dataset de calentamiento para poder visualizar árboles/reglas fácilmente.
- `cell-12`: `train_test_split(test_size=0.3, random_state=1,
  stratify=df["target"])`.
- `cell-15`: `sns.clustermap` de correlaciones como EDA rápido.
- `cell-17`: `RandomForestClassifier(random_state=1, max_depth=3)` como
  modelo supervisado clásico baseline; `cell-18` reporta accuracy train/test
  (`0.957…`/`0.918…`); `cell-19`-`cell-20` importancia de variables
  (`feature_importances_`).
- `cell-23`: accuracy simple del RF asumiendo que todos los errores penalizan
  igual (comentario explícito: "esto es óptimo si los errores penalizan
  igual") — el punto de partida antes de introducir coste asimétrico.

### 3.2 Barrido de umbral cost-sensitive

- `cell-24`: define la matriz de coste asimétrica con dos constantes,
  `PENALIZACION_FALSO_POSITIVO = 1` y `PENALIZACION_FALSO_NEGATIVO = 10`
  (valores concretos de ejemplo del ejercicio de calentamiento, no los del
  dataset real).
- `cell-25`: para cada umbral único de `predict_proba(X_test)[:,1]`, calcula
  la penalización media aplicando la matriz de coste sobre las predicciones
  binarizadas a ese umbral, y grafica penalización media vs. umbral —
  patrón de **optimización de umbral sensible al coste**, directamente
  reutilizable para los dos escenarios de coste del taller (FP=FN=1 y
  FP=FN=10, `docs/_fuentes/taller_XAI.pdf`, apartado "1. Objetivo de la
  práctica") sin más que cambiar las dos constantes y la fuente de
  `y_test`/`predict_proba`.

### 3.3 Entorno de refuerzo a medida y Multiarmed Bandit contextual

- `cell-28`-`cell-32`: introducción pedagógica a clases de Python (clase
  `persona` de juguete) como paso previo a construir el entorno de RL.
- `cell-33`: clase `entorno_tumores(df, penalizacion_FP, penalizacion_FN)`
  con métodos `nuevo_paciente()` (elige una fila al azar de `df` con
  `np.random.randint`, sin semilla fija), `datos_paciente()` (devuelve las
  features de esa fila) y `act(accion)` (devuelve el refuerzo: `0` si acierta,
  `-penalizacion_FP`/`-penalizacion_FN` si se equivoca, según la clase real).
  `cell-34` instancia `entorno_train`/`entorno_test` a partir de
  `df_train`/`df_test`.
- `cell-38`: benchmark de referencia "médico random" (predicción binaria
  aleatoria uniforme), refuerzo medio de `-2.1866` sobre `N=10000` pacientes
  simulados.
- `cell-40`: descarga (`wget`+`unzip`) del paquete `space_bandits` y de las
  librerías auxiliares `my_library.py`/`libreria_aux_arboles.py` desde Google
  Drive (mismos ficheros que están ya en `docs/_fuentes/` como
  `space_bandits.zip`, `my_library.py.txt`, `libreria_aux_arboles.py.txt`).
- `cell-42`: instancia `LinearBandits(num_actions=2,
  num_features=len(X_train.columns))` del paquete `space_bandits` — el bandit
  contextual lineal, con 2 acciones (predecir 0/1) y tantas features como
  columnas de `X_train`.
- `cell-45`: bucle de entrenamiento online (`N=1000` iteraciones con `tqdm`):
  en cada paso, `entorno_train.nuevo_paciente()` → `model.predict(datos)` →
  `entorno_train.act(prediccion)` (refuerzo) → `model.update(datos,
  prediccion, refuerzo)`. Refuerzo medio acumulado `-0.366` (celda anota en
  comentario un valor de referencia `-0.335` de una ejecución anterior del
  profesor, ambos mejores que el benchmark aleatorio `-2.1866`).
- `cell-46`: evaluación en test (mismo bucle, sin `model.update`), refuerzo
  medio `-0.156` en la ejecución cacheada de este fichero (el comentario de la
  celda anota `-0.289` de otra ejecución previa) — el entorno no fija semilla
  aleatoria, por lo que los resultados varían entre ejecuciones y no son
  directamente reproducibles tal cual.

### 3.4 Modelo subrogado con reglas sobre las decisiones del bandit

- `cell-49`: genera `X_caja_negra = X_test.values` y
  `y_predicha_caja_negra = model.predict(X_caja_negra)` — importante: el
  "modelo caja negra" que se explica aquí es el **bandit** entrenado en 3.3
  (la variable `model` se reasigna de `RandomForestClassifier` a
  `LinearBandits` entre `cell-17` y `cell-42`), no el `RandomForestClassifier`
  de 3.1, que solo se usó para el barrido de umbral.
- `cell-51`: entrena un `DecisionTreeClassifier(max_leaf_nodes=4)` como
  subrogado, ajustándolo a `(X_caja_negra, y_predicha_caja_negra)` —es decir,
  a imitar las **predicciones del bandit**, no la clase real—; `.score()`
  sobre esos mismos datos da `0.9942` (fidelidad de la réplica, no accuracy
  frente al target real).
- `cell-53`-`cell-54`: visualiza el árbol subrogado con
  `graphviz`/`export_graphviz`.
- `cell-55`: usa `get_rules_from_tree` (de `my_library.py.txt`) para extraer
  11 reglas legibles por hoja, con el soporte y % de casos de cada clase de
  salida del bandit por regla (p. ej. `(worst perimeter <= 96.20) and (worst
  perimeter <= 87.11) and (compactness error <= 0.0795)` → 100 % de los casos
  con salida "1").
- `cell-56`-`cell-58`: definición de las dos acciones
  (`ACCION_PREDECIR_MALIGNO=0`, `ACCION_PREDECIR_BENIGNO=1`) y el enunciado
  del ejercicio embebido como celda markdown: "construir un Multiarmed Bandit
  Contextual... optimizar el reward total... Aplica también IA Explicable
  (XAI)... para intentar interpretar la solución obtenida".
- `cell-59`: importa `tree_to_code`/`tree_to_pseudo` de
  `libreria_aux_arboles.py.txt` pero las llamadas están comentadas (código
  muerto/no usado en la versión cacheada); `cell-60`-`cell-61` quedan vacías.

## 4. Qué queda pendiente de adaptar al dataset real de crédito

> **Actualización (2026-07-10):** varios puntos de esta lista ya se han resuelto
> en el repositorio (no en el material del profesor, que seguía sin cubrirlos).
> En concreto: el **EDA del dataset real** se amplió por completo en
> `notebooks/01_EDA.ipynb` (correlaciones, VIF, información mutua, PCA/t-SNE,
> análisis de los valores centinela 96/98 y de outliers, y tests del patrón de
> nulos); y el **preprocesado real** —que el notebook de partida no contenía—
> se construyó desde cero en `notebooks/02_preprocesado.ipynb` y en el pipeline
> serializable `src/preprocessing.py` (imputación de `MonthlyIncome` por tramo
> de edad con flag MNAR `is_missing_monthlyincome`, recálculo de `DebtRatio`
> como numerador/ingreso, winsorización por columna y `log1p`+`StandardScaler`
> parametrizables mediante el flag `escalar`). La decisión de imputación quedó
> registrada en `docs/DECISIONES.md` (D-1.1). Siguen **pendientes** el modelado,
> el barrido de coste, el eventual entorno de RL, el subrogado, SHAP, los
> contrafactuales y la generación de predicciones sobre `cs_produccion.csv`.

- **Reescribir el pipeline completo de las secciones 3.1-3.4 sobre
  `cs_construccion.csv`/`cs_produccion.csv`**: hoy solo existe para
  `breast_cancer` (en ambos notebooks de ejercicio). Ninguno de los dos
  notebooks locales llega a construir `X_train`/`X_test` con las columnas
  reales del crédito (sección 1.1), ni un `RandomForestClassifier` (u otro
  modelo supervisado) ni un `LinearBandits` entrenado sobre ellas.
- **Tratar los nulos** de `MonthlyIncome` (19.8 %) y `NumberOfDependents`
  (2.6 %) antes de poder alimentar cualquier modelo (`RandomForestClassifier`
  y `LinearBandits` no aceptan `NaN`); no hay ninguna celda de imputación en
  el material del profesor, pero esta decisión de diseño ya se tomó e implementó
  en `02_preprocesado.ipynb`/`src/preprocessing.py` (registrada en
  `docs/DECISIONES.md`, D-1.1; ver la nota de actualización al inicio de esta
  sección).
- **Revisar los valores extremos** detectados en `df.describe()` (sección
  1.5: `DebtRatio` hasta `329664`, `RevolvingUtilizationOfUnsecuredLines`
  hasta `22198` pese a describirse como `percentage`, `age` con mínimo `0`) —
  no hay tratamiento de outliers en el material del profesor, pero ya se abordó
  en `02_preprocesado.ipynb` mediante winsorización por columna y en el EDA de
  `01_EDA.ipynb` (ver la nota de actualización al inicio de esta sección).
- **Duplicar el barrido de umbral cost-sensitive (3.2) y/o el entorno de
  RL (3.3) para los dos escenarios de coste exigidos por el enunciado**:
  FP=FN=1 (salida `cs_produccion1.csv`) y FP=FN=10 (salida
  `cs_produccion2.csv`) (fuente: `docs/_fuentes/taller_XAI.pdf`, apartado "1.
  Objetivo de la práctica"). El patrón de `ejercicio1_AR.ipynb` solo
  implementa un único par de constantes de penalización fijas (1 y 10, pero
  usadas como valores de ejemplo sobre tumores, no parametrizadas por
  escenario); sigue abierta en `docs/DECISIONES.md` (`D-0.2`) si esto se
  resuelve con dos umbrales sobre un mismo modelo o con dos modelos
  independientes.
- **Adaptar la semántica de acciones/FP/FN al dominio de crédito**: en el
  ejercicio de calentamiento, acción 0 = "predecir maligno" y acción 1 =
  "predecir benigno" (`cell-56`-`cell-58` de `ejercicio1_AR.ipynb`); para
  crédito hay que redefinir qué significa cada acción (conceder/denegar
  crédito) en relación con `SeriousDlqin2yrs` (1 = impago grave), evitando
  intercambiar sin querer FP y FN al trasladar el patrón.
- **Generar las predicciones sobre `cs_produccion.csv`** (45 000 filas, target
  vacío, sección 1.3) para ambos escenarios de coste y entregarlas como
  `cs_produccion1.csv`/`cs_produccion2.csv` — no existe ninguna celda en los
  notebooks de ejercicio que toque `cs_produccion.csv` en absoluto.
- **Añadir las técnicas de auditoría que exige el enunciado y que no aparecen
  en ningún notebook de ejercicio**: análisis de contrafactuales sobre
  ejemplos seleccionados de clase real 0 y 1, y análisis SHAP global y local
  (`docs/_fuentes/taller_XAI.pdf`, apartado "1. Objetivo de la práctica"); el
  patrón de modelo subrogado con reglas (3.4) sí está implementado y es
  directamente adaptable, cambiando qué modelo se trata como "caja negra".
- **Decidir la familia de modelo** (supervisado clásico vs. Multiarmed
  Bandit, ambos permitidos por el enunciado) — recogido como decisión abierta
  `D-0.1` en `docs/DECISIONES.md`.
- **Fijar semillas aleatorias** en el entorno de RL (`nuevo_paciente` usa
  `np.random.randint` sin semilla) y en el bucle de entrenamiento del bandit,
  para que los resultados numéricos (hoy dispares entre el comentario de una
  celda y su propia salida cacheada, ver 3.3) sean reproducibles.
- **Resolver las descargas por `wget`/Google Drive** (`COLAB = True` en ambos
  notebooks) hacia rutas locales: el dataset, `space_bandits.zip`,
  `my_library.py.txt` y `libreria_aux_arboles.py.txt` ya están disponibles en
  `docs/_fuentes/`, pero como ficheros brutos (el `.zip` sin descomprimir, los
  `.py` guardados como `.txt`), no como módulos importables en un entorno de
  trabajo local.
- **Ampliar el EDA del dataset real**: en el notebook de partida se limita a
  `describe()`, un histograma de `age` y el cálculo de % de nulos (sección 2.1),
  sin análisis de correlaciones, distribución de las demás variables ni revisión
  sistemática de outliers. Este EDA ampliado ya se realizó en
  `notebooks/01_EDA.ipynb` (ver la nota de actualización al inicio de esta
  sección), por lo que este punto está resuelto.
- Confirmar si `ejercicio2_proyecto_XAI.ipynb` es en efecto el "notebook de
  partida oficial" o solo el candidato local más cercano: el enunciado remite
  a una URL externa (ver introducción) que no se ha podido descargar en este
  entorno de trabajo.

## 5. Convenciones detectadas

Consolidación de patrones ya descritos en las secciones anteriores (introducción,
2.1 y 3.3) bajo un encabezado único, por consistencia estructural con el resto
de la serie de documentos de teoría (`shap.md`, `contrafactuals.md`,
`subrogados.md`, `cost_sensitive.md`, `bandits.md`). No hay contenido nuevo
respecto a lo ya citado arriba.

- **Patrón de descarga `COLAB = True` + `wget`**: ambos notebooks de ejercicio
  anteponen a la carga de datos un flag `COLAB = True` seguido de descargas por
  `wget` desde URLs de Google Drive. En `ejercicio2_proyecto_XAI.ipynb` va todo
  en una sola celda (`cell-1`, sección 2.1: descarga `cs_construccion.csv` y
  `DataDictionary.csv`); en `ejercicio1_AR.ipynb` está repartido en dos celdas
  consecutivas (`cell-39` fija `COLAB = True`, `cell-40` hace el `wget`/`unzip`
  de `space_bandits.zip`, `my_library.py` y `libreria_aux_arboles.py`, ver
  sección 3.3). Es una convención repetida en los dos notebooks del profesor,
  no una instrucción explícita del enunciado (`taller_XAI.pdf`); el grupo
  puede mantenerla (con `COLAB = False` para trabajo local) o sustituirla por
  lectura directa de fichero, dado que todo lo referenciado ya está disponible
  en `docs/_fuentes/`.
- **Referencia explícita a un notebook base no incluido**: `ejercicio1_AR.ipynb`
  remite a un notebook externo concreto — "Toma el notebook
  `6-credit_scoring.ipynb` de la carpeta `AR/2-contextuales` y tómalo como base
  para resolver el siguiente problema" (`cell-2`, ver introducción y sección 3)
  — que no está presente en `docs/_fuentes/` ni se ha podido descargar (la URL
  de Drive citada en el enunciado tampoco se ha resuelto, ver introducción).
  Se deja constancia de esta instrucción del profesor sobre la procedencia del
  material, aunque el fichero en sí sea inaccesible en este entorno de trabajo.
- **Nombrado homogéneo de secciones markdown entre notebooks**: los dos
  notebooks de ejercicio usan el mismo título de celda markdown para arrancar
  el bloque de modelado (`## Modelo supervisado clásico`, `cell-15` en
  `ejercicio2_proyecto_XAI.ipynb` ↔ `cell-7` en `ejercicio1_AR.ipynb`, ver
  sección 2.2) — evidencia de que ambos notebooks comparten una plantilla y un
  nombrado de secciones consistente, aunque solo uno de los dos
  (`ejercicio1_AR.ipynb`) llega a completarse sobre su dataset.

## 6. Fuentes citadas

- `docs/_fuentes/taller_XAI.pdf` — enunciado oficial de la práctica.
- `docs/_fuentes/DataDictionary.csv` — diccionario de columnas.
- `docs/_fuentes/cs_construccion.csv` — dataset de construcción (105 000
  filas × 11 columnas, verificado por conteo de líneas y por las celdas
  `cell-2`/`cell-5`/`cell-7` de `ejercicio2_proyecto_XAI.ipynb`).
- `docs/_fuentes/cs_produccion.csv` — dataset de producción (45 000 filas ×
  11 columnas, verificado por conteo de líneas e inspección directa de las
  primeras filas del fichero).
- `docs/_fuentes/ejercicio2_proyecto_XAI.ipynb` — candidato principal a
  notebook de partida (celdas `cell-0` a `cell-69`).
- `docs/_fuentes/ejercicio1_AR.ipynb` — notebook hermano con el mismo patrón
  resuelto sobre `breast_cancer` (celdas `cell-0` a `cell-61`).
- `docs/_fuentes/INVENTARIO.md` — inventario previo de `docs/_fuentes/` que
  corrobora tamaños de fichero y clasificación de cada fuente.
- `docs/DECISIONES.md` — decisiones abiertas `D-0.1` (supervisado vs. bandit)
  y `D-0.2` (un modelo con dos umbrales vs. dos modelos), referenciadas en la
  sección 4.
