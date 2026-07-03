# Modelo subrogado con reglas

> Documento de teoría de referencia del Taller B4-T2 (XAI aplicado a concesión de
> crédito, dataset "Give Me Some Credit"). Es **solo teoría**: no contiene código
> de modelo ni decide implementación; eso corresponde a `src/` y a los notebooks
> del grupo.

## 1. Introducción

El enunciado oficial de la práctica exige explícitamente, como parte de la
auditoría del modelo de concesión de crédito, un **"Modelo subrogado con reglas
que lo expliquen"** (fuente: `taller_XAI.pdf`, apartado "1. Objetivo de la
práctica"), junto con el análisis de contrafactuales y el análisis SHAP global y
local (documentados en `docs/teoria/contrafactuals.md` y `docs/teoria/shap.md`
respectivamente). Este documento cubre qué dice el material del profesor sobre
el **modelo subrogado global** — su definición, el concepto de **fidelidad**
(qué tan bien el subrogado imita al modelo original, no la variable objetivo
real) y cómo se extraen reglas legibles de un árbol de decisión entrenado como
subrogado — y lo contrasta con el patrón de código que aparece efectivamente
aplicado en los notebooks de ejercicio del profesor, que sí implementan este
patrón paso a paso (a diferencia de SHAP y contrafactuales, para los que el
material solo da la definición conceptual sin código de ejemplo).

La fuente teórica principal es `XAI.pdf` (slides de Manuel Sánchez-Montañés,
UAM), sección **"Modelo Subrogado Global"** en las páginas 176-180, dentro del
bloque más amplio de páginas 136-180 que el propio material dedica al panorama
de "Métodos Agnósticos al Modelo" (PDP/ICE/ALE, importancia por permutación y,
finalmente, modelo subrogado). Como complemento práctico, se han revisado los
notebooks `ejercicio1_AR.ipynb` y `ejercicio2_proyecto_XAI.ipynb` (idéntico
patrón de código en ambos) y `XAI_predicción_series_temporales_LSTM.ipynb`, así
como las librerías de utilidades `my_library.py.txt` y
`libreria_aux_arboles.py.txt` que esos notebooks importan para extraer reglas
de un árbol ya entrenado.

## 2. Panorama: dónde encaja el modelo subrogado entre los métodos agnósticos de XAI

Antes de entrar en el detalle del subrogado, el material sitúa el tema dentro
de un panorama más amplio de algoritmos de XAI que conviene dejar constancia
aquí por haberse leído en el mismo rango de páginas encargado (136-180), aunque
no sean el objeto central de este documento:

- El material abre el bloque con una lista general de "Algoritmos típicos" de
  XAI: **"Modelo surrogado: normalmente un árbol de decisión, conjunto de
  reglas etc."**, **"Valores SHAP"**, **"LIME"** y **"Análisis What-If"**
  (fuente: XAI.pdf, pp. 138-140). Sobre este último, el material añade una
  frase breve que no se desarrolla más adelante en el rango leído: "Análisis
  What-If: son modelos de explicabilidad a nivel de ejemplo. En este caso se
  calcula el cambio mínimo para que el ejemplo cambie la clase" (fuente:
  XAI.pdf, p. 141) — nótese el parecido conceptual con la idea de
  contrafactuales, aunque el material no los identifica explícitamente como la
  misma técnica en estas páginas.
- El material clasifica los métodos de XAI según dos ejes: **"Ante-Hoc versus
  Post-Hoc"** y **"Métodos agnósticos al modelo versus Métodos específicos para
  el modelo"** (fuente: XAI.pdf, p. 143). El bloque que sigue inmediatamente
  después se titula **"Métodos Agnósticos al Modelo"** (fuente: XAI.pdf, pp.
  145-146) y bajo ese mismo título se desarrollan, en este orden, **PDP/ICE/ALE**
  (pp. 147-165), **importancia por permutación** (pp. 166-175) y, por último,
  el **modelo subrogado global** (pp. 176-180) — es decir, el propio material
  encuadra el modelo subrogado como un método agnóstico al modelo (no depende
  de que el modelo original sea de una familia concreta), aunque las páginas
  176-180 no repiten literalmente la etiqueta "agnóstico" para el subrogado en
  particular; es una lectura de la estructura del documento, no una frase
  citada textualmente sobre el subrogado.
- Muy resumidamente, para dejar constancia de qué contienen esas páginas sin
  inventar detalle que no se ha pedido desarrollar aquí: **PDP** (Partial
  Dependence Plot) mide la predicción media del modelo forzando a un
  subconjunto de variables a un valor fijo (fuente: XAI.pdf, pp. 147-148);
  **ICE** (Individual Conditional Expectation) es "en el fondo un PDP para un
  caso particular (no se promedia)" (fuente: XAI.pdf, p. 158); **ALE**
  (Accumulated Local Effects) "hace algo parecido al PDP pero tiene en cuenta
  relaciones estadísticas entre diferentes variables (no asume que son
  independientes)" (fuente: XAI.pdf, p. 165); y la **importancia por
  permutación** mide "el nivel de degradación del modelo cuando la variable se
  permuta aleatoriamente" (fuente: XAI.pdf, p. 166), con la ventaja de dar "una
  visión global y muy compacta del comportamiento del modelo" (fuente: XAI.pdf,
  p. 171) pero con el riesgo de generar "casos poco realistas" si hay variables
  correlacionadas (fuente: XAI.pdf, pp. 173-175). Estas cuatro técnicas no son
  el tema de este documento (que es el modelo subrogado con reglas) y **el
  material no las conecta explícitamente con el subrogado ni con el dataset de
  crédito de este taller**; se documentan aquí solo para dejar constancia de
  que se han revisado y de dónde queda situado el subrogado respecto a ellas.
  Si el grupo decide usarlas como "otras técnicas que los estudiantes
  consideren oportunas" (fuente: `taller_XAI.pdf`), este documento no es la
  fuente de teoría para ellas.

## 3. Fundamento técnico del modelo subrogado global

### 3.1 Definición: un modelo interpretable que imita al modelo caja negra

El material presenta la idea en dos diapositivas consecutivas, con un diagrama
que se construye en dos pasos:

- Paso 1: "Modelo subrogado: se toma un modelo black-box" — el diagrama muestra
  una caja "Modelo black box" que recibe una "Entrada" y produce una
  "Predicción modelo black box" (fuente: XAI.pdf, p. 176).
- Paso 2: "...y se entrena un modelo interpretable que lo imite" — el diagrama
  añade una segunda caja, "Modelo interpretable", que recibe la **misma
  Entrada** que el modelo black-box y produce una "Estimación predicción salida
  modelo black box" (fuente: XAI.pdf, p. 177).

Este segundo diagrama es la definición operativa más precisa que da el
material: el modelo interpretable (el subrogado) no se entrena para predecir la
variable objetivo real del problema, sino para **estimar la salida del modelo
black-box** dada la misma entrada. Es la base directa del concepto de
fidelidad que se desarrolla en la sección 3.4.

### 3.2 Ejemplo del profesor

El material da un único ejemplo concreto de qué combinación de modelos
"original / subrogado" tiene en mente: **"Modelo original: gradient boosting
(no interpretable). Modelo subrogado: árbol de decisión"** (fuente: XAI.pdf,
p. 178). No se desarrolla en el texto extraíble de esa página ningún detalle
adicional (dataset, métricas) más allá de nombrar esa pareja de modelos.

### 3.3 Ventajas señaladas por el profesor

La siguiente diapositiva resume explícitamente las ventajas (fuente: XAI.pdf,
p. 179):

- "El método es sencillo y fácil de entender para personas no técnicas."
- "Flexibilidad: puede usarse cualquiera de los modelos directamente
  interpretables (árbol de decisión, regresión lineal, regresión logística,
  conjunto de reglas etc.)."

Es decir, el material no impone que el subrogado tenga que ser necesariamente
un árbol de decisión: lo presenta como la opción más habitual ("normalmente un
árbol de decisión, conjunto de reglas etc.", fuente: XAI.pdf, p. 139) entre
varias posibles. El enunciado de este taller, sin embargo, sí concreta la
elección al pedir explícitamente un modelo subrogado **"con reglas"** (fuente:
`taller_XAI.pdf`) — ver la discusión en la sección 5.

### 3.4 Desventajas y el concepto de FIDELIDAD

La última diapositiva de la sección es la más relevante para este documento,
porque introduce explícitamente la idea de que el subrogado es una
**aproximación**, no una réplica exacta, y plantea la pregunta central sobre
qué tan buena tiene que ser esa aproximación (fuente: XAI.pdf, p. 180):

- "El modelo subrogado no es equivalente al modelo que deseamos explicar. Es
  una aproximación."
- "¿Qué nivel de ajuste queremos? ¿el 99%? ¿el 95%? Normalmente, cuanto más
  aproximado más complejo, y menos interpretable es el modelo subrogado."
- "Cuidado: el modelo subrogado puede aproximar muy bien el modelo original en
  un tipo de casos, pero mal en otros."

Esto es, en esencia, la definición de **fidelidad** (aunque el material no usa
literalmente la palabra "fidelidad" en estas páginas): el "nivel de ajuste" al
que se refiere la pregunta "¿el 99%? ¿el 95%?" no se mide comparando las
predicciones del subrogado contra la variable objetivo real del problema, sino
comparándolas contra **las predicciones del propio modelo caja-negra** — tal y
como muestra el diagrama de la sección 3.1 (p. 177), donde el modelo
interpretable estima la "salida modelo black box", no el target real. Esta
distinción — fidelidad al modelo original, no *accuracy* frente a la etiqueta
verdadera — es la idea más importante que debe manejar el grupo al construir el
subrogado de este taller, y aparece confirmada de forma muy explícita en el
código de ejemplo del profesor (sección 3.5) y en un segundo ejemplo con
números concretos (sección 3.6).

Además, el material señala explícitamente el trade-off complejidad↔fidelidad
("cuanto más aproximado más complejo, y menos interpretable es el modelo
subrogado", p. 180) sin dar una regla concreta de cuándo parar (no hay un
umbral recomendado universal en estas páginas); y advierte de un riesgo
adicional: la fidelidad puede no ser homogénea ("puede aproximar muy bien el
modelo original en un tipo de casos, pero mal en otros", p. 180), es decir, una
fidelidad global alta no garantiza que el subrogado sea fiel para todos los
subgrupos de casos (p. ej. podría ser muy fiel para solicitantes con score bajo
y poco fiel cerca del punto de corte de decisión, que es precisamente la zona
más sensible para la auditoría). **El material no cubre esto** con más detalle
ni da una métrica concreta de fidelidad "por subgrupo"; es una lectura directa
de la propia frase de la p. 180 aplicada al contexto de auditoría de crédito.

### 3.5 Confirmación práctica: el patrón de código del profesor (`ejercicio1_AR.ipynb` / `ejercicio2_proyecto_XAI.ipynb`)

Los dos notebooks de ejercicio del profesor (idénticos en esta parte,
diferenciándose solo en que `ejercicio2_proyecto_XAI.ipynb` añade al principio
una celda de EDA sobre el dataset real de crédito sin llegar a aplicar el resto
del notebook sobre él — ver `docs/_fuentes/INVENTARIO.md`) implementan
literalmente el patrón de las secciones 3.1 y 3.4, con un comentario del propio
profesor que resume la idea con una metáfora propia:

> "Modelo surrogado. 'Gemelo digital' del que queremos interpretar. Pero mucho
> más sencillo. Idealmente: 'caja blanca'." (fuente: `ejercicio1_AR.ipynb` y
> `ejercicio2_proyecto_XAI.ipynb`, celda de comentario justo antes de entrenar
> el árbol subrogado.)

El código que seguía a ese comentario (sobre un `RandomForestClassifier` como
caja negra en el ejercicio, o sobre las acciones del bandit `LinearBandits`
entrenado antes en el mismo notebook) sigue exactamente el diagrama de la
sección 3.1:

1. Se genera `X_caja_negra` (las entradas) y `y_predicha_caja_negra =
   model.predict(X_caja_negra)` — es decir, **la salida del modelo caja negra
   sobre esas mismas entradas**, no la variable objetivo real del dataset
   (fuente: `ejercicio1_AR.ipynb`/`ejercicio2_proyecto_XAI.ipynb`).
2. Se entrena `surrogate_model = DecisionTreeClassifier(max_leaf_nodes=4)` con
   `surrogate_model.fit(X_caja_negra, y_predicha_caja_negra)` — el árbol
   aprende a predecir la salida del modelo caja negra, exactamente como en el
   diagrama de la p. 177 (fuente: `ejercicio1_AR.ipynb`/
   `ejercicio2_proyecto_XAI.ipynb`).
3. La fidelidad se mide con `surrogate_model.score(X_caja_negra,
   y_predicha_caja_negra)`, con un comentario explícito del profesor en la
   misma línea: **"# cómo de buena es la réplica"** (fuente:
   `ejercicio1_AR.ipynb`/`ejercicio2_proyecto_XAI.ipynb`). En la ejecución
   concreta del notebook (sobre el problema de calentamiento con
   `breast_cancer` y el bandit `LinearBandits` como caja negra) ese valor fue
   `0.9941520467836257` (≈ 99.4%), es decir, se mide contra
   `y_predicha_caja_negra` (la salida del modelo caja negra), nunca contra
   `y_test` (la variable objetivo real) — confirmación directa, con código y
   número concretos, del concepto de fidelidad descrito en la sección 3.4.
4. Al nombrar las clases para la extracción de reglas, el propio código usa la
   etiqueta `"Salida modelo caja negra={}"` (no "diagnóstico real" ni nombres
   de la variable objetivo original) para las dos clases del árbol (fuente:
   `ejercicio1_AR.ipynb`/`ejercicio2_proyecto_XAI.ipynb`) — otra confirmación,
   esta vez en el nombrado de variables, de que lo que el árbol explica es la
   decisión del modelo caja negra, no la realidad subyacente.

### 3.6 Un segundo ejemplo con umbral de fidelidad explícito: el notebook de LSTM

El notebook `XAI_predicción_series_temporales_LSTM.ipynb` (dominio distinto:
predicción de demanda/serie temporal con una red LSTM, no el dataset de
crédito, pero mismo patrón conceptual de auditoría) aplica el mismo patrón de
forma más explícita, con una definición en una celda de markdown del propio
profesor:

> "Para entender el LSTM, que no es interpretable, montamos un modelo
> interpretable, en este caso un árbol de decisión, que explique la salida del
> LSTM en función de las entradas. A este modelo interpretable se le llama
> 'modelo surrogado' (surrogate model)." (fuente:
> `XAI_predicción_series_temporales_LSTM.ipynb`.)

Y, de forma más relevante aún para el concepto de fidelidad, el código de ese
notebook define explícitamente `y_LSTM` como **las predicciones del propio
LSTM** (no la serie real observada): `y_LSTM = np.array(target_train_pred.tolist()
+ target_test_pred.tolist())` (fuente:
`XAI_predicción_series_temporales_LSTM.ipynb`). A partir de ahí, entrena un
`DecisionTreeRegressor` de forma iterativa, aumentando el número de hojas hasta
alcanzar una fidelidad objetivo:

> "objetivo = 0.9 # precisión deseada de la explicación (de 0 a 1; 1:
> perfecto)" — y un bucle que aumenta `n_hojas` mientras
> `R2_score(y_LSTM, y_surrogated) < objetivo` (fuente:
> `XAI_predicción_series_temporales_LSTM.ipynb`).

Esto es una implementación concreta y numérica exacta de la pregunta retórica
de la p. 180 de `XAI.pdf` ("¿Qué nivel de ajuste queremos? ¿el 99%? ¿el 95%?"):
aquí el profesor fija ese nivel en 0.9 (R², medido contra las predicciones del
LSTM, no contra la serie real) y usa un bucle de complejidad creciente
(número de hojas del árbol) hasta alcanzarlo — el mismo trade-off
complejidad↔fidelidad señalado en la teoría. Además, las propias diapositivas
de `XAI.pdf` dan un número de fidelidad concreto para este mismo ejemplo de
demanda: **"XAI: modelo surrogado (explica el 95% del LSTM)"** (fuente:
XAI.pdf, p. 218) — confirmando, con un caso real del profesor, que la cifra que
se reporta como resultado del subrogado es un porcentaje de fidelidad al modelo
original (aquí, un LSTM), no una métrica sobre la variable objetivo real de la
serie temporal.

Nótese que este ejemplo (pp. 213-219 de `XAI.pdf` y el notebook de LSTM)
aparece formalmente dentro del rango de páginas que `docs/teoria/contrafactuals.md`
documenta como "Explicaciones Basadas en Ejemplos" (contrafactuales,
prototipos/excepciones), pero — como ya se señala en ese documento — el
contenido real de esas páginas concretas (213-219) es un ejemplo de **modelo
subrogado**, no de contrafactual; se cita aquí, en el documento correcto, por
esa razón.

### 3.7 Extracción de reglas legibles de un árbol de decisión ya entrenado

El requisito del enunciado no es solo "un modelo subrogado", sino un modelo
subrogado **"con reglas que lo expliquen"** (fuente: `taller_XAI.pdf`). Las
diapositivas de `XAI.pdf` (pp. 176-180) no detallan ningún algoritmo concreto
para convertir un árbol ya entrenado en reglas de texto legibles — solo
mencionan el árbol/conjunto de reglas como familia de modelo subrogado posible
(p. 179). **El material no cubre esto en el PDF de teoría**; el detalle de
cómo extraer esas reglas en la práctica proviene únicamente de las librerías de
utilidades que los notebooks de ejercicio descargan e importan. Se documenta
aquí, de forma conceptual (sin transcribir el código línea a línea), qué hace
cada función:

- **`get_rules_from_tree(clf, feature_names, class_values, X, y)`** (fuente:
  `my_library.py.txt`, importada en los notebooks con
  `from my_library import get_rules_from_tree`): recorre recursivamente la
  estructura interna de un `DecisionTreeClassifier` de scikit-learn (los
  arrays `tree_.children_left`, `tree_.children_right`, `tree_.feature`,
  `tree_.threshold`) acumulando, para cada nodo hoja, la conjunción de
  condiciones `(variable <= umbral)` / `(variable > umbral)` de todos los
  nodos de decisión atravesados desde la raíz hasta esa hoja — es decir,
  construye literalmente la regla "SI ... Y ... Y ... ENTONCES" que describe
  esa hoja. A continuación, usa `clf.apply(X)` para averiguar en qué hoja cae
  cada caso del conjunto `X` proporcionado, y cruza esa asignación con las
  etiquetas `y` (en el patrón del profesor, `y` es `y_predicha_caja_negra`,
  la salida del modelo caja negra — ver sección 3.5) para reportar, por cada
  regla/hoja, cuántos casos de cada clase caen en ella y qué porcentaje
  representan. En la salida impresa en los notebooks, cada regla aparece con
  un formato como `(worst perimeter <= 96.2000) and (worst perimeter <=
  87.1100) and (compactness error <= 0.0795)` seguido del reparto de clases
  observado en esa hoja (p. ej. "Salida modelo caja negra=1: 53 cases
  (100.00%)") (fuente: `ejercicio1_AR.ipynb`/`ejercicio2_proyecto_XAI.ipynb`,
  celda de ejecución de `get_rules_from_tree`). El **soporte** (número de casos)
  y el **porcentaje de pureza** de cada regla son exactamente la evidencia
  cuantitativa de fidelidad *local* (por regla) que complementa la fidelidad
  *global* del `.score()` de la sección 3.5: una regla con "100.00%" de una
  sola clase es una regla perfectamente fiel a la decisión del modelo caja
  negra para los casos que caen en ella; una regla con clases mezcladas sería
  una zona donde el árbol subrogado no logra replicar con precisión la
  decisión del modelo original.
- **`get_rules_from_regression_tree(...)`** (fuente:
  `XAI_predicción_series_temporales_LSTM.ipynb`, celda 89): es la misma idea
  aplicada a un `DecisionTreeRegressor` en vez de a un clasificador — en vez de
  reportar el reparto de clases por hoja, reporta el **valor medio** de `y` (en
  ese notebook, las predicciones del LSTM) para los casos que caen en cada
  hoja, de forma que cada regla se lee como "SI ... ENTONCES prediction =
  &lt;valor medio&gt;" (fuente: celda de ejecución del notebook, formato
  observado: `"IF " + rule[0] + " THEN prediction = {}".format(rule[1])`).
  Aunque este notebook no trata el dataset de crédito, es directamente
  aplicable si el grupo decide auditar con un árbol de **regresión** (por
  ejemplo, si el subrogado se entrena sobre la probabilidad continua de
  impago predicha por el modelo caja negra, en vez de sobre la clase discreta
  0/1).
- **`train_val_test_split(...)`** (fuente: `my_library.py.txt`): utilidad
  genérica de partición de datos en train/val/test; no es específica del
  modelo subrogado, se documenta aquí solo por completitud del inventario de
  `my_library.py.txt`.
- **`tree_to_code(tree, feature_names, ...)`** (fuente:
  `libreria_aux_arboles.py.txt`, importada con
  `from libreria_aux_arboles import tree_to_code, tree_to_pseudo`): recorre el
  árbol de forma recursiva y **imprime una función Python ejecutable**
  (`def tree(...): if variable <= umbral: ... else: ... return valor`) que
  reproduce exactamente las decisiones del árbol como código real, no solo
  como texto descriptivo. Es una alternativa a `get_rules_from_tree` que sirve
  para poder ejecutar la lógica del árbol fuera de scikit-learn (p. ej. para
  documentarla o incrustarla en otro sistema).
- **`tree_to_pseudo(tree, feature_names)`** (fuente:
  `libreria_aux_arboles.py.txt`): variante de `tree_to_code` que en vez de
  imprimir Python imprime **pseudocódigo con llaves al estilo C**
  (`if ( variable <= umbral ) { ... } else { ... }`), pensado para que las
  reglas sean legibles por alguien no familiarizado con Python — encaja
  directamente con el requisito del enunciado de que el subrogado tenga
  "reglas que lo expliquen" de forma comprensible.
- En los notebooks de ejercicio (`ejercicio1_AR.ipynb`/
  `ejercicio2_proyecto_XAI.ipynb`), las llamadas a `tree_to_code` y a la
  representación gráfica con `export_graphviz`/`graphviz.Source` aparecen
  **comentadas** (no ejecutadas) en la celda final, mientras que
  `get_rules_from_tree` sí aparece ejecutada con su salida impresa completa —
  es decir, en el ejemplo de código que dejó el profesor, la vía de extracción
  de reglas efectivamente usada y mostrada con resultado es
  `get_rules_from_tree`, no `tree_to_code`/`tree_to_pseudo` (fuente:
  `ejercicio1_AR.ipynb`/`ejercicio2_proyecto_XAI.ipynb`). Esto no es una
  prohibición de usar las otras dos funciones, pero sí es un dato objetivo
  sobre cuál de las utilidades disponibles llegó a ejecutarse en el material
  de partida.

## 4. Aplicación práctica a este taller

### 4.1 Qué pide el enunciado y qué aporta el material revisado

El enunciado pide, sin más matización, un **"Modelo subrogado con reglas que lo
expliquen"** como parte obligatoria de la auditoría (fuente: `taller_XAI.pdf`).
Aplicando lo documentado en las secciones 2-3 al dataset de crédito de este
taller ("Give Me Some Credit", variable objetivo `SeriousDlqin2yrs`, variables
como `RevolvingUtilizationOfUnsecuredLines`, `age`,
`NumberOfTime30-59DaysPastDueNotWorse`, `DebtRatio`, `MonthlyIncome`,
`NumberOfOpenCreditLinesAndLoans`, `NumberOfTimes90DaysLate`,
`NumberRealEstateLoansOrLines`, `NumberOfTime60-89DaysPastDueNotWorse`,
`NumberOfDependents`; fuente: `DataDictionary.csv`), el patrón conceptual que
sí está cubierto por el material (secciones 3.1, 3.4 y 3.5) sería:

1. Tomar el modelo de concesión de crédito ya entrenado (el "modelo caja
   negra" de este taller: el modelo supervisado clásico o el Multiarmed Bandit,
   según lo que decida el grupo — ver `docs/DECISIONES.md`, decisión D-0.1,
   abierta en el momento de escribir este documento).
2. Generar las predicciones de ese modelo sobre un conjunto de solicitantes
   (`y_predicha_caja_negra` en la terminología del notebook del profesor,
   sección 3.5) — **no** usar `SeriousDlqin2yrs` (la variable objetivo real)
   como target del árbol subrogado.
3. Entrenar un árbol de decisión (`DecisionTreeClassifier`, siguiendo el
   patrón de `ejercicio1_AR.ipynb`/`ejercicio2_proyecto_XAI.ipynb`) para
   predecir esas predicciones del modelo caja negra a partir de las mismas
   variables explicativas del dataset de crédito.
4. Medir la fidelidad con `.score()` del árbol contra
   `y_predicha_caja_negra` (no contra `y_test`), tal como hace el notebook del
   profesor (sección 3.5), y decidir el nivel de complejidad (número de hojas)
   en función del trade-off fidelidad↔interpretabilidad señalado en XAI.pdf,
   p. 180 (sección 3.4) — opcionalmente con el mismo patrón iterativo del
   notebook de LSTM (aumentar hojas hasta una fidelidad objetivo, sección 3.6),
   aunque el material no fija qué umbral concreto (0.9, 0.95, 0.99...) sería
   apropiado para este dataset de crédito en particular.
5. Extraer las reglas del árbol resultante con `get_rules_from_tree` (o,
   alternativamente, `tree_to_pseudo`/`tree_to_code`), reportando para cada
   regla el soporte y el porcentaje de pureza respecto a la decisión del
   modelo caja negra, siguiendo exactamente el formato de salida mostrado en
   los notebooks de ejercicio (sección 3.7).

### 4.2 Relación con los dos escenarios de coste del taller (FP=FN=1 y FP=FN=10)

El taller pide auditar el modelo en dos condiciones de coste: Falso
Positivo = Falso Negativo = 1 y Falso Positivo = Falso Negativo = 10 (fuente:
`taller_XAI.pdf`). **Las páginas 176-180 de `XAI.pdf` (la sección específica
de modelo subrogado) no mencionan en ningún momento coste ni escenarios FP/FN
— el material no cubre esto.** Tampoco lo hacen los notebooks de ejercicio en
la parte del subrogado en sí (las celdas de `PENALIZACION_FALSO_POSITIVO` /
`PENALIZACION_FALSO_NEGATIVO` aparecen antes, en la parte de modelo
supervisado/bandit, no en la parte de subrogado).

Dicho esto, la propia lógica de la definición (sección 3.1: el subrogado imita
"la salida del modelo caja negra") permite un puente conceptual razonable, no
una fórmula inventada: si el escenario de coste (FP=FN=1 vs FP=FN=10) cambia el
umbral de decisión del modelo de crédito o lleva a reentrenarlo (ver
`docs/DECISIONES.md`, decisiones D-0.2 y D-0.3, abiertas en el momento de
escribir este documento), entonces `y_predicha_caja_negra` sería distinta en
cada escenario — y, por la propia definición del subrogado (que aprende a
imitar esa salida concreta), en principio haría falta **entrenar y evaluar un
árbol subrogado distinto para cada uno de los dos escenarios de coste**, en
vez de asumir que las reglas extraídas para un escenario siguen siendo fieles
al modelo del otro escenario. Esta es una inferencia razonada a partir de la
definición de la sección 3.1, **no una instrucción explícita de la fuente**, y
debería verificarse empíricamente en el notebook del grupo (comparando, por
ejemplo, si las reglas y la fidelidad del árbol cambian de forma relevante
entre ambos escenarios).

### 4.3 Sobre la "regla con reglas": qué variables tienen sentido en las reglas del subrogado de crédito

El material no dice nada específico sobre qué variables del dataset de crédito
deberían aparecer en las reglas del subrogado (esto no está cubierto por
`XAI.pdf` ni por los notebooks, que usan otro dataset de ejemplo). Lo único que
aporta el material de forma directamente aplicable es el formato de salida de
`get_rules_from_tree` (sección 3.7): cada regla combina condiciones sobre las
variables explicativas reales del dataset (aquí, las de `DataDictionary.csv`)
con el soporte y pureza respecto a la decisión del modelo. Por ejemplo, una
regla hipotética con ese mismo formato aplicada a este dataset podría leerse
como "SI `NumberOfTimes90DaysLate` > 2 Y `RevolvingUtilizationOfUnsecuredLines`
> 0.8 ENTONCES Salida modelo caja negra = deniega crédito (soporte: N casos,
pureza: X%)" — pero esta frase concreta es un **ejemplo ilustrativo construido
para este documento**, no una regla real extraída de ningún modelo, y no debe
confundirse con contenido citado de las fuentes.

## 5. Preferencias del profesor detectadas

- El enunciado oficial (`taller_XAI.pdf`) exige, de forma explícita y
  obligatoria (no como "otras técnicas que los estudiantes consideren
  oportunas"), un **"Modelo subrogado con reglas que lo expliquen"** — el
  énfasis en "con reglas" (no solo "un modelo interpretable" en abstracto)
  sugiere una preferencia clara por que el subrogado sea, en concreto, un
  árbol de decisión (u otro modelo del que se puedan extraer reglas
  if/then), en vez de, por ejemplo, una regresión lineal/logística — aunque
  el propio `XAI.pdf` (p. 179) presenta la regresión lineal/logística como
  alternativas igualmente válidas de "modelo directamente interpretable" para
  un subrogado en general, sin restringirlo a árboles.
- El material de ejercicio del profesor (`ejercicio1_AR.ipynb` y
  `ejercicio2_proyecto_XAI.ipynb`, ambos con código idéntico en esta parte)
  usa consistentemente `DecisionTreeClassifier(max_leaf_nodes=4)` — es decir,
  controla la complejidad del árbol limitando el **número de hojas**, no la
  profundidad (`max_depth` no se usa en la llamada real, aunque aparece
  comentado como alternativa en el notebook de LSTM: `#max_depth=4`). Es una
  convención concreta observada en el código de partida del profesor, aunque
  no se declara explícitamente como "regla obligatoria" en ninguna parte del
  material.
- El propio profesor mide la fidelidad del subrogado con el comentario
  explícito **"# cómo de buena es la réplica"** junto a la llamada a
  `.score()` contra `y_predicha_caja_negra` (fuente:
  `ejercicio1_AR.ipynb`/`ejercicio2_proyecto_XAI.ipynb`) — deja claro, con su
  propio comentario en el código, que la métrica de bondad del subrogado es la
  réplica de la salida del modelo, no el acierto sobre la etiqueta real. Esta
  es la preferencia metodológica más importante y mejor evidenciada de todo
  el material revisado para este documento.
- En el notebook de LSTM, el profesor fija un umbral numérico concreto de
  fidelidad objetivo, `objetivo = 0.9` (R², medido contra las predicciones del
  LSTM), y itera la complejidad del árbol (número de hojas) hasta alcanzarlo
  (fuente: `XAI_predicción_series_temporales_LSTM.ipynb`). **Este 0.9 es
  específico de ese ejemplo de demanda/LSTM, no una cifra que el material
  extienda explícitamente al dataset de crédito de este taller** — se señala
  aquí como una convención observada en un ejemplo del profesor, no como un
  requisito a aplicar sin más al taller de crédito.
- El profesor proporciona `my_library.py.txt` y `libreria_aux_arboles.py.txt`
  como utilidades descargables (vía `wget` desde su Google Drive, en las
  celdas `COLAB` de los notebooks de ejercicio) e importables directamente
  (`from my_library import get_rules_from_tree`,
  `from libreria_aux_arboles import tree_to_code, tree_to_pseudo`) — esto
  sugiere que la intención del profesor es que los alumnos reutilicen estas
  funciones ya hechas para la extracción de reglas, en vez de reimplementar el
  recorrido del árbol desde cero.
- De las tres utilidades de extracción de reglas disponibles
  (`get_rules_from_tree`, `tree_to_code`, `tree_to_pseudo`), en el código de
  ejemplo del profesor solo `get_rules_from_tree` aparece **efectivamente
  ejecutada con salida mostrada**; las llamadas a `tree_to_code` y a la
  visualización gráfica con `graphviz` aparecen comentadas en la celda final
  de ambos notebooks de ejercicio (fuente:
  `ejercicio1_AR.ipynb`/`ejercicio2_proyecto_XAI.ipynb`). Es un dato objetivo
  sobre qué vía llegó a mostrarse funcionando en el material de partida, no
  una prohibición explícita de usar las otras.
- No se ha detectado ninguna instrucción explícita del profesor sobre qué
  umbral de fidelidad usar para el dataset de crédito de este taller
  concretamente, ni sobre cómo adaptar el modelo subrogado a los dos
  escenarios de coste (FP=FN=1 y FP=FN=10). Se deja constancia explícita de
  esta ausencia en vez de suponer una preferencia no verificada (ver
  secciones 3.4, 4.2 y 4.3).

## 6. Fuentes citadas

- `docs/_fuentes/XAI.pdf`:
  - pp. 136-143 — panorama introductorio de algoritmos de XAI (modelo
    surrogado, SHAP, LIME, análisis What-If) y taxonomía Ante-Hoc/Post-Hoc,
    agnóstico/específico.
  - pp. 145-146 — cabecera de la sección "Métodos Agnósticos al Modelo", bajo
    la que se sitúan PDP/ICE/ALE, permutación y modelo subrogado.
  - pp. 147-165 — PDP, ICE (e ICE centrado), ALE: revisadas por completitud del
    rango de páginas encargado, mencionadas brevemente en la sección 2 pero no
    desarrolladas (no son el tema de este documento).
  - pp. 166-175 — importancia por permutación: ídem, mencionada brevemente en
    la sección 2.
  - **pp. 176-180 — Modelo Subrogado Global: sección núcleo de este documento**
    (definición con diagrama en dos pasos, ejemplo gradient boosting→árbol,
    ventajas, desventajas y la pregunta sobre nivel de ajuste/fidelidad).
  - p. 218 — número de fidelidad concreto ("modelo surrogado, explica el 95%
    del LSTM") del ejemplo de predicción de demanda, situado formalmente en el
    rango de páginas de "Explicaciones Basadas en Ejemplos" pero
    conceptualmente parte del tema de modelo subrogado (ver también
    `docs/teoria/contrafactuals.md`, que remite aquí para este mismo ejemplo).
  - Extraído con PyMuPDF (`fitz`) por exceder el PDF el límite de tamaño del
    tool `Read` y no estar disponible `poppler` para lectura por rango de
    páginas.
- `docs/_fuentes/taller_XAI.pdf` — enunciado oficial de la práctica, apartado
  "1. Objetivo de la práctica": requisito explícito de "Modelo subrogado con
  reglas que lo expliquen" y los dos escenarios de coste (FP=FN=1 y FP=FN=10).
- `docs/_fuentes/ejercicio1_AR.ipynb` y
  `docs/_fuentes/ejercicio2_proyecto_XAI.ipynb` (código idéntico en la parte de
  modelo subrogado): patrón completo de código de ejemplo del profesor —
  comentario "Gemelo digital... caja blanca", generación de
  `y_predicha_caja_negra`, entrenamiento de `DecisionTreeClassifier
  (max_leaf_nodes=4)`, medición de fidelidad con `.score()` ("cómo de buena es
  la réplica"), y extracción de reglas con `get_rules_from_tree` (11 reglas
  con soporte y pureza por hoja).
- `docs/_fuentes/XAI_predicción_series_temporales_LSTM.ipynb` — segundo
  ejemplo del patrón de subrogado (árbol de regresión sobre un LSTM), con
  definición explícita en markdown, umbral de fidelidad objetivo (R²=0.9)
  medido contra las propias predicciones del LSTM (`y_LSTM`), búsqueda
  iterativa del número de hojas, y `get_rules_from_regression_tree`
  (equivalente de `get_rules_from_tree` para árboles de regresión). No trata
  el dataset de crédito de este taller; se cita por ser el mismo patrón
  metodológico aplicado a otro dominio, y por dar un número de fidelidad que
  coincide con el de `XAI.pdf`, p. 218.
- `docs/_fuentes/my_library.py.txt` — funciones `get_rules_from_tree` (extrae
  reglas if/then con soporte y pureza por hoja de un `DecisionTreeClassifier`
  ya entrenado) y `train_val_test_split` (utilidad genérica de partición de
  datos, no específica del subrogado). Documentadas aquí de forma conceptual,
  sin transcribir el código literal.
- `docs/_fuentes/libreria_aux_arboles.py.txt` — funciones `tree_to_code`
  (convierte el árbol en una función Python ejecutable) y `tree_to_pseudo`
  (convierte el árbol en pseudocódigo if/then con llaves). Documentadas de
  forma conceptual, sin transcribir el código literal.
- `docs/_fuentes/DataDictionary.csv` — nombres y descripción de las variables
  del dataset de crédito, usadas para ilustrar (sección 4.3) cómo se vería una
  regla del subrogado aplicada a este dataset (ejemplo ilustrativo propio, no
  una regla real extraída de un modelo).
- `docs/_fuentes/INVENTARIO.md` — confirma el mapa de páginas de `XAI.pdf`, la
  clasificación de los notebooks de ejercicio y la nota de que
  `ejercicio2_proyecto_XAI.ipynb` carga el dataset real de crédito en una
  celda de EDA inicial sin aplicar todavía el resto del patrón (modelado,
  bandit, subrogado) sobre él.
- `docs/DECISIONES.md` — decisiones D-0.1, D-0.2 y D-0.3 (abiertas en el
  momento de redactar este documento), citadas solo como contexto de proyecto
  para la sección 4.2 (relación entre escenarios de coste y necesidad de
  reentrenar el subrogado), no como fuente teórica del modelo subrogado.

Los ficheros `00-fundamentos-dependencia.md`, `01-capa-custom.md`,
`02-fair-loss.md`, `03-keras-tuner.md`, `04-incertidumbre.md` y
`05-contexto-confiabilidad.md` (`docs/_fuentes/`) se revisaron (búsqueda de
términos "subrogado", "surrogate", "árbol", "regla"/"rule", "tree",
"interpretable") por completitud, pero pertenecen a otro taller
(fairness/incertidumbre en redes, dataset `CODE_GENDER`) y no contienen
ninguna idea genuinamente aplicable al modelo subrogado ni a la extracción de
reglas de un árbol; las únicas coincidencias fueron falsos positivos de la
palabra "regla" en el sentido de "regla práctica" genérica, no reglas de un
árbol de decisión. No se han usado como fuente de este documento.
