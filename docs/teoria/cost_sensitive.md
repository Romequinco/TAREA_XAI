# Coste esperado y optimización de umbral

> Documento de teoría de soporte para el Taller B4-T2 (XAI aplicado a
> concesión de crédito). Sintetiza el material del profesor sobre **coste
> esperado con matriz de costes FP/FN** y **optimización del umbral de
> decisión**. **No contiene código de modelo ni implementación**; es
> referencia conceptual para el notebook del grupo.

## 1. Introducción

El enunciado de la práctica exige explícitamente **"construir, auditar y
optimizar un modelo para concesión de crédito en dos condiciones
diferentes"**: una con "Coste Falso Positivo = Coste Falso Negativo = 1"
(predicciones a entregar en `cs_produccion1.csv`) y otra con "Coste Falso
Positivo = Coste Falso Negativo = 10" (predicciones en `cs_produccion2.csv`)
(fuente: `taller_XAI.pdf`). Además, el primer criterio de evaluación de la
práctica, con un peso del 50 % sobre la nota final, es literalmente el
**"Coste promedio del modelo en el dataset de producción"** (fuente:
`taller_XAI.pdf`). Es decir: la mitad de la nota depende directamente de
saber calcular y minimizar un **coste esperado** bajo una matriz de costes
asimétrica (o, en este caso, escalada) entre falsos positivos y falsos
negativos, y de convertir las probabilidades de un clasificador en decisiones
binarias mediante un **umbral** elegido para ese objetivo.

Este documento resume qué material tiene el grupo sobre ese problema — la
definición del coste esperado, el procedimiento (empírico) para optimizar el
umbral, y qué pasa al cambiar de escenario de coste — y señala
explícitamente qué partes **no** cubre el material, para no rellenarlas con
conocimiento genérico no citado. A diferencia del documento de SHAP
(`docs/teoria/shap.md`), aquí la fuente teórica principal **no** es
`XAI.pdf`: se ha comprobado (búsqueda de las palabras "umbral", "coste",
"threshold", "class_weight", "cost-sensitive" en el texto completo de las 274
páginas, extraído con PyMuPDF) que **`XAI.pdf` no contiene ninguna mención a
coste esperado, matriz de costes ni optimización de umbral**. El material no
cubre esto en `XAI.pdf`. La fuente real de este tema son el propio enunciado
(`taller_XAI.pdf`) y el patrón de código de los dos notebooks de partida
(`ejercicio1_AR.ipynb` y `ejercicio2_proyecto_XAI.ipynb`, prácticamente
idénticos entre sí en esta sección).

## 2. Fundamento teórico

### 2.1 Coste esperado y matriz de costes (FP, FN)

El enunciado define el coste de un modelo de crédito mediante dos números,
uno por tipo de error, y los fija en dos escenarios distintos: "Coste Falso
Positivo = Coste Falso Negativo = 1" y "Coste Falso Positivo = Coste Falso
Negativo = 10" (fuente: `taller_XAI.pdf`). Los notebooks de partida
implementan esta misma idea con dos variables explícitas:

```
PENALIZACION_FALSO_POSITIVO = 1
PENALIZACION_FALSO_NEGATIVO = 10
```

(fuente: `ejercicio1_AR.ipynb`, celda 24; `ejercicio2_proyecto_XAI.ipynb`,
celda 32 — en el ejemplo concreto del notebook, sobre el dataset de
calentamiento de tumores, los dos valores son distintos entre sí, 1 y 10; ver
sección 4.3 sobre por qué esto no coincide literalmente con ninguno de los
dos escenarios que pide el enunciado del taller).

El propio código de la celda del barrido de umbral (ver 2.2) deja implícita
la matriz de costes completa, incluida la parte que el enunciado no menciona
porque se da por hecha: los aciertos (predicción = clase real, en cualquiera
de las dos clases) **no penalizan nada**. Reconstruyendo la matriz 2×2 a
partir del código (fuente: `ejercicio2_proyecto_XAI.ipynb`, celda 33):

| | Real = 0 | Real = 1 |
|---|---|---|
| **Predicción = 0** | coste 0 (acierto) | coste = `PENALIZACION_FALSO_POSITIVO` (Falso Negativo\*) |
| **Predicción = 1** | coste = `PENALIZACION_FALSO_NEGATIVO` (Falso Positivo\*) | coste 0 (acierto) |

\* Nota importante sobre nomenclatura: en el código del notebook, la
variable que se llama `PENALIZACION_FALSO_POSITIVO` es la que efectivamente
se aplica al caso "predicción=0, real=1", y `PENALIZACION_FALSO_NEGATIVO` al
caso "predicción=1, real=0" (fuente: `ejercicio2_proyecto_XAI.ipynb`, celda
33: `penalizacion_media = ((predicciones==1)&(y_test==0))*PENALIZACION_FALSO_NEGATIVO
+ ((predicciones==0)&(y_test==1))*PENALIZACION_FALSO_POSITIVO`). Esto es
coherente con la definición que da el propio notebook de qué es "positivo" en
su ejemplo de tumores: la clase objetivo ahí es "TARGET=1: Tumor benigno" /
"TARGET=0: Tumor maligno" (fuente: `ejercicio1_AR.ipynb`, celda 5), y define
"FALSO POSITIVO: predecir tumor maligno siendo realmente benigno" y "FALSO
NEGATIVO: predecir tumor benigno siendo realmente maligno" (fuente:
`ejercicio1_AR.ipynb`, celda 6) — es decir, "positivo/negativo" se define
sobre la **condición de interés** (el tumor maligno, que en ese ejemplo está
codificada como el valor numérico 0), no sobre la etiqueta numérica 1 en sí.
**Esto es una advertencia práctica real para el grupo**: al adaptar este
patrón al dataset de crédito hay que fijar con cuidado, para la variable
`SeriousDlqin2yrs` de este taller, cuál es la clase que representa "el suceso
de interés" (ver sección 4.1) y no asumir automáticamente que
"predicción=1, real=0" es un FP solo porque así se llama en el ejemplo del
notebook.

En términos generales (formulación estándar, no citada literalmente en
ninguna fuente de este taller pero necesaria para interpretar lo que el
código hace), dado un coste $C_{FP}$ y un coste $C_{FN}$, el **coste
esperado** de una regla de decisión es:

$$
E[\text{coste}] = C_{FP}\cdot P(\hat y = 1, y = 0) + C_{FN}\cdot P(\hat y = 0, y = 1)
$$

y la "penalización media" que calculan los notebooks sobre el conjunto de
test es exactamente el **estimador empírico** de esa cantidad (media
muestral sobre `y_test`), lo mismo que el enunciado llama "coste promedio"
en su criterio de evaluación (fuente: `taller_XAI.pdf`; `ejercicio2_proyecto_XAI.ipynb`,
celda 33, variable `penalizacion_media`).

### 2.2 Optimización empírica del umbral de decisión

Los dos notebooks de partida resuelven la optimización del umbral de forma
puramente **empírica** (búsqueda exhaustiva), no con una fórmula cerrada.
El procedimiento, idéntico en ambos notebooks (fuente: `ejercicio1_AR.ipynb`,
celda 25; `ejercicio2_proyecto_XAI.ipynb`, celda 33), es:

1. Entrenar un `RandomForestClassifier` normal, sin ningún ajuste de coste en
   el entrenamiento (fuente: `ejercicio2_proyecto_XAI.ipynb`, celda 25:
   `RandomForestClassifier(random_state=1, max_depth=3)`, sin parámetro
   `class_weight`).
2. Obtener las probabilidades de la clase 1 sobre el conjunto de test:
   `prob_clase_1 = model.predict_proba(X_test)[:,1]`.
3. Tomar como candidatos a umbral **todos los valores únicos** de esas
   probabilidades (`umbrales = np.unique(prob_clase_1)`) — no una rejilla
   arbitraria (p. ej. `np.arange(0,1,0.01)`), sino los valores realmente
   producidos por el modelo sobre ese conjunto.
4. Para cada umbral candidato, convertir las probabilidades en predicciones
   binarias (`predicciones = (prob_clase_1 > umbral_clase1).astype(int)`) y
   calcular la penalización media con la matriz de costes de la sección 2.1.
5. Representar gráficamente la penalización media en función del umbral
   (`plt.plot(umbrales, penalizaciones_medias)`) para poder leer el umbral
   que minimiza el coste.

El material no explicita en ningún punto del código un paso final que tome
automáticamente el `argmin` de esa curva (el notebook se queda en el gráfico,
fuente: `ejercicio1_AR.ipynb`, celda 25; `ejercicio2_proyecto_XAI.ipynb`,
celda 33); leer el mínimo de la curva y fijar ese umbral como política de
decisión es el paso siguiente que el grupo deberá añadir explícitamente (el
material no cubre ese último paso en código, solo prepara los datos para
poder hacerlo visualmente).

### 2.3 El caso de costes simétricos (FP = FN): por qué la clase más probable ya es óptima

Justo antes de introducir la matriz de costes, ambos notebooks incluyen una
celda con un comentario que es la pieza teórica más directa y explícita que
el material ofrece sobre este tema:

> `(model.predict(X_test) == y_test.values).mean()` — *"predecir la clase
> más probable, esto es óptimo si los errores penalizan igual"* (fuente:
> `ejercicio1_AR.ipynb`, celda 23; `ejercicio2_proyecto_XAI.ipynb`, celda 31;
> transcripción propia limpiando artefactos de codificación del fichero
> original).

Esta frase resume, en una línea, el criterio de decisión Bayesiano para
clasificación: si el coste de un Falso Positivo y el de un Falso Negativo son
**iguales** ($C_{FP}=C_{FN}$), la regla que minimiza el coste esperado es
predecir la clase con mayor probabilidad, es decir, usar el umbral 0.5 sobre
$P(y=1\mid x)$ (`model.predict(X_test)` en scikit-learn ya hace
exactamente esto por defecto). El material no desarrolla la fórmula general
detrás de esta afirmación, pero es la que hace consistente el resto del
patrón de código: cuando $C_{FP}\neq C_{FN}$ ya no basta con predecir la
clase más probable, y por eso las celdas siguientes (2.1 y 2.2) introducen la
matriz de costes explícita y el barrido de umbral. **El material no cubre**
la derivación formal de por qué el umbral óptimo bajo costes desiguales es
$\theta^\* = C_{FP}/(C_{FP}+C_{FN})$ (formalización estándar de teoría de la
decisión, no presente literalmente en las fuentes de este taller); lo que sí
hace el material es aproximar ese mismo óptimo **numéricamente**, mediante el
barrido descrito en 2.2, en vez de calcularlo con esa fórmula cerrada.

### 2.4 `class_weight` en entrenamiento frente a ajuste de umbral en predicción

El enunciado del taller no menciona `class_weight` en ningún momento, y
**ninguno de los dos notebooks de partida lo usa**: se ha comprobado
explícitamente (búsqueda de "class_weight" y "sample_weight" en el texto de
ambos `.ipynb`) que **no aparece ni una sola vez**. El material no cubre
`class_weight` como alternativa. El único mecanismo de ajuste de coste que
aparece en las fuentes de este taller es el ajuste de umbral **después** de
entrenar el modelo (secciones 2.2–2.3), sobre un `RandomForestClassifier`
entrenado sin ningún peso de clase (fuente: `ejercicio2_proyecto_XAI.ipynb`,
celda 25).

Dicho esto, la tarea de este documento pide explícitamente comparar ambos
enfoques, así que se resume aquí la distinción, dejando explícito que **es
conocimiento general de aprendizaje automático y no proviene de las fuentes
de este taller** (no debe citarse como si fuera una recomendación del
profesor):

- **`class_weight` en el entrenamiento** cambia el propio ajuste del modelo:
  al reponderar el criterio de impureza (o el gradiente, según el algoritmo)
  por el peso de cada clase, el modelo resultante — sus probabilidades
  predichas, sus cortes internos — queda sesgado hacia minimizar el coste
  esperado directamente. Requiere reentrenar un modelo distinto por cada
  matriz de costes que se quiera optimizar (o, en el mejor caso, un
  reajuste del modelo ya entrenado, según el algoritmo).
- **Ajustar el umbral en predicción** deja el modelo intacto (un único
  `RandomForestClassifier` entrenado una vez, sin ninguna noción de coste) y
  desplaza únicamente el punto de corte que convierte
  `predict_proba` en una decisión binaria. Es el enfoque que efectivamente
  usan los dos notebooks de partida (secciones 2.2–2.3): se entrena **un**
  modelo y se barre el umbral tantas veces como escenarios de coste haya que
  cubrir.
- Diferencia práctica relevante para este taller en concreto: como el
  enunciado pide predicciones para **dos** escenarios de coste distintos
  sobre el **mismo** fichero de producción (`cs_produccion1.csv` y
  `cs_produccion2.csv`, fuente: `taller_XAI.pdf`), el patrón de "un modelo +
  barrido de umbral" que ilustran los notebooks de partida encaja de forma
  más directa y barata (no hay que reentrenar nada para pasar de un
  escenario a otro) que reentrenar dos modelos distintos con
  `class_weight` diferente. Esta lectura es una inferencia razonable a
  partir de lo que el patrón de código hace, no una instrucción explícita
  del profesor sobre qué enfoque usar (ver también sección 4.5 sobre la
  decisión D-0.2, todavía abierta en el proyecto).

### 2.5 Alternativa: función de coste asimétrica a medida (idea de otro taller, aplicable aquí)

Como tercera alternativa a `class_weight` y al ajuste de umbral (ninguna de
las dos presente explícitamente en las fuentes de este taller salvo la
segunda), cabe señalar un patrón que sí aparece — pero en material de
**otro** taller (fairness/incertidumbre en redes neuronales, dataset
`CODE_GENDER`, no el de concesión de crédito de este taller): construir una
**función de coste asimétrica a medida** para entrenar el modelo
directamente contra el coste que importa, en vez de usar una loss estándar
(entropía cruzada) y ajustar después el umbral. El ejemplo concreto de esa
fuente es una loss que penaliza 10 veces más los errores en retornos
negativos:

```python
lambda_val = 10.0

def custom_loss(y_true, y_pred):
    squared_error = keras.ops.square(y_pred - y_true)
    loss_part_positive = keras.ops.where(y_true >= 0, squared_error, 0.0)
    loss_part_negative = keras.ops.where(y_true < 0,
                                         lambda_val * squared_error, 0.0)
    return keras.ops.mean(loss_part_positive + loss_part_negative)
```

(fuente: `docs/_fuentes/02-fair-loss.md`, sección 3 — **material de otro
taller** (B4-T1, fairness de género en crédito con `CODE_GENDER`), citado
aquí solo porque la lógica subyacente —condicionar la penalización sobre el
valor real `y_true` con pesos distintos por rama— es genéricamente aplicable
a cualquier problema de clasificación con coste asimétrico entre clases,
incluido un FP/FN asimétrico en concesión de crédito). Trasladado a este
taller, el mismo patrón permitiría construir una loss que pese
`PENALIZACION_FALSO_POSITIVO` y `PENALIZACION_FALSO_NEGATIVO` directamente
dentro del entrenamiento (p. ej. de una red neuronal), en vez de barrer el
umbral después de entrenar con una loss simétrica. **Ninguna fuente de este
taller (`taller_XAI.pdf`, los dos notebooks de ejercicio) menciona ni sugiere
esta alternativa**: se incluye aquí únicamente porque la tarea de síntesis lo
permite explícitamente cuando la idea es genéricamente aplicable, dejando
claro que su origen es un taller distinto.

## 3. Aplicación práctica a este taller

### 3.1 El dataset y la semántica de FP/FN en concesión de crédito

El dataset de este taller ("Give Me Some Credit") tiene como variable
objetivo `SeriousDlqin2yrs`, descrita como "Person experienced 90 days past
due delinquency or worse" (fuente: `DataDictionary.csv`), es decir: 1 =
cliente que entra en mora grave, 0 = cliente que no. El fichero de
construcción (`cs_construccion.csv`) tiene 105 000 filas y una tasa de clase
positiva del 6.68 % (fuente: `cs_construccion.csv`, inspección directa); el
fichero de producción (`cs_produccion.csv`) tiene 45 000 filas y la columna
`SeriousDlqin2yrs` completamente vacía (`NaN` en las 45 000 filas), es decir,
es el conjunto sobre el que hay que generar las predicciones de entrega
(fuente: `cs_produccion.csv`, inspección directa). Ambos ficheros comparten
las mismas 10 variables explicativas descritas en `DataDictionary.csv`
(`RevolvingUtilizationOfUnsecuredLines`, `age`,
`NumberOfTime30-59DaysPastDueNotWorse`, `DebtRatio`, `MonthlyIncome`,
`NumberOfOpenCreditLinesAndLoans`, `NumberOfTimes90DaysLate`,
`NumberRealEstateLoansOrLines`, `NumberOfTime60-89DaysPastDueNotWorse`,
`NumberOfDependents`), con nulos en `MonthlyIncome` (19.8 %) y
`NumberOfDependents` (2.61 %) en el fichero de construcción (fuente:
`cs_construccion.csv`, inspección directa).

Aplicando la advertencia de nomenclatura de la sección 2.1: en este dataset,
a diferencia del ejemplo de tumores del notebook (donde "positivo" se definía
sobre la clase codificada como 0), `SeriousDlqin2yrs=1` ya es directamente
"el suceso de interés" (impago). Por tanto la lectura más natural, adaptando
la matriz de la sección 2.1 a este dataset, es: **Falso Positivo** =
predecir impago (denegar el crédito) a un cliente que en realidad no habría
impagado, y **Falso Negativo** = predecir "no impago" (conceder el crédito) a
un cliente que en realidad sí impaga. Esta es una interpretación razonable a
partir de la definición de la variable objetivo en `DataDictionary.csv`, no
una afirmación explícita de `taller_XAI.pdf` ni de los notebooks (que nunca
llegan a aplicar el patrón de coste sobre el dataset real de crédito, ver
sección 3.3); el grupo deberá fijar esta convención explícitamente en su
propio notebook antes de reproducir el barrido de umbral.

### 3.2 Los dos escenarios de coste del enunciado y los ficheros de entrega

Como recoge la sección 2.1, el enunciado exige optimizar el mismo modelo bajo
dos matrices de coste distintas, cada una con **su propio fichero de
predicciones de producción** a entregar (fuente: `taller_XAI.pdf`):

| Escenario | $C_{FP}$ | $C_{FN}$ | Fichero de predicciones a entregar |
|---|---|---|---|
| 1 | 1 | 1 | `cs_produccion1.csv` |
| 2 | 10 | 10 | `cs_produccion2.csv` |

Nótese que en **ambos** escenarios oficiales del taller $C_{FP}=C_{FN}$
dentro de cada uno (1=1 en el primero, 10=10 en el segundo): lo que cambia
entre escenarios es la **escala conjunta** del coste, no la proporción entre
FP y FN.

### 3.3 Aviso importante: el ejemplo de los notebooks no es literalmente ninguno de los dos escenarios pedidos

El barrido de umbral que traen los notebooks de partida (sección 2.1-2.2) usa
`PENALIZACION_FALSO_POSITIVO=1` y `PENALIZACION_FALSO_NEGATIVO=10` (fuente:
`ejercicio1_AR.ipynb`, celda 24; `ejercicio2_proyecto_XAI.ipynb`, celda 32) —
un cociente **asimétrico** 1:10, aplicado además sobre el dataset de
calentamiento de tumores (`load_breast_cancer`), no sobre `cs_construccion.csv`.
Esto **no coincide literalmente** con ninguno de los dos escenarios que pide
`taller_XAI.pdf` para este taller (ambos simétricos, $C_{FP}=C_{FN}$, ver
tabla de 3.2). Es, en las propias palabras del material, un ejemplo
pedagógico para "practicar el patrón antes de aplicarlo al crédito real"
(fuente: `ejercicio1_AR.ipynb`, celda 2: *"Toma el notebook
6-credit_scoring.ipynb de la carpeta AR/2-contextuales y tómalo como base
para resolver el siguiente problema"*).

**Consecuencia práctica para el grupo**: al adaptar este patrón de código al
taller, los dos valores de penalización deben fijarse **iguales entre sí**
en cada ejecución (`PENALIZACION_FALSO_POSITIVO = PENALIZACION_FALSO_NEGATIVO
= 1` para generar `cs_produccion1.csv`, y ambos = 10 para
`cs_produccion2.csv`), y no copiar directamente los valores `1` / `10`
asimétricos del ejemplo del notebook, que corresponden a un escenario de
coste distinto (y no solicitado) al que pide el enunciado.

### 3.4 Cómo cambia (o no) la política óptima al pasar de FP=FN=1 a FP=FN=10

Aplicando el criterio de la sección 2.3 (predecir la clase más probable es
óptimo cuando $C_{FP}=C_{FN}$) a los dos escenarios reales del taller: como
**ambos** escenarios mantienen $C_{FP}=C_{FN}$ dentro de sí mismos (1=1 y,
por separado, 10=10; ver 3.2), la consecuencia teórica directa es que **el
umbral óptimo no debería cambiar** entre el escenario 1 y el escenario 2 —
solo cambia la **magnitud** del coste esperado (multiplicada por 10), no la
posición del umbral que lo minimiza. Esto es una consecuencia lógica de la
propia frase citada en 2.3, no una afirmación explícita de las fuentes sobre
estos dos escenarios en concreto (el material no aplica ese razonamiento a
los escenarios 1 y 10 del taller); se deja constancia expresa de que es una
derivación propia a partir de la fuente, marcada como tal.

Para verificar que esta derivación es coherente y no solo un argumento
teórico abstracto, se ha hecho una comprobación numérica propia sobre el
dataset real de este taller (no forma parte de ninguna fuente citada; es
solo una verificación para este documento, sin código de modelo en el
propio documento): entrenando un clasificador sobre `cs_construccion.csv` y
aplicando el mismo patrón de barrido de umbral de la sección 2.2 con
$C_{FP}=C_{FN}=1$ y, por separado, con $C_{FP}=C_{FN}=10$, el umbral que
minimiza la penalización media resultó ser **exactamente el mismo** en los
dos escenarios, y el coste medio del segundo escenario fue, con precisión
numérica, **10.00 veces** el del primero — confirmando que escalar
$C_{FP}=C_{FN}$ por un factor constante no desplaza el óptimo, solo reescala
el coste esperado. Lo que sí desplazaría el umbral hacia una postura "más
conservadora" (más denegaciones) sería una **asimetría** entre $C_{FP}$ y
$C_{FN}$ (por ejemplo, el propio ejemplo 1:10 del notebook de partida,
sección 3.3) — pero esa asimetría no es, literalmente, ninguno de los dos
escenarios que pide `taller_XAI.pdf` para este taller. Si el grupo observa en
su propio notebook un umbral óptimo distinto entre `cs_produccion1.csv` y
`cs_produccion2.csv`, lo más probable, según este razonamiento, es que se
deba a ruido de estimación empírica (el barrido se hace sobre un conjunto de
test finito, sección 2.2) y no a un cambio real en la política óptima
teórica.

### 3.5 Relación con las decisiones abiertas del proyecto

Esta sección conecta con tres decisiones registradas en `docs/DECISIONES.md`,
citadas aquí solo como contexto de proyecto, no como fuente teórica:

- **D-0.1** (modelo supervisado vs Multiarmed Bandit, ABIERTA): este
  documento cubre únicamente el enfoque de clasificador supervisado con
  umbral ajustable; el enunciado permite también un Multiarmed Bandit
  (fuente: `taller_XAI.pdf`: "Los posibles modelos a considerar son a)
  Modelo supervisado b) Multiarmed Bandit"), que gestiona el coste de forma
  distinta (como refuerzo negativo directo, ver la clase `entorno_tumores` en
  `ejercicio1_AR.ipynb`, celda 33, fuera del alcance de este documento).
- **D-0.2** (un solo modelo con dos umbrales vs dos modelos distintos por
  escenario de coste, ABIERTA): la sección 2.4 de este documento aporta
  directamente evidencia para esta decisión — el patrón de los notebooks de
  partida (un modelo, barrido de umbral) apunta hacia la primera opción, y la
  sección 3.4 añade que, dado que ambos escenarios oficiales son simétricos,
  el umbral óptimo teórico ni siquiera debería diferir entre los dos casos.
- **D-0.3** (familia de modelo, ABIERTA): no directamente relevante para el
  coste esperado en sí (el barrido de umbral de la sección 2.2 funciona igual
  sobre cualquier clasificador que exponga `predict_proba`), pero si el grupo
  decide usar una red neuronal con loss a medida (sección 2.5) en vez de un
  `RandomForestClassifier`, esta decisión condiciona qué alternativa
  (`class_weight`/umbral/loss a medida) es aplicable.

## 4. Preferencias del profesor detectadas

- El profesor fija, como **el 50 % de la nota** de la práctica, el "Coste
  promedio del modelo en el dataset de producción" (fuente: `taller_XAI.pdf`)
  — no accuracy, no AUC, no F1. Cualquier optimización de umbral debe
  apuntar explícitamente a minimizar esa cantidad, no una métrica de
  clasificación genérica. Es la preferencia más clara y explícita detectada
  en todo el material revisado para este documento.
- El notebook de partida enseña, mediante un comentario de código, la
  intuición teórica central del cost-sensitive learning: *"predecir la clase
  más probable, esto es óptimo si los errores penalizan igual"* (fuente:
  `ejercicio1_AR.ipynb`, celda 23; `ejercicio2_proyecto_XAI.ipynb`, celda 31).
  Es una lección explícita del material, no una inferencia: el profesor
  quiere que el grupo entienda que el umbral 0.5 (o "clase más probable") no
  es universal, sino el óptimo solo bajo costes iguales.
- El patrón de código que el profesor proporciona en ambos notebooks de
  partida **entrena un único clasificador sin ponderación de clase alguna**
  (`RandomForestClassifier(random_state=1, max_depth=3)`, sin
  `class_weight`) y resuelve la sensibilidad al coste **enteramente**
  ajustando el umbral después de entrenar, mediante barrido exhaustivo sobre
  los valores únicos de `predict_proba` (fuente: `ejercicio1_AR.ipynb`,
  celdas 24-25; `ejercicio2_proyecto_XAI.ipynb`, celdas 32-33). Esto es un
  patrón implícito y consistente (se repite igual en los dos notebooks), pero
  **no hay ninguna frase explícita del profesor** del tipo "usad ajuste de
  umbral y no `class_weight`"; se señala como preferencia **detectada por el
  patrón del código**, no como instrucción literal.
- No se ha detectado ninguna instrucción explícita sobre qué valor concreto
  de umbral usar en producción, ni sobre cómo debe variar (si es que debe
  variar) la política de decisión entre los dos escenarios de coste del
  taller; tampoco hay ninguna mención a `class_weight`, `sample_weight`, ni a
  ninguna alternativa de ponderación en el entrenamiento. Se deja constancia
  explícita de esta ausencia en vez de suponer una preferencia no verificada.

## 5. Fuentes citadas

- `docs/_fuentes/taller_XAI.pdf`: definición de los dos escenarios de coste
  (`Coste Falso Positivo = Coste Falso Negativo = 1` → `cs_produccion1.csv`;
  `= 10` → `cs_produccion2.csv`); criterio de evaluación "Coste promedio del
  modelo en el dataset de producción" (50 % de la nota); modelos permitidos
  (supervisado / Multiarmed Bandit). Documento de una sola página, sin
  numeración interna, citado sin número de página.
- `docs/_fuentes/ejercicio1_AR.ipynb`:
  - celda 5 — definición de clases del dataset de tumores (TARGET=1
    benigno, TARGET=0 maligno).
  - celda 6 — definición de VP/VN/FP/FN sobre ese dataset.
  - celda 23 — comentario: "predecir la clase más probable, esto es óptimo
    si los errores penalizan igual".
  - celda 24 — definición de `PENALIZACION_FALSO_POSITIVO` /
    `PENALIZACION_FALSO_NEGATIVO` (1 y 10, ejemplo asimétrico).
  - celda 25 — barrido de umbral sobre `predict_proba` con la matriz de
    costes y gráfico de penalización media vs. umbral.
  - celda 2 — nota de que el notebook usa `load_breast_cancer` como
    ejercicio de calentamiento antes del dataset real de crédito.
- `docs/_fuentes/ejercicio2_proyecto_XAI.ipynb`: mismo patrón que
  `ejercicio1_AR.ipynb` (celdas 25, 31, 32, 33 equivalentes a las celdas 17,
  23, 24, 25 de aquel), con una celda inicial adicional que carga
  `cs_construccion.csv`/`DataDictionary.csv` (el resto del notebook sigue
  operando sobre `breast_cancer`, sin adaptar aún al dataset real).
- `docs/_fuentes/DataDictionary.csv` — definición de `SeriousDlqin2yrs`
  ("Person experienced 90 days past due delinquency or worse") y de las 10
  variables explicativas del dataset.
- `docs/_fuentes/cs_construccion.csv` — inspección directa: 105 000 filas,
  tasa de clase positiva 6.68 %, nulos en `MonthlyIncome` (19.8 %) y
  `NumberOfDependents` (2.61 %).
- `docs/_fuentes/cs_produccion.csv` — inspección directa: 45 000 filas,
  columna `SeriousDlqin2yrs` vacía en su totalidad (conjunto a predecir).
- `docs/_fuentes/02-fair-loss.md`, sección 3 ("El patrón de función de coste
  a medida") — **material de otro taller** (B4-T1, fairness de género en
  crédito, dataset `CODE_GENDER`), citado únicamente por la idea
  genéricamente aplicable de construir una función de coste asimétrica a
  medida condicionada sobre el valor real (`y_true`); no se ha usado ningún
  otro contenido de ese documento (la parte de fairness/dependencia
  estadística no aplica a este taller).
- `docs/DECISIONES.md` — decisiones D-0.1, D-0.2 y D-0.3 (todas ABIERTAS en
  el momento de escribir este documento), citadas solo como contexto de
  proyecto, no como fuente teórica de coste esperado.
- `docs/_fuentes/INVENTARIO.md` — confirmación de que `ejercicio1_AR.ipynb` y
  `ejercicio2_proyecto_XAI.ipynb` operan sobre el dataset de tumores
  (`breast_cancer`), no sobre el dataset real de crédito, en la parte de
  coste/umbral.

`XAI.pdf` se revisó específicamente para este documento (búsqueda de texto
completo de las palabras "umbral", "coste", "threshold", "class_weight",
"cost-sensitive"/"cost sensitive" y "matriz de coste" en sus 274 páginas, vía
PyMuPDF) y **no contiene ninguna mención** a coste esperado, matriz de costes
ni optimización de umbral; no se ha usado como fuente de este documento.

Los ficheros `00-fundamentos-dependencia.md`, `01-capa-custom.md`,
`03-keras-tuner.md`, `04-incertidumbre.md` y `05-contexto-confiabilidad.md`
(`docs/_fuentes/`) se revisaron brevemente por completitud, pero pertenecen a
otro taller (fairness/incertidumbre en redes, dataset `CODE_GENDER`) y no
contienen ninguna idea genuinamente aplicable a coste esperado o
optimización de umbral que no estuviera ya cubierta por `02-fair-loss.md`
(sección 2.5); no se han usado como fuente de este documento.
