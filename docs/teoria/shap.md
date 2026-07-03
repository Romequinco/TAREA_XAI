# SHAP (SHapley Additive exPlanations): fundamento global y local

> Documento de teoría de soporte para el Taller B4-T2 (XAI aplicado a
> concesión de crédito). Sintetiza el material del profesor sobre valores de
> Shapley/SHAP. **No contiene código ni implementación**; es referencia
> conceptual para el notebook del grupo.

## 1. Introducción

El enunciado de la práctica exige explícitamente, como parte de la auditoría
del modelo de concesión de crédito, un **"Análisis SHAP global y local"**,
junto con un modelo subrogado con reglas y un análisis de contrafactuales
(fuente: `taller_XAI.pdf`). Este documento resume qué dice el material del
profesor sobre SHAP — su fundamento teórico, cómo se interpreta a nivel local
(una predicción concreta) y qué implica a nivel global (importancia de
variables) — y señala explícitamente los puntos que el material **no** cubre,
para no rellenarlos con conocimiento genérico no citado.

La fuente principal es `XAI.pdf` (slides del profesor, Manuel
Sánchez-Montañés, UAM), concretamente la sección **"Valores de Shapley / SHAP"**
en las páginas 200-204 (5 diapositivas). Se ha revisado también la sección de
LIME (pp. 181-199) del mismo documento para poder contrastar ambos métodos,
tal como el propio material hace en su página final sobre SHAP.

## 2. Fundamento teórico de SHAP

### 2.1 Origen: teoría de juegos de coalición

El material sitúa el origen de los valores de Shapley en la **teoría de
juegos de coalición**: "La idea viene de teoría de juegos de coalición"
(fuente: XAI.pdf, p. 200). SHAP traslada esa idea a la explicación de modelos
de aprendizaje automático mediante la siguiente analogía, explicada punto por
punto en el material:

- **El juego**: consiste en predecir la variable objetivo (target) para un
  caso concreto (fuente: XAI.pdf, p. 201).
- **Los jugadores**: son los valores de las variables en el ejemplo concreto
  que se quiere explicar (instancia $X_p$); "colaboran" para conseguir su
  premio (fuente: XAI.pdf, pp. 200-201).
- **El premio a repartir**: es la predicción realizada por el modelo para ese
  caso (fuente: XAI.pdf, p. 200).
- **La ganancia**: se define como "el valor de la predicción menos la
  predicción media" (fuente: XAI.pdf, p. 201). Es decir, SHAP no reparte el
  valor absoluto de la predicción, sino la **diferencia respecto a la
  predicción promedio del dataset**.
- Los valores de Shapley (método de teoría de juegos de coalición) indican
  "cómo distribuir de manera justa el 'premio' entre las variables" (fuente:
  XAI.pdf, p. 200).

De forma más general, en el resumen panorámico de algoritmos de XAI (antes de
entrar en el detalle de SHAP) el material ya adelanta la idea en una frase:
"Valores SHAP: miden el impacto que tiene cada variable en la estimación del
modelo. Basadas en Teoría de Juegos. Muy usados en XAI" (fuente: XAI.pdf,
p. 140).

**Nota sobre lo que el material no cubre**: las diapositivas pp. 200-201 no
incluyen la fórmula matemática formal del valor de Shapley (la suma sobre
coaliciones/permutaciones de variables, ponderada por el tamaño de la
coalición) ni nombran explícitamente los axiomas formales (eficiencia,
simetría, dummy/nulidad, aditividad) que sí se mencionan en la literatura
general de SHAP. El material solo dice, de forma genérica, que "los axiomas en
los que está basado dan a la explicación una base razonable" (fuente:
XAI.pdf, p. 204), sin detallarlos. **El material no cubre esto** más allá de
esa mención genérica, así que no se debe asumir que el grupo puede citar
axiomas concretos como si vinieran de esta fuente.

### 2.2 Ejemplos visuales incluidos en el material

Las páginas 202 y 203 del material no contienen texto (son diapositivas de
solo imagen), pero muestran dos ejemplos gráficos de explicación local que
ilustran directamente la definición anterior (fuente: XAI.pdf, pp. 202-203):

- Un ejemplo con un modelo de riesgo (variables tipo `STDs`, `Number of
  sexual partners`, `Age`, etc.) donde se reporta: `Actual prediction: 0.57`,
  `Average prediction: 0.03`, `Difference: 0.54`, y un gráfico de barras
  horizontales con la "Feature value contribution" de cada variable, ordenadas
  de mayor a menor contribución.
- Un ejemplo de un modelo de predicción de alquiler de bicicletas, con la
  misma estructura: `Actual prediction: 2409`, `Average prediction: 4518`,
  `Difference: -2108`, y las contribuciones (positivas y negativas) de
  variables como `temp`, `hum`, `weathersit`, `windspeed`, etc.

En ambos ejemplos se observa que la suma de las contribuciones individuales de
las variables reconstruye exactamente la diferencia entre la predicción real
para ese caso y la predicción media del conjunto — es la ilustración práctica
de la definición de "ganancia" dada en la p. 201. El material no explicita el
nombre técnico de esta propiedad (aditividad/eficiencia), pero la muestra de
forma empírica en ambos gráficos (fuente: XAI.pdf, pp. 202-203).

### 2.3 Fortalezas y debilidades señaladas por el profesor

La última diapositiva de la sección resume explícitamente ventajas y
desventajas de SHAP (fuente: XAI.pdf, p. 204):

- "SHAP tiene una sólida base teórica en la teoría de juegos."
- "La predicción se distribuye equitativamente entre los valores de las
  características."
- "Es el único método de XAI con una teoría sólida."
- "Los axiomas en los que está basado dan a la explicación una base
  razonable."
- Contraste explícito con LIME: "Métodos como LIME asumen un comportamiento
  lineal del modelo de aprendizaje automático a nivel local, pero no existe
  una teoría que explique por qué debería funcionar."
- Limitación explícita: "Puede ser muy lento de calcular."

### 2.4 Contraste con LIME (para situar SHAP)

El material dedica una sección extensa a LIME (pp. 181-199) justo antes de la
de SHAP, lo que permite contrastar ambos enfoques con la propia fuente:

- LIME también es un método de explicación local ("Es un modelo subrogado,
  pero local. Explica la decisión del modelo black-box para un caso
  particular $X_p$", fuente: XAI.pdf, p. 181), pero funciona generando casos
  artificiales alrededor de $X_p$, entrenando un modelo interpretable (lineal,
  típicamente con Lasso) ponderado por cercanía a $X_p$, y usando ese modelo
  local para explicar la predicción (fuente: XAI.pdf, pp. 182-183).
- LIME depende críticamente de un hiperparámetro sensible (el ancho del
  kernel que pesa los puntos por proximidad a $X_p$) y el propio material
  advierte: "Hay que tener mucho cuidado seleccionando el kernel width" y "el
  análisis LIME es inestable. Si se vuelven a crear ejemplos generados, puede
  dar una respuesta diferente" (fuente: XAI.pdf, pp. 185-186 y 199).
- La conclusión del profesor sobre LIME es cautelosa: "LIME es un algoritmo
  prometedor. Pero el método está aún en fase de desarrollo y hay que
  resolver muchos problemas antes de que pueda aplicarse con seguridad"
  (fuente: XAI.pdf, p. 199).
- Frente a esto, SHAP se presenta como el método con base teórica más sólida
  (p. 204, citado arriba), aunque con mayor coste computacional.

Esta comparación explícita en la fuente justifica, para este taller, dar
preferencia a SHAP como método principal de explicación (local y global) y
usar LIME, si acaso, como método secundario/contraste, dado que el propio
material señala menos garantías teóricas y mayor inestabilidad para LIME.

## 3. Interpretación de un valor SHAP local

Este es el caso que el material desarrolla con más detalle (pp. 200-203, ver
sección 2 arriba). La interpretación local es directa a partir de la propia
definición del "juego":

- Se fija una instancia concreta $X_p$ (p. ej., un solicitante de crédito
  concreto).
- Se calcula la predicción del modelo para $X_p$ y la predicción media del
  modelo sobre el dataset (fuente: XAI.pdf, p. 201, ejemplos en pp. 202-203).
- Cada variable de $X_p$ recibe un valor SHAP que representa su contribución
  (positiva o negativa) a la diferencia entre la predicción de ese caso y la
  predicción media.
- La interpretación práctica: variables con valor SHAP positivo grande
  "empujan" la predicción de ese caso concreto por encima de la media;
  variables con valor SHAP negativo grande la empujan por debajo. La magnitud
  indica el peso relativo de esa variable **para ese caso concreto**, no para
  el modelo en general.

Esta es exactamente la pregunta que el enunciado del taller plantea sobre
auditoría individual: "Si un cliente pide explicaciones sobre por qué se le
deniega un crédito, ¿qué información le damos?" (fuente: `taller_XAI.pdf`). El
material de SHAP (sección local) es la herramienta directamente pensada para
responder a esa pregunta: identificar, para el solicitante concreto, qué
variables (p. ej. una utilización de línea de crédito muy alta, o un número
elevado de impagos previos) explican por qué su score se aleja de la media.

## 4. Interpretación de un valor SHAP global (importancia de variables)

El enunciado exige explícitamente tanto el análisis local como el "global"
(fuente: `taller_XAI.pdf`), pero **la sección específica de SHAP en
`XAI.pdf` (pp. 200-204) se centra en el caso local (explicar una instancia
$X_p$) y no desarrolla explícitamente el procedimiento para construir la
versión global** (p. ej., la fórmula o el gráfico resumen para agregar los
valores SHAP de todas las instancias y obtener un ranking de importancia de
variables). **El material no cubre esto** en detalle en las páginas leídas.

Dicho esto, la propia lógica del método, tal como está descrita en la fuente,
permite un puente conceptual razonable y mínimo (no una fórmula inventada):
si cada instancia del dataset tiene su propia descomposición local de
"contribución por variable" (sección 3), agregar esas contribuciones
individuales a lo largo de muchos casos (p. ej. muchos solicitantes de
crédito) da una noción de qué variables, en conjunto, más influyen en las
predicciones del modelo — es decir, una importancia "global" construida a
partir de muchas explicaciones "locales". Esto es coherente con la
definición de "juego" de la p. 201 aplicada repetidamente sobre el dataset,
pero se señala aquí explícitamente que **el mecanismo concreto de agregación
(qué estadístico usar, cómo visualizarlo) no aparece en el material
revisado**; si el grupo necesita ese detalle deberá documentarlo aparte como
conocimiento no cubierto por esta fuente.

## 5. Aplicación práctica a este taller

Contexto del taller (fuente: `taller_XAI.pdf`): se debe construir, auditar y
optimizar un modelo de concesión de crédito en **dos condiciones de coste**:

- Coste Falso Positivo = Coste Falso Negativo = 1 (predicciones para
  `cs_produccion1.csv`).
- Coste Falso Positivo = Coste Falso Negativo = 10 (predicciones para
  `cs_produccion2.csv`).

El dataset ("Give Me Some Credit") tiene como variable objetivo
`SeriousDlqin2yrs` y variables explicativas como
`RevolvingUtilizationOfUnsecuredLines`, `age`,
`NumberOfTime30-59DaysPastDueNotWorse`, `DebtRatio`, `MonthlyIncome`,
`NumberOfOpenCreditLinesAndLoans`, `NumberOfTimes90DaysLate`,
`NumberRealEstateLoansOrLines`, `NumberOfTime60-89DaysPastDueNotWorse` y
`NumberOfDependents` (fuente: `DataDictionary.csv`).

Aplicando lo anterior (secciones 3 y 4) a este contexto, sin inventar nada
más allá de lo que dice la fuente:

- **SHAP local** encaja directamente con el requisito del enunciado de poder
  explicar una denegación de crédito a un cliente concreto (fuente:
  `taller_XAI.pdf`): para un solicitante $X_p$ con crédito denegado, se
  calcularía la diferencia entre su score y el score medio, y se
  identificarían qué variables (p. ej. `NumberOfTimes90DaysLate` alto,
  `RevolvingUtilizationOfUnsecuredLines` alto) explican esa diferencia — el
  mismo patrón de los dos ejemplos de las pp. 202-203 de la fuente, aplicado
  aquí al dominio de crédito en vez de riesgo médico o alquiler de
  bicicletas.
- **SHAP global** encajaría con la parte de auditoría orientada a entender,
  en conjunto, qué variables pesan más en las decisiones del modelo de
  crédito (más allá de un cliente concreto) — pero, como se indica en la
  sección 4, el material no detalla el mecanismo de agregación, por lo que
  su construcción exacta queda fuera de lo cubierto por esta fuente.
- Respecto a los **dos escenarios de coste (FP=FN=1 vs FP=FN=10)**: el
  material de SHAP (pp. 200-204) no menciona en ningún momento cómo el coste
  FP/FN debería afectar al cálculo o interpretación de los valores SHAP.
  **El material no cubre esto**. Lo único que sí es coherente con la propia
  definición de SHAP (sección 2.1) es que, si cambiar el escenario de coste
  lleva a reentrenar el modelo o a cambiar el umbral de decisión (ver
  `docs/DECISIONES.md`, decisiones D-0.2 y D-0.3, abiertas en el momento de
  escribir este documento), el "juego" a explicar (la predicción del modelo)
  sería distinto en cada escenario, por lo que en principio habría que
  recalcular los valores SHAP para cada modelo/escenario por separado en
  lugar de asumir que la explicación de un escenario es válida para el otro.
  Esta última frase es una inferencia razonable a partir de la definición de
  la fuente (sección 2.1), no una instrucción explícita de la fuente.
- El propio enunciado pide también un **modelo subrogado con reglas** y un
  **análisis de contrafactuales** (fuente: `taller_XAI.pdf`) como
  complementos a SHAP; esos dos elementos se documentan en
  `docs/teoria/subrogados.md` y `docs/teoria/contrafactuals.md`
  respectivamente y no se desarrollan aquí.
- Sobre el **coste computacional**: el material advierte que SHAP "puede ser
  muy lento de calcular" (fuente: XAI.pdf, p. 204). No especifica qué
  implementación o algoritmo mitiga ese coste (el material no cubre esto);
  es una consideración práctica a tener en cuenta al planificar el análisis
  sobre el dataset de producción (45 000 filas en `cs_produccion.csv`, fuente:
  `docs/_fuentes/INVENTARIO.md`), pero su solución técnica concreta queda
  fuera de las fuentes leídas para este documento.

### Sobre la elección de "Explainer" según familia de modelo

La consigna de esta tarea pedía señalar, si estaba en las fuentes, qué
implementación de SHAP usar según la familia de modelo (p. ej.
`TreeExplainer` para árboles/boosting, `KernelExplainer` o `LinearExplainer`
para otras familias). **Tras revisar íntegramente las páginas 200-204 de
`XAI.pdf` (la sección específica de SHAP) y su contexto inmediato (pp.
136-199), el material no menciona en ningún momento estos nombres concretos
de implementación ni da ninguna recomendación explícita de qué Explainer usar
según la familia de modelo. El material no cubre esto.** No se debe atribuir
a esta fuente ninguna preferencia por `TreeExplainer`, `KernelExplainer` o
`LinearExplainer`: esa elección, si se necesita, tendrá que apoyarse en otra
fuente o en la propia librería `shap` (fuera del alcance de este documento de
teoría).

## 6. Preferencias del profesor detectadas

Revisando el material de SHAP (`XAI.pdf`, pp. 200-204 y contexto pp. 136-199)
y el enunciado (`taller_XAI.pdf`):

- El profesor pide **explícitamente** análisis SHAP tanto global como local
  como parte obligatoria de la auditoría del modelo (no es opcional, a
  diferencia de "otras técnicas que los estudiantes consideren oportunas")
  (fuente: `taller_XAI.pdf`).
- El profesor presenta SHAP con un tono claramente más favorable que LIME:
  lo describe como "el único método de XAI con una teoría sólida" y dedica su
  última diapositiva de LIME a una conclusión de cautela ("está aún en fase
  de desarrollo y hay que resolver muchos problemas antes de que pueda
  aplicarse con seguridad"), mientras que a SHAP solo le atribuye como pega
  la lentitud de cálculo (fuente: XAI.pdf, pp. 199 y 204). Esto sugiere una
  preferencia implícita del profesor por SHAP sobre LIME cuando haya que
  elegir un método de explicación para el modelo de crédito, aunque no hay
  una frase literal tipo "usad SHAP y no LIME" en el material.
- No se ha detectado ninguna instrucción explícita del profesor sobre qué
  implementación concreta de SHAP usar (`TreeExplainer`, `KernelExplainer`,
  `LinearExplainer`, etc.), ni sobre cómo adaptar el análisis SHAP a los dos
  escenarios de coste (FP=FN=1 y FP=FN=10) del taller. Se deja constancia
  explícita de esta ausencia en vez de suponer una preferencia no verificada.

## 7. Fuentes citadas

- `docs/_fuentes/XAI.pdf`:
  - pp. 200-201 — definición del juego de Shapley/SHAP (jugadores, premio,
    ganancia).
  - pp. 202-203 — dos ejemplos visuales de explicación local (gráficos de
    contribución por variable).
  - p. 204 — fortalezas/debilidades de SHAP y comparación explícita con
    LIME.
  - p. 140 — mención general de SHAP en el panorama de algoritmos de XAI.
  - pp. 181-199 — sección de LIME, usada aquí solo como contraste con SHAP
    (pp. 181-183 definición y algoritmo; pp. 185-186, 198-199 desventajas y
    conclusión).
- `docs/_fuentes/taller_XAI.pdf` — enunciado oficial: requisito explícito de
  "Análisis SHAP global y local", los dos escenarios de coste (FP=FN=1 y
  FP=FN=10) y los ficheros de entrega asociados.
- `docs/_fuentes/DataDictionary.csv` — nombres y descripción de las variables
  del dataset de crédito usadas como ejemplo de aplicación práctica.
- `docs/_fuentes/INVENTARIO.md` — tamaño del fichero de producción
  (`cs_produccion.csv`, 45 000 filas), citado como contexto del coste
  computacional de SHAP.
- `docs/DECISIONES.md` — decisiones D-0.2 y D-0.3 (abiertas en el momento de
  redactar este documento), citadas solo como contexto de proyecto, no como
  fuente teórica de SHAP.

Los ficheros `00-fundamentos-dependencia.md`, `01-capa-custom.md`,
`02-fair-loss.md`, `03-keras-tuner.md`, `04-incertidumbre.md` y
`05-contexto-confiabilidad.md` (`docs/_fuentes/`) se revisaron brevemente por
completitud, pero pertenecen a otro taller (fairness/incertidumbre en redes,
dataset `CODE_GENDER`) y no contienen ninguna idea genuinamente aplicable a
SHAP ni a la auditoría de este taller de crédito; no se han usado como
fuente de este documento.
