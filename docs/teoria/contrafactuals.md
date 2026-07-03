# Contrafactuales (explicaciones contrafácticas)

> Documento de teoría de referencia del Taller B4-T2 (XAI aplicado a concesión de
> crédito, dataset "Give Me Some Credit"). Es **solo teoría**: no contiene código
> de modelo ni decide implementación; eso corresponde a `src/` y a los notebooks
> del grupo.

## 1. Introducción

Cuando un modelo de *machine learning* deniega un crédito, la pregunta que
importa al cliente no es "¿qué variables usa el modelo?" sino algo mucho más
concreto: **¿qué tendría que ser distinto en mi situación para que me lo
concedieran?**. Las explicaciones contrafactuales responden exactamente a esa
pregunta: en vez de describir el modelo en abstracto (como hacen SHAP o un
modelo subrogado), buscan un caso hipotético, muy parecido al caso real, para
el que el modelo habría decidido lo contrario. El material de la asignatura
introduce esta familia de técnicas bajo el epígrafe **"Explicaciones Basadas en
Ejemplos"** (fuente: XAI.pdf, p. 205), inmediatamente después del bloque de
SHAP (pp. 200-204) y antes de las herramientas de fairness/regulación
(pp. 220 en adelante).

Este documento cubre: (1) qué es un contrafactual y cómo lo presenta el
material del máster, (2) qué otras técnicas de "explicación por ejemplos"
aparecen junto a los contrafactuales en las mismas diapositivas, y (3) —el
punto central pedido por el enunciado de la práctica— qué hace que un
contrafactual sea **accionable** y útil de cara a explicarle a un cliente por
qué se le deniega un crédito.

## 2. Fundamento técnico

### 2.1 Definición: la estructura causal "si no X, no Y"

El PDF define la explicación contrafactual de forma progresiva, diapositiva a
diapositiva (pp. 206-209), construyendo el mismo ejemplo con más detalle en
cada paso:

- "Una explicación contrafáctica describe una situación causal de la forma:
  'Si no hubiera ocurrido X, no habría ocurrido Y'" (fuente: XAI.pdf, p. 206).
- Ejemplo dado en el material: "Si no hubiera tomado un sorbo de este café
  caliente, no me habría quemado la lengua" (fuente: XAI.pdf, p. 207).
- El material identifica explícitamente los roles: "el suceso Y es que me
  quemé la lengua; la causa X es que me tomé un café caliente" (fuente:
  XAI.pdf, p. 208).
- Y cierra la idea señalando que razonar en términos de contrafactuales
  "requiere imaginar una realidad hipotética que contradiga los hechos
  observados […], de ahí el nombre de 'contrafáctico'" (fuente: XAI.pdf,
  p. 209).

Es decir: un contrafactual no es una medida de importancia de variables (como
SHAP) ni una aproximación global del modelo (como un subrogado); es un **caso
hipotético alternativo**, construido para invertir el resultado observado.

### 2.2 Aplicación explícita al caso de concesión de crédito

El material da el salto de la analogía del café al caso que nos ocupa en este
taller, de forma explícita y en términos casi idénticos al escenario de la
práctica:

> "Imaginemos que hemos entrenado un modelo de Machine Learning que asigna una
> alta [de] probabilidad de impago a una solicitud de crédito S. Los ejemplos
> contrafácticos son casos muy parecidos a S pero que, según el modelo, tienen
> una baja probabilidad de impago. La idea es muy sencilla de entender y de
> implementar en XAI" (fuente: XAI.pdf, p. 210).

Esta es la única frase del material que conecta explícitamente contrafactuales
con concesión de crédito, y define el patrón general del método: **buscar una
instancia próxima a la solicitud real S, pero con predicción opuesta**. El
PDF no aporta, en el texto extraíble de estas páginas, una fórmula, un
algoritmo de optimización o una métrica de distancia concreta para "próxima" —
se queda en la idea conceptual citada arriba. (Nota de método: la extracción
se hizo con PyMuPDF sobre el texto de las diapositivas; si las páginas 206-210
contenían diagramas o ejemplos visuales adicionales, esos gráficos no forman
parte del texto extraído y no se han podido revisar).

### 2.3 Prototipos y excepciones (técnica relacionada, distinta de un contrafactual)

Inmediatamente después de contrafactuales, dentro del mismo bloque
"Explicaciones Basadas en Ejemplos", el material presenta una segunda técnica
de explicación por ejemplos que **no es un contrafactual** pero comparte
familia:

- "Un prototipo es una instancia real que es representativa de un conjunto de
  datos" y "una excepción es una instancia real que no está bien representada
  por el conjunto de prototipos" (fuente: XAI.pdf, p. 211).
- El procedimiento que describe el material: "Encontrar prototipos y
  excepciones con algún algoritmo como el **MMD-critic**. Extraer la
  predicción del modelo para cada prototipo y excepción. Analizar las
  predicciones: ¿en qué casos se equivocó el algoritmo?" (fuente: XAI.pdf,
  p. 212).

Diferencia clave con un contrafactual: un prototipo/excepción es una instancia
**real** que ya existe en los datos y sirve para auditar dónde el modelo
acierta o falla como patrón general; un contrafactual, en cambio, es un caso
**hipotético** construido específicamente a partir de una instancia concreta
(la solicitud S) para explicar *esa* decisión individual. Para el objetivo de
este taller (explicarle a un cliente concreto por qué se le deniega *su*
crédito) la técnica pertinente es el contrafactual, no el prototipo/excepción
— aunque prototipos/excepciones podrían usarse de forma complementaria en la
auditoría global del modelo.

### 2.4 Lo que el material NO cubre (y conviene no inventar)

Siguiendo la instrucción de no rellenar huecos con conocimiento genérico no
citado, se deja constancia explícita de lo que el material de este taller
**no** dice sobre contrafactuales, tras revisar el texto completo del PDF
(274 páginas) buscando términos como "DiCE", "accionable"/"actionable",
"inmutable"/"immutable", "plausible", "sparsity" y "Wachter":

- **No se menciona la herramienta DiCE** (Diverse Counterfactual Explanations)
  ni ninguna otra librería o paquete concreto para generar contrafactuales.
  El enunciado de la tarea (`taller_XAI.pdf`, ver §3) tampoco la nombra. Si el
  grupo decide usar DiCE (u otra librería) en el notebook, es una decisión de
  implementación del equipo, no una instrucción del material teórico.
- **No se formalizan criterios de "accionabilidad"** (p. ej. distancia
  mínima, dispersión/*sparsity* del cambio, plausibilidad respecto a la
  distribución de datos, o la distinción formal entre variables mutables e
  inmutables tal como la trata la literatura académica sobre contrafactuales,
  p. ej. Wachter et al.). El material se limita a la idea conceptual de las
  pp. 206-210 citada arriba.
- El ejemplo trabajado que sigue en las diapositivas (pp. 213-219,
  "Predicción de demanda de un producto") **no es un ejemplo de
  contrafactual**: es un caso de estudio sobre un modelo LSTM de predicción de
  demanda, interpretado mediante un **modelo subrogado** ("XAI: modelo
  surrogado (explica el 95% del LSTM)", fuente: XAI.pdf, p. 218) y lógica
  difusa (fuente: XAI.pdf, p. 219). Aparece dentro del rango de páginas de
  esta sección pero pertenece conceptualmente al tema de modelos subrogados
  (`docs/teoria/subrogados.md`), no al de contrafactuales; se menciona aquí
  solo para dejar constancia de que el material no incluye, en este rango de
  páginas, un ejemplo trabajado paso a paso de contrafactuales sobre datos
  tabulares.
- Las diapositivas 00-05 de `docs/_fuentes/` (`00-fundamentos-dependencia.md`
  … `05-contexto-confiabilidad.md`) pertenecen a otro taller (fairness/
  incertidumbre en redes, dataset CODE_GENDER) y, tras revisarlas, **no
  contienen ninguna idea aplicable a contrafactuales** (solo coincidencias
  falsas de la palabra "dice" como verbo, no la herramienta DiCE). No se han
  usado como fuente de esta sección.

## 3. Aplicación práctica a este taller

### 3.1 Contexto: qué pide el enunciado

El enunciado oficial de la práctica exige explícitamente, como parte de la
auditoría del modelo: "Análisis de contrafactuals para varios ejemplos de
clase real 0 y 1 seleccionados. Si un cliente pide explicaciones sobre por qué
se le deniega un crédito, ¿qué información le damos?" (fuente: taller_XAI.pdf,
apartado "1. Objetivo de la práctica"). Esto fija dos requisitos concretos:

1. Analizar contrafactuales tanto para clientes de **clase real 0** (no
   impago) como de **clase real 1** (impago), no solo para los casos donde el
   modelo se equivoca.
2. Orientar el análisis hacia una **explicación comunicable a un cliente**, no
   solo hacia un diagnóstico técnico del modelo.

### 3.2 Qué información darle a un cliente al que se le deniega un crédito

Aplicando el patrón conceptual de la p. 210 del PDF (buscar un caso muy
parecido al real pero con predicción opuesta) al dataset de este taller,
la información útil para un cliente denegado no es "estas son las variables
más importantes del modelo" (eso es SHAP/subrogado), sino algo del tipo: *"con
tu situación actual el modelo predice riesgo de impago; si tu utilización de
las líneas de crédito revolving fuera, por ejemplo, un 20% en vez de un 80%,
el modelo te habría aprobado"*. Es decir, el contrafactual traduce la decisión
del modelo en **un conjunto mínimo de cambios concretos** sobre las variables
de la solicitud que habrían bastado para obtener la decisión contraria.

Para que esa explicación sea realmente **accionable** (y no solo matemáticamente
válida), conviene distinguir, para las variables del dataset del taller
(`DataDictionary.csv` — fichero de datos del taller, no parte de la teoría de
XAI.pdf), entre variables sobre las que el cliente puede actuar y variables
que no tiene sentido plantear como "cambiables". **Esta clasificación es una
aplicación razonada de la idea general del material a las variables concretas
de este dataset; el PDF de teoría no enumera esta lista, que es síntesis
propia para este taller**:

- **Variables potencialmente accionables** (tiene sentido decirle al cliente
  "si esto cambiara, tu situación mejoraría"):
  - `RevolvingUtilizationOfUnsecuredLines` (% de uso de líneas de crédito
    revolving/tarjetas): el cliente puede reducirlo pagando saldo.
  - `DebtRatio` (deuda mensual / ingreso bruto mensual): accionable reduciendo
    deuda o, más lentamente, aumentando ingresos.
  - `NumberOfOpenCreditLinesAndLoans` (nº de líneas de crédito y préstamos
    abiertos): accionable cancelando líneas de crédito no usadas.
  - `MonthlyIncome`: accionable en el medio plazo (cambio de empleo, ingresos
    adicionales), aunque no es un cambio inmediato ni siempre realista a
    corto plazo — conviene presentarlo con esa matización.
- **Variables no accionables o de dudosa utilidad como "palanca" para el
  cliente**:
  - `age` (edad del solicitante): es el ejemplo canónico de variable
    **inmutable** — nadie puede "hacerse más joven o más mayor" para obtener
    el crédito, y plantearla como palanca de cambio no tiene sentido (y further
    puede rozar terreno sensible de discriminación por edad). Un contrafactual
    que dijera "si tuvieras 10 años menos se te concedería" no es una
    explicación útil ni razonable para el cliente.
  - `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`,
    `NumberOfTime60-89DaysPastDueNotWorse` (recuentos de impagos pasados en
    los últimos 2 años): son **hechos históricos ya ocurridos**; el cliente no
    puede "deshacerlos" hoy para conseguir el crédito ahora mismo, aunque sí
    describen una vía de mejora futura (ir sin nuevos retrasos hará que estos
    contadores mejoren con el tiempo). No son accionables de forma inmediata.
  - `NumberOfDependents` (nº de personas dependientes): no es una variable
    sobre la que tenga sentido decirle al cliente que actúe para obtener
    crédito.
  - `NumberRealEstateLoansOrLines`: parcialmente accionable (cancelar una
    hipoteca no es una decisión trivial ni rápida), requiere matizar igual que
    `MonthlyIncome`.

En síntesis, un contrafactual útil para explicar una denegación de crédito
debería, como mínimo: (a) mantener fijas las variables inmutables o de
hechos-pasados (como `age`) y buscar el cambio solo sobre variables
verdaderamente accionables; (b) preferir el cambio **más pequeño** posible
(cuantos menos variables haya que tocar y menor sea la magnitud del cambio,
más creíble y más fácil de comunicar es la explicación); y (c) comprobar que
el caso hipotético resultante sea razonable dentro del rango de valores reales
del dataset (evitar, por ejemplo, un `DebtRatio` negativo o un
`MonthlyIncome` fuera de todo rango observado). El material de XAI.pdf no
formaliza estos tres criterios explícitamente (ver §2.4); se presentan aquí
como la interpretación práctica más directa de la idea de la p. 210 aplicada
al objetivo declarado en el enunciado ("¿qué información le damos?").

### 3.3 Clase real 0 vs. clase real 1

El enunciado pide contrafactuales para ejemplos de ambas clases reales
(fuente: taller_XAI.pdf). Aplicando la idea de la p. 210:

- Para un cliente de **clase real 0** (no tuvo impago) al que el modelo
  deniega el crédito (falso positivo): el contrafactual es directamente
  explicativo de cara al cliente — muestra qué cambio mínimo habría evitado un
  error del modelo sobre alguien que, de hecho, no era de riesgo.
- Para un cliente de **clase real 1** (sí tuvo impago) correctamente denegado:
  el contrafactual tiene más valor como **auditoría interna** del modelo (¿es
  razonable la distancia entre este caso y el punto de corte de aprobación?)
  que como explicación a comunicar tal cual al cliente, aunque el enunciado
  pide igualmente generarlo.

### 3.4 Relación con los dos escenarios de coste del taller (FP=FN=1 y FP=FN=10)

El taller pide auditar el modelo en dos condiciones de coste: Falso
Positivo = Falso Negativo = 1, y Falso Positivo = Falso Negativo = 10 (fuente:
taller_XAI.pdf, apartado "1. Objetivo de la práctica"). El material de
contrafactuales (XAI.pdf, pp. 205-219) no menciona coste ni umbrales de
decisión — esa conexión no está en la fuente y se deja aquí solo como
observación de aplicación práctica, no como contenido citado: el punto de
corte de probabilidad que separa "se concede" de "se deniega" es el que define,
para cada solicitante concreto, si hace falta o no un contrafactual (solo lo
necesitan quienes caen del lado de la denegación). Como ambos escenarios de
coste del enunciado mantienen la misma proporción entre coste de Falso
Positivo y coste de Falso Negativo (1:1 en los dos casos, solo cambia la
magnitud absoluta), es razonable esperar que el umbral de decisión óptimo —y,
por tanto, el conjunto de solicitantes denegados para los que hay que generar
contrafactuales— sea similar en ambos escenarios; esto último es una
deducción propia a partir de la definición del problema, no una afirmación
del material de teoría, y debería verificarse empíricamente en el notebook del
grupo (fuera del alcance de este documento, que es solo teoría).

### 3.5 Motivación regulatoria (breve, pp. 220-235)

El propio material, en el bloque posterior a XAI (herramientas de fairness y
regulación europea), da un argumento adicional de por qué esto le importa a un
cliente al que se le deniega un crédito: la propuesta de Reglamento europeo de
IA clasifica explícitamente como **de riesgo alto** los "sistemas que evalúan
la solvencia crediticia del consumidor" (fuente: XAI.pdf, p. 232), junto con
sistemas de justicia, educación o contratación de personal. El mismo bloque
señala que quedan prohibidos los sistemas de IA que "explotan las
vulnerabilidades de grupos específicos de personas, incluyendo también a las
personas vulnerables por su situación social o económica" (fuente: XAI.pdf,
p. 231). Ninguna de estas dos páginas menciona contrafactuales ni un "derecho
a explicación" de forma literal (ese término no aparece en el rango pp.
220-235 revisado); se citan aquí únicamente como motivación de contexto — un
sistema de concesión de crédito es, según esta propuesta normativa, de alto
riesgo y por tanto sujeto a mayor exigencia de transparencia y de
justificación individual de sus decisiones, lo que refuerza por qué una
explicación contrafactual clara para el cliente no es solo una buena
práctica sino algo alineado con la dirección regulatoria descrita en el
material.

## 4. Preferencias del profesor detectadas

- El enunciado oficial de la práctica (`taller_XAI.pdf`) contiene una
  instrucción explícita y específica sobre el alcance del análisis de
  contrafactuales: debe hacerse **"para varios ejemplos de clase real 0 y 1
  seleccionados"**, no solo sobre errores del modelo ni solo sobre una clase.
  Se ha reflejado en el §3.3.
- El mismo enunciado enmarca explícitamente el propósito del análisis como
  una respuesta a la pregunta **"si un cliente pide explicaciones sobre por
  qué se le deniega un crédito, ¿qué información le damos?"** — es decir, el
  profesor pide orientar el contrafactual hacia una explicación comunicable al
  cliente, no hacia una descripción técnica del modelo. Se ha usado como eje
  del §3.2.
- En las diapositivas de teoría, al describir prototipos y excepciones (técnica
  relacionada pero distinta de los contrafactuales), el material nombra un
  algoritmo concreto por su nombre — **MMD-critic** — como ejemplo de
  herramienta a usar (fuente: XAI.pdf, p. 212). No se ha detectado una
  recomendación equivalente de herramienta concreta para generar los
  contrafactuales en sí (ni DiCE ni ninguna otra), por lo que en ese punto no
  hay preferencia explícita del profesor que seguir, más allá de la idea
  conceptual de la p. 210.
- No se ha detectado ninguna preferencia explícita sobre distancia, métrica de
  similitud, número de contrafactuales por caso, ni sobre cómo tratar
  variables inmutables — estos criterios, ausentes del material, se han
  razonado en el §3.2 como aplicación propia y se señalan como tal.

## 5. Fuentes citadas

- `docs/_fuentes/XAI.pdf`, pp. 205 (título de sección), 206-209 (definición y
  ejemplo del café), 210 (ejemplo de concesión de crédito), 211-212
  (prototipos y excepciones, MMD-critic), 213-219 (ejemplo de predicción de
  demanda con modelo subrogado — no es un ejemplo de contrafactual, se cita
  para dejar constancia de qué contiene realmente ese rango), 231-232
  (regulación europea de IA, sistemas de alto riesgo y prohibiciones,
  motivación de contexto). Extraído con PyMuPDF (`fitz`) por exceder el PDF el
  límite de tamaño del tool `Read` y no estar disponible `poppler` para lectura
  por rango de páginas.
- `docs/_fuentes/taller_XAI.pdf` — enunciado oficial de la práctica, apartado
  "1. Objetivo de la práctica" (requisito de contrafactuales para clase real 0
  y 1, y pregunta sobre qué información darle al cliente) y matrices de coste
  FP=FN=1 / FP=FN=10.
- `docs/_fuentes/DataDictionary.csv` — diccionario de variables del dataset
  "Give Me Some Credit", usado para la clasificación de variables
  accionables/inmutables del §3.2 (no es una fuente de teoría de XAI, es el
  propio dataset del taller).
- `docs/_fuentes/INVENTARIO.md` — confirma el mapa de páginas de `XAI.pdf` y la
  advertencia de que los ficheros `00-fundamentos-dependencia.md` a
  `05-contexto-confiabilidad.md` pertenecen a otro taller.
- Revisados y descartados por no contener nada aplicable a contrafactuales:
  `docs/_fuentes/00-fundamentos-dependencia.md`, `01-capa-custom.md`,
  `02-fair-loss.md`, `03-keras-tuner.md`, `04-incertidumbre.md`,
  `05-contexto-confiabilidad.md` (otro taller, fairness/incertidumbre,
  dataset CODE_GENDER), y `ejercicio1_AR.ipynb` / `ejercicio2_proyecto_XAI.ipynb`
  (patrón de bandits/subrogado/coste sobre `breast_cancer`, sin contenido de
  contrafactuales).
