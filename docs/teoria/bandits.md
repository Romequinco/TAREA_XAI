# Multiarmed Bandits: fundamento y su encaje (o no) en concesión de crédito

> Documento de teoría de soporte para el Taller B4-T2 (XAI aplicado a
> concesión de crédito). Sintetiza el material del profesor sobre
> Multiarmed Bandits — formalización, algoritmos no contextuales y bandits
> contextuales lineales — y, sobre todo, qué argumentos a favor y en contra
> ofrece (o no ofrece) el material para elegir un bandit frente a un modelo
> supervisado clásico en este taller. **No contiene código ni
> implementación**; es referencia conceptual para alimentar la decisión
> `D-0.1` de `docs/DECISIONES.md`.

## 1. Introducción

El enunciado de la práctica plantea explícitamente dos familias de modelo
posibles para la concesión de crédito: "a) Modelo supervisado b) Multiarmed
Bandit" (fuente: `taller_XAI.pdf`, sección 1), sin dar ningún criterio
adicional sobre cuándo preferir una u otra. Esta es precisamente la
decisión `D-0.1`, registrada como abierta en `docs/DECISIONES.md`.

La fuente principal de este documento es `AR_Multiarmed_Bandits.pdf`
(48 diapositivas, Manuel Sánchez-Montañés, UAM, traducción y extensión de
las "Lectures on Reinforcement Learning" de David Silver, 2015 — fuente:
AR_Multiarmed_Bandits.pdf, p. 3), que cubre la formalización completa de
bandits no contextuales y contextuales. Se complementa con la inspección de
los dos notebooks de ejercicio del profesor (`ejercicio1_AR.ipynb` y
`ejercicio2_proyecto_XAI.ipynb`), que muestran el patrón concreto de código
con el que se espera aplicar un bandit contextual lineal (librería
`space_bandits`, clase `LinearBandits`) sobre un problema de clasificación
binaria con coste asimétrico. Importante: **el propio material teórico
(`AR_Multiarmed_Bandits.pdf`) no menciona en ningún momento, en sus 48
páginas, el caso de uso "concesión de crédito"** ni compara explícitamente
bandits contra modelos supervisados; los argumentos a favor/en contra que se
recogen en la sección 4 son, salvo que se indique lo contrario, inferencias
razonadas construidas a partir de la formalización general del material y de
lo que sí implementan los notebooks de ejercicio, no citas literales de una
comparación ya hecha por el profesor. Esto se señala explícitamente para no
atribuir a la fuente una conclusión que no está escrita en ella.

## 2. Fundamento teórico

### 2.1 La idea central: exploración vs explotación

El material sitúa los multiarmed bandits como el caso más simple dentro de
una taxonomía de tres tipos de aprendizaje por refuerzo, según cuánta
estructura secuencial tiene el problema (fuente: AR_Multiarmed_Bandits.pdf,
pp. 5-7):

- **Multi-armed Bandit**: `acción → recompensa`. No hay estado.
- **Contextual Bandit**: `estado → acción → recompensa`, donde el estado
  (contexto) condiciona la recompensa esperada, pero no se realimenta en un
  estado siguiente.
- **Problemas secuenciales** (RL completo / MDP): `estado → acción →
  recompensa → nuevo estado`, con retroalimentación completa a lo largo del
  tiempo.

La definición formal de un multiarmed bandit (fuente: p. 8) es: existe un
conjunto **conocido** de `m` acciones ("brazos"); en cada paso `t` el agente
elige una acción `a_t`; esa acción produce una recompensa `r_t`; la
distribución de esa recompensa es **inicialmente desconocida** y depende
solo de la acción tomada (no de la historia previa); el objetivo es
**maximizar la recompensa total obtenida**.

La idea central que resuelve el dilema exploración/explotación se plantea
así (fuente: p. 10): si se conocieran de antemano las distribuciones de
recompensa de cada acción, el problema sería trivial (elegir siempre la de
mayor media). El problema real es que esas distribuciones son
desconocidas al principio, así que hay que *explorar* (probar acciones para
aprender sus recompensas) y *explotar* (usar lo aprendido para maximizar
recompensa) al mismo tiempo. El material advierte explícitamente que la
"estrategia obvia" de separar ambas fases (explorar un tiempo fijo, luego
explotar siempre la mejor) "puede estar muy lejos de ser óptima" y que "la
recompensa total en exploración puede ser muy pobre" (fuente: p. 10).

Los ejemplos prácticos que el propio material usa para motivar los bandits
son recomendación de productos, marketing digital (click/no-click en un
anuncio) y asignación de tratamientos médicos experimentales (fuente: p. 9).
**El material no incluye concesión de crédito entre estos ejemplos
introductorios**; el vínculo con crédito aparece únicamente en el ejercicio
práctico del profesor (ver sección 3), que a su vez remite a un notebook
externo no presente en `docs/_fuentes/` ("6-credit_scoring.ipynb" de la
carpeta "AR/2-contextuales", fuente: `ejercicio1_AR.ipynb`, celda 2).

### 2.2 Formalización: valor de una acción y regret

El material formaliza (fuente: pp. 12-14):

- **Valor de una acción**: `Q(a) = E[r|a]` (refuerzo medio de la acción `a`).
- **Valor óptimo**: `V* = max_a Q(a)`.
- **Regret** de un paso: `l_t = E[V* − Q(a_t)]`; **total regret**:
  `L_T = E[Σ_{t=1}^{T} (V* − Q(a_t))]`. Minimizar el regret total equivale a
  maximizar el refuerzo total.
- **Gap** `Δ_a = V* − Q(a)`: el coste de oportunidad de elegir la acción `a`
  en vez de la óptima.
- **Teorema (Lai y Robbins)**: asintóticamente, el regret total de
  *cualquier* algoritmo de bandit posible **no puede ser menor que
  `c·log T`** (con `c` una constante que depende del problema) — es decir,
  hay un suelo teórico de coste de exploración que ningún algoritmo puede
  evitar (fuente: pp. 14 y 21).

### 2.3 Algoritmos no contextuales (resumen)

El material presenta varias familias de estrategias (fuente: p. 16:
"exploración naïve", "inicialización optimista", "optimismo ante la
incertidumbre", "probability matching", "búsqueda sobre estados de
información") y desarrolla en detalle:

- **Greedy**: estima `Q̂(a)` como la media de refuerzos observados y
  siempre elige la acción de mayor `Q̂(a)`. Puede "bloquearse para siempre"
  en una acción subóptima por simple mala suerte estadística en las
  primeras observaciones, y tiene **regret total lineal en el tiempo**
  (fuente: p. 18).
- **ε-greedy**: con probabilidad `ε` elige una acción al azar, si no actúa
  greedy. También tiene regret lineal (fuente: p. 19); con `ε` decreciente
  en el tiempo (`ε_t = min(1, k/t)`) el regret pasa a ser asintóticamente
  logarítmico, pero requiere conocer `k` (que depende del gap mínimo, que en
  la práctica se desconoce) — fuente: p. 20.
- **Valores iniciales optimistas (OIV)**: inicializar `Q̂(a)` alto para
  fomentar exploración sistemática al principio; combinado con greedy o
  ε-greedy sigue teniendo regret lineal (fuente: pp. 22-23).
- **UCB1 (Upper Confidence Bound)**: en vez de una sola estimación puntual,
  se estima una cota superior de confianza `Û_t(a)` para cada `Q(a)`
  (basada en la desigualdad de Hoeffding) y se elige la acción que maximiza
  `Q̂_t(a) + Û_t(a)`, con `Û_t(a) = sqrt(2·log t / N_t(a))` (fuente: pp.
  24-32). UCB1 alcanza un regret total **asintóticamente logarítmico**
  (fuente: p. 32), acercándose al límite teórico de Lai-Robbins.
- **Bandits bayesianos / muestreo de Thompson**: modelan la incertidumbre
  sobre `Q(a)` con una distribución de probabilidad (para bandits tipo
  Bernoulli, la distribución Beta, actualizada de forma cerrada con cada
  observación: `Beta(1+n1_a, 1+n0_a)` partiendo de un prior
  `α0=β0=1`, fuente: pp. 36-41). El algoritmo (fuente: p. 42) muestrea un
  `p_a` de cada `Beta(α_a, β_a)`, elige la acción con `p_a` muestreado más
  alto, observa el refuerzo y actualiza `α`/`β` según el resultado. El
  material afirma explícitamente que este algoritmo es "sencillo de
  implementar", que "se puede demostrar que alcanza el límite mínimo
  teórico en el total regret (cota de Lai y Robbins)" y que "se puede
  extender... para otras distribuciones (bandits gaussianos, etc.)"
  (fuente: p. 43).

### 2.4 Bandits contextuales y bandits contextuales lineales

Un **bandit contextual** añade información adicional (el "contexto" `s`) de
la que depende la recompensa esperada: `Q(s,a) = E[r|s,a]` (fuente: p. 45).
Un **bandit contextual lineal** aproxima esa función con un modelo lineal
distinto por cada acción: `Q_θ(s,a) = φ(s,a)^T θ ≈ Q(s,a)`, cuyos parámetros
`θ` se ajustan por mínimos cuadrados: `θ_t = A_t^{-1} b_t`, con
`A_t = Σ φ(s_τ,a_τ)φ(s_τ,a_τ)^T` y `b_t = Σ φ(s_τ,a_τ)·r_τ` (fuente: p. 45).

La regresión por mínimos cuadrados no solo da la estimación puntual `θ`,
sino también la incertidumbre (varianza) de esa estimación:
`σ²_θ(s,a) = φ(s,a)^T A^{-1} φ(s,a)` (fuente: p. 46). Esto permite construir
un **UCB lineal**: elegir la acción que maximiza
`Q_θ(s_t,a) + k·sqrt(φ(s_t,a)^T A_t^{-1} φ(s_t,a))`, donde `k` es un
hiperparámetro que regula cuánto se explora (fuente: p. 47). El material
señala explícitamente que "esta idea se puede extender a modelos no
lineales (por ejemplo, modelar la estimación de las Q con redes
neuronales)" (fuente: p. 47) — coherente con que el paquete `space_bandits`
usado en los notebooks de ejercicio incluya, además de `linear.py`, los
módulos `neural_linear.py` y `bayesian_nn.py` (fuente:
`docs/_fuentes/INVENTARIO.md`, entrada de `space_bandits.zip`), aunque los
notebooks de ejercicio solo instancian la variante `LinearBandits`.

## 3. Aplicación práctica a este taller

### 3.1 El patrón de los notebooks de ejercicio

Los dos notebooks de ejercicio disponibles (`ejercicio1_AR.ipynb`, sobre el
dataset `breast_cancer` como calentamiento, y `ejercicio2_proyecto_XAI.ipynb`,
que añade al principio la carga del dataset real de crédito
`cs_construccion.csv`/`DataDictionary.csv` pero sin adaptar el resto del
notebook a esas variables — fuente: `docs/_fuentes/INVENTARIO.md`) siguen
un mismo patrón reutilizable de cuatro pasos:

1. **Modelo supervisado clásico de referencia**: un `RandomForestClassifier`
   entrenado sobre los datos, con un barrido de umbral de probabilidad
   contra una matriz de penalización asimétrica
   (`PENALIZACION_FALSO_POSITIVO = 1`, `PENALIZACION_FALSO_NEGATIVO = 10` en
   el ejemplo médico) y un gráfico de penalización media vs. umbral (fuente:
   `ejercicio1_AR.ipynb`, celdas 17-25).
2. **Entorno de refuerzo a medida**: una clase Python (`entorno_tumores`)
   que expone `nuevo_paciente()` (elige una fila al azar del dataframe),
   `datos_paciente()` (el contexto, es decir, las variables explicativas de
   ese caso) y `act(accion)`, que devuelve el refuerzo: `0` si la acción
   coincide con la clase real, y `-penalizacion_FP` o `-penalizacion_FN`
   según el tipo de error cometido (fuente: `ejercicio1_AR.ipynb`, celda 33).
3. **Bandit contextual**: se instancia `LinearBandits(num_actions=2,
   num_features=...)` de la librería `space_bandits` y se ejecuta un bucle
   que, en cada paso, obtiene un nuevo caso del entorno, predice una acción
   (`model.predict(...)`), observa el refuerzo (`entorno.act(...)`) y
   actualiza el modelo (`model.update(datos_paciente, prediccion,
   refuerzo)`) — fuente: `ejercicio1_AR.ipynb`, celdas 41-46.
4. **Modelo subrogado**: sobre las predicciones del bandit ya entrenado se
   entrena un `DecisionTreeClassifier(max_leaf_nodes=4)` que imita esas
   decisiones, y se extraen reglas legibles con `get_rules_from_tree`
   (`my_library.py.txt`) — con el comentario explícito en el propio
   notebook: "Modelo surrogado ... 'Gemelo digital' del que queremos
   interpretar. Pero mucho más sencillo. Idealmente: 'caja blanca'" (fuente:
   `ejercicio1_AR.ipynb`, celdas 48-55).

**Advertencia importante sobre este patrón**: la clase `entorno_tumores`
conoce siempre la etiqueta real (`target`) de cada caso al calcular el
refuerzo, independientemente de qué acción se tomó (fuente:
`ejercicio1_AR.ipynb`, celda 33, ambas ramas del `if accion == 0` / `else`
consultan `self.df.iloc[self.i]["target"]`). Es decir, el "entorno" no es un
sistema online real con feedback parcial, sino una simulación por
remuestreo de un dataset ya completamente etiquetado
(`nuevo_paciente()` elige una fila al azar con `np.random.randint`, fuente:
misma celda y celda 34). Esta observación es una inferencia de este
documento a partir de la lectura del código del notebook, no una afirmación
del material teórico; se retoma en la sección 4.2 porque matiza uno de los
argumentos a favor del bandit.

### 3.2 Encaje conceptual con la concesión de crédito

Aplicando la formalización de la sección 2.4 a este taller: el contexto `s`
serían las variables del solicitante — `RevolvingUtilizationOfUnsecuredLines`,
`age`, `NumberOfTime30-59DaysPastDueNotWorse`, `DebtRatio`,
`MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`,
`NumberOfTimes90DaysLate`, `NumberRealEstateLoansOrLines`,
`NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfDependents` (fuente:
`DataDictionary.csv`) —, las dos acciones `a` serían "conceder crédito" /
"denegar crédito", y la recompensa `r` sería el negativo del coste según el
tipo de error cometido (impago tras concesión = falso negativo; denegación
a quien no habría impagado = falso positivo), exactamente como en el patrón
`entorno.act()` descrito arriba pero sustituyendo las penalizaciones
médicas por la matriz de coste del taller. **Esto es una traslación directa
del patrón del notebook al dominio de crédito construida para este
documento; el material no realiza esta traslación de forma explícita** —
los notebooks de ejercicio, aunque cargan el dataset real de crédito en
`ejercicio2_proyecto_XAI.ipynb`, no llegan a adaptar el resto del código
(entorno, bandit, subrogado) a esas variables (fuente:
`docs/_fuentes/INVENTARIO.md`, entrada de `ejercicio2_proyecto_XAI.ipynb`).

### 3.3 Los dos escenarios de coste (FP=FN=1 y FP=FN=10)

El taller exige optimizar el modelo bajo dos matrices de coste distintas:
Coste Falso Positivo = Coste Falso Negativo = 1, y Coste Falso Positivo =
Coste Falso Negativo = 10 (fuente: `taller_XAI.pdf`, sección 1). Nótese que,
a diferencia del ejemplo médico del notebook de ejercicio (donde
`PENALIZACION_FALSO_POSITIVO=1` y `PENALIZACION_FALSO_NEGATIVO=10` son
**distintas entre sí**, dentro del mismo escenario — fuente:
`ejercicio1_AR.ipynb`, celda 24), en este taller el coste es **simétrico
dentro de cada escenario** (FP y FN valen lo mismo) y lo que cambia es la
magnitud absoluta entre un escenario y otro (1 vs. 10). El mecanismo de
`entorno.act()` (recompensa = `-coste` según el tipo de error, fuente:
`ejercicio1_AR.ipynb`, celda 33) es igualmente válido para este caso
simétrico: solo cambiaría el valor numérico de la penalización aplicada a
ambos tipos de error por igual, en vez de usar dos constantes distintas.
El material no dice explícitamente cómo debería cambiar la política del
bandit (o del modelo supervisado) al pasar de un escenario a otro más allá
de este cambio de magnitud en la recompensa/coste; **el material no cubre**
si, por ejemplo, debería reentrenarse un bandit desde cero para cada
escenario o si un mismo bandit podría servir a ambos con solo cambiar la
función de recompensa durante el entrenamiento.

## 4. Argumentos a favor y en contra de un Bandit frente a un modelo supervisado

Esta es la pregunta central de la decisión `D-0.1`. Como se advierte en la
sección 1, el material no compara explícitamente ambos enfoques para este
caso concreto; lo que sigue son argumentos construidos combinando (a) la
formalización general de `AR_Multiarmed_Bandits.pdf` y (b) lo que
efectivamente implementan y advierten los notebooks de ejercicio y el
enunciado, señalando en cada punto si es una cita directa o una inferencia.

### 4.1 A favor del Multiarmed Bandit

- **Encaje conceptual del contexto**: el framework de bandit contextual
  (`Q(s,a)=E[r|s,a]`, fuente: p. 45) mapea de forma natural el contexto `s`
  a las variables del solicitante y la acción `a` a
  conceder/denegar — es exactamente el patrón que instancian los notebooks
  de ejercicio (`datos_paciente()` como contexto, `act(accion)` como
  recompensa, fuente: `ejercicio1_AR.ipynb`, celda 33). No hace falta la
  maquinaria de un problema secuencial completo (con estado que se
  realimenta): la propia taxonomía del material (pp. 5-7) sitúa el caso
  "cada solicitante es independiente, sin estado que evolucione entre
  decisiones" como un Contextual Bandit, no como un problema de RL
  secuencial completo — una lectura razonable de esa taxonomía aplicada a
  este dominio, no una afirmación literal sobre crédito.
- **El coste asimétrico entra de forma natural como recompensa**: el patrón
  `entorno.act()` (fuente: `ejercicio1_AR.ipynb`, celda 33) codifica
  directamente la penalización por tipo de error como refuerzo negativo, en
  vez de como un ajuste de umbral aplicado *después* de entrenar un modelo
  de probabilidad. Esto permite, en principio, que la sensibilidad al coste
  (los dos escenarios FP=FN=1 y FP=FN=10 del taller) forme parte del propio
  proceso de decisión del bandit desde el principio, en lugar de ser un paso
  separado de post-procesado sobre un modelo entrenado de forma neutra al
  coste.
- **Conserva parte de la interpretabilidad de un modelo lineal**: un bandit
  contextual lineal es, en su núcleo matemático, un modelo lineal por
  acción con coeficientes `θ` ajustados por mínimos cuadrados (fuente: p.
  45) — en principio tan interpretable como una regresión lineal/logística,
  relevante en un taller centrado en explicabilidad. (Ver también la
  advertencia de la sección 4.2 sobre cómo el propio notebook trata este
  modelo como caja negra en la práctica).
- **Sencillez y garantías teóricas del enfoque bayesiano**: si se optara por
  un bandit no lineal/no contextual como aproximación inicial, el material
  destaca que el algoritmo de Thompson sampling con bandits Bernoulli es
  "sencillo de implementar" y alcanza el "límite mínimo teórico en el total
  regret" (fuente: p. 43) — una garantía formal de eficiencia que un
  modelo supervisado clásico no ofrece de la misma manera (el supervisado
  no tiene una noción de "regret" porque no interactúa secuencialmente con
  el problema).

### 4.2 En contra del Multiarmed Bandit (a favor de un modelo supervisado clásico)

- **La distribución de recompensas no es realmente desconocida aquí**: la
  propia definición de bandit asume que "la señal de recompensa...
  inicialmente se desconoce" y debe aprenderse por interacción (fuente: p.
  8). Sin embargo, este taller dispone de un dataset histórico completo y
  ya etiquetado, `cs_construccion.csv` (105.000 filas con
  `SeriousDlqin2yrs` conocido, fuente: `docs/_fuentes/INVENTARIO.md`). La
  tasa de impago condicionada a las variables del solicitante ya está
  observada en esos datos; no hace falta "descubrirla" por interacción
  secuencial. Un modelo supervisado puede explotar directamente ese
  conocimiento completo desde el primer momento. Esta es una inferencia
  razonada (contraste entre la definición de p. 8 y los datos disponibles),
  no una afirmación explícita del material sobre este taller.
- **Coste de exploración/regret no evitable**: el material advierte que "en
  la estrategia obvia, la recompensa total en exploración puede ser muy
  pobre" (fuente: p. 10) y demuestra que **todo** algoritmo de bandit tiene
  un regret total que asintóticamente no puede bajar de `c·log T` (Teorema
  de Lai y Robbins, fuente: pp. 14 y 21). Trasladado a crédito: un bandit,
  mientras aprende, tomará necesariamente algunas decisiones subóptimas
  (conceder a quien impagará, o denegar a quien no lo habría hecho) solo
  para explorar — un coste real en un dominio financiero. Un modelo
  supervisado entrenado una vez sobre datos históricos etiquetados no paga
  ese coste de exploración en producción, porque no tiene que "descubrir"
  nada mediante prueba y error sobre solicitantes reales.
- **Problema del feedback contrafactual (no abordado por el material)**: la
  definición formal asume que cada acción tomada produce una recompensa
  observable (fuente: p. 8: la acción "da lugar a una señal de recompensa
  `r_t`"). En concesión de crédito real, si se **deniega** un crédito, no
  se observa si esa persona habría impagado o no — no hay recompensa
  observable para la acción no tomada (el clásico problema de "feedback
  parcial"/censura en scoring). **El material no aborda este problema en
  ningún momento de las 48 páginas revisadas.** Además, al inspeccionar el
  patrón de los notebooks de ejercicio, el "entorno" simulado sí conoce
  siempre la etiqueta verdadera para ambas acciones (fuente:
  `ejercicio1_AR.ipynb`, celda 33, ver sección 3.1) — es decir, el
  simulador de referencia disponible **no reproduce** el problema real de
  feedback censurado propio de la concesión de crédito, sino que lo evita
  usando un dataset ya completamente etiquetado. Esto matiza el argumento
  "a favor" de la sección 4.1: la ventaja teórica de un bandit (aprender
  bajo feedback parcial) no está realmente demostrada ni ejercitada por el
  material disponible sobre este dominio.
- **No hay ventaja de explicabilidad "gratis" en la práctica**: aunque
  matemáticamente un bandit contextual lineal es un modelo lineal (p. 45),
  el propio notebook de ejercicio trata el bandit ya entrenado como una
  caja negra que necesita un modelo subrogado para interpretarse
  ("'Gemelo digital'... Idealmente: 'caja blanca'", fuente:
  `ejercicio1_AR.ipynb`, celdas 48-55). En la práctica de este taller,
  tanto el bandit como un modelo supervisado (p. ej. el `RandomForestClassifier`
  de referencia del mismo notebook) acabarían necesitando el mismo tipo de
  auditoría exigida por el enunciado (subrogado con reglas, y
  presumiblemente SHAP y contrafactuales — fuente: `taller_XAI.pdf`,
  sección 1); el material no muestra ninguna ventaja de auditoría
  específica del bandit frente al supervisado.

### 4.3 Lo que el material no cubre

Para no atribuir a la fuente conclusiones que no contiene, se deja constancia
explícita de los siguientes vacíos:

- El material no compara nunca, ni en `AR_Multiarmed_Bandits.pdf` ni en el
  enunciado (`taller_XAI.pdf`), el rendimiento esperado (coste medio en
  producción) de un bandit contextual lineal frente a un modelo supervisado
  clásico sobre un dataset de crédito real.
- El material no discute el coste computacional relativo de entrenar/servir
  un `LinearBandits` frente a, por ejemplo, un árbol boosted, sobre un
  dataset del tamaño del de este taller (105.000 filas de construcción, 45.000
  de producción, fuente: `docs/_fuentes/INVENTARIO.md`).
- El material no dice si, ante los dos escenarios de coste del taller
  (FP=FN=1 y FP=FN=10), conviene entrenar un bandit distinto por escenario
  o uno solo con la recompensa reponderada (pregunta análoga a la decisión
  `D-0.2` sobre el modelo supervisado, que tampoco está cerrada).
- El material no menciona el problema de feedback contrafactual/parcial en
  scoring de crédito (ver 4.2); toda esa discusión es una construcción de
  este documento, no una cita de la fuente.

## 5. Relevancia para D-0.1

Resumen directamente utilizable para la decisión `D-0.1` (¿Modelo
supervisado o Multiarmed Bandit?, actualmente `ABIERTA` en
`docs/DECISIONES.md`):

- **El enunciado no marca preferencia**: ambos modelos están permitidos
  explícitamente y sin jerarquía (fuente: `taller_XAI.pdf`, sección 1).
- **A favor del bandit** (sección 4.1): encaje conceptual limpio
  contexto→acción→recompensa con las variables del solicitante; permite
  incorporar el coste asimétrico directamente en la recompensa en vez de
  como umbral post-hoc; en su variante lineal conserva, en teoría,
  coeficientes interpretables por variable y por acción; y, si se usa la
  variante bayesiana/Thompson sampling, tiene garantías teóricas de
  eficiencia (regret) que el material demuestra formalmente.
- **En contra del bandit** (sección 4.2): el dataset de este taller ya es
  histórico y está completamente etiquetado, por lo que la premisa central
  del bandit ("la recompensa se desconoce y hay que aprenderla por
  interacción") aporta menos valor aquí que en un problema genuinamente
  online; cualquier bandit paga inevitablemente un coste de exploración/regret
  no nulo (demostrado formalmente por el propio material, teorema de Lai y
  Robbins) que un supervisado entrenado de una vez no paga; el problema de
  feedback contrafactual propio de la concesión de crédito real (no se
  observa el resultado de un crédito denegado) no está resuelto ni siquiera
  simulado de forma realista en el material/patrón de ejercicio disponible;
  y, en la práctica del propio notebook del profesor, el bandit acaba
  auditándose igual que un modelo supervisado (mismo paso de subrogado), sin
  ventaja de explicabilidad evidente.
- **Conclusión para el grupo**: el material teórico revisado no zanja la
  decisión — no hay una recomendación explícita del profesor ni una
  comparación empírica ya hecha. Los argumentos recogidos aquí son
  insumo teórico para la justificación de `D-0.1`, pero, tal como registra
  `docs/DECISIONES.md`, la decisión final requiere evidencia empírica
  (comparar el coste medio en `cs_produccion.csv` bajo ambos enfoques) que
  queda fuera del alcance de este documento de teoría.

## 6. Preferencias del profesor detectadas

- El enunciado no expresa ninguna preferencia entre modelo supervisado y
  bandit: los presenta como dos alternativas igualmente válidas, sin
  calificarlas ("Los posibles modelos a considerar son a) Modelo
  supervisado b) Multiarmed Bandit", fuente: `taller_XAI.pdf`, sección 1).
  Se deja constancia explícita de que **no hay preferencia detectada aquí**,
  en vez de suponer una.
- Si el grupo opta por un bandit, el patrón de ambos notebooks de ejercicio
  usa consistentemente la librería `space_bandits` y, en concreto, su clase
  `LinearBandits` (fuente: `ejercicio1_AR.ipynb`, celdas 40-42; mismo patrón
  en `ejercicio2_proyecto_XAI.ipynb`) — es una convención repetida en ambos
  notebooks del profesor, aunque no hay una frase explícita del tipo "usad
  `space_bandits`"; se señala como preferencia implícita detectada por
  repetición del patrón, no como instrucción literal.
- Ambos notebooks de ejercicio encadenan siempre el bandit con un paso de
  **modelo subrogado** (`DecisionTreeClassifier(max_leaf_nodes=4)` +
  `get_rules_from_tree`) inmediatamente después de entrenarlo (fuente:
  `ejercicio1_AR.ipynb`, celdas 48-55) — sugiere que, si se usa un bandit,
  el profesor espera que se le aplique el mismo tratamiento de auditoría con
  reglas que a un modelo supervisado, no que quede exento de esa parte del
  enunciado.
- El notebook de ejercicio remite explícitamente a un notebook base
  concreto no incluido en `docs/_fuentes/`: "Toma el notebook
  `6-credit_scoring.ipynb` de la carpeta `AR/2-contextuales` y tómalo como
  base" (fuente: `ejercicio1_AR.ipynb`, celda 2). Es una instrucción
  explícita del profesor sobre qué material de partida usar, pero el
  fichero en sí no está disponible localmente; el grupo tendría que
  localizarlo en la URL oficial de Google Drive citada en `taller_XAI.pdf`
  si quiere seguir esa referencia al pie de la letra.
- Para el caso concreto de Thompson sampling con bandits Bernoulli, el
  material sugiere una función concreta para el muestreo: "Por ejemplo, con
  función `scipy.stats.beta.rvs(alfa, beta)` de librería `scipy` de Python"
  (fuente: AR_Multiarmed_Bandits.pdf, p. 42). Es una sugerencia de
  implementación explícita, aunque solo aplica si se usa la variante
  bayesiana no contextual (no la `LinearBandits` que usan los notebooks de
  ejercicio).

## 7. Fuentes citadas

- `docs/_fuentes/AR_Multiarmed_Bandits.pdf` (48 páginas, Manuel
  Sánchez-Montañés, UAM):
  - p. 3 — atribución del material (traducción/extensión de David Silver,
    2015).
  - pp. 5-7 — taxonomía Multi-armed Bandit / Contextual Bandit / Problemas
    secuenciales.
  - p. 8 — definición formal de multiarmed bandit.
  - p. 9 — ejemplos prácticos motivadores (recomendación de productos,
    marketing digital, tratamientos médicos).
  - p. 10 — dilema exploración/explotación, coste de la "estrategia obvia".
  - pp. 12-14 — formalización (`Q(a)`, `V*`, regret, gap, teorema de Lai y
    Robbins).
  - p. 16 — panorama de estrategias no contextuales.
  - pp. 17-21 — algoritmo greedy y ε-greedy (con y sin decaimiento de ε),
    regret lineal vs. logarítmico.
  - pp. 22-23 — valores iniciales optimistas (OIV).
  - pp. 24-33 — UCB / UCB1 (optimismo ante la incertidumbre, desigualdad de
    Hoeffding, cota de regret logarítmica).
  - pp. 34-43 — bandits bayesianos, distribución Beta, muestreo de
    Thompson para bandits Bernoulli, algoritmo paso a paso.
  - pp. 44-47 — bandits contextuales y bandits contextuales lineales (UCB
    lineal), extensión a modelos no lineales.
- `docs/_fuentes/taller_XAI.pdf` — enunciado oficial: los dos modelos
  permitidos (supervisado / Multiarmed Bandit), los dos escenarios de coste
  (FP=FN=1 y FP=FN=10), y los requisitos de auditoría (subrogado, SHAP,
  contrafactuales).
- `docs/_fuentes/ejercicio1_AR.ipynb` (celdas citadas: 2, 17-25, 33-34,
  40-46, 48-55) — patrón de código de referencia: modelo supervisado con
  barrido de umbral, entorno de RL a medida, `LinearBandits` de
  `space_bandits`, y modelo subrogado con reglas.
- `docs/_fuentes/ejercicio2_proyecto_XAI.ipynb` (celdas citadas: 1-7) —
  misma estructura que `ejercicio1_AR.ipynb`, con una celda inicial que
  carga el dataset real de crédito (`cs_construccion.csv`,
  `DataDictionary.csv`), sin adaptar el resto del notebook a esas
  variables.
- `docs/_fuentes/DataDictionary.csv` — nombres y descripción de las
  variables del dataset de crédito usadas como ejemplo de contexto.
- `docs/_fuentes/INVENTARIO.md` — tamaño de los datasets de construcción y
  producción, y contenido de `space_bandits.zip` (módulos `linear.py`,
  `neural_linear.py`, `bayesian_nn.py`, etc.), citados como contexto.
- `docs/DECISIONES.md` — decisión `D-0.1` (abierta en el momento de
  redactar este documento), a la que alimenta directamente este documento,
  y `D-0.2` (citada solo como paralelismo sobre los dos escenarios de
  coste).

Los ficheros `00-fundamentos-dependencia.md`, `01-capa-custom.md`,
`02-fair-loss.md`, `03-keras-tuner.md`, `04-incertidumbre.md` y
`05-contexto-confiabilidad.md` (`docs/_fuentes/`) se revisaron brevemente
por completitud (búsqueda de menciones a "bandit", "exploración/explotación",
"online", "refuerzo" y a la matriz de coste FP/FN); no contienen ninguna
idea genuinamente aplicable a Multiarmed Bandits que no esté ya cubierta por
las fuentes anteriores, así que no se han usado como fuente de este
documento.
