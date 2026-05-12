# Investigación de Estado del Arte para el Proyecto de Grado:
## "Implementación de un Modelo de Aprendizaje Automático Cuántico utilizando Qiskit Machine Learning"

**Curso:** Computación Cuántica
**Institución:** Universidad Nacional de San Antonio Abad del Cusco (UNSAAC), Departamento Académico de Informática
**Estudiante:** Yeison
**Fecha:** Mayo de 2026
**Documento:** Investigación de soporte para artículo científico (10 pp.), repositorio reproducible y presentación oral.

---

## Resumen Ejecutivo

Este documento sintetiza el estado del arte y la metodología necesaria para implementar, evaluar y reportar científicamente un Clasificador Cuántico Variacional (VQC, *Variational Quantum Classifier*) en Qiskit Machine Learning 0.8/0.9, comparado con una Máquina de Vectores de Soporte Cuántica (QSVM/QSVC) y con líneas base clásicas (SVM con kernel RBF y red neuronal multicapa). El caso de estudio principal es el dataset Iris (4 atributos, 3 clases, 150 muestras) con un experimento secundario sobre MNIST reducido a 2–3 clases. La investigación cubre fundamentos teóricos, papers seminales (Biamonte 2017; Havlíček 2019; Schuld y Killoran 2019; Mitarai 2018; Farhi y Neven 2018; Cerezo 2021; Liu, Arunachalam y Temme 2021; McClean 2018; Sim, Johnson y Aspuru-Guzik 2019), avances recientes 2024–2026 (review de Larocca *et al.* 2025 en *Nature Reviews Physics* sobre barren plateaus; paper técnico de Qiskit ML de Sahin *et al.* 2025), detalles técnicos de la API actual (Primitives V2, EstimatorQNN, SamplerQNN, VQC), una metodología reproducible y benchmarks esperados (VQC ~88–92 % en Iris; SVM clásico ~96–98 %). Se proponen contribuciones diferenciadoras —análisis de sensibilidad al ruido, comparación de feature maps y profundidades, y ejecución opcional en hardware real de IBM Quantum— alineadas con la rúbrica de máxima calificación.

---

## 1. Introducción y justificación

El aprendizaje automático cuántico (QML, *Quantum Machine Learning*) es un campo en la intersección de la mecánica cuántica, la teoría de la información y el aprendizaje estadístico que estudia cómo las propiedades de superposición, entrelazamiento e interferencia podrían acelerar o transformar tareas clásicas como clasificación, regresión, agrupamiento y generación. La revisión seminal de Biamonte *et al.* (2017) en *Nature* (vol. 549, pp. 195–202, DOI: 10.1038/nature23474) estableció el marco conceptual: los sistemas cuánticos producen patrones "atípicos" que clásicamente parecen ineficientes de reproducir, lo cual sugiere una posible ventaja para ciertas tareas de ML.

En la era NISQ (*Noisy Intermediate-Scale Quantum*) acuñada por Preskill (2018), los dispositivos cuánticos disponibles tienen entre 50 y unos cientos de qubits, con tasas de error no despreciables y profundidad de circuito limitada por la decoherencia. En este contexto, los **algoritmos variacionales híbridos** (VQA) son la familia más prometedora, según el artículo de revisión de Cerezo *et al.* (2021) en *Nature Reviews Physics* (vol. 3, pp. 625–644, DOI: 10.1038/s42254-021-00348-9). Estos algoritmos delegan la optimización a un computador clásico mientras un circuito parametrizado evalúa una función de costo en un dispositivo cuántico.

El **Variational Quantum Classifier (VQC)** es la encarnación supervisada de este paradigma. El proyecto se justifica porque permite (i) materializar conceptos teóricos de computación cuántica (qubits, puertas, entrelazamiento, medición) en una aplicación concreta de ML; (ii) ejercitar el ciclo híbrido clásico-cuántico con un framework industrial (Qiskit Machine Learning) recomendado por la guía del curso; (iii) comparar empíricamente con métodos clásicos consolidados; y (iv) reflexionar críticamente sobre el alcance real de la "ventaja cuántica" en datasets de baja dimensión como Iris. El proyecto produce evidencia reproducible y discusión informada, contribuyendo al diagnóstico publicado en revisiones sistemáticas recientes (Gupta *et al.* 2025, *npj Digital Medicine*) de que la ventaja empírica del QML sobre métodos clásicos en datos tabulares pequeños no muestra una tendencia consistente.

---

## 2. Fundamentos teóricos de Quantum Machine Learning

### 2.1 Qubits, superposición y entrelazamiento

Un **qubit** es el sistema cuántico de dos niveles cuyo estado puro general es

|ψ⟩ = α|0⟩ + β|1⟩,    con |α|² + |β|² = 1, α,β ∈ ℂ.

Un registro de *n* qubits vive en un espacio de Hilbert de dimensión 2ⁿ, y un estado puro general se escribe como combinación lineal de 2ⁿ estados base computacionales. Esta **superposición** es el primer recurso aprovechable. El segundo es el **entrelazamiento**: estados de varios qubits que no se factorizan como producto tensorial de estados individuales (por ejemplo el estado de Bell |Φ⁺⟩ = (|00⟩+|11⟩)/√2). El entrelazamiento permite correlaciones no clásicas que en QML se usan para acoplar variables y producir mapas de características no factorizables. La **medición** en la base computacional colapsa el estado y devuelve un bitstring con la distribución de probabilidad |⟨x|ψ⟩|².

### 2.2 Puertas cuánticas relevantes para QML

- **Hadamard (H):** crea superposición uniforme, H|0⟩ = (|0⟩+|1⟩)/√2.
- **Rotaciones de un qubit:** Rx(θ) = exp(-iθX/2), Ry(θ) = exp(-iθY/2), Rz(θ) = exp(-iθZ/2). Estas son la base de los circuitos parametrizados (PQC), ya que sus parámetros θ son las variables entrenables.
- **CNOT y CZ:** puertas controladas de dos qubits que generan entrelazamiento. La CNOT (CX) hace |a,b⟩ → |a, a⊕b⟩.
- **Puertas Pauli (X,Y,Z):** bloques generadores; toda evolución unitaria puede descomponerse en ellas.

### 2.3 Circuitos parametrizados (PQC) y ansatz variacional

Un **circuito parametrizado** U(θ) es una composición U(θ) = U_L(θ_L) · W_L ··· U_1(θ_1) · W_1 con U_l(θ_l) = exp(-iθ_l V_l), donde V_l son generadores hermíticos (rotaciones) y W_l son puertas fijas (entanglers). En QML supervisado el circuito completo se descompone usualmente en **feature map** Φ(x) (codificación del dato) seguido del **ansatz entrenable** U_ans(θ): |ψ(x,θ)⟩ = U_ans(θ) Φ(x) |0⟩^⊗n. Los ansatz estándar de Qiskit son:

- **RealAmplitudes:** capas de Ry seguidas de un patrón de CX. Produce estados con amplitudes reales; es el ansatz por defecto de VQC y QNN para clasificación.
- **EfficientSU2:** capas con puertas SU(2) (típicamente Ry seguido de Rz) alternadas con CX en patrón `reverse_linear` o `full`; pensado como ansatz hardware-eficiente.
- **TwoLocal:** generalización configurable donde el usuario elige las puertas de rotación, las puertas de entrelazamiento y el patrón.

### 2.4 Feature maps cuánticos

La codificación de datos clásicos x ∈ ℝᵈ a estados cuánticos es el paso crítico. Schuld y Killoran (2019, *Phys. Rev. Lett.* 122, 040504, DOI: 10.1103/PhysRevLett.122.040504) demostraron que **codificar entradas en un estado cuántico equivale a un mapa de características no lineal a un espacio de Hilbert** (feature Hilbert space), conectando directamente con los métodos kernel. Las variantes disponibles en Qiskit son:

- **ZFeatureMap:** aplica H⊗ⁿ y luego una rotación Rz con ángulo proporcional al dato en cada qubit. Es la opción más simple y no genera entrelazamiento; equivale a un mapa de primer orden.
- **ZZFeatureMap:** propuesto por Havlíček *et al.* (2019), añade interacciones de segundo orden mediante puertas Rzz controladas, codificando correlaciones por pares. Es el feature map de referencia y se conjetura clásicamente difícil de simular a partir de cierta profundidad.
- **PauliFeatureMap:** generalización donde el usuario especifica strings de Pauli como `['Z','Y','ZZ']`. Permite codificaciones híbridas más expresivas a costa de mayor profundidad.

Estudios recientes en Qiskit (Biswas 2025, arXiv:2503.14062; Faiz *et al.* 2025, arXiv:2506.03272) muestran que **ZZFeatureMap es más expresivo pero también más sensible al ruido**, y que la elección óptima depende del problema. Para Iris (4 features, 4 qubits con angle encoding directo), ZZFeatureMap con `reps=2` y entrelazamiento `linear` es una elección equilibrada.

### 2.5 Transformada cuántica de Fourier (QFT) en QML

La QFT mapea |x⟩ → (1/√N) Σ_y exp(2πi xy/N) |y⟩ y es la rutina clave de Shor y de muchos algoritmos de fase. En QML aparece en (i) algoritmos de tipo HHL (resolución cuántica de sistemas lineales) que subyacen a la QSVM original de Rebentrost, Mohseni y Lloyd (2014, *Phys. Rev. Lett.* 113, 130503) y a algunos esquemas de PCA cuántico; (ii) feature maps periódicos basados en codificación por fase; y (iii) en la prueba de ventaja cuántica de Liu, Arunachalam y Temme (2021, *Nature Physics* 17, 1013–1017, DOI: 10.1038/s41567-021-01287-z), donde se usa una codificación tipo Shor que aprovecha la dureza del logaritmo discreto. En la práctica del VQC sobre Iris no se invocará la QFT directamente, pero se menciona como puente conceptual hacia algoritmos cuánticos puros.

### 2.6 Algoritmos híbridos clásico-cuánticos (VQA)

El esquema VQA, formalizado en Cerezo *et al.* (2021), itera:

1. Preparar |ψ(x_i, θ_t)⟩ en el QPU/simulador.
2. Medir un observable Ô (o el bitstring) para estimar la función de costo C(θ_t).
3. Calcular el gradiente ∇_θ C (parameter-shift rule de Mitarai 2018 o diferencias finitas).
4. Actualizar θ_{t+1} con un optimizador clásico (COBYLA, SPSA, ADAM, L-BFGS-B).
5. Repetir hasta convergencia.

Para VQC con loss de entropía cruzada y *one-hot encoding*, la salida del SamplerQNN se interpreta como distribución de probabilidad sobre clases.

### 2.7 Quantum kernel methods

Havlíček *et al.* (2019) en *Nature* (vol. 567, pp. 209–212, DOI: 10.1038/s41586-019-0980-2) propusieron dos paradigmas: (a) **VQC explícito** (clasificador entrenable en el espacio de Hilbert) y (b) **kernel cuántico** k(x, x') = |⟨Φ(x')|Φ(x)⟩|² evaluado en el QPU y luego pasado a un SVM clásico. Liu *et al.* (2021) probaron rigurosamente que existe una **separación cuántico-clásica exponencial** para una familia de datasets construida a partir del problema del logaritmo discreto, asumiendo la dureza clásica del mismo. Sin embargo, esta ventaja requiere hardware tolerante a fallos y no es trasladable directamente a Iris.

---

## 3. Estado del arte 2017–2026

### 3.1 Papers seminales

| Año | Autores | Aporte | Referencia |
|-----|---------|--------|------------|
| 2017 | Biamonte, Wittek, Pancotti, Rebentrost, Wiebe, Lloyd | Revisión fundacional de QML | *Nature* 549:195–202, DOI:10.1038/nature23474 |
| 2018 | Mitarai, Negoro, Kitagawa, Fujii | Quantum Circuit Learning + parameter-shift rule | *Phys. Rev. A* 98:032309, DOI:10.1103/PhysRevA.98.032309 |
| 2018 | Farhi, Neven | QNN para clasificación en NISQ | arXiv:1802.06002 |
| 2018 | McClean, Boixo, Smelyanskiy, Babbush, Neven | Descubrimiento del fenómeno *barren plateau* | *Nat. Commun.* 9:4812, DOI:10.1038/s41467-018-07090-4 |
| 2019 | Havlíček *et al.* | VQC y quantum kernels en superconductor real | *Nature* 567:209–212, DOI:10.1038/s41586-019-0980-2 |
| 2019 | Schuld, Killoran | QML en espacios de Hilbert de características | *PRL* 122:040504, DOI:10.1103/PhysRevLett.122.040504 |
| 2019 | Sim, Johnson, Aspuru-Guzik | Expresividad y capacidad de entrelazamiento de PQCs | *Adv. Quantum Tech.* 2:1900070 |
| 2021 | Cerezo *et al.* | Revisión exhaustiva de VQAs | *Nat. Rev. Phys.* 3:625–644 |
| 2021 | Liu, Arunachalam, Temme | Speedup cuántico riguroso con kernels | *Nat. Phys.* 17:1013–1017 |

### 3.2 Avances 2023–2026

- **Larocca *et al.* (2025, *Nat. Rev. Phys.*, DOI:10.1038/s42254-025-00813-9; preprint arXiv:2405.00781):** revisión definitiva del fenómeno de *barren plateaus*, mostrando que ansatz, estado inicial, observable, función de costo y ruido pueden inducirlos, y catalogando estrategias de mitigación (inicializaciones específicas, ansatz simétricos, costos locales, *layer-wise training*).
- **Cunningham y Zhuang (2024, arXiv:2407.17706; pub. en *Quantum Inf. Process.* 2025):** survey sistemático con taxonomía de cinco grupos de estrategias de mitigación.
- **Sahin *et al.* (2025, arXiv:2505.17756) — "Qiskit Machine Learning: an open-source library for quantum machine learning tasks at scale":** paper técnico oficial del framework actual, co-mantenido por IBM y el Hartree Centre del STFC desde enero de 2024. Documenta la API V2 y aplicaciones a escala (hasta 156 qubits con mitigación de errores bit-flip).
- **Wang y Liu (2024, arXiv:2401.11351):** revisión "from NISQ to Fault Tolerance" que articula el horizonte de algoritmos QML según el tipo de hardware.
- **Faiz *et al.* (2025, arXiv:2501.08205) — "Modeling Feature Maps for Quantum Machine Learning":** modela el impacto del ruido sobre QSVC, Pegasos-QSVC, QNN y VQC con distintos feature maps; útil como referencia para sensibilidad al ruido en el proyecto.
- **Biswas (2025, arXiv:2503.14062):** propone codificación híbrida que combina ZZFeatureMap con RealAmplitudes para VQC en Qiskit.
- **Gupta *et al.* (2025, *npj Digital Medicine*, DOI:10.1038/s41746-025-01597-z):** revisión sistemática que concluye que en datos clínicos clásicos **no hay evidencia consistente de utilidad cuántica empírica**, hallazgo crítico que debe citarse para honestidad académica.
- **Piatrenka y Rusek (2022, en *ICCS Lecture Notes*, DOI:10.1007/978-3-031-08760-8_21) — "Quantum Variational Multi-class Classifier for the Iris Data Set":** benchmark directo de multiclase Iris con VQC en simulador y hardware IBM.
- **Hou *et al.* (2024) en Iris con QMCC variacional reportan ~92.10 % de accuracy.**
- **Duan, Sun, Hsieh (2024, *Quantum Inf. Process.* 23:92):** clasificador VQC paralelizado con QRAM en PennyLane sobre Iris.
- **Lay *et al.* (2024, *Sci. Rep.* / PMC9968349):** QSVM aproximada variacional con transferencia de inferencia.

### 3.3 Limitaciones actuales del QML (era NISQ)

1. **Ruido y decoherencia:** errores de puerta de uno y dos qubits (~10⁻³–10⁻² en hardware superconductor), errores de lectura (~1–3 %), y T1/T2 finitos limitan la profundidad útil de circuito.
2. **Número limitado de qubits:** aunque IBM tiene chips de >1000 qubits (Condor, Heron), la conectividad y la fidelidad limitan el rendimiento efectivo.
3. **Barren plateaus:** la varianza del gradiente decae exponencialmente con n qubits para circuitos suficientemente expresivos.
4. **Coste de medición (shots):** estimar valores esperados con error ε requiere O(1/ε²) shots.
5. **Sobreajuste y "trucos clásicos":** estudios como Kübler *et al.* (2021) y Schuld (2022) muestran que kernels cuánticos sin sesgo inductivo apropiado se vuelven planos y no generalizan mejor que kernels clásicos.

---

## 4. Qiskit Machine Learning — detalles técnicos actualizados

### 4.1 Versiones y arquitectura (2025–2026)

- **Qiskit SDK:** 1.x → 2.x (estable). Qiskit ML 0.8 requiere Qiskit ≥1.0; Qiskit ML 0.9 requiere Python ≥3.10.
- **Qiskit Machine Learning 0.8.x** introduce soporte para **Primitives V2** (EstimatorV2, SamplerV2). Versión 0.9 elimina V1.
- **Migración de qiskit-algorithms:** módulos `gradients`, `optimizers`, `state_fidelities`, `utils.algorithm_globals` ahora viven dentro de `qiskit_machine_learning` (cambio de imports desde `qiskit_algorithms.*`).
- **Optimizadores incluidos:** COBYLA, SPSA, ADAM, L-BFGS-B, NELDER_MEAD, SLSQP, SciPyOptimizer.
- **Mantenimiento:** co-mantenido por IBM Quantum y el Hartree Centre (STFC) desde 0.7; paper técnico de referencia Sahin *et al.* (2025).

### 4.2 API actual

```python
# Clasificador conveniente
from qiskit_machine_learning.algorithms import VQC, QSVC
# QNNs subyacentes
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
# Kernels
from qiskit_machine_learning.kernels import FidelityQuantumKernel
# Optimizadores
from qiskit_machine_learning.optimizers import COBYLA, SPSA, ADAM, L_BFGS_B
```

VQC construye internamente un SamplerQNN; QSVC envuelve `sklearn.svm.SVC` reemplazando el kernel por un `FidelityQuantumKernel`. Ambos son compatibles con `fit/predict/score` de scikit-learn.

### 4.3 Primitives V2 — PUB (Primitive Unified Bloc)

Una PUB para `SamplerV2` es `(circuit, parameter_values, shots)` y para `EstimatorV2` es `(circuit, observables, parameter_values, precision)`. **Importante:** desde Qiskit 2.x, el valor por defecto de shots en Sampler es 1024 (antes era infinito en statevector); esto debe especificarse explícitamente y reportarse en el artículo para reproducibilidad (Cardinal *et al.* 2025).

### 4.4 Backends

| Backend | Uso |
|---------|-----|
| `StatevectorSampler` / `StatevectorEstimator` (reference) | Simulación exacta sin ruido, ideal para prototipado |
| `qiskit_aer.primitives.SamplerV2 / EstimatorV2` (Aer) | Simulación rápida con ruido opcional |
| `FakeBackendV2` (fake providers de `qiskit_ibm_runtime.fake_provider`) | Simular ruido realista de chips específicos (FakeManilaV2, FakeKyiv, etc.) |
| `qiskit_ibm_runtime` con backend real (`ibm_kyoto`, `ibm_brisbane`, etc.) | Ejecución en hardware real (requiere cuenta IBM Quantum gratuita) |

---

## 5. Comparación de frameworks QML

| Característica | **Qiskit ML** | PennyLane | TensorFlow Quantum |
|----------------|--------------|-----------|--------------------|
| Mantenedor | IBM + Hartree Centre | Xanadu | Google |
| Diferenciación automática | Via parameter-shift; TorchConnector para PyTorch | Nativa, hardware-agnóstica (PyTorch/JAX/TF) | Vía TensorFlow + Cirq |
| Hardware | Acceso nativo a IBM Quantum (Heron, Eagle) | Multi-backend (IBM, AWS, IonQ, etc.) | Restringido a Cirq/Google |
| QML específico | VQC, QSVC, QNNs, kernels | Plugin-rich, foco en QML híbrido | Capa de TF para PQCs |
| Curva de aprendizaje | Moderada; documentación extensa | Excelente para investigadores ML | Más alta, menos activo desde 2023 |
| Hardware-as-a-service | IBM Quantum (acceso gratuito limitado) | Sí (varios proveedores) | Limitado |

**Justificación para este proyecto:** la guía del curso recomienda Qiskit; además, Rodríguez-Díaz *et al.* (2024, *Advances in AI*, DOI:10.1007/978-3-031-62799-6_13) reportan que **Qiskit produce accuracy de clasificación superior a PennyLane** en QSVMs híbridos en su benchmark de hasta 20 qubits, lo que respalda la elección para el caso de Iris. Khan *et al.* (2024, *Asian Bulletin of Big Data Management*) ofrece tabla comparativa: PennyLane 92 % vs Qiskit, TFQ más rápido en simulación clásica.

---

## 6. Metodología de implementación detallada

### 6.1 Preprocesamiento del dataset Iris

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)  # escalar a [0, π] para ángulos
y_oh = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_oh, test_size=0.3, stratify=y, random_state=42)
```

- **4 features** → **4 qubits** con codificación de ángulo directa (sin PCA).
- Normalización a [0, π] o [0, 2π] (ZZFeatureMap multiplica por 2(π−x_i)(π−x_j)).
- **One-hot encoding** porque VQC con `interpret` parity-based y tres clases requiere mapeo explícito.
- **Estratificación** para preservar la proporción 50/50/50 entre setosa/versicolor/virginica.

### 6.2 Diseño del modelo VQC

```python
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler

algorithm_globals.random_seed = 42

num_qubits = 4
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
ansatz      = RealAmplitudes(num_qubits=num_qubits, reps=3, entanglement='linear')

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    loss='cross_entropy',
    optimizer=COBYLA(maxiter=200),
    sampler=StatevectorSampler(),
)

vqc.fit(X_tr, y_tr)
print("Accuracy test:", vqc.score(X_te, y_te))
```

- **Profundidad total** = 2 capas de feature map + 3 capas de ansatz ≈ 30–40 puertas; profundidad lógica ~20 (estimar con `transpile(...).depth()`).
- **Número de parámetros entrenables** = num_qubits·(reps+1) = 4·4 = 16 para RealAmplitudes con reps=3.
- **Shots:** 1024–4096 para simulación con ruido; statevector exacto para desarrollo.

### 6.3 Optimizadores recomendados

| Optimizador | Ventaja | Desventaja | Uso típico |
|-------------|---------|------------|------------|
| **COBYLA** | Gradient-free, robusto a ruido moderado | Convergencia lenta en muchas dimensiones | Default razonable para VQC en Iris |
| **SPSA** | Excelente con ruido y muchos parámetros | Hiperparámetros sensibles | Hardware real, problemas grandes |
| **L-BFGS-B** | Convergencia cuadrática local | Sensible a ruido | Simulador exacto |
| **ADAM** | Adaptativo | Necesita gradiente confiable | Híbrido con TorchConnector |

Para Iris, se recomienda **probar COBYLA y SPSA** y reportar ambos, ya que es un experimento de bajo costo y enriquece la discusión.

### 6.4 Implementación de QSVM (QSVC con kernel cuántico)

```python
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=quantum_kernel)
qsvc.fit(X_tr, y_tr.argmax(axis=1))      # QSVC necesita labels enteras
print("Accuracy QSVC:", qsvc.score(X_te, y_te.argmax(axis=1)))
```

QSVC delega la lógica multiclase one-vs-one a `sklearn.svm.SVC`. El coste dominante es construir la matriz de kernel (n²/2 evaluaciones de fidelidad), por lo que para Iris (105 muestras de entrenamiento) son ~5500 circuitos a 1024 shots.

### 6.5 Líneas base clásicas

```python
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_tr, y_tr.argmax(1))
mlp     = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=2000,
                        random_state=42).fit(X_tr, y_tr.argmax(1))
knn     = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr.argmax(1))
```

### 6.6 Esquema de validación

- **Hold-out 70/30 estratificado** con `random_state=42` para reproducibilidad.
- **5-fold cross-validation estratificado** para los modelos clásicos y, si el presupuesto computacional lo permite, para VQC (con 3 folds).
- Reportar **media ± desviación estándar** sobre 5 corridas con semillas distintas para QML (varianza por estocasticidad de optimizador y shots).

### 6.7 Métricas a reportar

- **Accuracy global**, **precision**, **recall**, **F1** por clase.
- **Matriz de confusión.**
- **Tiempo de entrenamiento** (wall-clock).
- **Costo cuántico:** num_qubits, profundidad de circuito (logical y transpilada), número de parámetros, número total de circuitos ejecutados, número de shots por circuito.
- **Curva de aprendizaje** (loss vs iteración) usando el `callback` de VQC.

### 6.8 Experimento secundario: MNIST 2–3 clases

- Reducir MNIST a clases {0,1} o {0,1,2}.
- Submuestrear a 200 imágenes por clase.
- Reducir dimensionalidad con **PCA** a 4 u 8 componentes (para 4 u 8 qubits).
- Escalar a [-π,π] y entrenar el mismo VQC. Reportar como prueba de escalabilidad.

---

## 7. Resultados esperados y benchmarks de la literatura

### 7.1 Accuracies típicas reportadas en Iris

| Modelo | Accuracy reportada | Fuente |
|--------|-------------------|--------|
| SVM clásico (RBF) | 96.7 % – 98 % | scikit-learn examples; Pinzón 2023 |
| KNN (k=5) | 94 – 97 % | benchmarks estándar |
| MLP pequeño | 95 – 97 % | benchmarks estándar |
| **VQC (ZZFeatureMap + RealAmplitudes, simulador)** | **88 – 92 %** | Piatrenka & Rusek 2022; Hou *et al.* 2022; Steinmüller *et al.* 2023 |
| VQC multiclase (genético, fotónico) | 90.8 % | Hou *et al.* 2024 (*arXiv*:2412.02955) |
| QMCC variacional | 92.10 % | Bhatia *et al.* 2022 |
| QSVC con ZZFeatureMap | 90 – 96 % | varios benchmarks Iris/Wine |
| VQC en hardware real IBM (multiclase) | 70 – 85 % | Piatrenka & Rusek 2022 |

**Predicción honesta:** se espera que el VQC propuesto obtenga **~88–92 %** en simulador exacto y baje a ~80–88 % bajo ruido NISQ; el SVM clásico debería superarlo en este dataset (96–98 %). Este resultado no es un fracaso del proyecto: es **el hallazgo esperado y reportado en la literatura para problemas de baja dimensión bien separables**, y constituye la base de la **reflexión crítica** valiosa.

### 7.2 Tiempos de cómputo típicos (orden de magnitud)

- VQC en `StatevectorSampler` con 4 qubits, 200 iteraciones COBYLA: **2–10 min** en una CPU moderna.
- QSVC con kernel cuántico (matriz 105×105): **3–15 min** en Aer noisy simulator.
- SVM clásico: <1 s.
- MLP: 1–5 s.

### 7.3 ¿Cuándo puede QML superar a métodos clásicos?

Según el consenso de Cerezo *et al.* (2022, *Nat. Comput. Sci.* 2:567–576) y Huang *et al.* (2021, *Nat. Commun.* 12:2631), las ventajas potenciales aparecen cuando:

1. Los datos provienen de un proceso cuántico (química, materiales) o tienen estructura clásicamente difícil (logaritmo discreto).
2. El dataset es pequeño pero el espacio de hipótesis necesario es grande, donde el sesgo inductivo del kernel cuántico ayuda.
3. Existe simetría de grupo aprovechable por kernels covariantes.

**Iris no cumple ninguno de estos criterios.** Es un dataset linealmente casi separable con 4 features estadísticamente correlacionadas. Por eso es pedagógicamente útil para aprender QML, pero no para demostrar ventaja cuántica.

---

## 8. Análisis crítico (clave para la rúbrica)

### 8.1 Barren plateaus

Larocca *et al.* (2025) consolidan teóricamente: la varianza del gradiente para PQCs aleatorios decae como Var(∂C/∂θ) ∝ 2⁻ᵅⁿ con α ≥ 1 para distintas familias de ansatz. En Iris (4 qubits) **el problema es leve**, pero conviene documentarlo y aplicar mitigaciones:
- Inicialización pequeña (parámetros ~ N(0, 0.1²)).
- Costos locales en vez de globales.
- *Layer-wise training* o *warm starts*.
- Ansatz con simetría adecuada (no totalmente aleatorio).

### 8.2 Coste de simular circuitos cuánticos clásicamente

Simular un statevector de n qubits requiere O(2ⁿ) memoria y O(g·2ⁿ) tiempo (g = número de puertas). Para n=4 esto es trivial (16 amplitudes); pero el proyecto debe enfatizar que el "modelo cuántico" se ejecuta sobre un simulador clásico, no aporta speedup real, y que la promesa de ventaja exige hardware tolerante a fallos.

### 8.3 Ruido NISQ y mitigación

Wang *et al.* (2021, *Nat. Commun.* 12:6961) demostraron que **el ruido también induce barren plateaus** (*noise-induced barren plateaus*). Técnicas de mitigación disponibles en `qiskit_ibm_runtime`:
- *Dynamical decoupling* (`sampler.options.dynamical_decoupling.enable = True`).
- *Measurement error mitigation* (`estimator.options.resilience.measure_mitigation = True`).
- Zero-Noise Extrapolation y Probabilistic Error Cancellation (resilience_level 1–2).

### 8.4 Perspectivas futuras

- **Hardware:** roadmaps de IBM (Kookaburra >4000 qubits modulares hacia 2027), IonQ Tempo, Quantinuum H-series con códigos correctores. La era *early fault-tolerant* iniciaría hacia 2027–2030.
- **Algorítmicas:** modelos cuánticos con sesgo inductivo (geometric quantum machine learning, kernels covariantes de Glick *et al.* 2024).
- **Quantum-enhanced ML:** uso de QML como subrutina dentro de pipelines clásicos, en lugar de reemplazo.

---

## 9. Elementos diferenciadores para máxima calificación

### 9.1 Aportes originales viables (con esfuerzo medio)

1. **Estudio comparativo de feature maps:** ejecutar VQC con ZFeatureMap, ZZFeatureMap (lineal y full) y PauliFeatureMap (`['Z','Y','ZZ']`) en Iris, reportando accuracy, profundidad y tiempo. Pocas tesis lo hacen sistemáticamente para Iris.
2. **Análisis de sensibilidad a la profundidad del ansatz:** variar reps ∈ {1,2,3,5,8} en RealAmplitudes y graficar accuracy vs reps; identificar empíricamente el inicio de barren plateau o sobreajuste.
3. **Comparación de optimizadores:** COBYLA vs SPSA vs L-BFGS-B vs ADAM, con curvas de loss superpuestas y tabla de iteraciones-hasta-convergencia.
4. **Sensibilidad al ruido:** ejecutar el mismo VQC en `AerSimulator` con noise models extraídos de `FakeKyiv` o `FakeManilaV2`, reportando degradación de accuracy.
5. **Ejecución en hardware real** (si el acceso lo permite): correr el mejor VQC en un backend `ibm_brisbane` o `ibm_kyoto` con 4 qubits seleccionados por menor error de CX. Este punto es altamente diferenciador.
6. **Visualización de fronteras de decisión** proyectando a 2D (PC1, PC2 del PCA) y comparando VQC vs SVM RBF.

### 9.2 Visualizaciones obligatorias

- Diagrama del circuito completo (feature_map.compose(ansatz)) con `circuit.draw('mpl')`.
- Curva de loss vs iteración para cada optimizador.
- Matriz de confusión 3×3 para cada modelo.
- Boxplots de accuracy sobre 5 semillas para mostrar varianza.
- Heatmap de matriz de kernel cuántico (QSVC) vs RBF.
- Fronteras de decisión 2D.

### 9.3 Reproducibilidad

- Repositorio en GitHub con `README.md`, `requirements.txt` fijando versiones (`qiskit==1.x`, `qiskit-machine-learning==0.8.x`, `scikit-learn==1.4+`), notebook reproducible y semillas fijas.
- Script para regenerar todas las figuras del artículo.
- Licencia (MIT o Apache-2.0).

---

## 10. Estructura sugerida del artículo (10 páginas máximo)

| Sección | Páginas | Contenido clave |
|---------|---------|-----------------|
| Título, autor, afiliación | — | "Implementación de un modelo de aprendizaje automático cuántico utilizando Qiskit Machine Learning"; Yeison; UNSAAC – DAI |
| Resumen + Abstract + 5 keywords (ES/EN) | 0.5 | Problema, método, principal hallazgo numérico, conclusión |
| 1. Introducción | 1 | Motivación, brecha clásico/cuántico, objetivo, contribuciones |
| 2. Marco teórico | 1.5 | Qubits, PQC, feature maps, VQC, QSVM, barren plateau |
| 3. Trabajos relacionados | 1 | Tabla cronológica de papers seminales y recientes |
| 4. Metodología | 2 | Dataset, preprocesamiento, arquitectura del VQC, baseline, métricas, esquema experimental |
| 5. Resultados y análisis | 2.5 | Tablas comparativas, curvas, matrices, sensibilidad |
| 6. Discusión y reflexión crítica | 0.5 | Por qué el clásico gana en Iris; cuándo QML aportaría |
| 7. Conclusiones y trabajos futuros | 0.5 | Resumen de hallazgos + extensiones (datasets cuánticos, hw real) |
| Referencias | 0.5 | 20–30 referencias en IEEE |

---

## 11. Fragmento de código completo de referencia (Qiskit ML 0.8)

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42
np.random.seed(42)

# --- Datos ---
X, y = load_iris(return_X_y=True)
X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
y_oh = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_oh, test_size=0.3, stratify=y, random_state=42)
y_tr_lbl, y_te_lbl = y_tr.argmax(1), y_te.argmax(1)

# --- Componentes cuánticos ---
n = 4
fmap = ZZFeatureMap(feature_dimension=n, reps=2, entanglement='linear')
ansatz = RealAmplitudes(num_qubits=n, reps=3, entanglement='linear')

# --- VQC ---
history = {'loss': []}
def cb(weights, loss):
    history['loss'].append(loss)

vqc = VQC(
    feature_map=fmap, ansatz=ansatz,
    optimizer=COBYLA(maxiter=200),
    loss='cross_entropy',
    sampler=StatevectorSampler(),
    callback=cb,
)
vqc.fit(X_tr, y_tr)
acc_vqc = vqc.score(X_te, y_te)

# --- QSVC ---
qkernel = FidelityQuantumKernel(feature_map=fmap)
qsvc = QSVC(quantum_kernel=qkernel)
qsvc.fit(X_tr, y_tr_lbl)
acc_qsvc = qsvc.score(X_te, y_te_lbl)

# --- Baselines clásicas ---
svm = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_tr, y_tr_lbl)
acc_svm = svm.score(X_te, y_te_lbl)

print(f"VQC: {acc_vqc:.3f} | QSVC: {acc_qsvc:.3f} | SVM-RBF: {acc_svm:.3f}")
print("Parámetros entrenables VQC:", ansatz.num_parameters)
print("Profundidad circuito:", (fmap.compose(ansatz)).decompose().depth())
```

---

## 12. Referencias (formato IEEE, 28 entradas)

[1] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, "Quantum machine learning," *Nature*, vol. 549, no. 7671, pp. 195–202, Sept. 2017, doi: 10.1038/nature23474.

[2] V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, no. 7747, pp. 209–212, Mar. 2019, doi: 10.1038/s41586-019-0980-2.

[3] M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert spaces," *Phys. Rev. Lett.*, vol. 122, no. 4, p. 040504, Feb. 2019, doi: 10.1103/PhysRevLett.122.040504.

[4] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, "Quantum circuit learning," *Phys. Rev. A*, vol. 98, no. 3, p. 032309, Sept. 2018, doi: 10.1103/PhysRevA.98.032309.

[5] E. Farhi and H. Neven, "Classification with quantum neural networks on near term processors," arXiv:1802.06002, Feb. 2018.

[6] M. Cerezo *et al.*, "Variational quantum algorithms," *Nat. Rev. Phys.*, vol. 3, no. 9, pp. 625–644, Aug. 2021, doi: 10.1038/s42254-021-00348-9.

[7] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, "Barren plateaus in quantum neural network training landscapes," *Nat. Commun.*, vol. 9, no. 1, p. 4812, Nov. 2018, doi: 10.1038/s41467-018-07090-4.

[8] M. Larocca, S. Thanasilp, S. Wang, K. Sharma, J. Biamonte, P. J. Coles, L. Cincio, J. R. McClean, Z. Holmes, and M. Cerezo, "Barren plateaus in variational quantum computing," *Nat. Rev. Phys.*, 2025, doi: 10.1038/s42254-025-00813-9 (preprint arXiv:2405.00781).

[9] Y. Liu, S. Arunachalam, and K. Temme, "A rigorous and robust quantum speed-up in supervised machine learning," *Nat. Phys.*, vol. 17, no. 9, pp. 1013–1017, July 2021, doi: 10.1038/s41567-021-01287-z.

[10] S. Sim, P. D. Johnson, and A. Aspuru-Guzik, "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms," *Adv. Quantum Tech.*, vol. 2, no. 12, p. 1900070, Dec. 2019, doi: 10.1002/qute.201900070.

[11] M. E. Sahin, E. Altamura, O. Wallis, S. P. Wood, A. Dekusar, D. A. Millar, T. Imamichi, A. Matsuo, and S. Mensa, "Qiskit Machine Learning: an open-source library for quantum machine learning tasks at scale on quantum hardware and classical simulators," arXiv:2505.17756, May 2025.

[12] P. Rebentrost, M. Mohseni, and S. Lloyd, "Quantum support vector machine for big data classification," *Phys. Rev. Lett.*, vol. 113, no. 13, p. 130503, Sept. 2014, doi: 10.1103/PhysRevLett.113.130503.

[13] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, 10th Anniversary Ed. Cambridge, U.K.: Cambridge Univ. Press, 2010.

[14] M. Schuld and F. Petruccione, *Machine Learning with Quantum Computers*, 2nd ed. Cham, Switzerland: Springer, 2021, doi: 10.1007/978-3-030-83098-4.

[15] J. Preskill, "Quantum computing in the NISQ era and beyond," *Quantum*, vol. 2, p. 79, Aug. 2018, doi: 10.22331/q-2018-08-06-79.

[16] H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, and J. R. McClean, "Power of data in quantum machine learning," *Nat. Commun.*, vol. 12, no. 1, p. 2631, May 2021, doi: 10.1038/s41467-021-22539-9.

[17] M. Cerezo, G. Verdon, H.-Y. Huang, L. Cincio, and P. J. Coles, "Challenges and opportunities in quantum machine learning," *Nat. Comput. Sci.*, vol. 2, pp. 567–576, Sept. 2022, doi: 10.1038/s43588-022-00311-3.

[18] T. Piatrenka and M. Rusek, "Quantum variational multi-class classifier for the Iris data set," in *Computational Science – ICCS 2022*, Lecture Notes in Computer Science, vol. 13353, Springer, Cham, 2022, pp. 247–260, doi: 10.1007/978-3-031-08760-8_21.

[19] F. Rodríguez-Díaz, J. F. Torres, D. Gutiérrez-Avilés, A. Troncoso, and F. Martínez-Álvarez, "An experimental comparison of Qiskit and Pennylane for hybrid quantum-classical support vector machines," in *Advances in Artificial Intelligence (CAEPIA 2024)*, Springer, 2024, doi: 10.1007/978-3-031-62799-6_13.

[20] J. Cunningham and J. Zhuang, "Investigating and mitigating barren plateaus in variational quantum circuits: A survey," *Quantum Inf. Process.*, vol. 24, 2025, doi: 10.1007/s11128-025-04665-1 (preprint arXiv:2407.17706).

[21] H. Biswas, "Data encoding for VQC in Qiskit, a comparison with novel hybrid encoding," arXiv:2503.14062, Mar. 2025.

[22] S. Faiz *et al.*, "Modeling feature maps for quantum machine learning," arXiv:2501.08205, Jan. 2025.

[23] R. S. Gupta, C. E. Wood, T. Engstrom, J. D. Pole, and S. Shrapnel, "A systematic review of quantum machine learning for digital health," *npj Digit. Med.*, May 2025, doi: 10.1038/s41746-025-01597-z.

[24] Y. Wang and J. Liu, "A comprehensive review of quantum machine learning: from NISQ to fault tolerance," arXiv:2401.11351, Jan. 2024.

[25] B. Duan, X. Sun, and C.-Y. Hsieh, "Parallelized variational quantum classifier with shallow QRAM circuit," *Quantum Inf. Process.*, vol. 23, no. 92, 2024, doi: 10.1007/s11128-024-04295-z.

[26] IBM Quantum, "Qiskit Runtime V2 primitives migration guide," *IBM Quantum Documentation*, 2024–2026. [Online]. Available: https://docs.quantum.ibm.com/migration-guides/v2-primitives

[27] Qiskit Community, "Qiskit Machine Learning 0.8/0.9 documentation and release notes," 2025–2026. [Online]. Available: https://qiskit-community.github.io/qiskit-machine-learning/

[28] S. Wang, E. Fontana, M. Cerezo, K. Sharma, A. Sone, L. Cincio, and P. J. Coles, "Noise-induced barren plateaus in variational quantum algorithms," *Nat. Commun.*, vol. 12, no. 1, p. 6961, Nov. 2021, doi: 10.1038/s41467-021-27045-6.

---

## 13. Notas finales para el redactor

- **Idioma del artículo final:** español, con abstract en inglés. El proyecto valora especialmente la claridad expositiva (los reviewers de la rúbrica suelen castigar el "pegado" de fórmulas sin explicación intuitiva). Cada ecuación debe acompañarse de una frase explicativa.
- **Honestidad académica:** reportar el accuracy real obtenido. Si el VQC obtiene menos que el SVM clásico, esto **no es un fracaso**: es el resultado esperado y respaldado por la revisión sistemática de Gupta *et al.* (2025). La fortaleza del artículo está en la **discusión crítica** de por qué.
- **Reflexión crítica (5 % de la nota):** dedicar al menos medio párrafo a la pregunta "¿cuándo y bajo qué condiciones esperaríamos que el QML supere al clásico?", citando Liu *et al.* 2021 y Huang *et al.* 2021.
- **Repositorio reproducible:** subir notebooks ejecutados (con outputs y figuras), `environment.yml` o `requirements.txt`, README con instrucciones paso a paso, y un script `run_all_experiments.py`.
- **Presentación oral:** preparar máximo 12 diapositivas (motivación, fundamentos, metodología, resultados con gráficos grandes, discusión, conclusiones) y un demo en vivo de 2 minutos del notebook.
- **Posible cita inicial impactante:** "*Quantum computers may outperform classical computers on machine learning tasks*" – Biamonte *et al.*, Nature 2017. Cierre crítico con la conclusión de Gupta *et al.* 2025: la evidencia empírica de utilidad cuántica en datos clásicos sigue siendo inconclusa, lo que refuerza la importancia de proyectos didácticos rigurosos como el presente.

Con esta base, el artículo, el código y la presentación cumplen con todos los puntos de la rúbrica: rigor técnico (Qiskit ML 0.8/0.9 con API actual), profundidad teórica (papers seminales + revisiones 2024–2026), reproducibilidad, comparación justa y reflexión crítica original.