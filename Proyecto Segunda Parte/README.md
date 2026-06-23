# Implementación en Hardware Real (IBM Quantum) - Segunda Parte

Este subproyecto documenta la ejecución de algoritmos de Machine Learning Cuántico (VQC y QSVC) sobre procesadores físicos ruidosos de la era NISQ, a través de la plataforma en la nube **IBM Quantum**.

## 1. Backend Utilizado
Para el desarrollo de los experimentos se utilizó el backend `ibm_marrakesh`. 
* **Arquitectura:** Procesador de grado de utilidad IBM Heron r2.
* **Topología:** Acoplamiento hexagonal pesado enfocado en reducir la interferencia de canal (*cross-talk*).
* **Qubits:** 156 qubits físicos disponibles.

## 2. Primitivas de Ejecución (Qiskit Runtime)
La transición del simulador `Statevector` al hardware real requirió el uso de Primitivas V2 de Qiskit Runtime, específicamente `SamplerV2`.
* Se ejecutaron los circuitos parametrizados enviando los "Pubs" (Primitive Unified Blocs) optimizados para el hardware con el *transpiler* `OptimizationLevel=3`.
* Las mediciones de los vectores de expectación estadísticos se corrieron a **2048 shots** por circuito.

## 3. Limitaciones de Cuota (Open Plan)
Dado que el procesamiento asíncrono se ejecutó en la capa de acceso gratuito (*Open Plan*), que restringe el cómputo a 10 minutos de QPU reales por mes:
* **Subset de Evaluación Extrema:** El tamaño muestral del kernel debió encogerse masivamente. De las 150 muestras canónicas del dataset Iris, la matriz del Kernel se construyó evaluando únicamente **10 vectores de entrenamiento** y **5 de prueba**.
* **Impacto del Cuello de Botella:** Al carecer de suficiente densidad estadística en un hiperplano, el modelo degeneró estancando sus predicciones en un 40% de *accuracy*. Este problema subyace en el subajuste de los datos de entrada (*underfitting* estadístico), enmascarando cualquier mejora potencial del procesador.

## 4. Mitigación de Errores (TREX)
Se implementó activamente la técnica nativa **Twirled Readout Error Extinction (TREX)**.
* **Mecanismo:** El compilador inserta puertas de Pauli-X aleatorias justo antes de los pulsos de medición para "simetrizar" y transformar el asimétrico ruido del readout (SPAM) en un canal de depolización inversible clásico.
* **Resultado:** Ante el extremo sesgo producido por la baja cuota de datos (10 instancias), la limpieza fina de la distribución de medición generada por TREX fue imperceptible. El *accuracy* se mantuvo bloqueado, lo que demuestra que los métodos algorítmicos no subsanan la sub-representatividad de un dataset.

---
**Nota:** El código fuente completo (incluida la limpieza de dimensionalidad con PCA para el modelo visual de MNIST y el rediseño de las épocas de optimización con COBYLA) está ubicado en los *Notebooks* (.ipynb) de este directorio.
