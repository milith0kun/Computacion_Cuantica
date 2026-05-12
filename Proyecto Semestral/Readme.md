Gu ́ıa de Proyecto

Aprendizaje Autom ́atico con Computaci ́on Cu ́antica

Universidad Nacional de San Antonio Abad del Cusco
Departamento Acad ́emico de Inform ́atica
Curso: Computaci ́on Cu ́antica

Implementaci ́on de un modelo de aprendizaje autom ́ati-
co cu ́antico utilizando Qiskit Machine Learning

1. Objetivo general
Dise ̃nar e implementar un modelo de aprendizaje autom ́atico basado en principios de
computaci ́on cu ́antica, evaluando su desempe ̃no frente a m ́etodos cl ́asicos.
2. Objetivos espec ́ıficos
Comprender los fundamentos del Machine Learning cu ́antico (Quantum ML).
Implementar un modelo de clasificaci ́on o regresi ́on usando Qiskit Machine Learning u
otro framework cu ́antico.
Comparar los resultados con un modelo cl ́asico equivalente (por ejemplo, SVM o red
neuronal tradicional).
Analizar el impacto y las limitaciones actuales del aprendizaje cu ́antico en problemas
reales.

3. Justificaci ́on

El aprendizaje autom ́atico cu ́antico es un campo emergente que busca aprovechar la su-
perposici ́on y el entrelazamiento cu ́antico para acelerar tareas de aprendizaje y optimizaci ́on.

Este proyecto permitir ́a explorar sus ventajas potenciales frente a la computaci ́on cl ́asica,
contribuyendo al entendimiento de las tecnolog ́ıas h ́ıbridas cu ́antico–cl ́asicas.

1

4. Marco te ́orico
El informe debe incluir conceptos de:
Computaci ́on cu ́antica: qubits, puertas cu ́anticas, circuitos, medici ́on y transformada
cu ́antica de Fourier.
Machine Learning cl ́asico: clasificaci ́on, regresi ́on, aprendizaje supervisado y no
supervisado.
Quantum Machine Learning (QML):
• Quantum Support Vector Machine (QSVM)
• Variational Quantum Classifier (VQC)
• Quantum Neural Networks (QNN)
Frameworks: Qiskit Machine Learning, PennyLane o TensorFlow Quantum.
5. Metodolog ́ıa
5.1. Etapa 1: Investigaci ́on y dise ̃no
Revisi ́on bibliogr ́afica sobre QML.
Selecci ́on del tipo de modelo cu ́antico (VQC, QSVM, QNN, etc.).
Elecci ́on de un dataset peque ̃no y representativo (por ejemplo, Iris, MNIST reducido o
datos sint ́eticos).
5.2. Etapa 2: Implementaci ́on
Construcci ́on del modelo cl ́asico de referencia (Scikit-Learn).
Implementaci ́on del modelo cu ́antico usando Qiskit o PennyLane.
Entrenamiento y validaci ́on.
5.3. Etapa 3: Evaluaci ́on
Comparaci ́on entre desempe ̃no cl ́asico y cu ́antico (accuracy, tiempo de c ́omputo, costo
cu ́antico).
Discusi ́on sobre ventajas y limitaciones observadas.

2

5.4. Etapa 4: Documentaci ́on y presentaci ́on
Redacci ́on del informe cient ́ıfico.
Creaci ́on de un repositorio reproducible (c ́odigo, datos, documentaci ́on).
Preparaci ́on de una presentaci ́on oral con resultados.
6. Entregables
6.1. Informe en formato art ́ıculo cient ́ıfico
Estructura sugerida (m ́ax. 10 p ́aginas):
1. T ́ıtulo, autores y afiliaci ́on
2. Resumen y palabras clave
3. Introducci ́on y justificaci ́on
4. Marco te ́orico y antecedentes
5. Metodolog ́ıa
6. Resultados y an ́alisis
7. Conclusiones y trabajos futuros
8. Referencias (APA o IEEE)
6.2. Repositorio (GitHub o GitLab)
Debe incluir:
C ́odigo fuente (Jupyter Notebook o Python)
Dataset usado
Resultados o gr ́aficos
Archivo README.md con:
• Descripci ́on del proyecto
• Instrucciones de instalaci ́on
• Ejemplo de ejecuci ́on

3

6.3. Presentaci ́on oral
Duraci ́on: 15–20 minutos
Contenido:
• Problema abordado y objetivos
• Breve explicaci ́on te ́orica del modelo cu ́antico
• Demostraci ́on de resultados
• Conclusiones e impacto
7. Herramientas recomendadas
Lenguaje: Python
Frameworks cu ́anticos
Qiskit Machine Learning
PennyLane
TensorFlow Quantum
Frameworks cl ́asicos
Scikit-learn
NumPy
Matplotlib
Plataformas
IBM Quantum Experience
Google Colab
GitHub
Edici ́on
Overleaf (para el art ́ıculo)
PowerPoint o Google Slides (para la presentaci ́on)

4

8. Cronograma sugerido

Semana Actividades Entregable
1 Revisi ́on te ́orica y definici ́on del modelo cu ́antico Marco te ́orico y propuesta
2 Implementaci ́on inicial y dataset Primer notebook funcional
3 Entrenamiento, resultados y comparaci ́on C ́odigo final y gr ́aficos
4 Redacci ́on del informe y exposici ́on Informe, repositorio y presentaci ́on
9. R ́ubrica de Evaluaci ́on

Criterio Descripci ́on Ponderaci ́on Puntaje
M ́ax.

Informe cient ́ıfico Claridad, profundidad te ́orica, estruc-
tura, redacci ́on y referencias acad ́emi-
cas.

35 % 7

Implementaci ́on
t ́ecnica

Correcta implementaci ́on del modelo
cu ́antico y comparaci ́on con modelo
cl ́asico; c ́odigo reproducible.

30 % 6

Repositorio Organizaci ́on del c ́odigo, documenta-
ci ́on clara (README) y uso adecuado

de control de versiones.

15 % 3

Presentaci ́on oral Claridad expositiva, dominio del tema,
calidad del material visual y an ́alisis de
resultados.

15 % 3

Innovaci ́on y refle-
xi ́on cr ́ıtica

Creatividad, aporte original y discusi ́on

sobre el impacto del aprendizaje au-
tom ́atico cu ́antico.

5 % 1

Total 100 % 20
10. Conclusi ́on esperada

El proyecto permitir ́a demostrar c ́omo la computaci ́on cu ́antica puede aplicarse al apren-
dizaje autom ́atico, evaluando su potencial frente a enfoques cl ́asicos y fomentando una com-
prensi ́on integral de los algoritmos h ́ıbridos cu ́antico–cl ́asicos.