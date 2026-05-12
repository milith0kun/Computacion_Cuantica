# Aprendizaje Automático con Computación Cuántica

**Universidad Nacional de San Antonio Abad del Cusco**  
**Departamento Académico de Informática**  
**Curso:** Computación Cuántica  

## 📌 Descripción del Proyecto

Este proyecto implementa y evalúa modelos de **Aprendizaje Automático Cuántico (QML)** utilizando **Qiskit Machine Learning**, comparando su desempeño con métodos clásicos de Machine Learning (SVM-RBF, MLP, KNN). 

El objetivo es clasificar el clásico dataset de **Iris** mediante modelos cuánticos como:
- **VQC (Variational Quantum Classifier)**
- **QSVC (Quantum Support Vector Classifier con Kernel Cuántico)**

Se analiza el impacto de diferentes *feature maps*, la profundidad del *ansatz*, el uso de distintos optimizadores (COBYLA, SPSA, etc.) y la sensibilidad al ruido en entornos cuánticos simulados.

## 🛠️ Requisitos e Instrucciones de Instalación

Para ejecutar este proyecto, necesitas Python 3.8+ y las siguientes librerías. Se recomienda usar un entorno virtual.

### 1. Crear y activar el entorno virtual

En Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

En Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar las dependencias

Instala los paquetes necesarios utilizando `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn
pip install qiskit qiskit-machine-learning
```

*Nota: También puedes instalar jupyter si deseas ejecutar los notebooks incluidos.*
```bash
pip install jupyter
```

## 🚀 Ejemplo de Ejecución

El proyecto está dividido en scripts de Python que puedes ejecutar directamente. 

### Parte 1: Modelos Base y Comparación Inicial
Ejecuta el primer script para entrenar los modelos clásicos (SVM, MLP, KNN) y los modelos cuánticos base (VQC, QSVC), y generar gráficas comparativas.

```bash
python QML_Proyecto_Parte1.py
```
*(Los gráficos generados se guardarán automáticamente en la carpeta `figuras/` o se mostrarán en pantalla).*

### Parte 2: Experimentos Avanzados
Ejecuta el segundo script para pruebas más avanzadas como la evaluación de diferentes Feature Maps (ZZFeatureMap vs ZFeatureMap), profundidad del circuito, diferentes optimizadores cuánticos y simulación con ruido.

```bash
python QML_Proyecto_Parte2.py
```

## 📊 Resultados Esperados

Al ejecutar los scripts, el sistema imprimirá las métricas de evaluación (Accuracy, Precision, Recall, F1-Score) y el tiempo de entrenamiento de cada modelo. Además, generará visualizaciones como:
- Matrices de confusión.
- Gráficos comparativos de rendimiento y tiempo.
- Gráficos de curvas de pérdida para los algoritmos variacionales cuánticos.
- Fronteras de decisión proyectadas mediante PCA.

---
*Este proyecto fue desarrollado como parte de la evaluación semestral del curso de Computación Cuántica.*