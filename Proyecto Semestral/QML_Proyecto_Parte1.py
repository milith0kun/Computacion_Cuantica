# %% [markdown]
# # Implementación de un Modelo de Aprendizaje Automático Cuántico
# ## Utilizando Qiskit Machine Learning
# **Universidad Nacional de San Antonio Abad del Cusco** — Departamento Académico de Informática
#
# **Curso:** Computación Cuántica | **Estudiante:** Yeison | **Fecha:** Mayo 2026
#
# ---
# Este notebook implementa un Clasificador Cuántico Variacional (VQC) y una Máquina de
# Vectores de Soporte Cuántica (QSVC), comparándolos con modelos clásicos (SVM-RBF, MLP, KNN)
# sobre el dataset Iris. Se incluyen análisis de feature maps, profundidad de ansatz,
# comparación de optimizadores y sensibilidad al ruido.

# %% [markdown]
# ## 1. Configuración e Importaciones

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA

from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap, RealAmplitudes, EfficientSU2
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.optimizers import COBYLA, SPSA, L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42
np.random.seed(42)

plt.rcParams.update({
    'figure.figsize': (10, 6), 'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 12,
})

print("✅ Todas las librerías cargadas correctamente.")

# %% [markdown]
# ## 2. Carga y Preprocesamiento del Dataset Iris
# El dataset Iris tiene 150 muestras, 4 características y 3 clases (setosa, versicolor, virginica).
# Se normaliza a [0, π] para codificación angular en circuitos cuánticos.

# %%
X, y = load_iris(return_X_y=True)
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Normalizar a [0, π] para codificación angular
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# One-hot encoding para VQC (requiere etiquetas one-hot)
ohe = OneHotEncoder(sparse_output=False)
y_oh = ohe.fit_transform(y.reshape(-1, 1))

# Split estratificado 70/30
X_tr, X_te, y_tr_oh, y_te_oh = train_test_split(
    X_scaled, y_oh, test_size=0.3, stratify=y, random_state=42)
y_tr_lbl = y_tr_oh.argmax(axis=1)
y_te_lbl = y_te_oh.argmax(axis=1)

print(f"Entrenamiento: {X_tr.shape[0]} muestras | Test: {X_te.shape[0]} muestras")
print(f"Distribución entrenamiento: {np.bincount(y_tr_lbl)}")
print(f"Distribución test:          {np.bincount(y_te_lbl)}")

# %% [markdown]
# ### 2.1 Análisis Exploratorio de Datos (EDA)

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Análisis Exploratorio del Dataset Iris', fontsize=16, fontweight='bold')

# Scatter plots de pares de features
pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
colors = ['#2196F3', '#FF9800', '#4CAF50']
for idx, (i, j) in enumerate(pairs):
    ax = axes[idx//3][idx%3]
    for c in range(3):
        mask = y == c
        ax.scatter(X[mask, i], X[mask, j], c=colors[c], label=target_names[c],
                   alpha=0.7, edgecolors='k', linewidth=0.5, s=50)
    ax.set_xlabel(feature_names[i], fontsize=9)
    ax.set_ylabel(feature_names[j], fontsize=9)
    if idx == 0: ax.legend(fontsize=8)
axes[0][0].set_title('Distribución por pares de características')
plt.tight_layout()
plt.savefig('figuras/eda_iris.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 El gráfico muestra la separabilidad entre clases. Setosa es linealmente separable.")

# %% [markdown]
# ## 3. Modelos Clásicos de Referencia (Baseline)
# Se implementan SVM con kernel RBF, Red Neuronal (MLP) y KNN como líneas base.

# %%
resultados = {}

# --- SVM con kernel RBF ---
t0 = time.time()
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_tr, y_tr_lbl)
t_svm = time.time() - t0
y_pred_svm = svm_rbf.predict(X_te)
acc_svm = accuracy_score(y_te_lbl, y_pred_svm)
resultados['SVM-RBF'] = {
    'accuracy': acc_svm, 'tiempo': t_svm,
    'y_pred': y_pred_svm, 'tipo': 'Clásico'
}

# --- MLP (Red Neuronal) ---
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42)
mlp.fit(X_tr, y_tr_lbl)
t_mlp = time.time() - t0
y_pred_mlp = mlp.predict(X_te)
acc_mlp = accuracy_score(y_te_lbl, y_pred_mlp)
resultados['MLP'] = {
    'accuracy': acc_mlp, 'tiempo': t_mlp,
    'y_pred': y_pred_mlp, 'tipo': 'Clásico'
}

# --- KNN ---
t0 = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_tr, y_tr_lbl)
t_knn = time.time() - t0
y_pred_knn = knn.predict(X_te)
acc_knn = accuracy_score(y_te_lbl, y_pred_knn)
resultados['KNN'] = {
    'accuracy': acc_knn, 'tiempo': t_knn,
    'y_pred': y_pred_knn, 'tipo': 'Clásico'
}

print("=" * 55)
print("RESULTADOS MODELOS CLÁSICOS")
print("=" * 55)
for name, r in resultados.items():
    print(f"  {name:12s} → Accuracy: {r['accuracy']:.4f} | Tiempo: {r['tiempo']:.4f}s")

# Validación cruzada 5-fold para los modelos clásicos
print("\n📋 Validación Cruzada 5-Fold:")
for name, model in [('SVM-RBF', svm_rbf), ('MLP', mlp), ('KNN', knn)]:
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"  {name:12s} → {scores.mean():.4f} ± {scores.std():.4f}")

# %% [markdown]
# ## 4. Diseño del Circuito Cuántico
# Visualización del feature map (ZZFeatureMap) y el ansatz (RealAmplitudes).

# %%
num_qubits = 4
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3, entanglement='linear')
circuito_completo = feature_map.compose(ansatz)

print(f"Número de qubits: {num_qubits}")
print(f"Parámetros del feature map: {feature_map.num_parameters}")
print(f"Parámetros entrenables (ansatz): {ansatz.num_parameters}")
print(f"Profundidad del circuito completo: {circuito_completo.decompose().depth()}")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))
feature_map.decompose().draw('mpl', ax=axes[0])
axes[0].set_title('Feature Map: ZZFeatureMap (reps=2, linear)', fontsize=13)
ansatz.decompose().draw('mpl', ax=axes[1])
axes[1].set_title('Ansatz: RealAmplitudes (reps=3, linear)', fontsize=13)
circuito_completo.decompose().draw('mpl', ax=axes[2])
axes[2].set_title('Circuito Completo: Feature Map + Ansatz', fontsize=13)
plt.tight_layout()
plt.savefig('figuras/circuitos.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Modelo Cuántico 1: VQC (Variational Quantum Classifier)
# Clasificador cuántico variacional con ZZFeatureMap + RealAmplitudes + COBYLA.

# %%
# Callback para registrar la curva de pérdida
history_vqc = {'loss': [], 'weights': []}
def callback_vqc(weights, loss, *args):
    history_vqc['loss'].append(loss)

print("🔄 Entrenando VQC (esto puede tardar ~5-10 minutos)...")
t0 = time.time()

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    loss='cross_entropy',
    optimizer=COBYLA(maxiter=200),
    sampler=StatevectorSampler(),
    callback=callback_vqc,
)
vqc.fit(X_tr, y_tr_oh)
t_vqc = time.time() - t0

acc_vqc = vqc.score(X_te, y_te_oh)
y_pred_vqc = vqc.predict(X_te).argmax(axis=1) if len(vqc.predict(X_te).shape) > 1 else vqc.predict(X_te)
# Manejar predicciones one-hot vs label
y_pred_raw = vqc.predict(X_te)
if len(y_pred_raw.shape) > 1:
    y_pred_vqc = y_pred_raw.argmax(axis=1)
else:
    y_pred_vqc = y_pred_raw.astype(int)

resultados['VQC'] = {
    'accuracy': acc_vqc, 'tiempo': t_vqc,
    'y_pred': y_pred_vqc, 'tipo': 'Cuántico'
}

print(f"✅ VQC entrenado en {t_vqc:.1f}s")
print(f"   Accuracy en test: {acc_vqc:.4f}")
print(f"   Iteraciones de pérdida registradas: {len(history_vqc['loss'])}")

# Curva de pérdida
plt.figure(figsize=(10, 5))
plt.plot(history_vqc['loss'], color='#7B1FA2', linewidth=2)
plt.xlabel('Iteración')
plt.ylabel('Cross-Entropy Loss')
plt.title('Curva de Pérdida del VQC (COBYLA)')
plt.grid(True, alpha=0.3)
plt.savefig('figuras/vqc_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Modelo Cuántico 2: QSVC (Quantum SVM con Kernel Cuántico)
# Utiliza FidelityQuantumKernel con ZZFeatureMap y delega a sklearn.svm.SVC.

# %%
print("🔄 Entrenando QSVC (construyendo matriz de kernel cuántico)...")
t0 = time.time()

qkernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=qkernel)
qsvc.fit(X_tr, y_tr_lbl)
t_qsvc = time.time() - t0

y_pred_qsvc = qsvc.predict(X_te)
acc_qsvc = accuracy_score(y_te_lbl, y_pred_qsvc)

resultados['QSVC'] = {
    'accuracy': acc_qsvc, 'tiempo': t_qsvc,
    'y_pred': y_pred_qsvc, 'tipo': 'Cuántico'
}

print(f"✅ QSVC entrenado en {t_qsvc:.1f}s")
print(f"   Accuracy en test: {acc_qsvc:.4f}")

# %% [markdown]
# ## 7. Comparación General: Clásico vs Cuántico

# %%
print("\n" + "=" * 70)
print("TABLA COMPARATIVA: TODOS LOS MODELOS")
print("=" * 70)
print(f"{'Modelo':15s} {'Tipo':10s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Tiempo':>10s}")
print("-" * 70)

for name, r in resultados.items():
    yp = r['y_pred']
    yt = y_te_lbl
    prec = precision_score(yt, yp, average='macro', zero_division=0)
    rec = recall_score(yt, yp, average='macro', zero_division=0)
    f1 = f1_score(yt, yp, average='macro', zero_division=0)
    print(f"{name:15s} {r['tipo']:10s} {r['accuracy']:10.4f} {prec:10.4f} {rec:10.4f} {f1:10.4f} {r['tiempo']:8.2f}s")

# Gráfico de barras comparativo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
names = list(resultados.keys())
accs = [resultados[n]['accuracy'] for n in names]
times = [resultados[n]['tiempo'] for n in names]
colors_bar = ['#2196F3' if resultados[n]['tipo']=='Clásico' else '#9C27B0' for n in names]

bars = ax1.bar(names, accs, color=colors_bar, edgecolor='white', linewidth=1.5)
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy: Clásico vs Cuántico')
ax1.set_ylim(0, 1.1)
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc:.1%}', ha='center', fontweight='bold')
ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
ax1.legend()

ax2.bar(names, times, color=colors_bar, edgecolor='white', linewidth=1.5)
ax2.set_ylabel('Tiempo (s)')
ax2.set_title('Tiempo de Entrenamiento')
ax2.set_yscale('log')
plt.tight_layout()
plt.savefig('figuras/comparacion_general.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Matrices de Confusión

# %%
fig, axes = plt.subplots(1, len(resultados), figsize=(5*len(resultados), 4))
if len(resultados) == 1: axes = [axes]
for idx, (name, r) in enumerate(resultados.items()):
    cm = confusion_matrix(y_te_lbl, r['y_pred'])
    im = axes[idx].imshow(cm, cmap='Purples', interpolation='nearest')
    axes[idx].set_title(f'{name}\nAcc: {r["accuracy"]:.2%}', fontweight='bold')
    axes[idx].set_xlabel('Predicción')
    axes[idx].set_ylabel('Real')
    axes[idx].set_xticks(range(3))
    axes[idx].set_yticks(range(3))
    axes[idx].set_xticklabels(target_names, rotation=45, fontsize=8)
    axes[idx].set_yticklabels(target_names, fontsize=8)
    for i in range(3):
        for j in range(3):
            axes[idx].text(j, i, str(cm[i,j]), ha='center', va='center',
                          fontweight='bold', fontsize=14,
                          color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig('figuras/matrices_confusion.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Fronteras de Decisión (Proyección PCA 2D)

# %%
pca = PCA(n_components=2, random_state=42)
X_tr_2d = pca.fit_transform(X_tr)
X_te_2d = pca.transform(X_te)
x_min, x_max = X_tr_2d[:,0].min()-0.5, X_tr_2d[:,0].max()+0.5
y_min, y_max = X_tr_2d[:,1].min()-0.5, X_tr_2d[:,1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Entrenar SVM en 2D para visualizar fronteras
scaler_2d = MinMaxScaler(feature_range=(0, np.pi))
X_tr_2d_s = scaler_2d.fit_transform(X_tr_2d)
svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_tr_2d_s, y_tr_lbl)

grid_2d = scaler_2d.transform(np.c_[xx.ravel(), yy.ravel()])
Z_svm = svm_2d.predict(grid_2d).reshape(xx.shape)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.contourf(xx, yy, Z_svm, alpha=0.3, cmap='coolwarm')
for c, color, name in zip([0,1,2], ['#2196F3','#FF9800','#4CAF50'], target_names):
    mask = y_te_lbl == c
    ax.scatter(X_te_2d[mask,0], X_te_2d[mask,1], c=color, label=name,
               edgecolors='k', s=80, linewidth=0.8)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
ax.set_title('Frontera de Decisión SVM-RBF (proyección PCA 2D)')
ax.legend()
plt.savefig('figuras/frontera_decision.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Parte 1 completada. Ejecutar Parte 2 para experimentos avanzados.")
