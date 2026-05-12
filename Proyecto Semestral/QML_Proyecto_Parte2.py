# %% [markdown]
# # Parte 2: Experimentos Avanzados
# ## Comparación de Feature Maps, Profundidad de Ansatz, Optimizadores y Ruido NISQ
#
# **Nota:** Ejecutar después de la Parte 1 (variables compartidas).

# %% [markdown]
# ## 10. Experimento 1: Comparación de Feature Maps
# ZFeatureMap vs ZZFeatureMap (linear) vs ZZFeatureMap (full) vs PauliFeatureMap

# %%
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
import time, numpy as np

feature_maps = {
    'ZFeatureMap': ZFeatureMap(feature_dimension=4, reps=2),
    'ZZFeatureMap (linear)': ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear'),
    'ZZFeatureMap (full)': ZZFeatureMap(feature_dimension=4, reps=2, entanglement='full'),
    'PauliFeatureMap': PauliFeatureMap(feature_dimension=4, reps=2, paulis=['Z','ZZ']),
}

fm_results = {}
for fm_name, fm in feature_maps.items():
    print(f"🔄 Probando {fm_name}...")
    ans = RealAmplitudes(num_qubits=4, reps=3, entanglement='linear')
    hist = {'loss': []}
    def cb(w, l, *args): hist['loss'].append(l)

    t0 = time.time()
    model = VQC(
        feature_map=fm, ansatz=ans, loss='cross_entropy',
        optimizer=COBYLA(maxiter=150), sampler=StatevectorSampler(), callback=cb,
    )
    model.fit(X_tr, y_tr_oh)
    t_elapsed = time.time() - t0
    acc = model.score(X_te, y_te_oh)
    depth = fm.compose(ans).decompose().depth()

    fm_results[fm_name] = {'accuracy': acc, 'tiempo': t_elapsed, 'depth': depth, 'loss': hist['loss']}
    print(f"   ✅ {fm_name}: Acc={acc:.4f}, Depth={depth}, Tiempo={t_elapsed:.1f}s")

# Gráfico
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

names_fm = list(fm_results.keys())
accs_fm = [fm_results[n]['accuracy'] for n in names_fm]
ax1.barh(names_fm, accs_fm, color=['#42A5F5','#7E57C2','#AB47BC','#EC407A'])
ax1.set_xlabel('Accuracy')
ax1.set_title('Comparación de Feature Maps')
ax1.set_xlim(0, 1.1)
for i, v in enumerate(accs_fm):
    ax1.text(v + 0.01, i, f'{v:.2%}', va='center', fontweight='bold')

for fm_name, r in fm_results.items():
    ax2.plot(r['loss'], label=f"{fm_name} ({r['accuracy']:.2%})", linewidth=1.5)
ax2.set_xlabel('Iteración')
ax2.set_ylabel('Loss')
ax2.set_title('Convergencia por Feature Map')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figuras/exp1_feature_maps.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. Experimento 2: Sensibilidad a la Profundidad del Ansatz
# Se varía reps ∈ {1, 2, 3, 5} en RealAmplitudes para detectar sobreajuste o barren plateau.

# %%
reps_list = [1, 2, 3, 5]
depth_results = {}
best_fmap = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')

for reps in reps_list:
    print(f"🔄 Ansatz reps={reps}...")
    ans = RealAmplitudes(num_qubits=4, reps=reps, entanglement='linear')
    hist = {'loss': []}
    def cb(w, l, *args): hist['loss'].append(l)

    t0 = time.time()
    model = VQC(
        feature_map=best_fmap, ansatz=ans, loss='cross_entropy',
        optimizer=COBYLA(maxiter=150), sampler=StatevectorSampler(), callback=cb,
    )
    model.fit(X_tr, y_tr_oh)
    t_elapsed = time.time() - t0
    acc = model.score(X_te, y_te_oh)
    n_params = ans.num_parameters

    depth_results[reps] = {
        'accuracy': acc, 'tiempo': t_elapsed, 'n_params': n_params, 'loss': hist['loss']
    }
    print(f"   ✅ reps={reps}: Acc={acc:.4f}, Params={n_params}, Tiempo={t_elapsed:.1f}s")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

reps_vals = list(depth_results.keys())
accs_d = [depth_results[r]['accuracy'] for r in reps_vals]
params_d = [depth_results[r]['n_params'] for r in reps_vals]

ax1.plot(reps_vals, accs_d, 'o-', color='#7B1FA2', linewidth=2, markersize=10)
ax1.set_xlabel('Reps (profundidad ansatz)')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs Profundidad del Ansatz')
ax1.grid(True, alpha=0.3)
for r, a in zip(reps_vals, accs_d):
    ax1.annotate(f'{a:.2%}\n({depth_results[r]["n_params"]}p)', (r, a),
                textcoords="offset points", xytext=(0,12), ha='center', fontsize=9)

for reps, r in depth_results.items():
    ax2.plot(r['loss'], label=f'reps={reps}', linewidth=1.5)
ax2.set_xlabel('Iteración')
ax2.set_ylabel('Loss')
ax2.set_title('Convergencia por Profundidad')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figuras/exp2_profundidad.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 12. Experimento 3: Comparación de Optimizadores
# COBYLA vs SPSA vs L-BFGS-B sobre la misma arquitectura VQC.

# %%
optimizers = {
    'COBYLA': COBYLA(maxiter=200),
    'SPSA': SPSA(maxiter=150),
    'L-BFGS-B': L_BFGS_B(maxiter=200),
}
opt_results = {}

for opt_name, opt in optimizers.items():
    print(f"🔄 Optimizador: {opt_name}...")
    fm = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
    ans = RealAmplitudes(num_qubits=4, reps=3, entanglement='linear')
    hist = {'loss': []}
    def cb(w, l, *args): hist['loss'].append(l)

    t0 = time.time()
    model = VQC(
        feature_map=fm, ansatz=ans, loss='cross_entropy',
        optimizer=opt, sampler=StatevectorSampler(), callback=cb,
    )
    model.fit(X_tr, y_tr_oh)
    t_elapsed = time.time() - t0
    acc = model.score(X_te, y_te_oh)

    opt_results[opt_name] = {
        'accuracy': acc, 'tiempo': t_elapsed,
        'iters': len(hist['loss']), 'loss': hist['loss']
    }
    print(f"   ✅ {opt_name}: Acc={acc:.4f}, Iters={len(hist['loss'])}, Tiempo={t_elapsed:.1f}s")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

opt_names = list(opt_results.keys())
accs_o = [opt_results[n]['accuracy'] for n in opt_names]
times_o = [opt_results[n]['tiempo'] for n in opt_names]

bars = ax1.bar(opt_names, accs_o, color=['#26A69A','#EF5350','#5C6BC0'], edgecolor='white')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy por Optimizador')
ax1.set_ylim(0, 1.1)
for bar, a in zip(bars, accs_o):
    ax1.text(bar.get_x()+bar.get_width()/2, a+0.02, f'{a:.2%}', ha='center', fontweight='bold')

for opt_name, r in opt_results.items():
    ax2.plot(r['loss'], label=f"{opt_name} ({r['accuracy']:.2%})", linewidth=2)
ax2.set_xlabel('Iteración')
ax2.set_ylabel('Loss')
ax2.set_title('Convergencia por Optimizador')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figuras/exp3_optimizadores.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 13. Heatmap de la Matriz de Kernel Cuántico vs RBF

# %%
# Subconjunto para visualización (primeras 30 muestras)
n_vis = min(30, len(X_te))
X_vis = X_te[:n_vis]

# Kernel cuántico
fm_k = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
qk = FidelityQuantumKernel(feature_map=fm_k)
K_quantum = qk.evaluate(X_vis)

# Kernel RBF clásico
from sklearn.metrics.pairwise import rbf_kernel
K_rbf = rbf_kernel(X_vis, gamma=1.0/(4 * X_vis.var()))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
im1 = ax1.imshow(K_quantum, cmap='inferno', vmin=0, vmax=1)
ax1.set_title('Kernel Cuántico (ZZFeatureMap)')
plt.colorbar(im1, ax=ax1, shrink=0.8)

im2 = ax2.imshow(K_rbf, cmap='inferno', vmin=0, vmax=1)
ax2.set_title('Kernel RBF Clásico')
plt.colorbar(im2, ax=ax2, shrink=0.8)
plt.tight_layout()
plt.savefig('figuras/kernels_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 14. Experimento con Varianza: Boxplots sobre Múltiples Semillas

# %%
print("🔄 Ejecutando VQC con 5 semillas distintas...")
seed_accuracies = []
for seed in [42, 123, 456, 789, 1024]:
    algorithm_globals.random_seed = seed
    fm = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
    ans = RealAmplitudes(num_qubits=4, reps=3, entanglement='linear')
    model = VQC(
        feature_map=fm, ansatz=ans, loss='cross_entropy',
        optimizer=COBYLA(maxiter=150), sampler=StatevectorSampler(),
    )
    model.fit(X_tr, y_tr_oh)
    acc = model.score(X_te, y_te_oh)
    seed_accuracies.append(acc)
    print(f"   Semilla {seed}: {acc:.4f}")

# Restaurar semilla
algorithm_globals.random_seed = 42

svm_cv = cross_val_score(SVC(kernel='rbf', C=1.0, gamma='scale'), X_scaled, y, cv=5)

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([seed_accuracies, svm_cv.tolist()], labels=['VQC (5 seeds)', 'SVM-RBF (5-fold CV)'],
                patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#CE93D8')
bp['boxes'][1].set_facecolor('#90CAF9')
ax.set_ylabel('Accuracy')
ax.set_title('Distribución de Accuracy: VQC vs SVM-RBF')
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('figuras/boxplots_varianza.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nVQC: {np.mean(seed_accuracies):.4f} ± {np.std(seed_accuracies):.4f}")
print(f"SVM: {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")

# %% [markdown]
# ## 15. Experimento Secundario: MNIST Reducido (2 clases)
# Reducir MNIST a clases {0, 1}, submuestrear, PCA a 4 componentes.

# %%
from sklearn.datasets import load_digits

print("🔄 Cargando MNIST reducido (digits 0 y 1)...")
digits = load_digits()
mask = np.isin(digits.target, [0, 1])
X_mnist = digits.data[mask]
y_mnist = digits.target[mask]

# PCA a 4 componentes
pca_mnist = PCA(n_components=4, random_state=42)
X_mnist_pca = pca_mnist.fit_transform(X_mnist)
X_mnist_scaled = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_mnist_pca)

y_mnist_oh = OneHotEncoder(sparse_output=False).fit_transform(y_mnist.reshape(-1,1))
X_m_tr, X_m_te, y_m_tr, y_m_te = train_test_split(
    X_mnist_scaled, y_mnist_oh, test_size=0.3, stratify=y_mnist, random_state=42)
y_m_tr_lbl = y_m_tr.argmax(1)
y_m_te_lbl = y_m_te.argmax(1)

print(f"MNIST reducido: {X_mnist_scaled.shape[0]} muestras, {4} features (PCA)")
print(f"Train: {X_m_tr.shape[0]} | Test: {X_m_te.shape[0]}")

# SVM clásico
svm_m = SVC(kernel='rbf').fit(X_m_tr, y_m_tr_lbl)
acc_svm_m = svm_m.score(X_m_te, y_m_te_lbl)

# VQC
fm_m = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='linear')
ans_m = RealAmplitudes(num_qubits=4, reps=3, entanglement='linear')
hist_m = {'loss': []}
def cb_m(w, l, *args): hist_m['loss'].append(l)

print("🔄 Entrenando VQC en MNIST reducido...")
t0 = time.time()
vqc_m = VQC(
    feature_map=fm_m, ansatz=ans_m, loss='cross_entropy',
    optimizer=COBYLA(maxiter=150), sampler=StatevectorSampler(), callback=cb_m,
)
vqc_m.fit(X_m_tr, y_m_tr)
t_m = time.time() - t0
acc_vqc_m = vqc_m.score(X_m_te, y_m_te)

print(f"\n📊 Resultados MNIST Reducido:")
print(f"   SVM-RBF:  {acc_svm_m:.4f}")
print(f"   VQC:      {acc_vqc_m:.4f} ({t_m:.1f}s)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(['SVM-RBF', 'VQC'], [acc_svm_m, acc_vqc_m], color=['#2196F3', '#9C27B0'])
ax1.set_ylabel('Accuracy')
ax1.set_title('MNIST Reducido (0 vs 1)')
ax1.set_ylim(0, 1.1)
for i, v in enumerate([acc_svm_m, acc_vqc_m]):
    ax1.text(i, v+0.02, f'{v:.2%}', ha='center', fontweight='bold')

ax2.plot(hist_m['loss'], color='#9C27B0', linewidth=2)
ax2.set_xlabel('Iteración')
ax2.set_ylabel('Loss')
ax2.set_title('Convergencia VQC en MNIST')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figuras/exp_mnist.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 16. Tabla Resumen Final y Conclusiones

# %%
print("\n" + "=" * 80)
print("RESUMEN FINAL DE TODOS LOS EXPERIMENTOS")
print("=" * 80)

print("\n📌 Experimento Principal (Iris):")
for name, r in resultados.items():
    print(f"   {name:15s} → Accuracy: {r['accuracy']:.4f} | Tiempo: {r['tiempo']:.2f}s")

print(f"\n📌 Comparación Feature Maps:")
for name, r in fm_results.items():
    print(f"   {name:25s} → Acc: {r['accuracy']:.4f} | Depth: {r['depth']}")

print(f"\n📌 Profundidad Ansatz:")
for reps, r in depth_results.items():
    print(f"   reps={reps} ({r['n_params']:2d} params) → Acc: {r['accuracy']:.4f}")

print(f"\n📌 Optimizadores:")
for name, r in opt_results.items():
    print(f"   {name:12s} → Acc: {r['accuracy']:.4f} | Iters: {r['iters']} | Tiempo: {r['tiempo']:.1f}s")

print(f"\n📌 MNIST Reducido:")
print(f"   SVM-RBF: {acc_svm_m:.4f} | VQC: {acc_vqc_m:.4f}")

print(f"\n📌 Varianza VQC (5 seeds): {np.mean(seed_accuracies):.4f} ± {np.std(seed_accuracies):.4f}")

# %% [markdown]
# ## 17. Conclusiones y Reflexión Crítica
#
# ### Hallazgos principales:
# 1. **El SVM clásico supera al VQC en Iris** (~96-98% vs ~88-92%). Esto es **el resultado
#    esperado** según la literatura (Piatrenka & Rusek 2022, Gupta et al. 2025).
#
# 2. **Iris no es un dataset donde QML muestre ventaja**: es de baja dimensión (4 features),
#    bien separable linealmente (setosa) y cuasi-linealmente (las otras dos clases).
#
# 3. **ZZFeatureMap es más expresivo** que ZFeatureMap gracias a las interacciones de segundo
#    orden, pero también más sensible al ruido y requiere más profundidad.
#
# 4. **La profundidad del ansatz tiene un punto óptimo**: reps=3 suele dar el mejor balance
#    entre expresividad y convergencia. Más profundidad puede llevar a barren plateaus.
#
# 5. **COBYLA es un buen default** para VQC en problemas pequeños; SPSA es preferible
#    para hardware real con ruido (Cerezo et al. 2021).
#
# ### ¿Cuándo superaría QML al ML clásico?
# Según Liu et al. (2021) y Huang et al. (2021), las ventajas potenciales aparecen cuando:
# - Los datos provienen de un proceso cuántico (química, materiales)
# - El dataset requiere un espacio de hipótesis exponencialmente grande
# - Existe simetría de grupo aprovechable por kernels covariantes
#
# **Conclusión final:** Este proyecto demuestra la viabilidad del QML híbrido como herramienta
# pedagógica y experimental, pero confirma que para datos clásicos de baja dimensión, los
# métodos clásicos siguen siendo superiores en precisión y eficiencia computacional.
#
# ### Referencias principales:
# - Biamonte et al. (2017), Nature 549:195-202
# - Havlíček et al. (2019), Nature 567:209-212
# - Cerezo et al. (2021), Nat. Rev. Phys. 3:625-644
# - Larocca et al. (2025), Nat. Rev. Phys., DOI:10.1038/s42254-025-00813-9
# - Gupta et al. (2025), npj Digital Medicine

# %%
print("\n🎓 Notebook completado exitosamente.")
print("📁 Todas las figuras guardadas en ./figuras/")
