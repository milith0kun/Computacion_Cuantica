import numpy as np
import matplotlib.pyplot as plt
import os
import json

output_dir = r"d:\Proyectos\Cuantica\Labos\Labo 5\Informe\imagenes"
os.makedirs(output_dir, exist_ok=True)

# ----------------- EJERCICIO 3: CONVERGENCIA ESTACIONARIA -----------------
P3 = np.array([[0.9, 0.1],
               [0.5, 0.5]])
pi_0 = np.array([1.0, 0.0])
pi_est = np.array([5/6, 1/6])

pasos = 15
historial = [pi_0.copy()]
pi_temp = pi_0.copy()
for _ in range(pasos):
    pi_temp = pi_temp @ P3
    historial.append(pi_temp.copy())
historial = np.array(historial)

# Cargar conteos experimentales del JSON si existe
p0_est_exp = 0.8235
p1_est_exp = 0.1765
try:
    with open(r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\resultados_ibm\resultado_ejercicio_3.json", "r") as f:
        data_ej3 = json.load(f)
        p0_est_exp = data_ej3["probabilidades_obtenidas"][0]
        p1_est_exp = data_ej3["probabilidades_obtenidas"][1]
except Exception as e:
    print("No se pudo cargar el JSON del Ejercicio 3, usando fallback. Error:", e)

plt.figure(figsize=(9, 5))
plt.plot(range(pasos + 1), historial[:, 0], 'o-', color='#0D47A1', linewidth=2, label='Teórico P(Estado 0)')
plt.plot(range(pasos + 1), historial[:, 1], 's-', color='#D84315', linewidth=2, label='Teórico P(Estado 1)')
plt.axhline(y=5/6, color='#0D47A1', linestyle='--', alpha=0.5, label='Estacionario Límite 0 (5/6)')
plt.axhline(y=1/6, color='#D84315', linestyle='--', alpha=0.5, label='Estacionario Límite 1 (1/6)')

# Graficar el resultado experimental de Qiskit como puntos al final (paso 15, que representa el límite estacionario)
plt.scatter(pasos, p0_est_exp, color='#1565C0', s=120, zorder=5, marker='X', label=f'Qiskit Realidad Ruidosa 0 ({p0_est_exp:.4f})')
plt.scatter(pasos, p1_est_exp, color='#E65100', s=120, zorder=5, marker='X', label=f'Qiskit Realidad Ruidosa 1 ({p1_est_exp:.4f})')

plt.xlabel('Paso n')
plt.ylabel('Probabilidad')
plt.title('Evolución y Convergencia hacia el Estado Estacionario (Ejercicio 3)')
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.ylim(-0.05, 1.05)
plt.savefig(os.path.join(output_dir, "ej3_estacionario.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico ej3_estacionario.png generado.")


# ----------------- EJERCICIO 5: DECAIMIENTO CLÁSICO -----------------
p_values = [0.05, 0.2, 0.6]
plt.figure(figsize=(15, 4.5))

for idx, p_param in enumerate(p_values):
    P5 = np.array([[1.0, 0.0],
                   [p_param, 1.0 - p_param]])
    pi = np.array([0.0, 1.0])
    hist_5 = [pi.copy()]
    for _ in range(30):
        pi = pi @ P5
        hist_5.append(pi.copy())
    hist_5 = np.array(hist_5)
    
    plt.subplot(1, 3, idx + 1)
    plt.plot(hist_5[:, 0], '-', color='#2E7D32', linewidth=2, label='P(Estado 0) - Absorbente')
    plt.plot(hist_5[:, 1], '-', color='#C62828', linewidth=2, label='P(Estado 1) - Transitorio')
    plt.title(f"Decaimiento con p = {p_param}")
    plt.xlabel("Pasos")
    plt.ylabel("Probabilidad")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ej5_decaimiento_clasico.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico ej5_decaimiento_clasico.png generado.")


# ----------------- EJERCICIO 5: EVOLUCIÓN DE PUREZA -----------------
alpha = np.cos(np.pi / 6)
beta = np.sin(np.pi / 6) * np.exp(1j * np.pi / 4)
psi = np.array([alpha, beta])
rho_init = np.outer(psi, psi.conj())

p_param = 0.2
K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p_param)]])
K1 = np.array([[0.0, np.sqrt(p_param)], [0.0, 0.0]])

rho_temp = rho_init.copy()
purezas = [np.trace(rho_temp @ rho_temp).real]

for _ in range(25):
    rho_temp = K0 @ rho_temp @ K0.conj().T + K1 @ rho_temp @ K1.conj().T
    purezas.append(np.trace(rho_temp @ rho_temp).real)

plt.figure(figsize=(8, 4))
plt.plot(range(26), purezas, 'o-', color='#7B1FA2', linewidth=2, label='Tr(rho^2)')
plt.xlabel("Número de aplicaciones del canal")
plt.ylabel("Pureza del estado")
plt.title("Evolución de la Pureza bajo Amplitude Damping (p=0.2)")
plt.grid(True, alpha=0.3)
plt.ylim(0.45, 1.05)
plt.legend()
plt.savefig(os.path.join(output_dir, "ej5_pureza.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico ej5_pureza.png generado.")
