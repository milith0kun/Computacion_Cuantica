import json

json_path = r"d:\Proyectos\Cuantica\Labos\Labo 5\Colab\resultados_ibm\resultado_ejercicio_4.json"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

hist_q = data["historial_qiskit"]
hist_np = data["historial_numpy"]
backends = data["backends_solicitados"]

print("Paso n | NumPy | Qiskit | Backend")
for idx, (np_val, q_val) in enumerate(zip(hist_np, hist_q)):
    b = backends[idx-1] if idx > 0 else "Inicial"
    print(f"  {idx:2d}   | {np_val[0]:.8f} | {q_val[0]:.8f} | {b}")
