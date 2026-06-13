import json
import subprocess
import sys

print("Running jupyter nbconvert to execute the notebook and save images")
try:
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "Lista_Ejercicios_01.ipynb"], check=True)
except Exception as e:
    print(f"Error running nbconvert: {e}")
    # Fallback: run the python code directly to generate images
    with open("Lista_Ejercicios_01.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(source, globals())
print("Done executing.")
