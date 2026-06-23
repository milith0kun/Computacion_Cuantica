import io
import re
import os
import subprocess

html_path = 'D:/Proyectos/Cuantica/Proyecto Segunda Parte/Diapo/presentacion_experimentacion.html'
pdf_path = 'D:/Proyectos/Cuantica/Proyecto Segunda Parte/Diapo/presentacion_experimentacion.pdf'

long_html_path = 'D:/Proyectos/Cuantica/Proyecto Segunda Parte/Diapo/GRUPO 1_SAIRE BUSTAMANTE-LOZANO LLACCTAHUAMAN-PUMA POTOSINO_IMPLEMENTACION DE UN MODELO DE APRENDIZAJE AUTOMATICO CUANTICO UTILIZANDO QISKIT MACHINE LEARNING_DIAPOSITIVAS.html'
long_pdf_path = 'D:/Proyectos/Cuantica/Proyecto Segunda Parte/Diapo/GRUPO 1_SAIRE BUSTAMANTE-LOZANO LLACCTAHUAMAN-PUMA POTOSINO_IMPLEMENTACION DE UN MODELO DE APRENDIZAJE AUTOMATICO CUANTICO UTILIZANDO QISKIT MACHINE LEARNING_DIAPOSITIVAS.pdf'

with io.open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Swap logic
# First, let's swap presenters dictionary:
# Saire Bustamante: s09 to s15
# Lozano Llactahuaman: s02 to s08
# Puma Potosino: s16 to s21

presenters = {}
for idx in range(2, 9):
    presenters[f"s{idx:02d}"] = "Lozano Llactahuaman"
for idx in range(9, 16):
    presenters[f"s{idx:02d}"] = "Saire Bustamante"
for idx in range(16, 22):
    presenters[f"s{idx:02d}"] = "Puma Potosino"

for s_id, name in presenters.items():
    # Replace existing presenter pill or add it
    # We find the section with id="{s_id}" and then replace the presenter-pill
    # Pattern to find the section block
    pattern = rf'(<section class="[^"]*" id="{s_id}">.*?<span class="tag-pill presenter-pill">Presenta: )(.*?)(</span>)'
    
    # We search for it
    match = re.search(pattern, html, re.DOTALL)
    if match:
        old_full = match.group(0)
        new_full = match.group(1) + name + match.group(3)
        html = html.replace(old_full, new_full)
        print(f"Updated presenter in {s_id} to {name}")
    else:
        # If not present, search for s-header to inject
        pattern_header = rf'(<section class="[^"]*" id="{s_id}">.*?<div class="s-header[^"]*">.*?)(</div>)'
        match_header = re.search(pattern_header, html, re.DOTALL)
        if match_header:
            header_start = match_header.group(1)
            pill = f'<span class="tag-pill presenter-pill">Presenta: {name}</span>'
            html = html.replace(header_start + '</div>', header_start + pill + '</div>')
            print(f"Injected presenter pill in {s_id} for {name}")
        else:
            print(f"Header not found for {s_id}")

with io.open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print("Updated presentation HTML file!")

# Re-export PDF
edge_paths = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
]
chrome_paths = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
]
exe_path = None
for p in edge_paths + chrome_paths:
    if os.path.exists(p):
        exe_path = p
        break

if not exe_path:
    exe_path = 'msedge'

print(f"Using browser: {exe_path}")
print("Generating PDF...")
cmd = [
    exe_path,
    '--headless',
    '--disable-gpu',
    f'--print-to-pdf={pdf_path}',
    html_path
]

try:
    subprocess.run(cmd, check=True)
    print("PDF exported successfully!")
except Exception as e:
    print("Failed to export PDF. Error:", e)

# Copy to long filenames
try:
    import shutil
    shutil.copy2(html_path, long_html_path)
    shutil.copy2(pdf_path, long_pdf_path)
    print("Copied updated files to official long names successfully!")
except Exception as e:
    print("Failed to copy files to long names. Error:", e)
