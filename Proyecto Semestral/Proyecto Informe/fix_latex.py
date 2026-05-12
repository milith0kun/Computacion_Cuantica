import os
import glob
import re

content_dir = r"d:\Proyectos\Cuantica\Proyecto Semestral\Proyecto Informe\contenido"
files = glob.glob(os.path.join(content_dir, "cap_*.tex"))

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # We must do it in reverse order to not overwrite our own changes
    content = re.sub(r'\\subsubsection{', r'\\paragraph{', content)
    content = re.sub(r'\\subsection{', r'\\subsubsection{', content)
    content = re.sub(r'\\section{', r'\\subsection{', content)
    content = re.sub(r'\\chapter{', r'\\section{', content)
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

print("Files modified.")
