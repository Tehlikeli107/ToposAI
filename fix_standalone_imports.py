import os
import glob

# Bu script repo içindeki her bir 'application' ve 'benchmark' scriptinin
# en tepesine `sys.path.append` ekleyerek kullanıcıların pip install yapmadan
# dahi klasörün herhangi bir yerinde `python app.py` diyebilmesini sağlar.

dirs = ['ToposAI/applications', 'ToposAI/benchmarks']
header = "import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"

for d in dirs:
    for f in glob.glob(d + '/*.py'):
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Eğer bu dosya zaten düzeltildiyse (sys.path.append varsa) atla
        if 'sys.path.append' not in content:
            with open(f, 'w', encoding='utf-8') as file:
                file.write(header + content)
            print(f"Sabitlendi (Fixed Import Error): {f}")
