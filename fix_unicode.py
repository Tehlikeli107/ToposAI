import os
import glob

dirs = ['ToposAI/applications', 'ToposAI/benchmarks', 'ToposAI/experiments']

for d in dirs:
    for f in glob.glob(d + '/*.py'):
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Sadece sys_path fixi yapılmışsa ve reconfigure henüz yoksa
        if 'sys.path.append' in content and 'sys.stdout.reconfigure' not in content:
            new_content = content.replace("import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
                                          "import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nif hasattr(sys.stdout, 'reconfigure'):\n    sys.stdout.reconfigure(encoding='utf-8')")
            with open(f, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"Fixed Unicode: {f}")
