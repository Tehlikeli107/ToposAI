import os
import sys
import glob
import importlib.util
import pytest

# =====================================================================
# REPOSITORY WIDE SMOKE TEST
# İddia: Repodaki tüm uygulamalar, benchmarklar ve deneyler (50+ script)
# en azından syntax ve import (modül) hatası taşımamalıdır.
# Bu test her bir dosyayı dinamik olarak import ederek sözdizimi kırıklarını
# CI/CD aşamasında (Push öncesi) anında yakalar.
# =====================================================================

def get_all_python_scripts():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dirs = ['applications', 'benchmarks', 'experiments']
    scripts = []
    
    # 1. Alt Klasörler
    for d in target_dirs:
        dir_path = os.path.join(repo_root, d)
        for f in glob.glob(os.path.join(dir_path, '*.py')):
            if not os.path.basename(f).startswith('__'):
                scripts.append(f)
                
    # 2. Ana Dizin (Kök - Top Level)
    for f in glob.glob(os.path.join(repo_root, '*.py')):
        basename = os.path.basename(f)
        if not basename.startswith('__') and not basename.startswith('test_') and basename != 'setup.py':
            scripts.append(f)
            
    return scripts

@pytest.mark.parametrize("script_path", get_all_python_scripts())
def test_script_imports_without_syntax_errors(script_path):
    """
    Her bir script'in modül olarak başarıyla yüklenebildiğini (SyntaxError,
    IndentationError veya eksik import olmadığını) test eder.
    Live API call yapan veya uzun süren __main__ blokları tetiklenmez.
    """
    module_name = os.path.basename(script_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        # Script'i import et (Çalıştırmaz, sadece yükler ve tanımları okur)
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Script yüklenirken hata oluştu: {script_path}\nHata: {str(e)}")
