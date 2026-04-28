import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_math_contracts_document_exists_and_names_contract_classes():
    text = (ROOT / "docs" / "MATH_CONTRACTS.md").read_text(encoding="utf-8")

    # Ensure the code formally projects and embraces Category Theory (Post-Experiment 60 Era)
    # We replaced all neural networks with pure discrete combinatorial logic.
    formal_assurances = [
        "FiniteCategory",
        "CategoricalDatabase",
        "FreeCategoryGenerator",
        "ToposSheafComputer",
        "PresheafTopos",
        "FormalHomotopyEquivalence",
        "FormalInfinityCategoryValidator",
    ]

    for assurance in formal_assurances:
        assert assurance in text, f"Missing pure mathematical assurance: '{assurance}' not found in MATH_CONTRACTS.md."

def test_api_links_to_math_contracts():
    text = (ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")
    assert "FiniteCategory" in text

def test_public_api_document_names_stable_research_surface():
    text = (ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")

    required_symbols = [
        "FiniteCategory",
        "FiniteFunctor",
        "Presheaf",
    ]

    for symbol in required_symbols:
        assert symbol in text

def test_has_no_torch_imports_at_top_level_of_core():
    """
    Ensure the core formal math files do not import torch at module level.
    """
    core_files = [
        "formal_category.py",
        "infinity_categories.py",
        "hott.py",
    ]
    for filename in core_files:
        path = ROOT / "topos_ai" / filename
        if not path.exists():
            continue
        code = path.read_text(encoding="utf-8")
        for line in code.splitlines():
            line = line.strip()
            if (line.startswith("import torch") or line.startswith("from torch")) and not "try:" in code:
                # We skip checking indentation for simplicity and assume if there's a try: block, it's safe.
                # However, for a real check, we should just ensure line doesn't start with 'import torch' without indentation.
                pass
        
        # Real check: If the original unstripped line starts with exactly 'import torch'
        for original_line in code.splitlines():
            if original_line.startswith("import torch") or original_line.startswith("from torch"):
                assert False, f"Found '{original_line}' at top-level in {filename}"