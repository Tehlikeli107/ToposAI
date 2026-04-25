import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_math_contracts_document_exists_and_names_contract_classes():
    text = (ROOT / "docs" / "MATH_CONTRACTS.md").read_text(encoding="utf-8")

    required_phrases = [
        "Formal finite mathematics",
        "Neural and proxy research components",
        "FiniteCategory",
        "PresheafTopos",
        "GrothendieckTopology",
        "FiniteSimplicialSet",
        "FinitePathGroupoid",
        "HomotopyEquivalence is not a HoTT proof kernel",
        "InfinityCategoryLayer is not a full infinity-category engine",
    ]

    for phrase in required_phrases:
        assert phrase in text


def test_api_links_to_math_contracts():
    text = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    assert "MATH_CONTRACTS.md" in text


def test_public_api_document_names_stable_research_surface():
    text = (ROOT / "docs" / "PUBLIC_API.md").read_text(encoding="utf-8")

    required_symbols = [
        "FiniteCategory",
        "FiniteFunctor",
        "Presheaf",
        "PresheafTopos",
        "GrothendieckTopology",
        "category_of_elements",
        "yoneda_density_colimit",
        "FiniteSimplicialSet",
        "FiniteHorn",
        "nerve_2_skeleton",
        "nerve_3_skeleton",
        "FinitePathGroupoid",
        "PathFamily",
    ]

    for symbol in required_symbols:
        assert symbol in text


def test_tutorial_pages_are_linked_from_mkdocs():
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    required_pages = [
        "tutorials/yoneda_density.md",
        "tutorials/sheafification.md",
        "tutorials/internal_logic.md",
        "tutorials/quasi_categories_and_hott.md",
    ]
    for page in required_pages:
        assert page in mkdocs
        assert (ROOT / "docs" / page).exists()
    assert "docs/tutorials/" not in mkdocs
    assert "docs/MATH_CONTRACTS.md" not in mkdocs


def test_ci_builds_docs_and_runs_research_readiness_tests():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    required_fragments = [
        "tests/test_formal_properties.py",
        "tests/test_research_docs.py",
        "tests/test_examples.py",
        "mkdocs-material",
        "mkdocs build --strict",
        "--no-deps",
        "formal no-deps wheel smoke passed",
    ]

    for fragment in required_fragments:
        assert fragment in ci


def test_package_import_exposes_formal_core_without_torch():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ModuleNotFoundError("No module named 'torch'", name="torch")
                return None

        sys.meta_path.insert(0, BlockTorch())

        import topos_ai
        assert hasattr(topos_ai, "formal_category")
        assert hasattr(topos_ai, "hott")
        assert hasattr(topos_ai, "infinity_categories")

        from topos_ai.formal_category import FiniteCategory
        from topos_ai.infinity_categories import InfinityCategoryLayer

        category = FiniteCategory(
            objects=("*",),
            morphisms={"id": ("*", "*")},
            identities={"*": "id"},
            composition={("id", "id"): "id"},
        )
        assert category.validate_laws()

        try:
            InfinityCategoryLayer(1, 1, 1)
        except RuntimeError as exc:
            assert "requires PyTorch" in str(exc)
        else:
            raise AssertionError("InfinityCategoryLayer should require PyTorch when torch is absent.")
        """
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_readme_and_status_use_research_library_positioning():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "docs" / "PROJECT_STATUS.md").read_text(encoding="utf-8")

    assert "formal finite core" in readme
    assert "neural/proxy components" in readme
    assert "Maturity Matrix" in status
    assert "Research-library readiness" in status
