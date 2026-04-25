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
        "docs/tutorials/yoneda_density.md",
        "docs/tutorials/sheafification.md",
        "docs/tutorials/internal_logic.md",
        "docs/tutorials/quasi_categories_and_hott.md",
    ]
    for page in required_pages:
        assert page in mkdocs
        assert (ROOT / page).exists()


def test_readme_and_status_use_research_library_positioning():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "docs" / "PROJECT_STATUS.md").read_text(encoding="utf-8")

    assert "formal finite core" in readme
    assert "neural/proxy components" in readme
    assert "Maturity Matrix" in status
    assert "Research-library readiness" in status
