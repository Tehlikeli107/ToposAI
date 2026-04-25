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
