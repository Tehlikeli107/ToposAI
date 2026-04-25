import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_TARGETS = [
    ROOT / "topos_ai",
    ROOT / "applications",
    ROOT / "benchmarks",
    ROOT / "docs",
    ROOT / "experiments",
    ROOT / "tests",
    ROOT / "CHANGELOG.md",
    ROOT / "SECURITY.md",
    ROOT / "CITATION.cff",
    ROOT / "README.md",
    ROOT / "setup.py",
    ROOT / "pyproject.toml",
]

FORBIDDEN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        "BİLİMSEL " + "KANIT",
        "SCIENTIFIC " + "PROOF",
        "KUSUR" + "SUZ",
        "SIFIR " + "HATA",
        "SIFIR " + "DENEME",
        r"TRUE O" + r"\(1\)",
        "Halüsinasyon " + "garantili",
        "matematiksel olarak " + "engeller",
        "gerçekten " + "evrensel",
        "Sonsuz " + "bağlam",
        "Infinite " + "Context",
        r"\b" + "A" + "GI" + r"\b",
    )
]


def iter_text_files():
    for target in SCAN_TARGETS:
        if target.is_file():
            if target != Path(__file__).resolve():
                yield target
            continue

        for path in target.rglob("*"):
            if path == Path(__file__).resolve():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in {".py", ".md", ".toml", ".cff", ".yml", ".yaml"}:
                yield path


def test_repository_avoids_unqualified_grand_claims():
    violations = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_PATTERNS:
            match = pattern.search(text)
            if match:
                rel = path.relative_to(ROOT)
                violations.append(f"{rel}: matched {pattern.pattern!r} at offset {match.start()}")

    assert not violations, "Unqualified grand-claim language found:\n" + "\n".join(violations)
