import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples"


def load_example(name):
    path = EXAMPLE_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_yoneda_density_example_returns_isomorphism_summary():
    module = load_example("formal_yoneda_density.py")
    summary = module.main()
    assert summary["objects_in_category_of_elements"] == 3
    assert summary["round_trip_to_presheaf"] is True
    assert summary["round_trip_to_density"] is True


def test_sheafification_site_example_reports_reflector_summary():
    module = load_example("sheafification_site.py")
    summary = module.main()
    assert summary["is_sheaf_after_sheafification"] is True
    assert summary["factorization_valid"] is True


def test_kripke_joyal_forcing_example_reports_quantifier_laws():
    module = load_example("kripke_joyal_forcing.py")
    summary = module.main()
    assert summary["forces_closed_truth"] is True
    assert summary["quantifier_adjunctions_valid"] is True


def test_quasi_category_horns_example_reports_inner_kan():
    module = load_example("quasi_category_horns.py")
    summary = module.main()
    assert summary["inner_kan_2"] is True
    assert summary["inner_kan_3"] is True
    assert summary["assoc_3_simplices"] > 0


def test_hott_transport_example_reports_functorial_transport():
    module = load_example("hott_transport.py")
    summary = module.main()
    assert summary["identity_type_size"] == 1
    assert summary["transport_functorial"] is True
