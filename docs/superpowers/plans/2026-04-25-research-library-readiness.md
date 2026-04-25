# Research Library Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade ToposAI from a strong mathematical prototype into a credible, reproducible research-library prototype with clear formal contracts, runnable examples, stronger property tests, and CI-backed documentation smoke checks.

**Architecture:** Keep the existing installable package intact. Add a documentation contract layer that separates formal finite mathematics from neural/proxy demos, add examples that execute against the public API, and add deterministic property-style tests that verify the main categorical laws across several finite fixtures. Avoid new runtime dependencies and keep all new examples CPU-only.

**Tech Stack:** Python 3.11, pytest, pytest-cov, ruff, MkDocs markdown, current ToposAI modules, standard library only for new examples unless an existing module already requires PyTorch.

---

## File Structure

- Create: `docs/MATH_CONTRACTS.md`
  - Defines which modules are formal finite mathematics, which are neural/proxy research components, and which guarantees each formal module makes.
- Create: `docs/PUBLIC_API.md`
  - Freezes the research-library API surface for the formal core and gives import stability expectations.
- Create: `docs/tutorials/yoneda_density.md`
  - Tutorial for `category_of_elements` and `yoneda_density_colimit`.
- Create: `docs/tutorials/sheafification.md`
  - Tutorial for finite sites, covering sieves, plus construction, and sheafification.
- Create: `docs/tutorials/internal_logic.md`
  - Tutorial for subobjects, forcing, equality, and quantifiers.
- Create: `docs/tutorials/quasi_categories_and_hott.md`
  - Tutorial for finite horn fillers, nerve skeletons, identity types, and transport.
- Create: `examples/formal_yoneda_density.py`
  - Runnable script that reconstructs a walking-arrow presheaf from representables.
- Create: `examples/sheafification_site.py`
  - Runnable script that sheafifies a separated and non-separated presheaf on a tiny site.
- Create: `examples/kripke_joyal_forcing.py`
  - Runnable script that prints forcing sieves, equality truth values, and quantifier checks.
- Create: `examples/quasi_category_horns.py`
  - Runnable script that builds a 3-skeleton nerve and checks inner horn fillers.
- Create: `examples/hott_transport.py`
  - Runnable script that validates a finite path groupoid and dependent transport.
- Create: `tests/test_research_docs.py`
  - Documentation contract tests for required docs and explicit formal/proxy wording.
- Create: `tests/test_examples.py`
  - Executes all research examples and checks their returned summary dictionaries.
- Create: `tests/test_formal_properties.py`
  - Deterministic property-style tests over several finite categories and presheaves.
- Modify: `docs/api.md`
  - Link the new tutorials and public API contract.
- Modify: `docs/index.md`
  - Add a research-library navigation section.
- Modify: `mkdocs.yml`
  - Add the new documentation pages to the site navigation.
- Modify: `.github/workflows/ci.yml`
  - Add documentation and example smoke checks to CI.
- Modify: `README.md`
  - Reframe ToposAI as an experimental research library with a formal finite core and proxy ML components.
- Modify: `docs/PROJECT_STATUS.md`
  - Add a maturity matrix and release-readiness criteria.

---

## Execution Rules

- Work task-by-task.
- Use TDD for every code or testable documentation change.
- Run the exact verification command listed in each task before committing that task.
- Do not introduce new package dependencies.
- Do not edit generated build artifacts.
- Keep examples deterministic and CPU-only.
- Use small commits after each task.

---

### Task 1: Add Mathematical Contract Documentation

**Files:**
- Create: `tests/test_research_docs.py`
- Create: `docs/MATH_CONTRACTS.md`
- Modify: `docs/api.md`

- [ ] **Step 1: Write the failing documentation contract test**

Create `tests/test_research_docs.py` with:

```python
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
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python -m pytest tests/test_research_docs.py::test_math_contracts_document_exists_and_names_contract_classes -q
```

Expected: fail with `FileNotFoundError` for `docs/MATH_CONTRACTS.md`.

- [ ] **Step 3: Create the math contract document**

Create `docs/MATH_CONTRACTS.md` with this structure and content:

```markdown
# Mathematical Contracts

ToposAI has two kinds of modules: formal finite mathematics and neural or proxy research components. Formal modules expose explicit finite objects and validate algebraic laws. Proxy modules are useful experimental scaffolds but do not claim to implement the full mathematical theory named in their inspiration.

## Formal finite mathematics

| Module | Contract | Main limitations |
|--------|----------|------------------|
| `topos_ai.formal_category.FiniteCategory` | Explicit finite category with typed morphisms, identities, composition, identity laws, and associativity validation. | Finite categories only. No enriched, large, or higher categories. |
| `topos_ai.formal_category.PresheafTopos` | Finite fragment of `Set^(C^op)` with finite limits, colimits, exponentials, subobject classifier, Heyting operations, Kripke-Joyal forcing, quantifiers, sheafification, Lawvere-Tierney operators, and Yoneda density reconstruction. | Computes finite presheaf topoi. It is not a general theorem prover for arbitrary Grothendieck topoi. |
| `topos_ai.formal_category.GrothendieckTopology` | Finite Grothendieck topology with maximal-sieve, pullback-stability, and transitivity checks. | Covering data must be explicitly finite. |
| `topos_ai.infinity_categories.FiniteSimplicialSet` | Finite simplicial-set skeleton with face identities, optional degeneracy identities, horn enumeration, and finite inner-Kan checks. | Skeleton-limited. It checks enumerated finite horns up to supplied dimension. |
| `topos_ai.infinity_categories.nerve_3_skeleton` | Builds the 3-skeleton of the nerve of a finite category, including 2-horn composition and 3-simplex associativity coherence. | Not a complete infinity-category implementation beyond the represented skeleton. |
| `topos_ai.hott.FinitePathGroupoid` | 1-truncated HoTT identity-type semantics as a finite groupoid with reflexivity, inverse paths, composition, and associativity. | Models groupoid semantics. It is not a dependent type checker. |
| `topos_ai.hott.PathFamily` | Dependent family over a finite path groupoid with functorial transport validation. | Transport maps are finite and explicit. |

## Neural and proxy research components

| Module | Intended use | Non-claim |
|--------|--------------|-----------|
| `topos_ai.hott.HomotopyEquivalence` | Orthogonal Procrustes alignment for point clouds. | HomotopyEquivalence is not a HoTT proof kernel. |
| `topos_ai.infinity_categories.InfinityCategoryLayer` | Hodge message passing over finite simplicial complexes. | InfinityCategoryLayer is not a full infinity-category engine. |
| `topos_ai.yoneda.YonedaUniverse` | Probe-distance reconstruction experiment inspired by Yoneda-style observation. | It is not the categorical Yoneda lemma. Use `topos_ai.formal_category.yoneda_lemma_bijection` and `yoneda_density_colimit` for finite categorical Yoneda computations. |
| `topos_ai.logic.SubobjectClassifier` | Goedel-Heyting fuzzy logic layer for neural experiments. | It is a differentiable finite-valued algebra, not the subobject classifier of an arbitrary topos. |

## Publication language

Use precise claims:

- "finite presheaf topos computation" instead of "general topos engine"
- "finite quasi-category horn checks" instead of "full infinity-category implementation"
- "finite groupoid semantics for identity types" instead of "complete HoTT kernel"
- "neural proxy inspired by category theory" for differentiable modules that do not validate formal laws

## Verification policy

Formal modules must include tests for their defining laws. Proxy modules must include boundedness, shape, stability, or smoke tests and must document the mathematical non-claims above.
```

- [ ] **Step 4: Link contracts from the API docs**

Add this paragraph immediately after the `## topos_ai.formal_category` heading in `docs/api.md`:

```markdown
For the boundary between formal finite mathematics and neural/proxy components, see [Mathematical Contracts](MATH_CONTRACTS.md).
```

- [ ] **Step 5: Run the documentation contract tests**

Run:

```bash
python -m pytest tests/test_research_docs.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add docs/MATH_CONTRACTS.md docs/api.md tests/test_research_docs.py
git commit -m "docs: add mathematical contracts"
```

---

### Task 2: Freeze the Research Public API Contract

**Files:**
- Create: `docs/PUBLIC_API.md`
- Modify: `tests/test_research_docs.py`
- Modify: `docs/index.md`

- [ ] **Step 1: Add a failing public API documentation test**

Append to `tests/test_research_docs.py`:

```python
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
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python -m pytest tests/test_research_docs.py::test_public_api_document_names_stable_research_surface -q
```

Expected: fail with `FileNotFoundError` for `docs/PUBLIC_API.md`.

- [ ] **Step 3: Create `docs/PUBLIC_API.md`**

Create:

```markdown
# Public API Contract

This document defines the stable research-facing API surface for ToposAI. These names should remain importable across patch releases unless a release note announces a breaking change.

## Formal category and topos core

Import from `topos_ai.formal_category`:

- `FiniteCategory`
- `FiniteFunctor`
- `Presheaf`
- `NaturalTransformation`
- `FrozenNaturalTransformation`
- `Subpresheaf`
- `GrothendieckTopology`
- `PresheafTopos`
- `natural_transformations`
- `representable_presheaf`
- `yoneda_element_to_transformation`
- `yoneda_transformation_to_element`
- `yoneda_lemma_bijection`
- `category_of_elements`
- `yoneda_density_colimit`

## Finite simplicial and quasi-category core

Import from `topos_ai.infinity_categories`:

- `FiniteHorn`
- `FiniteSimplicialSet`
- `nerve_2_skeleton`
- `nerve_3_skeleton`
- `SimplicialComplexBuilder`
- `HodgeLaplacianEngine`
- `InfinityCategoryLayer`

## HoTT finite groupoid core

Import from `topos_ai.hott`:

- `FinitePathGroupoid`
- `PathFamily`
- `HomotopyEquivalence`

## Stability expectations

- Constructors should continue validating mathematical laws at construction time.
- Functions that return natural transformations should return validated transformations.
- Finite examples in `examples/` should run without network access.
- Experimental modules may evolve faster, but their non-claims must stay documented in `docs/MATH_CONTRACTS.md`.
```

- [ ] **Step 4: Link public API from docs index**

Add to `docs/index.md` under the main project overview:

```markdown
## Research Library Guides

- [Mathematical Contracts](MATH_CONTRACTS.md)
- [Public API Contract](PUBLIC_API.md)
- [API Reference](api.md)
```

- [ ] **Step 5: Run documentation tests**

Run:

```bash
python -m pytest tests/test_research_docs.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add docs/PUBLIC_API.md docs/index.md tests/test_research_docs.py
git commit -m "docs: define public research API"
```

---

### Task 3: Add Runnable Research Examples

**Files:**
- Create: `examples/formal_yoneda_density.py`
- Create: `examples/sheafification_site.py`
- Create: `examples/kripke_joyal_forcing.py`
- Create: `examples/quasi_category_horns.py`
- Create: `examples/hott_transport.py`
- Create: `tests/test_examples.py`

- [ ] **Step 1: Write failing example smoke tests**

Create `tests/test_examples.py`:

```python
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
```

- [ ] **Step 2: Run example smoke tests and verify they fail**

Run:

```bash
python -m pytest tests/test_examples.py -q
```

Expected: fail with `FileNotFoundError` for the first missing example script.

- [ ] **Step 3: Create `examples/formal_yoneda_density.py`**

Use this script:

```python
from topos_ai.formal_category import (
    FiniteCategory,
    Presheaf,
    PresheafTopos,
    category_of_elements,
    yoneda_density_colimit,
)


def walking_arrow_category():
    return FiniteCategory(
        objects=("0", "1"),
        morphisms={"id0": ("0", "0"), "id1": ("1", "1"), "up": ("0", "1")},
        identities={"0": "id0", "1": "id1"},
        composition={
            ("id0", "id0"): "id0",
            ("id1", "id1"): "id1",
            ("up", "id0"): "up",
            ("id1", "up"): "up",
        },
    )


def main():
    category = walking_arrow_category()
    presheaf = Presheaf(
        category,
        sets={"0": {"a", "b"}, "1": {"u"}},
        restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
    )
    topos = PresheafTopos(category)
    elements, projection = category_of_elements(presheaf)
    density, to_presheaf, from_presheaf = yoneda_density_colimit(presheaf)

    return {
        "objects_in_category_of_elements": len(elements.objects),
        "projection_target_objects": len(set(projection.object_map.values())),
        "round_trip_to_presheaf": topos.compose_transformations(
            to_presheaf,
            from_presheaf,
        ).components == topos.identity_transformation(presheaf).components,
        "round_trip_to_density": topos.compose_transformations(
            from_presheaf,
            to_presheaf,
        ).components == topos.identity_transformation(density).components,
    }


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 4: Create `examples/quasi_category_horns.py`**

Use this script:

```python
from topos_ai.formal_category import FiniteCategory
from topos_ai.infinity_categories import FiniteHorn, nerve_2_skeleton, nerve_3_skeleton


def thin_chain_category():
    objects = ("0", "1", "2", "3")
    morphisms = {}
    identities = {}
    for i, src in enumerate(objects):
        identity = f"id{src}"
        identities[src] = identity
        for dst in objects[i:]:
            label = identity if src == dst else f"m{src}{dst}"
            morphisms[label] = (src, dst)

    composition = {}
    for before, (src, middle) in morphisms.items():
        for after, (after_src, dst) in morphisms.items():
            if middle == after_src:
                composition[(after, before)] = identities[src] if src == dst else f"m{src}{dst}"

    return FiniteCategory(objects, morphisms, identities, composition)


def main():
    category = thin_chain_category()
    nerve_2 = nerve_2_skeleton(category)
    nerve_3 = nerve_3_skeleton(category)
    horn = FiniteHorn(2, 1, {0: ("mor", "m12"), 2: ("mor", "m01")})

    return {
        "inner_kan_2": nerve_2.is_inner_kan(max_dimension=2),
        "inner_kan_3": nerve_3.is_inner_kan(max_dimension=3),
        "composition_filler": nerve_2.horn_fillers(horn)[0],
        "assoc_3_simplices": len(nerve_3.simplices[3]),
        "degeneracy_identities": nerve_3.validate_degeneracy_identities(),
    }


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 5: Create `examples/hott_transport.py`**

Use this script:

```python
from topos_ai.hott import FinitePathGroupoid, PathFamily


def main():
    paths = FinitePathGroupoid(
        objects=("A", "B"),
        paths={"idA": ("A", "A"), "idB": ("B", "B"), "p": ("A", "B"), "p_inv": ("B", "A")},
        identities={"A": "idA", "B": "idB"},
        inverses={"idA": "idA", "idB": "idB", "p": "p_inv", "p_inv": "p"},
        composition={
            ("idA", "idA"): "idA",
            ("idB", "idB"): "idB",
            ("p", "idA"): "p",
            ("idB", "p"): "p",
            ("p_inv", "idB"): "p_inv",
            ("idA", "p_inv"): "p_inv",
            ("p_inv", "p"): "idA",
            ("p", "p_inv"): "idB",
        },
    )
    family = PathFamily(
        base=paths,
        fibers={"A": {0, 1}, "B": {"zero", "one"}},
        transports={
            "idA": {0: 0, 1: 1},
            "idB": {"zero": "zero", "one": "one"},
            "p": {0: "zero", 1: "one"},
            "p_inv": {"zero": 0, "one": 1},
        },
    )

    return {
        "identity_type_size": len(paths.identity_type("A", "B")),
        "transported_one": family.transport("p", 1),
        "transport_functorial": family.validate_functorial_transport(),
    }


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 6: Create `examples/kripke_joyal_forcing.py`**

Use the walking-arrow category and presheaf from Step 3 and this `main`:

```python
from examples.formal_yoneda_density import walking_arrow_category
from topos_ai.formal_category import Presheaf, PresheafTopos, Subpresheaf


def main():
    category = walking_arrow_category()
    presheaf = Presheaf(
        category,
        sets={"0": {"a", "b"}, "1": {"u"}},
        restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
    )
    topos = PresheafTopos(category)
    subobject = Subpresheaf(presheaf, {"0": {"a"}, "1": {"u"}})
    identity = topos.identity_transformation(presheaf)

    return {
        "forcing_at_1": subobject.parent.category.identities["1"] in topos.forcing_sieve(subobject, "1", "u"),
        "forces_closed_truth": topos.forces(subobject, "1", "u"),
        "equality_truth_is_maximal": topos.equality_truth(presheaf, "1", "u", "u") == topos.maximal_sieve("1"),
        "quantifier_adjunctions_valid": topos.validate_quantifier_adjunctions(identity),
    }


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 7: Create `examples/sheafification_site.py`**

Use this script:

```python
from examples.formal_yoneda_density import walking_arrow_category
from topos_ai.formal_category import GrothendieckTopology, Presheaf, PresheafTopos


def main():
    category = walking_arrow_category()
    topos = PresheafTopos(category)
    topology = GrothendieckTopology(
        category,
        covering_sieves={
            "0": {frozenset({"id0"})},
            "1": {frozenset({"id1", "up"}), frozenset({"up"})},
        },
    )
    presheaf = Presheaf(
        category,
        sets={"0": {"a"}, "1": {"u", "v"}},
        restrictions={"id0": {"a": "a"}, "id1": {"u": "u", "v": "v"}, "up": {"u": "a", "v": "a"}},
    )
    sheaf = topos.sheafification(presheaf, topology)
    identity = topos.identity_transformation(sheaf)
    factorization = topos.sheafification_factorization(identity, topology)

    return {
        "is_sheaf_before_sheafification": topos.is_sheaf(presheaf, topology),
        "is_sheaf_after_sheafification": topos.is_sheaf(sheaf, topology),
        "factorization_valid": factorization.validate_naturality(),
    }


if __name__ == "__main__":
    print(main())
```

- [ ] **Step 8: Run example smoke tests**

Run:

```bash
python -m pytest tests/test_examples.py -q
```

Expected: `5 passed`.

- [ ] **Step 9: Commit**

```bash
git add examples tests/test_examples.py
git commit -m "docs: add runnable research examples"
```

---

### Task 4: Add Deterministic Formal Property Tests

**Files:**
- Create: `tests/test_formal_properties.py`

- [ ] **Step 1: Write property-style tests**

Create `tests/test_formal_properties.py`:

```python
from topos_ai.formal_category import (
    FiniteCategory,
    Presheaf,
    PresheafTopos,
    Subpresheaf,
    category_of_elements,
    yoneda_density_colimit,
)
from topos_ai.infinity_categories import nerve_3_skeleton


def terminal_category():
    return FiniteCategory(
        objects=("*",),
        morphisms={"id": ("*", "*")},
        identities={"*": "id"},
        composition={("id", "id"): "id"},
    )


def walking_arrow_category():
    return FiniteCategory(
        objects=("0", "1"),
        morphisms={"id0": ("0", "0"), "id1": ("1", "1"), "up": ("0", "1")},
        identities={"0": "id0", "1": "id1"},
        composition={
            ("id0", "id0"): "id0",
            ("id1", "id1"): "id1",
            ("up", "id0"): "up",
            ("id1", "up"): "up",
        },
    )


def thin_chain_category():
    objects = ("0", "1", "2", "3")
    morphisms = {}
    identities = {}
    for i, src in enumerate(objects):
        identity = f"id{src}"
        identities[src] = identity
        for dst in objects[i:]:
            label = identity if src == dst else f"m{src}{dst}"
            morphisms[label] = (src, dst)
    composition = {}
    for before, (src, middle) in morphisms.items():
        for after, (after_src, dst) in morphisms.items():
            if middle == after_src:
                composition[(after, before)] = identities[src] if src == dst else f"m{src}{dst}"
    return FiniteCategory(objects, morphisms, identities, composition)


def test_yoneda_density_round_trip_across_small_presheaves():
    fixtures = [
        Presheaf(
            terminal_category(),
            sets={"*": {"x", "y"}},
            restrictions={"id": {"x": "x", "y": "y"}},
        ),
        Presheaf(
            walking_arrow_category(),
            sets={"0": {"a", "b"}, "1": {"u"}},
            restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
        ),
    ]

    for presheaf in fixtures:
        topos = PresheafTopos(presheaf.category)
        elements, projection = category_of_elements(presheaf)
        density, to_presheaf, from_presheaf = yoneda_density_colimit(presheaf)

        assert len(elements.objects) == sum(len(values) for values in presheaf.sets.values())
        assert set(projection.object_map.values()).issubset(set(presheaf.category.objects))
        assert topos.compose_transformations(to_presheaf, from_presheaf).components == (
            topos.identity_transformation(presheaf).components
        )
        assert topos.compose_transformations(from_presheaf, to_presheaf).components == (
            topos.identity_transformation(density).components
        )


def test_subobject_heyting_adjunction_across_all_subobjects_of_walking_arrow():
    category = walking_arrow_category()
    presheaf = Presheaf(
        category,
        sets={"0": {"a", "b"}, "1": {"u"}},
        restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
    )
    topos = PresheafTopos(category)
    subobjects = topos.subobjects(presheaf)

    def leq(left, right):
        return all(left.subsets[obj].issubset(right.subsets[obj]) for obj in category.objects)

    for a in subobjects:
        for b in subobjects:
            implication = topos.subobject_implication(a, b)
            for x in subobjects:
                assert leq(topos.subobject_meet(x, a), b) == leq(x, implication)


def test_inverse_image_preserves_meet_for_all_subobjects():
    category = walking_arrow_category()
    presheaf = Presheaf(
        category,
        sets={"0": {"a", "b"}, "1": {"u"}},
        restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
    )
    topos = PresheafTopos(category)
    identity = topos.identity_transformation(presheaf)
    subobjects = topos.subobjects(presheaf)

    for left in subobjects:
        for right in subobjects:
            pulled_meet = topos.inverse_image(identity, topos.subobject_meet(left, right))
            meet_pulled = topos.subobject_meet(
                topos.inverse_image(identity, left),
                topos.inverse_image(identity, right),
            )
            assert pulled_meet.subsets == meet_pulled.subsets


def test_nerve_3_skeleton_inner_kan_for_thin_chain():
    nerve = nerve_3_skeleton(thin_chain_category())
    assert nerve.validate_face_identities() is True
    assert nerve.validate_degeneracy_identities() is True
    assert nerve.is_inner_kan(max_dimension=3) is True
```

- [ ] **Step 2: Run the new tests**

Run:

```bash
python -m pytest tests/test_formal_properties.py -q
```

Expected: tests pass after the file is added because the implementation already exists.

- [ ] **Step 3: Run related formal suites**

Run:

```bash
python -m pytest tests/test_formal_category.py tests/test_category_core.py tests/test_formal_properties.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_formal_properties.py
git commit -m "test: add formal property checks"
```

---

### Task 5: Add Tutorial Pages and MkDocs Navigation

**Files:**
- Create: `docs/tutorials/yoneda_density.md`
- Create: `docs/tutorials/sheafification.md`
- Create: `docs/tutorials/internal_logic.md`
- Create: `docs/tutorials/quasi_categories_and_hott.md`
- Modify: `mkdocs.yml`
- Modify: `docs/index.md`
- Modify: `tests/test_research_docs.py`

- [ ] **Step 1: Add failing tutorial navigation test**

Append to `tests/test_research_docs.py`:

```python
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
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python -m pytest tests/test_research_docs.py::test_tutorial_pages_are_linked_from_mkdocs -q
```

Expected: fail because tutorial files are not linked yet.

- [ ] **Step 3: Create `docs/tutorials/yoneda_density.md`**

Create with:

````markdown
# Yoneda Density Tutorial

This tutorial uses the finite walking-arrow category `0 -> 1` and a presheaf `F` with `F(0) = {a, b}`, `F(1) = {u}`, and `F(up)(u) = a`.

Run:

```bash
python examples/formal_yoneda_density.py
```

The example constructs `int F`, the category of elements, and reconstructs `F` as `colim_{(c, x) in int F} y(c)`.

Expected summary keys:

- `objects_in_category_of_elements`
- `projection_target_objects`
- `round_trip_to_presheaf`
- `round_trip_to_density`

Mathematical statement: in the finite setting used here, every presheaf is recovered from representables indexed by its category of elements.
````

- [ ] **Step 4: Create `docs/tutorials/sheafification.md`**

Create with:

````markdown
# Sheafification Tutorial

This tutorial demonstrates a finite Grothendieck topology, matching families, and the associated sheaf construction.

Run:

```bash
python examples/sheafification_site.py
```

The example compares the original presheaf with its sheafification and checks that the resulting object satisfies the finite sheaf condition.

Mathematical statement: the finite plus-plus construction acts as a reflector from presheaves into sheaves for the tested finite site.
````

- [ ] **Step 5: Create `docs/tutorials/internal_logic.md`**

Create with:

````markdown
# Internal Logic Tutorial

This tutorial demonstrates Kripke-Joyal forcing, internal equality, and quantifier adjunctions in a finite presheaf topos.

Run:

```bash
python examples/kripke_joyal_forcing.py
```

The example reports forcing sieves, equality truth values, and a finite check of `exists_alpha -| alpha* -| forall_alpha`.

Mathematical statement: truth values in a presheaf topos are sieves, and quantifiers are adjoints to inverse image along a map.
````

- [ ] **Step 6: Create `docs/tutorials/quasi_categories_and_hott.md`**

Create with:

````markdown
# Quasi-Categories and HoTT Tutorial

This tutorial connects two finite formal layers:

- `FiniteSimplicialSet` checks face identities, degeneracy identities, horn fillers, and finite inner-Kan conditions.
- `FinitePathGroupoid` models 1-truncated identity types with functorial transport.

Run:

```bash
python examples/quasi_category_horns.py
python examples/hott_transport.py
```

Mathematical statements:

- Inner 2-horn fillers in the nerve of a category encode composition.
- Inner 3-horn coherence in the 3-skeleton encodes associativity.
- Finite path groupoids model identity proofs in a 1-truncated HoTT semantics.
- Path families validate transport as a functor out of the path groupoid.
````

- [ ] **Step 7: Modify `mkdocs.yml` navigation**

Add entries under `nav:`:

```yaml
  - Mathematical Contracts: docs/MATH_CONTRACTS.md
  - Public API: docs/PUBLIC_API.md
  - Tutorials:
      - Yoneda Density: docs/tutorials/yoneda_density.md
      - Sheafification: docs/tutorials/sheafification.md
      - Internal Logic: docs/tutorials/internal_logic.md
      - Quasi-Categories and HoTT: docs/tutorials/quasi_categories_and_hott.md
```

- [ ] **Step 8: Run docs tests**

Run:

```bash
python -m pytest tests/test_research_docs.py -q
```

Expected: all documentation tests pass.

- [ ] **Step 9: Commit**

```bash
git add docs/tutorials docs/index.md mkdocs.yml tests/test_research_docs.py
git commit -m "docs: add research tutorials"
```

---

### Task 6: Add CI Documentation and Example Smoke Checks

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add `test_examples.py`, `test_formal_properties.py`, and docs tests to CI**

Modify the `Run unit and hygiene tests` command in `.github/workflows/ci.yml` so it includes:

```yaml
          pytest \
            tests/test_core.py \
            tests/test_models.py \
            tests/test_generation.py \
            tests/test_category_core.py \
            tests/test_core_extensions.py \
            tests/test_formal_category.py \
            tests/test_formal_properties.py \
            tests/test_research_docs.py \
            tests/test_examples.py \
            tests/test_claim_hygiene.py \
            tests/test_runtime.py \
            -v --tb=short \
            -m "not cuda and not triton and not slow" \
            --cov=topos_ai --cov-report=term-missing --cov-fail-under=85
```

- [ ] **Step 2: Add MkDocs build smoke**

Add this step after the unit test step:

```yaml
      - name: Build documentation
        run: |
          python -m pip install mkdocs
          mkdocs build --strict
```

- [ ] **Step 3: Extend wheel smoke to import formal APIs**

Append this code inside the wheel smoke Python block:

```python
from topos_ai.formal_category import FiniteCategory, Presheaf, yoneda_density_colimit
from topos_ai.infinity_categories import nerve_3_skeleton
from topos_ai.hott import FinitePathGroupoid

category = FiniteCategory(
    objects=("*",),
    morphisms={"id": ("*", "*")},
    identities={"*": "id"},
    composition={("id", "id"): "id"},
)
presheaf = Presheaf(category, sets={"*": {"x"}}, restrictions={"id": {"x": "x"}})
assert yoneda_density_colimit(presheaf)[0].sets["*"]
assert nerve_3_skeleton(category).is_inner_kan(max_dimension=3)
assert FinitePathGroupoid(
    objects=("*",),
    paths={"id": ("*", "*")},
    identities={"*": "id"},
    inverses={"id": "id"},
    composition={("id", "id"): "id"},
).validate_groupoid_laws()
```

- [ ] **Step 4: Run local CI-equivalent commands**

Run:

```bash
python -m ruff check topos_ai tests
python -m compileall -q topos_ai tests experiments benchmarks
python -m pytest tests/test_core.py tests/test_models.py tests/test_generation.py tests/test_category_core.py tests/test_core_extensions.py tests/test_formal_category.py tests/test_formal_properties.py tests/test_research_docs.py tests/test_examples.py tests/test_claim_hygiene.py tests/test_runtime.py -q -m "not cuda and not triton and not slow" --cov=topos_ai --cov-report=term-missing --cov-fail-under=85
python -m build
```

Expected: all commands exit `0`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: verify docs examples and formal properties"
```

---

### Task 7: Reframe README and Project Status for Researchers

**Files:**
- Modify: `README.md`
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `tests/test_research_docs.py`

- [ ] **Step 1: Add failing docs wording test**

Append to `tests/test_research_docs.py`:

```python
def test_readme_and_status_use_research_library_positioning():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "docs" / "PROJECT_STATUS.md").read_text(encoding="utf-8")

    assert "formal finite core" in readme
    assert "neural/proxy components" in readme
    assert "Maturity Matrix" in status
    assert "Research-library readiness" in status
```

- [ ] **Step 2: Run focused test and verify it fails**

Run:

```bash
python -m pytest tests/test_research_docs.py::test_readme_and_status_use_research_library_positioning -q
```

Expected: fail because the exact phrases are not present yet.

- [ ] **Step 3: Modify README introduction**

Replace the first project description paragraph in `README.md` with:

```markdown
**ToposAI** is an experimental research library with a formal finite core for category/topos computations and neural/proxy components inspired by categorical structures. The formal core covers finite categories, presheaf topoi, Yoneda reconstruction, sheafification, Kripke-Joyal style internal logic, finite quasi-category horn checks, and 1-truncated HoTT path groupoid semantics.

The neural/proxy components explore how Goedel-Heyting logic, sheaf-style consistency, topological features, and categorical constraints can be embedded into PyTorch models. Proxy modules are documented as research scaffolds rather than complete implementations of the full mathematical theories that inspire them.
```

- [ ] **Step 4: Add a formal-core table to README**

Add after `Architecture at a Glance`:

```markdown
## Formal Finite Core

| Area | Implemented finite contract |
|------|-----------------------------|
| Category theory | `FiniteCategory`, `FiniteFunctor`, natural transformations, representables |
| Presheaf topoi | limits, colimits, exponentials, subobject classifier, Heyting operations |
| Sheaves | finite Grothendieck topologies, matching families, sheafification |
| Internal logic | forcing sieves, equality, existential and universal quantifiers |
| Yoneda | Yoneda lemma bijection and density reconstruction through category of elements |
| Quasi-categories | finite simplicial sets, horn fillers, 2- and 3-skeleton nerves |
| HoTT semantics | finite path groupoids and functorial transport |

See [Mathematical Contracts](docs/MATH_CONTRACTS.md) for exact guarantees and non-claims.
```

- [ ] **Step 5: Add maturity matrix to project status**

Append to `docs/PROJECT_STATUS.md`:

```markdown
## Maturity Matrix

| Area | Status | Evidence |
|------|--------|----------|
| Formal finite category/topos core | Research-library prototype | Law tests, Yoneda density tests, sheafification tests, quantifier tests |
| Finite quasi-category skeletons | Research-library prototype | Inner horn filler tests, degeneracy identity tests, 3-simplex associativity coherence tests |
| HoTT finite groupoid semantics | Research-library prototype | Identity type, inverse path, composition, and transport functoriality tests |
| Neural/proxy modules | Experimental | Shape, boundedness, runtime, and smoke tests |
| Benchmarks | Experimental | Reproducibility docs required before publication |
| GPU/Triton kernels | Experimental | CPU fallback tested; dedicated GPU CI still needed |

## Research-library readiness

The repository is ready to be presented as a research-library prototype when these checks pass together:

- mathematical contracts exist and are linked from API docs
- public API contract exists and names stable formal symbols
- examples run without network access
- formal property tests cover Yoneda density, Heyting adjunction, inverse image laws, and finite inner-Kan checks
- documentation builds in CI
- wheel smoke imports formal category, quasi-category, and HoTT APIs
```

- [ ] **Step 6: Run docs wording tests**

Run:

```bash
python -m pytest tests/test_research_docs.py -q
```

Expected: all documentation tests pass.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/PROJECT_STATUS.md tests/test_research_docs.py
git commit -m "docs: position project as research library prototype"
```

---

### Task 8: Final Research Readiness Verification

**Files:**
- No new files.

- [ ] **Step 1: Run lint**

Run:

```bash
python -m ruff check topos_ai tests
```

Expected: `All checks passed!`

- [ ] **Step 2: Run compile check**

Run:

```bash
python -m compileall -q topos_ai tests examples experiments benchmarks
```

Expected: exit `0`.

- [ ] **Step 3: Run focused research-library suite**

Run:

```bash
python -m pytest tests/test_research_docs.py tests/test_examples.py tests/test_formal_properties.py tests/test_formal_category.py tests/test_category_core.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run coverage suite**

Run:

```bash
python -m pytest tests/test_core.py tests/test_models.py tests/test_generation.py tests/test_category_core.py tests/test_core_extensions.py tests/test_formal_category.py tests/test_formal_properties.py tests/test_research_docs.py tests/test_examples.py tests/test_claim_hygiene.py tests/test_runtime.py -q -m "not cuda and not triton and not slow" --cov=topos_ai --cov-report=term-missing --cov-fail-under=85
```

Expected: coverage stays above `85%`.

- [ ] **Step 5: Run full suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass. Existing third-party deprecation warnings may remain if they match current smoke-test warnings.

- [ ] **Step 6: Build package**

Run:

```bash
python -m build
```

Expected: wheel and sdist build successfully.

- [ ] **Step 7: Run wheel smoke**

Run:

```powershell
@'
from pathlib import Path
import subprocess
import sys
import tempfile

wheel = next(Path("dist").glob("topos_ai-*.whl")).resolve()
with tempfile.TemporaryDirectory() as tmp:
    venv = Path(tmp) / "venv"
    subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.check_call([str(python), "-m", "pip", "install", "--no-deps", str(wheel)])
    code = """
from topos_ai.formal_category import FiniteCategory, Presheaf, yoneda_density_colimit
from topos_ai.infinity_categories import nerve_3_skeleton
from topos_ai.hott import FinitePathGroupoid
category = FiniteCategory(objects=('*',), morphisms={'id': ('*', '*')}, identities={'*': 'id'}, composition={('id', 'id'): 'id'})
presheaf = Presheaf(category, sets={'*': {'x'}}, restrictions={'id': {'x': 'x'}})
assert yoneda_density_colimit(presheaf)[0].sets['*']
assert nerve_3_skeleton(category).is_inner_kan(max_dimension=3)
assert FinitePathGroupoid(objects=('*',), paths={'id': ('*', '*')}, identities={'*': 'id'}, inverses={'id': 'id'}, composition={('id', 'id'): 'id'}).validate_groupoid_laws()
print('research wheel smoke passed')
"""
    subprocess.check_call([str(python), "-c", code])
'@ | python -
```

Expected: `research wheel smoke passed`.

- [ ] **Step 8: Run diff hygiene and tracked bytecode checks**

Run:

```bash
git diff --check
git ls-files | Select-String -Pattern '(__pycache__|\.pyc$)' | Measure-Object | Select-Object -ExpandProperty Count
```

Expected: `git diff --check` exits `0`; bytecode count is `0`. Existing CRLF warnings may appear and should be reported separately from whitespace errors.

- [ ] **Step 9: Commit verification updates if any**

If the previous tasks changed CI or docs after their task commits, commit the final changes:

```bash
git add .github/workflows/ci.yml README.md docs tests examples mkdocs.yml
git commit -m "chore: complete research readiness verification"
```

---

## Completion Criteria

The plan is complete when all of these are true:

- `docs/MATH_CONTRACTS.md` exists and explicitly separates formal finite mathematics from neural/proxy components.
- `docs/PUBLIC_API.md` exists and names the stable formal API.
- `examples/` contains five runnable, deterministic research examples.
- `tests/test_examples.py` executes the examples.
- `tests/test_formal_properties.py` checks formal laws across more than one finite fixture.
- MkDocs navigation includes contracts, public API, and tutorials.
- CI runs docs, examples, formal properties, coverage, build, and wheel smoke.
- README and project status describe ToposAI as a research-library prototype with a formal finite core.
- Final verification commands in Task 8 pass.

## Self-Review Checklist

- Spec coverage: the plan covers contracts, tutorials, examples, formal property tests, CI, README, project status, and final verification.
- Placeholder scan: the plan contains concrete file paths, concrete snippets, commands, and expected outputs.
- Type consistency: symbols used in examples match current public modules: `FiniteCategory`, `Presheaf`, `PresheafTopos`, `GrothendieckTopology`, `Subpresheaf`, `category_of_elements`, `yoneda_density_colimit`, `FiniteHorn`, `nerve_2_skeleton`, `nerve_3_skeleton`, `FinitePathGroupoid`, and `PathFamily`.
