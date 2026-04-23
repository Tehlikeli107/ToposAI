import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fast experiments that run in less than a second to ensure runtime logic is intact
from applications.generative_universal_grammar_topos import run_generative_grammar_experiment
from applications.semantic_universal_grammar_topos import run_semantic_grammar_experiment
from applications.categorical_kan_extensions_transfer_learning import run_kan_extension_experiment
from experiments.higher_category_theory_hypernetworks import run_2category_experiment

def test_generative_grammar_runtime():
    """Test that generative grammar runs without crashing."""
    run_generative_grammar_experiment()

def test_semantic_grammar_runtime():
    """Test that semantic grammar runs without crashing."""
    run_semantic_grammar_experiment()

def test_kan_extensions_runtime():
    """Test that Kan extensions transfer learning runs without crashing."""
    run_kan_extension_experiment()

def test_higher_category_hypernetworks_runtime():
    """Test that higher category theory hypernetworks run without crashing."""
    run_2category_experiment()
