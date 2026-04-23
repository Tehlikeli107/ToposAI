import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fast experiments that run in less than a second to ensure runtime logic is intact
from applications.generative_universal_grammar_topos import run_generative_grammar_experiment
from applications.semantic_universal_grammar_topos import run_semantic_grammar_experiment
from applications.categorical_kan_extensions_transfer_learning import run_kan_extension_experiment
from experiments.higher_category_theory_hypernetworks import run_2category_experiment

from benchmarks.real_world_needle_in_haystack import run_needle_in_haystack
from benchmarks.mamba_vs_attention_benchmark import run_speed_benchmark
from benchmarks.scaling_laws_benchmark import run_scaling_benchmark

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

def test_needle_benchmark_runtime():
    """Test that needle benchmark runs for a small dimension without crashing."""
    run_needle_in_haystack(128)

def test_mamba_vs_attention_runtime():
    """Test that mamba speed benchmark runs for small seqs without crashing."""
    run_speed_benchmark([64, 128])

def test_scaling_laws_runtime():
    """Test that scaling laws benchmark runs for small seqs without crashing."""
    run_scaling_benchmark([64, 128])
