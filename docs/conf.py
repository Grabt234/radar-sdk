# Configuration file for the Sphinx documentation builder.
import os
import sys
import plotly.io as pio

# 1. FORCE PLOTLY ENVIRONMENT VARIABLE (Must happen before everything else)
# This forces any Jupyter sub-kernel spawned by nbsphinx to use SVG/Iframe
os.environ["PLOTLY_RENDERER"] = (
    "notebook_connected"  # Or "iframe" if you prefer interactive plots
)

# -- Project information -----------------------------------------------------
project = "radar-sdk"
copyright = "2026, Grant Norrie"
author = "Grant Norrie"
release = "0.1.0"

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------
extensions = [
    "autodoc2",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

autodoc2_packages = [
    {"path": "../radar", "module": "radar"},
]
autodoc2_output_dir = "apidocs"
autodoc2_hidden_objects = []

templates_path = ["_templates"]

# Consolidated exclude patterns (added videos/.virtual_documents if needed)
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# 2. FORCE NBSPHINX TO RE-EXECUTE NOTEBOOKS
# This forces Sphinx to wipe old cached notebook outputs and re-render them using your Plotly settings
nbsphinx_execute = "always"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
