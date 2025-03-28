# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'OverLander'
copyright = '2025, Rohit Dhankar'
author = 'Rohit Dhankar'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

"""  
'myst_parser',
'sphinx.ext.autodoc',
'sphinx.ext.autosummary',
'sphinx.ext.intersphinx',
'sphinx.ext.todo',
'sphinx.ext.ifconfig',
'sphinx.ext.viewcode',
'sphinx.ext.graphviz',
'sphinx.ext.inheritance_diagram',
'sphinx.ext.napoleon',
#    'sphinxcontrib.fulltoc',
'sphinxcontrib.mermaid',

"""
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
            ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
