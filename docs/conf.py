from sphinx.ext import autodoc
import inspect
import os
import sys
import re
sys.path.insert(0, os.path.abspath('..'))

import pyrssa as prs

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyrssa'
copyright = '2023, Fleyderer'
author = 'Fleyderer'
release = '1.0.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.intersphinx',
    'autodocsumm'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
add_module_names = False
pyrssa_classes = [class_name for class_name, class_value in inspect.getmembers(prs) if class_name != '__builtins__'
                  and inspect.isclass(class_value)]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "packaging": ("https://packaging.pypa.io/en/latest", None),
    "numpy": ('https://numpy.org/doc/stable/', None)
}


class MockedClassDocumenter(autodoc.ClassDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)


class MockedFunctionDocumenter(autodoc.FunctionDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        for prs_class in pyrssa_classes:
            line = line.replace(f"|{prs_class}|", f":class:`.{prs_class}`")
        super().add_line(line, source, *lineno)

    format_signature = None


autodoc.ClassDocumenter = MockedClassDocumenter
autodoc.FunctionDocumenter = MockedFunctionDocumenter


# change the functions signature width to the maximum possible
def setup(app):
    app.add_css_file('css/funcsigs.css')
