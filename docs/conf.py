"""Sphinx configuration for the CRYSTALLine documentation.

The same theme CRYSTALClear's documentation uses (sphinx-book-theme), so the
two sites read as one family, with a small stylesheet of our own on top — see
_static/custom.css.

Pages are Markdown, through MyST. Build:

    sphinx-build -b html docs site
"""

import datetime
import pathlib
import re

project = "CRYSTALLine"
author = "@crystaldevs"   # the GitHub account, which is how the project signs
copyright = f"{datetime.datetime.now():%Y}, {author}"

# Read the version rather than repeat it: the release workflow already
# cross-checks pyproject.toml against src/crystalline/__init__.py, and a third
# copy here would be the one nobody remembers to bump.
_init = (pathlib.Path(__file__).parent.parent / "src" / "crystalline" / "__init__.py")
_match = re.search(r'^__version__\s*=\s*["\'](.+?)["\']', _init.read_text(), re.M)
release = version = _match.group(1) if _match else ""

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "attrs_inline",
    "attrs_block",
    "substitution",
]
myst_heading_anchors = 3

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"

# Developer notes that live in docs/ but are not pages of the site, and the
# build output if anyone points sphinx-build at the source tree.
exclude_patterns = ["CRYSTALClear_notes.md", "_build", "site", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_title = "CRYSTALLine"
html_logo = "logo.png"      # the light-mode file; see html_theme_options["logo"]
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_baseurl = "https://crystaldevs.github.io/CRYSTALLine/"

html_theme_options = {
    "announcement": (
        "CRYSTALLine is in beta. Errors are to be expected, and bug reports, "
        "suggestions, comments and requests for new features are all welcome: "
        "<a href='https://github.com/crystaldevs/CRYSTALLine/issues'>open an "
        "issue</a>."
    ),
    "repository_url": "https://github.com/crystaldevs/CRYSTALLine",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "home_page_in_toc": True,
    "navigation_with_keys": False,
    "show_navbar_depth": 1,
    "max_navbar_depth": 3,
    "logo": {
        "image_light": "logo.png",
        "image_dark": "_static/logo-dark.png",
    },
}

html_context = {
    # Dark by default. The theme reads this from the context, not from
    # html_theme_options, where it is rejected as an unknown option.
    "default_mode": "dark",
    "github_user": "crystaldevs",
    "github_repo": "CRYSTALLine",
    "github_version": "main",
    "doc_path": "docs",
}
