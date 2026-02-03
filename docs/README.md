# PySIPS Documentation

This directory contains the documentation for PySIPS.

## Building the Documentation Locally

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
pip install -e ..
```

### Build HTML Documentation

Using Make (recommended):

```bash
make html
```

Or using sphinx-build directly:

```bash
sphinx-build -b html . _build/html
cp .nojekyll _build/html/
```

The generated HTML files will be in `_build/html/`. Open `_build/html/index.html` in your browser to view the documentation.

### Clean Build Artifacts

```bash
make clean
```

## Documentation Structure

- `index.rst` - Main documentation page
- `quickstart.rst` - Quick start guide for new users
- `tutorial.rst` - Comprehensive tutorial with examples (text-based)
- `tutorial_notebook.ipynb` - Interactive Jupyter notebook tutorial (editable, downloadable)
- `api/` - API reference documentation (auto-generated from docstrings)
- `conf.py` - Sphinx configuration
- `requirements.txt` - Documentation build dependencies

## Automatic Deployment

The documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch. The workflow is defined in `.github/workflows/docs.yml`.

### Workflow Behavior

- **On pull requests**: Builds documentation and checks for errors (does not deploy)
- **On push to `develop`**: Builds documentation and checks for errors (does not deploy)
- **On push to `main`**: Builds documentation and deploys to GitHub Pages

## Viewing Published Documentation

Once deployed, the documentation is available at: https://nasa.github.io/pysips/

## Contributing to Documentation

When contributing to the documentation:

1. Update the relevant `.rst` files in this directory
2. For API documentation, update docstrings in the source code
3. Test your changes locally using `make html`
4. Ensure there are no build warnings or errors
5. Submit a pull request

### Writing Docstrings

PySIPS uses NumPy/Google style docstrings. Examples:

```python
def my_function(param1, param2):
    """
    Brief description of the function.

    Longer description with more details about what the function does.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    type
        Description of return value

    Examples
    --------
    >>> my_function(1, 2)
    3
    """
    return param1 + param2
```

## Troubleshooting

**Issue: Build fails with import errors**

- Make sure you've installed the package: `pip install -e ..`
- Make sure all dependencies are installed: `pip install -r requirements.txt`

**Issue: Warnings during build**

- Review the warnings to see if they're related to your changes
- Some warnings (like intersphinx network issues) are expected in restricted environments
- Use `sphinx-build -W` to treat warnings as errors (used in CI)

**Issue: Changes not appearing**

- Run `make clean` before `make html` to clear cached files
- Check that you're editing the right file (`.rst` vs `.py` docstrings)
