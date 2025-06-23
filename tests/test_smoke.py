import importlib

def test_package_import_and_version():
    """Basic smoke test to ensure the package can be imported and has a version string."""
    pkg = importlib.import_module("les_audits_affaires_eval")
    assert hasattr(pkg, "__version__"), "Package should define __version__"
    assert isinstance(pkg.__version__, str), "__version__ should be a string"
    assert pkg.__version__, "__version__ should not be empty"
    assert pkg.__version__ != "0.0.0", "__version__ should be set to a real version" 