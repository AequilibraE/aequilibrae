import sys
import pathlib
import importlib.util


def import_file_as_module(file: pathlib.Path, module_name, force: bool = False):
    """
    Import a file as a Python module.

    :Arguments:
        **file** (:obj:`pathlib.Path`): Path object pointing to the file to import

        **module_name**: Name to give the imported module

        **force**: Replace the module in ``sys.modules`` if it exists.

    :Returns:
        The imported module
    """
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None:
        raise ImportError(f"Could not find module spec for {file}")

    module = importlib.util.module_from_spec(spec)
    if module_name in sys.modules and not force:
        raise ImportError(f"Module name '{module_name}' already exists in sys.modules")
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
