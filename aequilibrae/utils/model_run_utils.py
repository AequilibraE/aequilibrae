import sys
import pathlib
import importlib.util


def import_directory_as_module(directory_path: pathlib.Path, module_name):
    """
    Import a directory as a Python module.

    :Arguments:
        **directory_path** (:obj:`pathlib.Path`): Path object pointing to the directory

        **module_name**: Name to give the imported module

    :Returns:
        The imported module
    """
    init_file = directory_path / "__init__.py"

    spec = importlib.util.spec_from_file_location(module_name, init_file)
    if spec is None:
        raise ImportError(f"Could not find module spec for {init_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
