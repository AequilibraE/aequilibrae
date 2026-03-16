import os
import pathlib

root_dir = "src/aequilibrae"

ext_modules = {
    "src/aequilibrae/paths/cython/AoN.pyx": "AoN",
    "src/aequilibrae/distribution/cython/ipf_core.pyx": "ipf_core",
    "src/aequilibrae/paths/cython/public_transport.pyx": "public_transport",
    "src/aequilibrae/paths/cython/route_choice_set.pyx": "route_choice_set",
    "src/aequilibrae/paths/cython/route_choice_link_loading_results.pyx": "route_choice_link_loading_results",
    "src/aequilibrae/paths/cython/route_choice_set_results.pyx": "route_choice_set_results",
    "src/aequilibrae/paths/cython/graph_building.pyx": "graph_building",
    "src/aequilibrae/matrix/sparse_matrix.pyx": "sparse_matrix",
    "src/aequilibrae/matrix/coo_demand.pyx": "coo_demand",
}


def clean_old_mesons(path):
    for root, dirs, files in os.walk(path):
        if "meson.build" in files:
            os.remove(os.path.join(root, "meson.build"))


def get_subdirs(path):
    subdirs = []
    for entry in os.scandir(path):
        if entry.is_dir() and entry.name != "__pycache__" and not entry.name.endswith("egg-info"):
            # Check if subdir actually has anything useful
            has_stuff = False
            for r, d, f in os.walk(entry.path):
                if any(x.endswith((".py", ".pyx", ".pxd", ".pxi", ".sql", ".sqlite", ".zip", ".yml")) for x in f):
                    has_stuff = True
                    break
            if has_stuff:
                subdirs.append(entry.name)
    return sorted(subdirs)


def get_files(path):
    py_files = []
    ext_files = []
    for entry in os.scandir(path):
        if entry.is_file():
            if entry.name.endswith((".py", ".sql", ".sqlite", ".zip", ".yml")):
                py_files.append(entry.name)
            elif entry.name.endswith(".pyx"):
                full_path = os.path.join(path, entry.name).replace("\\", "/")
                if full_path in ext_modules:
                    ext_files.append(ext_modules[full_path])
    return sorted(py_files), sorted(ext_files)


def write_meson(path):
    subdirs = get_subdirs(path)
    py_files, ext_files = get_files(path)

    if not subdirs and not py_files and not ext_files:
        return

    thisdir = path.replace(os.sep, "/").removeprefix("src/")

    content = []
    content.append(f"thisdir = '{thisdir}'\n")

    if py_files:
        content.append("pyfiles = [")
        for f in py_files:
            content.append(f"  '{f}',")
        content.append("]\n")

    if ext_files:
        content.append("exts = [")
        for ext in ext_files:
            content.append(f"  '{ext}',")
        content.append("]\n")

    if subdirs:
        content.append("pkgs = [")
        for d in subdirs:
            content.append(f"  '{d}',")
        content.append("]\n")

    if py_files:
        content.append("py.install_sources(")
        content.append("  pyfiles,")
        content.append("  pure: false,")
        content.append("  subdir: thisdir,")
        content.append(")\n")

    if ext_files:
        content.append("foreach ext : exts")
        content.append("  py.extension_module(")
        content.append("    ext,")
        content.append("    ext + '.pyx',")
        content.append("    dependencies: deps,")
        content.append("    install: true,")
        content.append("    subdir: thisdir,")
        content.append("  )")
        content.append("endforeach\n")

    if subdirs:
        content.append("foreach pkg : pkgs")
        content.append("  subdir(pkg)")
        content.append("endforeach\n")

    if content:
        with open(os.path.join(path, "meson.build"), "w") as f:
            f.write("\n".join(content))

    for d in subdirs:
        write_meson(os.path.join(path, d))


if __name__ == "__main__":
    clean_old_mesons(root_dir)
    write_meson(root_dir)
