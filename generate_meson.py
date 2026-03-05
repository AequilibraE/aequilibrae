import os

root_dir = "aequilibrae"

ext_modules = {
    "aequilibrae/paths/cython/AoN.pyx": "AoN",
    "aequilibrae/distribution/cython/ipf_core.pyx": "ipf_core",
    "aequilibrae/paths/cython/public_transport.pyx": "public_transport",
    "aequilibrae/paths/cython/route_choice_set.pyx": "route_choice_set",
    "aequilibrae/paths/cython/route_choice_link_loading_results.pyx": "route_choice_link_loading_results",
    "aequilibrae/paths/cython/route_choice_set_results.pyx": "route_choice_set_results",
    "aequilibrae/paths/cython/graph_building.pyx": "graph_building",
    "aequilibrae/matrix/sparse_matrix.pyx": "sparse_matrix",
    "aequilibrae/matrix/coo_demand.pyx": "coo_demand",
}


def get_subdirs(path):
    subdirs = []
    for entry in os.scandir(path):
        if entry.is_dir() and entry.name != "__pycache__" and not entry.name.endswith("egg-info"):
            subdirs.append(entry.name)
    return sorted(subdirs)


def get_files(path):
    py_files = []
    cy_files = []
    data_files = []
    for entry in os.scandir(path):
        if entry.is_file():
            if entry.name.endswith(".py"):
                py_files.append(entry.name)
            elif entry.name.endswith(".pyx") or entry.name.endswith(".pxd") or entry.name.endswith(".pxi"):
                cy_files.append(entry.name)
            elif (
                entry.name.endswith(".sql")
                or entry.name.endswith(".sqlite")
                or entry.name.endswith(".zip")
                or entry.name.endswith(".yml")
            ):
                data_files.append(entry.name)
    return sorted(py_files), sorted(cy_files), sorted(data_files)


def write_meson(path):
    subdirs = get_subdirs(path)
    py_files, cy_files, data_files = get_files(path)

    # We only create a meson.build if there are subdirs, py files, or data files
    if not subdirs and not py_files and not cy_files and not data_files:
        return

    content = []
    subdir_str = path.replace("/", ".")

    if py_files:
        content.append(f"py.install_sources(")
        for f in py_files:
            content.append(f"    '{f}',")
        content.append(f"    subdir: '{subdir_str}'")
        content.append(")")
        content.append("")

    if data_files:
        content.append(f"py.install_sources(")
        for f in data_files:
            content.append(f"    '{f}',")
        content.append(f"    subdir: '{subdir_str}'")
        content.append(")")
        content.append("")

    if cy_files:
        content.append(f"py.install_sources(")
        for f in cy_files:
            content.append(f"    '{f}',")
        content.append(f"    subdir: '{subdir_str}'")
        content.append(")")
        content.append("")

        for f in cy_files:
            if f.endswith(".pyx"):
                full_path = os.path.join(path, f)
                if full_path in ext_modules:
                    mod_name = ext_modules[full_path]
                    content.append(f"py.extension_module(")
                    content.append(f"    '{mod_name}',")
                    content.append(f"    '{f}',")
                    content.append(f"    cpp_args: cpp_args,")
                    if "link_args" in open("meson.build").read():  # wait wait we can just pass it if it exists
                        # but in our root meson.build we defined link_args? No, I need to fix root meson.build
                        content.append(f"    link_args: link_args,")
                    else:
                        content.append(f"    link_args: link_args,")  # We'll fix root
                    content.append(f"    install: true,")
                    content.append(f"    subdir: '{subdir_str}'")
                    content.append(")")
                    content.append("")

    for d in subdirs:
        content.append(f"subdir('{d}')")
        write_meson(os.path.join(path, d))

    with open(os.path.join(path, "meson.build"), "w") as f:
        f.write("\n".join(content))


write_meson(root_dir)
