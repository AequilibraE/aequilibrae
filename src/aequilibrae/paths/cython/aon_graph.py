"""Prepare assignment inputs from Graph and support legacy output buffers."""

from itertools import combinations
import operator

import numpy as np

from aequilibrae.paths.cython.aon_context import PreparedAoN
from aequilibrae.paths.cython.graph_context import NodeBasedContext, TurnBasedContext


def prepare_graph_inputs(matrix, graph):
    """Build a routing context, demand view and assignment options from Graph."""
    if matrix.matrix_view is None:
        raise ValueError("prepare the matrix computational view before assignment")
    if not np.array_equal(matrix.index, graph.centroids):
        raise ValueError("Matrix and graph do not have compatible sets of centroids")
    zones, links, nodes = graph.num_zones, graph.compact_num_links, graph.compact_num_nodes
    if not 0 < zones <= nodes:
        raise ValueError("assignment requires centroids")
    demand = matrix.matrix_view
    if demand.ndim == 2:
        demand = demand[:, :, None]
    if demand.ndim != 3 or demand.shape[:2] != (zones, zones) or demand.shape[2] < 1:
        raise ValueError("matrix view must have shape (zones, zones, classes)")
    if not np.array_equal(graph.compact_graph.id.to_numpy(), np.arange(links)):
        raise ValueError("compact graph IDs must be consecutive CSR positions")
    if not np.array_equal(graph.compact_nodes_to_indices[matrix.index], np.arange(zones)):
        raise ValueError("compact centroids must be the first nodes in matrix order")

    fs = graph.compact_fs.copy()
    # Graph may leave an isolated first centroid's offset at -1.
    if fs[0] == -1:
        fs[0] = 0
    heads = graph.compact_graph.b_node.to_numpy()
    costs = graph.compact_cost[:links]  # Exclude the unused sentinel.
    if graph.has_turn_restrictions:
        context = TurnBasedContext(fs, heads, costs, graph.compact_turn_fs,
                                   graph.compact_turn_to_arcs, graph.compact_turn_penalties,
                                   allow_uturns=graph.allow_path_uturns)
    else:
        context = NodeBasedContext(fs, heads, costs)
    if context.node_count != nodes:
        raise ValueError("compact graph node count does not match its forward star")

    field_count = len(graph.skim_fields)
    if field_count and (graph.compact_skims is None or graph.compact_skims.ndim != 2
                        or graph.compact_skims.shape[0] < links or graph.compact_skims.shape[1] < field_count):
        raise ValueError("compact_skims does not have enough link/field entries")
    fields = [graph.compact_skims[:links, field] for field in range(field_count)]
    penalty_names = graph.turn_skim_fields or ([graph.cost_field] if graph.cost_field else [])
    options = {
        "costs": costs,
        "skim_fields": fields,
        "skim_penalties": [name in penalty_names for name in graph.skim_fields],
        "block_centroids": graph.block_centroid_flows and not graph.has_turn_restrictions,
    }
    return context, demand, options


def prepare_aon(matrix, graph, *, cores=1, selected_links=None):
    """Prepare reusable assignment from Graph and a matrix demand view.

    Copies topology, demand and skim fields, but borrows graph.compact_cost.
    Call update_costs if that array is replaced; never change it during a run.
    selected_links maps names to compact directed-link indices (not external IDs).
    """
    context, demand, options = prepare_graph_inputs(matrix, graph)
    return PreparedAoN(context, demand, cores=cores, selected_links=selected_links, **options)


def validate_output_array(array, name, shape):
    """Check a writable float64 output's shape, alignment and contiguous layout."""
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be a float64 NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not array.flags.c_contiguous or not array.flags.aligned or not array.flags.writeable:
        raise ValueError(f"{name} must be writable, aligned and C-contiguous")
    return array


def validate_legacy_outputs(matrix, graph, result, aux, cores):
    """Check legacy options and output buffers before assignment writes to them."""
    if result._graph_id != graph._id:
        raise ValueError("Results object not prepared. Use --> results.prepare(graph)")
    for enabled, feature in ((result.save_path_file, "path-file saving"),
                             (result._heap != "4ary", "heaps other than 4ary")):
        if enabled:
            raise NotImplementedError(f"context AoN does not support {feature}")
    if not 1 <= cores <= result.cores:
        raise ValueError("aux_result does not have enough thread capacity (cores must be positive)")
    zones, classes = graph.num_zones, result.classes["number"]
    loads = validate_output_array(aux.temp_link_loads, "temp_link_loads", (result.cores, result.links + 1, classes))
    turns = validate_output_array(aux.turn_penalty_accumulator, "turn_penalty_accumulator", (result.cores,))
    if loads.shape[1] < graph.compact_num_links + 1:
        raise ValueError("aux_result does not have enough link capacity")
    fields = len(graph.skim_fields)
    skims = validate_output_array(result.skims.matrix_view, "skims", (zones, zones, fields)) if fields else None
    sl_loads = sl_od = None
    if result._selected_links:
        sets = len(result._selected_links)
        sl_loads = validate_output_array(aux.temp_sl_link_loading, "temp_sl_link_loading",
                                        (result.cores, sets, graph.compact_num_links, classes))
        sl_od = validate_output_array(aux.temp_sl_od_matrix, "temp_sl_od_matrix",
                                     (result.cores, sets, zones, zones, classes))
    arrays = [array for array in (matrix.matrix_view, loads, turns, skims, sl_loads, sl_od) if array is not None]
    if any(np.shares_memory(a, b) for a, b in combinations(arrays, 2)):
        raise ValueError("AoN input and output buffers must not overlap")
    return loads, turns, skims, sl_loads, sl_od


def legacy_select_links(result, aux, links):
    """Read named link sets from the old assignment buffers.

    Rows are padded with -1 because sets can have different lengths. Remove
    that padding so it cannot be mistaken for a link index in the new kernel.
    """
    names = result._selected_links
    if not names:
        return None
    indices = [operator.index(index) for index in names.values()]
    if sorted(indices) != list(range(len(names))):
        raise ValueError("selected-link results must be prepared with consecutive set indices")
    table = aux.select_links
    if (not isinstance(table, np.ndarray) or table.ndim != 2
            or table.shape[0] != len(names) or table.dtype.kind not in "iu"):
        raise ValueError("select_links must contain one integer row per set")
    selected = {}
    # Output rows follow stored row indices, which may differ from name order.
    # Preserve those indices so loads stay attached to the right set names.
    for name, row_index in sorted(names.items(), key=lambda item: item[1]):
        members = []
        padding = False
        for link in table[row_index]:
            if link == -1:
                padding = True
            elif padding or not 0 <= link < links:
                raise ValueError("select_links must contain valid compact links followed by -1 padding")
            else:
                members.append(int(link))
        selected[name] = members
    return selected


def choose_legacy_origins(matrix, graph, context, demand, skimming):
    """Choose active legacy origins and report centroids with no outgoing links."""
    report, origins = [], []
    active = np.nansum(demand, axis=(1, 2)) > 0
    fs = context.fs
    for origin, external in enumerate(matrix.index):
        if not (active[origin] or skimming):
            continue
        full = graph.nodes_to_indices[external]
        if graph.fs[full] == graph.fs[full + 1] or fs[origin] == fs[origin + 1]:
            report.append(f"Centroid {external} is not connected")
        else:
            origins.append(origin)
    return origins, report


def aon_parallel_context(matrix, graph, result, aux_result, cores, bridge=None):
    """Add loads and turn totals to legacy buffers and copy active skim rows.

    Prepares a new assignment each call; use prepare_aon for reuse. bridge is unused.
    """
    cores = operator.index(cores)
    loads, turns, skims, sl_loads, sl_od = validate_legacy_outputs(matrix, graph, result, aux_result, cores)
    context, demand, options = prepare_graph_inputs(matrix, graph)
    if demand.shape[2] != result.classes["number"]:
        raise ValueError("matrix view must have shape (zones, zones, classes)")
    origins, report = choose_legacy_origins(matrix, graph, context, demand, bool(options["skim_fields"]))
    selected = legacy_select_links(result, aux_result, context.link_count)
    prepared = PreparedAoN(context, demand, cores=cores, origins=origins, selected_links=selected, **options)
    outputs = prepared.make_outputs()
    prepared.run(outputs)
    thread_loads, thread_turns = prepared.thread_outputs
    loads[:cores, :context.link_count, :] += thread_loads
    turns[:cores] += thread_turns
    if selected:
        sl_loads[:cores] += prepared.thread_select_link_loads
        # The caller sums OD across workers. Store each row only once to avoid
        # counting it twice. Clear all worker copies, even if this run uses fewer
        # workers, so old OD values cannot survive in unused worker slots.
        for origin in origins:
            sl_od[:, :, origin, :, :] = 0
            sl_od[0, :, origin, :, :] = outputs.select_link_od[:, origin, :, :]
    if skims is not None:
        for origin in origins:
            skims[origin] = outputs.skims[origin]
    return report
