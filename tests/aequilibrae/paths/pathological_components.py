"""Reusable definitions for the pathological network fixture suite."""

from dataclasses import replace

from .pathological_network import BridgeDef, LinkDef, NetworkComponent, NodeDef, PathologicalNetwork, TurnDef


def make_finite_turn_component() -> NetworkComponent:
    """A finite movement cost makes the geometrically short branch suboptimal."""
    return NetworkComponent(
        name="finite_turn",
        purpose="A finite movement cost makes the geometrically short branch suboptimal.",
        nodes=(
            NodeDef("origin", 0, 0),
            NodeDef("via", 1, 0),
            NodeDef("detour", 1, 1),
            NodeDef("destination", 2, 0),
        ),
        links=(
            LinkDef("short_in", "origin", "via", cost=1.0, direction=1),
            LinkDef("short_out", "via", "destination", cost=1.0, direction=1),
            LinkDef("detour_in", "origin", "detour", cost=2.0, direction=1),
            LinkDef("detour_out", "detour", "destination", cost=2.0, direction=1),
        ),
        turns=(TurnDef("origin", "via", "destination", penalty=3.0),),
    )


def make_mixed_direction_component() -> NetworkComponent:
    """Every physical-link direction convention appears in one component."""
    return NetworkComponent(
        name="mixed_direction",
        purpose="Expand AB-only, BA-only, and bidirectional links into the correct directed arcs.",
        nodes=(
            NodeDef("west", 0, 0),
            NodeDef("centre", 1, 0),
            NodeDef("east", 2, 0),
            NodeDef("north", 1, 1),
        ),
        links=(
            LinkDef("two_way", "west", "centre", cost=1.0, direction=0),
            LinkDef("eastbound", "centre", "east", cost=1.0, direction=1),
            LinkDef("southbound", "centre", "north", cost=1.0, direction=-1),
        ),
    )


def make_shared_junction_centroid_component() -> NetworkComponent:
    """Two centroid connectors meet at an ordinary junction without blocking their OD path."""
    return NetworkComponent(
        name="shared_junction_centroid",
        purpose="Keep centroid blocking on centroid transitions, not connectors sharing an ordinary junction.",
        nodes=(
            NodeDef("origin", 0, 0, centroid=True),
            NodeDef("destination", 2, 0, centroid=True),
            NodeDef("junction", 1, 0),
            NodeDef("alternate", 1, 1),
        ),
        links=(
            LinkDef("origin_junction", "origin", "junction", cost=1.0, direction=0),
            LinkDef("junction_destination", "junction", "destination", cost=1.0, direction=1),
            LinkDef("origin_alternate", "origin", "alternate", cost=3.0, direction=1),
            LinkDef("alternate_destination", "alternate", "destination", cost=3.0, direction=1),
        ),
    )


def make_intermediate_centroid_component() -> NetworkComponent:
    """A cheap centroid-crossing route competes with one longer directed bypass."""
    return NetworkComponent(
        name="intermediate_centroid",
        purpose="Block only traversal through a true intermediate centroid while preserving endpoint access.",
        nodes=(
            NodeDef("origin", 0, 0, centroid=True),
            NodeDef("left", 1, 0),
            NodeDef("middle", 2, 0, centroid=True),
            NodeDef("right", 3, 0),
            NodeDef("destination", 4, 0, centroid=True),
            NodeDef("single", 1, -1, centroid=True),
        ),
        links=(
            LinkDef("origin_left", "origin", "left", cost=1.0, direction=0),
            LinkDef("left_middle", "left", "middle", cost=1.0, direction=1),
            LinkDef("middle_right", "middle", "right", cost=1.0, direction=1),
            LinkDef("right_destination", "right", "destination", cost=1.0, direction=0),
            LinkDef("left_right_bypass", "left", "right", cost=5.0, direction=1),
            LinkDef("single_connector", "single", "left", cost=2.0, direction=-1),
        ),
    )


def make_compression_component() -> NetworkComponent:
    """A prohibited movement must keep its degree-two via node out of contraction."""
    return NetworkComponent(
        name="compression",
        purpose="Preserve a restricted degree-two via while safely contracting an unrestricted bypass.",
        nodes=(
            NodeDef("origin", 0, 0, centroid=True),
            NodeDef("via", 1, 0),
            NodeDef("bypass", 1, 1),
            NodeDef("destination", 2, 0, centroid=True),
        ),
        links=(
            LinkDef("restricted_in", "origin", "via", cost=1.0, direction=1),
            LinkDef("restricted_out", "via", "destination", cost=1.0, direction=1),
            LinkDef("bypass_in", "origin", "bypass", cost=2.0, direction=1),
            LinkDef("bypass_out", "bypass", "destination", cost=2.0, direction=1),
        ),
        turns=(TurnDef("origin", "via", "destination", penalty=None),),
    )


def make_dead_end_component() -> NetworkComponent:
    """Three directed forms of dead end hang from an otherwise valid centroid route."""
    return NetworkComponent(
        name="dead_end",
        purpose="Remove sink-only, source-only, and bidirectional leaf spurs without changing centroid OD paths.",
        nodes=(
            NodeDef("origin", 0, 0, centroid=True),
            NodeDef("main", 1, 0),
            NodeDef("destination", 2, 0, centroid=True),
            NodeDef("sink", 1, -1),
            NodeDef("source", 0, 1),
            NodeDef("leaf", 2, 1),
        ),
        links=(
            LinkDef("origin_main", "origin", "main", cost=1.0, direction=1),
            LinkDef("main_destination", "main", "destination", cost=1.0, direction=1),
            LinkDef("sink_only", "main", "sink", cost=0.5, direction=1),
            LinkDef("source_only", "source", "main", cost=0.5, direction=1),
            LinkDef("bidirectional_leaf", "main", "leaf", cost=0.5, direction=0),
        ),
    )


def make_prohibited_parallel_component() -> NetworkComponent:
    """A node movement prohibition covers two equal-cost parallel incoming arcs."""
    return NetworkComponent(
        name="prohibited_parallel",
        purpose="Apply one prohibited node movement to every parallel incoming arc and force the visible bypass.",
        nodes=(
            NodeDef("origin", 0, 0),
            NodeDef("via", 1, 0),
            NodeDef("bypass", 1, 1),
            NodeDef("destination", 2, 0),
        ),
        links=(
            LinkDef("parallel_first", "origin", "via", cost=1.0, direction=1),
            LinkDef("parallel_second", "origin", "via", cost=1.0, direction=1),
            LinkDef("via_out", "via", "destination", cost=1.0, direction=1),
            LinkDef("bypass_in", "origin", "bypass", cost=2.0, direction=1),
            LinkDef("bypass_out", "bypass", "destination", cost=2.0, direction=1),
        ),
        turns=(TurnDef("origin", "via", "destination", penalty=None),),
    )


def make_uturn_component() -> NetworkComponent:
    """A finite branch reversal can bypass a prohibited direct movement."""
    return NetworkComponent(
        name="uturn",
        purpose="Contrast absolute global U-turn blocking with explicit finite and prohibited reversal controls.",
        nodes=(
            NodeDef("origin", 0, 0),
            NodeDef("junction", 1, 0),
            NodeDef("branch", 1, 1),
            NodeDef("destination", 2, 0),
        ),
        links=(
            LinkDef("origin_in", "origin", "junction", cost=1.0, direction=1),
            LinkDef("direct_out", "junction", "destination", cost=1.0, direction=1),
            LinkDef("branch_leg", "junction", "branch", cost=1.0, direction=0),
            LinkDef("branch_escape", "branch", "destination", cost=8.0, direction=1),
        ),
        turns=(
            TurnDef("origin", "junction", "destination", penalty=None),
            TurnDef("junction", "branch", "junction", penalty=2.0),
        ),
    )


def make_disconnected_component() -> NetworkComponent:
    """A disconnected destination-island pair sits apart from a reachable directed fragment."""
    return NetworkComponent(
        name="disconnected",
        purpose="Report no path between a reachable fragment and a visibly disconnected destination island.",
        nodes=(
            NodeDef("origin", 0, 0),
            NodeDef("reachable", 1, 0),
            NodeDef("destination", 3, 0),
            NodeDef("island", 4, 0),
        ),
        links=(
            LinkDef("reachable_link", "origin", "reachable", cost=1.0, direction=1),
            LinkDef("island_link", "destination", "island", cost=1.0, direction=1),
        ),
    )


BRIDGE_COST = 1_000_000.0


def make_equal_parallel_component() -> NetworkComponent:
    """Removes the parallel movement control so both equal incoming states remain valid."""
    return replace(
        make_prohibited_parallel_component(),
        name="equal_parallel",
        purpose="Keep two equal-cost parallel incoming states without prescribing tie identity.",
        turns=(),
    )


def make_prohibited_uturn_component() -> NetworkComponent:
    """Replaces the finite branch U-turn with a prohibition."""
    base = make_uturn_component()
    return replace(
        base,
        name="prohibited_uturn",
        purpose="Apply an explicit U-turn prohibition even when path U-turns are globally allowed.",
        turns=(base.turns[0], TurnDef("junction", "branch", "junction", penalty=None)),
    )


def make_duplicate_uturn_control_component() -> NetworkComponent:
    """Adds finite and prohibited duplicates for the branch U-turn."""
    base = make_uturn_component()
    movement = ("junction", "branch", "junction")
    return replace(
        base,
        name="duplicate_uturn_control",
        purpose="Let a prohibition dominate finite controls duplicated around the same U-turn movement.",
        turns=base.turns
        + (
            TurnDef(*movement, penalty=None),
            TurnDef(*movement, penalty=0.5),
        ),
    )


def make_selected_finite_turn_component() -> NetworkComponent:
    """Prohibits the detour so the finite controlled movement must be selected."""
    base = make_finite_turn_component()
    return replace(
        base,
        name="selected_finite_turn",
        purpose="Force a finite controlled movement to expose turn cost in the terminal milepost.",
        turns=base.turns + (TurnDef("origin", "detour", "destination", penalty=None),),
    )


def make_all_centroid_finite_turn_component() -> NetworkComponent:
    """Marks every finite-turn node as a centroid for lifecycle rebuilding."""
    base = make_finite_turn_component()
    return replace(
        base,
        name="finite_turn_all_centroids",
        purpose="Exercise turn lifecycle rebuilding without also contracting an intermediate node.",
        nodes=tuple(replace(node, centroid=True) for node in base.nodes),
    )


def make_finite_turn_no_controls_component() -> NetworkComponent:
    """Removes controls from the all-centroid lifecycle component."""
    return replace(
        make_all_centroid_finite_turn_component(),
        name="finite_turn_no_controls",
        purpose="Provide the no-turn oracle for clearing controls on the lifecycle network.",
        turns=(),
    )


def make_finite_turn_without_detour_component() -> NetworkComponent:
    """Removes the unique detour egress from the all-centroid lifecycle component."""
    base = make_all_centroid_finite_turn_component()
    return replace(
        base,
        name="finite_turn_without_detour",
        purpose="Provide the reduced-topology oracle after excluding the unique detour egress.",
        links=tuple(link for link in base.links if link.name != "detour_out"),
    )


def make_reversed_finite_turn_component() -> NetworkComponent:
    """Reverses endpoints and node-triple controls independently of Graph.reverse()."""
    base = make_all_centroid_finite_turn_component()
    return replace(
        base,
        name="finite_turn_reversed",
        purpose="Provide an independent oracle for reversing physical links and movement controls.",
        links=tuple(replace(link, a=link.b, b=link.a) for link in base.links),
        turns=tuple(replace(turn, from_node=turn.to_node, to_node=turn.from_node) for turn in base.turns),
    )


def make_base_pathological_components() -> tuple[NetworkComponent, ...]:
    """Builds the components embedded in the connected composed network."""
    return (
        make_finite_turn_component(),
        make_mixed_direction_component(),
        make_prohibited_parallel_component(),
        make_uturn_component(),
        make_disconnected_component(),
        make_compression_component(),
        make_dead_end_component(),
        make_shared_junction_centroid_component(),
        make_intermediate_centroid_component(),
    )


def make_pathological_components() -> tuple[NetworkComponent, ...]:
    """Builds every base and test-variant component for individual dumping."""
    return make_base_pathological_components() + (
        make_equal_parallel_component(),
        make_prohibited_uturn_component(),
        make_duplicate_uturn_control_component(),
        make_selected_finite_turn_component(),
        make_all_centroid_finite_turn_component(),
        make_finite_turn_no_controls_component(),
        make_finite_turn_without_detour_component(),
        make_reversed_finite_turn_component(),
    )


def make_composed_pathological_network() -> PathologicalNetwork:
    """Joins every component with costly links that cannot create shortcuts."""
    bridges = (
        BridgeDef(
            "finite_turn_to_mixed_direction",
            "finite_turn:destination",
            "mixed_direction:west",
            cost=BRIDGE_COST,
        ),
        BridgeDef(
            "mixed_direction_to_prohibited_parallel",
            "mixed_direction:east",
            "prohibited_parallel:origin",
            cost=BRIDGE_COST,
        ),
        BridgeDef(
            "prohibited_parallel_to_uturn",
            "prohibited_parallel:destination",
            "uturn:origin",
            cost=BRIDGE_COST,
        ),
        BridgeDef(
            "uturn_to_disconnected",
            "uturn:destination",
            "disconnected:origin",
            cost=BRIDGE_COST,
        ),
        BridgeDef(
            "disconnected_to_compression",
            "disconnected:reachable",
            "compression:origin",
            cost=BRIDGE_COST,
        ),
        BridgeDef(
            "compression_to_dead_end",
            "compression:destination",
            "dead_end:origin",
            cost=BRIDGE_COST,
        ),
        BridgeDef(
            "dead_end_to_shared_junction_centroid",
            "dead_end:destination",
            "shared_junction_centroid:origin",
            cost=BRIDGE_COST,
        ),
        BridgeDef(
            "shared_junction_centroid_to_intermediate_centroid",
            "shared_junction_centroid:destination",
            "intermediate_centroid:origin",
            cost=BRIDGE_COST,
        ),
    )
    return PathologicalNetwork.compose(*make_base_pathological_components(), bridges=bridges)
