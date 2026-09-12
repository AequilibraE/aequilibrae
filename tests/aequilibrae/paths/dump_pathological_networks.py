"""Helper module for dumping pathological networks to parquet files."""

from __future__ import annotations

from pathlib import Path

from .pathological_components import make_composed_pathological_network, make_pathological_components
from .pathological_network import PathologicalNetwork


def dump_pathological_networks(output_directory: Path | str) -> dict[str, dict[str, Path]]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    result = {}

    for comp in make_pathological_components():
        comp_net = PathologicalNetwork.compose(comp)
        comp_dir = output_directory / comp.name
        comp_dir.mkdir(parents=True, exist_ok=True)
        nodes_path = comp_dir / "nodes.parquet"
        links_path = comp_dir / "links.parquet"
        turns_path = comp_dir / "turns.parquet"
        comp_net.node_frame().to_parquet(nodes_path)
        comp_net.link_frame().to_parquet(links_path)
        comp_net.turn_frame().to_parquet(turns_path)
        result[comp.name] = {
            "nodes": nodes_path,
            "links": links_path,
            "turns": turns_path,
        }

    composed_net = make_composed_pathological_network()
    comp_dir = output_directory / "composed"
    comp_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = comp_dir / "nodes.parquet"
    links_path = comp_dir / "links.parquet"
    turns_path = comp_dir / "turns.parquet"
    composed_net.node_frame().to_parquet(nodes_path)
    composed_net.link_frame().to_parquet(links_path)
    composed_net.turn_frame().to_parquet(turns_path)
    result["composed"] = {
        "nodes": nodes_path,
        "links": links_path,
        "turns": turns_path,
    }
    return result
