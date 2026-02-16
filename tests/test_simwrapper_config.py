import json
from pathlib import Path

import pandas as pd

from aequilibrae.utils.simwrapper.generate_simwrapper_config import SimwrapperConfigGenerator
from aequilibrae.utils.simwrapper.simwrapper_panel import AequilibraEResultsMapPanel


def test_find_project_title_prefers_model_name(sioux_falls_example):
    prj = sioux_falls_example
    prj.about.model_name = "My Model"
    gen = SimwrapperConfigGenerator(prj)
    assert gen._find_project_title() == "My Model"


def test_output_dir_is_project_relative(sioux_falls_example, tmp_path):
    prj = sioux_falls_example

    # relative name -> should be created under project_base_path
    gen = SimwrapperConfigGenerator(prj, output_dir="my_sim")
    assert gen.output_dir.exists()
    assert gen.output_dir.parent == prj.project_base_path
    assert gen.output_dir.name == "my_sim"

    # absolute path outside project -> coerced to project/<basename>
    external = tmp_path / "outside_sim"
    gen2 = SimwrapperConfigGenerator(prj, output_dir=str(external))
    assert gen2.output_dir.parent == prj.project_base_path
    assert gen2.output_dir.name == external.name


def test_categorical_palette_returns_hex(sioux_falls_example):
    gen = SimwrapperConfigGenerator(sioux_falls_example)
    cols = gen._categorical_palette(5)
    assert len(cols) == 5
    for c in cols:
        assert isinstance(c, str) and c.startswith("#")


def test_links_info_row_legend_colors_hex(sioux_falls_example):
    gen = SimwrapperConfigGenerator(sioux_falls_example)
    panels = gen._links_info_row()
    assert panels and len(panels) == 1
    panel = panels[0].to_dict()
    legend = panel.get("legend", [])
    # legend should contain colour entries as hex strings
    assert any(isinstance(item.get("color"), str) and item.get("color").startswith("#") for item in legend)


def test_assignment_convergence_plot_uses_output_dir_name(sioux_falls_example):
    gen = SimwrapperConfigGenerator(sioux_falls_example, output_dir="simwrk")

    # build a fake results dataframe with procedure_report containing convergence JSON
    df = pd.DataFrame(
        [
            {
                "table_name": "scenario_a",
                "procedure_report": json.dumps({"convergence": {"iteration": [1, 2], "rgap": [0.1, 0.01]}}),
            }
        ]
    )

    panels = gen._assignment_convergence_plot(df)
    assert panels is not None
    pnl = panels[0].to_dict()
    assert pnl["config"].startswith(f"{gen.output_dir.name}/simwrapper_data/")


def test_results_map_registers_extra_db_and_compute_range_default(sioux_falls_example):
    prj = sioux_falls_example
    panel = AequilibraEResultsMapPanel("title", project=prj, results_table="assignment", colour_metric="VOC_max")

    # extraDatabases should be registered when a results_table is used
    assert panel.extra_databases == {"results": "results_database.sqlite"}

    # non-existent metric -> default range
    panel2 = AequilibraEResultsMapPanel("t2", project=prj, colour_metric="this_column_does_not_exist")
    assert panel2._compute_data_range(prj, "this_column_does_not_exist") == [0, 1]
