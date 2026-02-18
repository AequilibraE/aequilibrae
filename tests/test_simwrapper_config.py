import pytest

from aequilibrae.utils.simwrapper.generate_simwrapper_config import SimwrapperConfigGenerator
from aequilibrae.utils.simwrapper.simwrapper_panel import AequilibraEResultsMapPanel


def test_find_project_title_prefers_model_name(sioux_falls_example):
    prj = sioux_falls_example
    prj.about.model_name = "My Model"
    gen = SimwrapperConfigGenerator(prj)
    assert gen._find_project_title() == "My Model"


def test_output_dir_is_project_relative(sioux_falls_example, tmp_path):
    prj = sioux_falls_example

    # relative name -> created under project_base_path
    gen = SimwrapperConfigGenerator(prj, output_dir="my_sim")
    assert gen.output_dir.exists()
    assert gen.output_dir.parent == prj.project_base_path
    assert gen.output_dir.name == "my_sim"

    # absolute path inside project -> accepted as given
    inside = prj.project_base_path / "inside_sim"
    gen_inside = SimwrapperConfigGenerator(prj, output_dir=str(inside))
    assert gen_inside.output_dir == inside.resolve()

    # absolute path outside project -> rejected (use project's parent to guarantee 'outside')
    external = prj.project_base_path.parent / "outside_sim"
    with pytest.raises(ValueError):
        SimwrapperConfigGenerator(prj, output_dir=str(external))


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


def test_results_map_registers_extra_db_and_compute_range_default(sioux_falls_example):
    prj = sioux_falls_example
    panel = AequilibraEResultsMapPanel("title", project=prj, results_table="assignment", colour_metric="VOC_max")

    # extraDatabases should be registered when a results_table is used
    assert panel.extra_databases == {"results": "results_database.sqlite"}

    # non-existent metric -> default range
    panel2 = AequilibraEResultsMapPanel("t2", project=prj, colour_metric="this_column_does_not_exist")
    assert panel2._compute_data_range(prj, "this_column_does_not_exist") == [0, 1]


def test_cli_writes_dashboard(sioux_falls_example, tmp_path):
    """CLI should generate the dashboard YAML inside the project."""
    from aequilibrae.utils.simwrapper.generate_simwrapper_config import main

    prj = sioux_falls_example
    out_dir = "simwrapper-cli-test"

    # call CLI function directly (avoids packaging)
    rc = main(["--project", str(prj.project_base_path), "--output-dir", out_dir, "--quiet"])
    assert rc == 0

    expected = prj.project_base_path / out_dir / "dashboard.yaml"
    assert expected.exists()
