import tempfile
from pathlib import Path
from aequilibrae.project.network.ovm_downloader import OVMDownloader

data_dir = Path(__file__).parent.parent.parent.parent / "data" / "overture" / "theme=transportation"


def test_download():
    with tempfile.TemporaryDirectory() as output_dir:
        o = OVMDownloader(["car"], output_dir)

        box1 = [148.713909, -20.272261, 148.7206475, -20.2702697]
        gdf_link, gdf_node = o.downloadTransportation(bbox=box1, data_source=data_dir, output_dir=output_dir)

        for t in ["segment", "connector"]:
            output_dir = Path(output_dir)
            # bbo = [148.71641, -20.27082, 148.71861, -20.27001]
            # woolworths_parkinglot = [148.718, -20.27049, 148.71889, -20.27006]
            expected_file = output_dir / f"theme=transportation" / f"type={t}" / f"transportation_data_{t}.parquet"
            assert expected_file.exists()

            link_columns = ["ovm_id", "connectors", "direction", "link_type", "name", "speed", "road", "geometry"]
            for element in link_columns:
                assert element in gdf_link.columns

            node_columns = ["ovm_id", "geometry"]
            for element in node_columns:
                assert element in gdf_node.columns

            # assert 'is_centroid' in gdf_node.columns
            # assert ['unknown', 'secondary', 'residential', 'parkingAisle'] == list(list_gdf[0]['link_type'].unique())

