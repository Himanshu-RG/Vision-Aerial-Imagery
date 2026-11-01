import csv
from pathlib import Path

from aerial_ovd.data.splits import NWPU_NAMES, NWPU_TO_LAE, SPLITS, source_space

VOCAB = Path(__file__).resolve().parents[1] / "data" / "manifests" / "lae80c_vocabulary.csv"


def test_nwpu_mapping_matches_vocabulary_names():
    rows = {int(r["category_id"]): r for r in csv.DictReader(VOCAB.open(encoding="utf-8"))}
    for k, c in NWPU_TO_LAE.items():
        assert rows[c]["source"] == "DIOR-18C"
        assert rows[c]["raw_name"].replace(" ", "") == NWPU_NAMES[k].replace(" ", "").replace("diamond", "field")
    assert 2 not in NWPU_TO_LAE  # ship is dropped (D45)


def test_source_space():
    cat_source = {1: "DIOR-18C", 2: "DIOR-18C", 30: "FAIR-18C"}
    assert source_space("DIOR-18C", cat_source) == {1, 2}
    assert source_space("NWPU-VHR10", cat_source) == set(NWPU_TO_LAE.values())
    assert set(SPLITS) == {"dev", "test", "nwpu"}
