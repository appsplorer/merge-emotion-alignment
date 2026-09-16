import pandas as pd
from pathlib import Path
from merge_emotion.data.manifest import LABEL_TO_INDEX


def test_label_mapping_is_fixed():
    assert LABEL_TO_INDEX == {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
