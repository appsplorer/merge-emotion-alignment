import pandas as pd
import pytest

from merge_emotion.data.integrity import (
    canonical_song_ids,
    discover_metadata_file,
    find_split_overlaps,
    normalize_quadrant,
    safe_identity_key,
    select_metadata_id_column,
)


def test_normalize_quadrant_accepts_common_formats():
    values = pd.Series(["Q1", "q2", 3, "4"])

    result = normalize_quadrant(values)

    assert result.tolist() == ["Q1", "Q2", "Q3", "Q4"]


def test_normalize_quadrant_rejects_unknown_label():
    values = pd.Series(["Q1", "happy"])

    with pytest.raises(ValueError, match="Unsupported quadrant"):
        normalize_quadrant(values)


def test_canonical_song_ids_prefers_allmusic_id():
    frame = pd.DataFrame(
        {
            "Song": ["A001", "MT002", "L003"],
            "AllMusic Id": ["AM100", None, ""],
        }
    )

    result = canonical_song_ids(
        frame,
        song_column="Song",
        allmusic_column="AllMusic Id",
    )

    assert result.tolist() == ["AM100", "MT002", "L003"]


def test_find_split_overlaps_detects_song_leakage():
    splits = {
        "train": {"A001", "A002", "A003"},
        "validation": {"A004", "A005"},
        "test": {"A003", "A006"},
    }

    overlaps = find_split_overlaps(splits)

    assert overlaps["train__validation"] == []
    assert overlaps["train__test"] == ["A003"]
    assert overlaps["validation__test"] == []


def test_safe_identity_key_normalizes_artist_and_title():
    result = safe_identity_key(
        artist="  The Example ",
        title="My   Song!",
    )

    assert result == "the example::my song"


def test_discover_metadata_prefers_full_bimodal_metadata(tmp_path):
    metadata = pd.DataFrame(
        {
            "Audio_Song": ["A001", "A002"],
            "Lyric_Song": ["L001", "L002"],
            "Quadrant": ["Q1", "Q2"],
            "AllMusic Id": ["AM1", "AM2"],
            "Artist": ["Artist A", "Artist B"],
            "Title": ["Song A", "Song B"],
        }
    )

    split = pd.DataFrame(
        {
            "Song": ["A001"],
            "Quadrant": ["Q1"],
        }
    )

    metadata_path = tmp_path / "merge_bimodal_complete_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    split_dir = tmp_path / "tvt_dataframes" / "tvt_70_15_15"
    split_dir.mkdir(parents=True)

    split.to_csv(
        split_dir / "tvt_70_15_15_train_bimodal_complete.csv",
        index=False,
    )

    selected_path, selected_frame = discover_metadata_file(tmp_path)

    assert selected_path == metadata_path
    assert len(selected_frame) == 2


def test_select_metadata_id_column_uses_split_identity():
    metadata = pd.DataFrame(
        {
            "Audio_Song": ["A001", "A002", "A003"],
            "Lyric_Song": ["L101", "L102", "L103"],
            "Quadrant": ["Q1", "Q2", "Q3"],
        }
    )

    selected = select_metadata_id_column(
        metadata,
        {"L101", "L102", "L103"},
    )

    assert selected == "Lyric_Song"