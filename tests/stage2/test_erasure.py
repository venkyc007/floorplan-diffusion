"""Realistic erasure: inpainted from the room fill (never white, no wall smear), nothing changes away
from the spot, and the image given to the pipeline carries no figure outline."""

import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import draw_wall, erase, grown, open_gap
from src.stage2.pipeline import check_set

I = C16.index


def toy():
    """Grey room (180) with a dark wall on top (rows 20-27) and a black fixture against it."""
    img = np.full((80, 80), 180, np.uint8)
    lbl = np.full((80, 80), I["bedroom"], np.uint8)
    img[20:28], lbl[20:28] = 40, I["wall"]
    img[28:36, 30:40], lbl[28:36, 30:40] = 0, I["sink"]
    return img, lbl


def test_erase_fills_from_room_not_white_or_wall():
    img, lbl = toy()
    m = lbl == I["sink"]
    out = erase(img, m, lbl)
    assert out.ndim == 2 and out.dtype == np.uint8
    assert abs(float(np.median(out[m])) - 180) < 15           # room grey continues, no white patch
    assert np.array_equal(out[lbl == I["wall"]], img[lbl == I["wall"]])
    away = ~grown(m, 2)
    assert np.array_equal(out[away], img[away])


def test_door_kinds():
    img, lbl = toy()
    door = np.zeros_like(lbl, bool)
    door[20:28, 50:62] = True
    img[door], lbl[door] = 255, I["door"]
    a = draw_wall(img, door, lbl)
    assert (a[door] == 40).all()                               # local wall colour across the doorway
    b = open_gap(img, door, lbl)
    assert abs(float(np.median(b[door])) - 180) < 30           # gap filled like the rooms beside it


def test_pipeline_input_has_no_outline():
    """make_case returns the image that goes to run_pipeline: single channel, changed only at the spot."""
    img, lbl = toy()
    out, removed, expected = check_set.make_case("fixture", img, lbl, None)
    assert out.ndim == 2 and expected == "sink"
    away = ~grown(removed, 2)
    assert np.array_equal(out[away], img[away])
