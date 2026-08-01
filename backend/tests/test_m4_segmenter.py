"""
tests/test_m4_segmenter.py
=============================

Covers food segmentation (`05 §6`): the trained `FoodSegmenter` wrapper, the
improved GrabCut classical fallback, and the estimator's fallback chain.

The real DeepLabV3 weights are validated by the training pipeline's own IoU
(`training/portion`); here we stub the network so the wrapper logic — resize,
argmax over the food channel, resize the mask back to the original resolution — is
checked deterministically without weights.
"""

import numpy as np
import torch
from PIL import Image

from app.ml.portion import PortionEstimator
from app.ml.portion.segmenter import SEG_INPUT_SIZE, FoodSegmenter, _build_seg_transform

IMG_W, IMG_H = 640, 480


def _img() -> Image.Image:
    return Image.new("RGB", (IMG_W, IMG_H), (120, 120, 120))


class _StubSegModule:
    """Returns logits with 'food' (channel 1) winning in a central square."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _n, _c, h, w = x.shape
        logits = torch.zeros(1, 2, h, w)
        logits[:, 0] = 1.0  # background baseline
        logits[:, 1, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 2.0  # food in the centre
        return logits

    def eval(self):
        return self

    def to(self, _device):
        return self


# ------------------------------ FoodSegmenter --------------------------------
def test_segmenter_returns_mask_at_original_resolution():
    seg = FoodSegmenter(_StubSegModule(), _build_seg_transform(), torch.device("cpu"))
    mask = seg.segment(_img())
    assert mask.shape == (IMG_H, IMG_W)  # resized back to the original image size
    assert mask.dtype == bool


def test_segmenter_food_is_central_not_corners():
    seg = FoodSegmenter(_StubSegModule(), _build_seg_transform(), torch.device("cpu"))
    mask = seg.segment(_img())
    assert mask[IMG_H // 2, IMG_W // 2]  # centre is food
    assert not mask[0, 0] and not mask[-1, -1]  # corners are background
    # roughly the central quarter is food (allowing resize slack).
    assert 0.15 < mask.mean() < 0.45


def test_seg_transform_output_shape():
    x = _build_seg_transform()(_img())
    assert x.shape == (3, SEG_INPUT_SIZE, SEG_INPUT_SIZE)


# --------------------------- GrabCut classical fallback ----------------------
def _blob_image() -> Image.Image:
    """Textured dark table with a bright textured blob in the centre."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 40, (IMG_H, IMG_W, 3), dtype=np.uint8)  # dark noisy table
    cy, cx, r = IMG_H // 2, IMG_W // 2, 120
    yy, xx = np.ogrid[:IMG_H, :IMG_W]
    blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    arr[blob] = rng.integers(180, 255, (blob.sum(), 3), dtype=np.uint8)  # bright food
    return Image.fromarray(arr, "RGB")


def test_grabcut_fallback_isolates_central_blob():
    mask = PortionEstimator._classical_segment(_blob_image())
    assert mask is not None
    assert mask[IMG_H // 2, IMG_W // 2]  # centre kept
    assert not mask[0, 0]  # corner (table) dropped
    assert mask.mean() < 0.9  # not the whole frame


# ------------------------------ fallback chain -------------------------------
def test_estimator_uses_wired_segmenter():
    seg = FoodSegmenter(_StubSegModule(), _build_seg_transform(), torch.device("cpu"))
    est = PortionEstimator(segmenter=seg)
    r = est.estimate(_img(), "pizza")  # no mask -> must segment internally
    assert r.method == "shape_only"
    assert r.point_g > 0


def test_broken_segmenter_falls_through_to_classical_or_box():
    class _Boom:
        def segment(self, image):
            raise RuntimeError("seg model died")

    est = PortionEstimator(segmenter=_Boom())
    # Must not raise — degrades to classical/central and still returns a range.
    r = est.estimate(_img(), "pizza")
    assert r.min_g <= r.point_g <= r.max_g


def test_central_region_is_last_resort():
    mask = PortionEstimator._central_region(_img())
    assert mask.shape == (IMG_H, IMG_W)
    assert mask[IMG_H // 2, IMG_W // 2]
    assert not mask[0, 0]
