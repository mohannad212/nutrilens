"""
app/ml/preprocess.py
=======================

WHY THIS FILE EXISTS
---------------------
The trained EfficientNet-B4 model only produces correct predictions if the
image fed to it is prepared in *exactly* the same way the images were prepared
during evaluation in training (`04-ml-model.md §2`). This file owns that
preparation — the "eval transform" (resize -> centre-crop -> tensor -> ImageNet
normalise) — plus the FR03 upload-validation rules (size, format, resolution)
that a photo must pass before it is ever handed to the model.

WHAT PROBLEM IT SOLVES
------------------------
Two problems, both about drift and trust:

1. **Train/serve skew.** If serving resized to 224 while training used 380, or
   used different normalisation stats, accuracy silently collapses and it is
   maddening to debug. Pinning the transform here, next to a comment pointing at
   `04 §2`, makes the contract explicit and impossible to fork by accident.

2. **Bad uploads reaching the model.** A 40 MB file, a PDF renamed `.jpg`, or a
   120x90 thumbnail should be rejected *up front* with a precise 4xx, not crash
   the model or produce a garbage prediction on a garbage image. This file turns
   each failure mode into the specific error the `/recognise` contract promises
   (`03 §4`, §9). On top of the hard format/size/resolution rules, a *quality
   gate* (`_check_quality`) rejects photos that are too dark/bright or too blurry
   to recognise, asking the user to retake. Crucially this gate only *rejects* —
   it never enhances or relights the pixels, because feeding the model images
   preprocessed differently from its training-time distribution would cause
   train/serve skew and could lower accuracy. Robustness to *moderate* lighting
   variation is instead trained into the model via augmentation (`04 §2`); the
   gate just stops the hopeless cases.

HOW IT WORKS INTERNALLY
--------------------------
- `build_eval_transform()` returns a torchvision `Compose` that mirrors
  `eval_tf` in `04 §2` byte-for-byte (Resize 400 -> CenterCrop 380 -> ToTensor ->
  Normalize with ImageNet mean/std). The classifier builds this once at startup.
- `validate_and_load_image(raw, content_type)` takes the raw request bytes and
  the declared content type, and:
    * rejects anything over `MAX_UPLOAD_MB` -> `ImageTooLargeError` (413),
    * decodes with Pillow and confirms the *actual* format is JPEG or PNG
      (never trusting the client's content-type header alone) ->
      `ImageBadFormatError` (400),
    * confirms width/height meet the FR03 minimum -> `ImageTooSmallError` (400),
    * returns a decoded, RGB, in-memory `PIL.Image` — never touching disk.
  It intentionally does NOT resize/normalise; that is the transform's job and
  happens later, inside the model's inference path.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Validating the *decoded* image's real format and dimensions (chosen) rather
  than trusting the multipart `Content-Type` header: the header is client-
  supplied and trivially spoofed, so "is this really a JPEG of adequate size?"
  can only be answered by actually decoding it.
- Keeping the transform as a module-level factory rather than inlining it in
  `classifier.py`: it lets tests assert the exact preprocessing, and keeps the
  single source of truth for "how a pixel becomes a tensor" in one file.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why check the size in bytes here as well as relying on the web server?"
A: Defence in depth. The route also enforces limits, but validating the bytes
   we actually received means the rule holds no matter how the endpoint is
   called, and lets us return the exact `IMAGE_TOO_LARGE` envelope with the
   received size in the details.

Q: "Why 380x380 and those specific normalisation numbers?"
A: 380 is EfficientNet-B4's native input resolution; the mean/std are the
   ImageNet statistics the pretrained backbone was trained with. Both must match
   training (`04 §2/§3`) or the learned features are fed the wrong distribution.

Q: "Where is the image saved?"
A: Nowhere. It lives only as bytes -> PIL object -> tensor in memory and is
   discarded when the request ends. The M3 DoD explicitly requires that no image
   is written to disk.

COMMON STUDENT MISTAKES
--------------------------
- Copy-pasting the *training* augmentation transform (RandomResizedCrop, flips,
  RandAugment) into serving — augmentation belongs to training only; at serve
  time you want the deterministic eval transform.
- Trusting `UploadFile.content_type` for validation instead of decoding.
- Forgetting `.convert("RGB")`, so a PNG with an alpha channel or a greyscale
  JPEG yields the wrong number of channels and the model errors or mispredicts.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.core.errors import (
    ImageBadFormatError,
    ImagePoorLightingError,
    ImageTooBlurryError,
    ImageTooLargeError,
    ImageTooSmallError,
)

# Must match `04-ml-model.md §2` (eval_tf) and the model's native resolution.
IMAGE_SIZE = 380
RESIZE_SHORT_SIDE = 400
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Pillow format names we accept. FR03 allows JPEG or PNG only.
_ALLOWED_FORMATS = {"JPEG", "PNG"}


def build_eval_transform() -> T.Compose:
    """The deterministic serving transform — identical to `eval_tf` in `04 §2`.

    Resize the short side to 400, centre-crop to 380x380, convert to a float
    tensor in [0, 1], then normalise with ImageNet statistics. Built once and
    reused for every request (it holds no per-image state).
    """
    return T.Compose(
        [
            T.Resize(RESIZE_SHORT_SIDE),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _check_quality(image: Image.Image) -> None:
    """Reject clearly un-recognisable photos (too dark/bright, or too blurry).

    A *gate*, not a fix: it never modifies the image, it just raises so the app
    can prompt a retake — which avoids the train/serve skew that pixel
    enhancement would cause. All thresholds live in `settings` and the whole
    gate can be turned off with `quality_gate_enabled`.

    - **Lighting:** mean luminance (ITU-R 601 luma via Pillow "L") outside
      `[brightness_min, brightness_max]` -> `ImagePoorLightingError`.
    - **Blur:** variance of the Laplacian (the standard focus measure — a sharp
      image has strong second derivatives, a blurry one doesn't) below
      `blur_laplacian_min` -> `ImageTooBlurryError`.
    """
    if not settings.quality_gate_enabled:
        return

    gray = np.asarray(image.convert("L"))

    mean_luma = float(gray.mean())
    if mean_luma < settings.brightness_min or mean_luma > settings.brightness_max:
        raise ImagePoorLightingError(
            details={
                "mean_luminance": round(mean_luma, 1),
                "min": settings.brightness_min,
                "max": settings.brightness_max,
            },
        )

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < settings.blur_laplacian_min:
        raise ImageTooBlurryError(
            details={
                "sharpness": round(laplacian_var, 1),
                "min": settings.blur_laplacian_min,
            },
        )


def validate_and_load_image(raw: bytes, content_type: str | None = None) -> Image.Image:
    """Validate raw upload bytes against FR03 and return a decoded RGB image.

    Raises the specific `AppError` subclass for each failure mode so the
    `/recognise` route returns the exact 4xx envelope from `03 §9`:
      * too big      -> `ImageTooLargeError`  (413, IMAGE_TOO_LARGE)
      * undecodable / wrong type -> `ImageBadFormatError` (400, IMAGE_BAD_FORMAT)
      * below min resolution     -> `ImageTooSmallError`  (400, IMAGE_TOO_SMALL)

    The returned image lives only in memory; nothing is written to disk.
    """
    # 1) Size (bytes) — cheapest check, do it before decoding megapixels.
    max_bytes = settings.max_upload_mb * 1024 * 1024
    received_mb = round(len(raw) / (1024 * 1024), 1)
    if len(raw) > max_bytes:
        raise ImageTooLargeError(
            f"Image exceeds the {settings.max_upload_mb} MB limit.",
            details={"max_mb": settings.max_upload_mb, "received_mb": received_mb},
        )
    if not raw:
        raise ImageBadFormatError("The uploaded image is empty.")

    # 2) Format — decode with Pillow and trust the *decoded* format, not the
    #    client-declared content type (which is spoofable).
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force full decode now so truncated files fail here, not later
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageBadFormatError() from exc

    if image.format not in _ALLOWED_FORMATS:
        raise ImageBadFormatError(
            "Image must be a JPEG or PNG.",
            details={"received_format": image.format, "allowed": sorted(_ALLOWED_FORMATS)},
        )

    # 3) Resolution — FR03 minimum. Check the real decoded dimensions.
    width, height = image.size
    if width < settings.min_image_width or height < settings.min_image_height:
        raise ImageTooSmallError(
            f"Image must be at least {settings.min_image_width}x{settings.min_image_height}.",
            details={
                "min_width": settings.min_image_width,
                "min_height": settings.min_image_height,
                "received_width": width,
                "received_height": height,
            },
        )

    # 4) Quality gate — reject too-dark/bright or too-blurry photos with a
    #    retake prompt (FR03 extension). Never alters the image.
    _check_quality(image)

    # Normalise channels so downstream tensor shape is always (3, H, W).
    return image.convert("RGB")
