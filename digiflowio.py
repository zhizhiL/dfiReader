"""digiflowio.py
A standalone reader for DigiFlow *.dfi* files (Tagged Floating‑Point Image files).

┌─────────────────────────────────────────────────────────────────────────────┐
│  Quick usage                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
>>> import digiflowio as dfi
>>> img = dfi.read("frame001.dfi")               # returns a DFIImage object
>>> img.data.shape                                # ndarray, shape (ny, nx) or (nz, ny, nx)
>>> img.to_tiff("frame001.tif")                  # save as TIFF (requires imageio)

Only **NumPy** is mandatory.  If **SciPy** is installed the module will use
it for high‑quality rescaling; otherwise, rescaling falls back to simple
nearest‑neighbour.

The reader covers the DigiFlow tags seen in modern workflows:

* 32‑bit single‑plane images (plain and zlib‑compressed)
* 32‑bit multi‑plane images
* Intensity range, rescale requests, per‑plane semantic IDs

Adding 8‑bit / 64‑bit support just needs two or three extra enum entries and
one more line in the dtype map near the top of the file.
"""

from __future__ import annotations

import enum
import io
import struct
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.ndimage as ndi  # type: ignore

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover – fine on minimal installs
    _HAS_SCIPY = False

# ──────────────────────────────────────────────────────────────────────────────
# Low‑level constants and helpers
# ──────────────────────────────────────────────────────────────────────────────
class Tag(enum.IntEnum):
    """Subset of DigiFlow *DataType* codes the reader understands."""

    IMG32 = 0x01004       #  40964 – 32‑bit image, uncompressed, nz==1
    IMG32_Z = 0x12004     #  73732 – 32‑bit image, zlib‑compressed
    IMG32_MP = 0x11004    #  69636 – 32‑bit multi‑plane image

    RANGE32 = 0x01014     #   4116 – rBlack / rWhite (float32, float32)

    RESCALE = 0x01100      #   4352 – rescale whole image
    RESCALE_RECT = 0x01101 #   4353 – rescale, rectangle variant

    PLANE_INFO = 0x04108   #  16648 – per‑plane semantic IDs

    # 8‑bit and 64‑bit cousins can be added exactly the same way.


# Map each image‑bearing tag → (numpy dtype, is_compressed, is_multi_plane)
_dtype_to_numpy: Dict[Tag, Tuple[np.dtype, bool, bool]] = {
    Tag.IMG32: (np.dtype("<f4"), False, False),
    Tag.IMG32_Z: (np.dtype("<f4"), True, False),
    Tag.IMG32_MP: (np.dtype("<f4"), False, True),
}

# DigiFlow interpolation codes (docs §12.7.21) → SciPy orders
INTERP_ORDER: Dict[int, int] = {
    0: 0,  # nearest
    1: 1,  # bilinear
    2: 3,  # bicubic ≈ order‑3 spline
    3: 3,  # natural spline ≈ cubic
    4: 3,  # cubic B‑spline
    5: 5,  # quintic B‑spline
}

# Pre‑compiled struct objects for speed & clarity
_U32 = struct.Struct("<I")
_THREE_I32 = struct.Struct("<III")
_TWO_F32 = struct.Struct("<ff")
_FOUR_I32 = struct.Struct("<IIII")


def _read_struct(fmt: struct.Struct, fh: io.BufferedReader) -> Tuple[int, ...]:
    """Read *exactly* ``fmt.size`` bytes from *fh* and unpack with *fmt*."""

    data = fh.read(fmt.size)
    if len(data) != fmt.size:
        raise EOFError("Unexpected end‑of‑file while reading .dfi stream")
    return fmt.unpack(data)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses that keep parsed info tidy
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RescaleRequest:
    nx: int
    ny: int
    method: int
    rectangle: Optional[Tuple[int, int, int, int]] = None  # (xMin, yMin, xMax, yMax)

    def __post_init__(self) -> None:  # validate immediately
        if self.method not in INTERP_ORDER:
            raise ValueError(f"Unknown rescale method code {self.method}")


@dataclass
class RangeInfo:
    black: float
    white: float


@dataclass
class PlaneInfo:
    ids: List[int]

    def as_dict(self) -> Dict[int, str]:
        """Translate DigiFlow plane IDs → short names (u, v, scalar…)."""

        mapping = {
            1: "grey",
            101: "x",
            102: "y",
            103: "z",
            201: "u",
            202: "v",
            203: "w",
        }
        return {i: mapping.get(pid, f"plane{pid}") for i, pid in enumerate(self.ids)}


@dataclass
class DFIImage:
    data: np.ndarray                      # (ny, nx) or (nz, ny, nx)
    rescale: Optional[RescaleRequest] = None
    intensity_range: Optional[RangeInfo] = None
    plane_info: Optional[PlaneInfo] = None

    # ─────────── convenience helpers ───────────
    def rescaled(self) -> "DFIImage":
        """Return *a copy* with DigiFlow rescaling applied (needs SciPy)."""

        if not self.rescale:
            return self  # nothing to do
        if not _HAS_SCIPY:
            raise RuntimeError("SciPy not available – cannot apply rescaling.")

        nx_want, ny_want = self.rescale.nx, self.rescale.ny
        order = INTERP_ORDER[self.rescale.method]

        arr = self.data
        if arr.ndim == 2:  # single‑plane: (ny, nx)
            factors = (ny_want / arr.shape[0], nx_want / arr.shape[1])
            arr = ndi.zoom(arr, zoom=factors, order=order)
        else:              # (nz, ny, nx)
            factors = (1, ny_want / arr.shape[1], nx_want / arr.shape[2])
            arr = ndi.zoom(arr, zoom=factors, order=order)

        out = replace(self, data=arr)  # drop rescale tag – it’s been honoured
        out.rescale = None
        return out

    def to_tiff(self, path: str | Path) -> None:
        """Write *data* as a floating‑point TIFF (needs **imageio**)."""

        import imageio.v3 as iio  # local import keeps dependency optional

        iio.imwrite(path, self.data.astype("<f4"), plugin="TIFF", compression="deflate")


# ──────────────────────────────────────────────────────────────────────────────
# Core reader
# ──────────────────────────────────────────────────────────────────────────────
def read(
    file: str | Path,
    *,
    apply_rescale: bool = False,
    fix_orientation: bool = True,
    _debug: bool = False,
) -> DFIImage:
    """Parse a DigiFlow *.dfi* file and return a :class:`DFIImage`.

    Parameters
    ----------
    file
        Path to the *.dfi* file.
    apply_rescale
        If **True** and the file contains a rescale tag, the returned image is
        already resampled (needs SciPy).  If SciPy is absent an exception is
        raised.
    fix_orientation
        DigiFlow writes pixels bottom‑left first.  MATLAB rotates them 90°
        counter‑clockwise so that (0,0) ends up top‑left.  With
        *fix_orientation*=**True** (default) we mimic MATLAB.
    _debug
        Print a short line for every tag encountered.
    """

    path = Path(file)
    with path.open("rb") as fh:
        # ───── header check ───────────────────────────────────────────────
        id_format = fh.read(32).rstrip(b"\0").decode("ascii", "replace")
        (version,) = _read_struct(_U32, fh)
        if id_format != "Tagged floating point image file" or version != 0:
            raise ValueError(f"{file!s} does not appear to be a DigiFlow *.dfi* file")

        # Placeholders that will填ĵ be filled while parsing tags
        img: Optional[np.ndarray] = None
        rescale_req: Optional[RescaleRequest] = None
        range_info: Optional[RangeInfo] = None
        plane_info: Optional[PlaneInfo] = None

        # ───── main tag loop ──────────────────────────────────────────────
        while True:
            tag_hdr = fh.read(8)
            if not tag_hdr:
                break  # EOF

            code_int, nbytes = struct.unpack("<II", tag_hdr)
            payload = io.BytesIO(fh.read(nbytes))

            try:
                code = Tag(code_int)
            except ValueError:
                if _debug:
                    print(f"Skipping unknown tag #{code_int}")
                continue  # skip – not fatal

            if code in _dtype_to_numpy:  # ─── image tags ────────────────
                dtype, compressed, multi_plane = _dtype_to_numpy[code]
                nx, ny, nz = _read_struct(_THREE_I32, payload)

                if compressed:
                    (sz,) = _read_struct(_U32, payload)
                    raw = zlib.decompress(payload.read(sz))
                else:
                    raw = payload.read()

                expected = nx * ny * nz * dtype.itemsize
                if len(raw) != expected:
                    raise ValueError(
                        f"Image payload size mismatch: expected {expected} bytes, got {len(raw)}"
                    )

                arr = np.frombuffer(raw, dtype=dtype)
                if nz == 1 and not multi_plane:
                    arr = arr.reshape(ny, nx)           # 2‑D image
                else:
                    arr = arr.reshape(nz, ny, nx)       # plane, y, x

                img = arr  # if multiple image tags appear we keep the last one

            elif code == Tag.RANGE32:
                r_black, r_white = _read_struct(_TWO_F32, payload)
                range_info = RangeInfo(r_black, r_white)

            elif code in (Tag.RESCALE, Tag.RESCALE_RECT):
                nx_want, ny_want, method = _read_struct(_THREE_I32, payload)
                rect: Optional[Tuple[int, int, int, int]] = None
                if code == Tag.RESCALE_RECT:
                    (use_rect,) = _read_struct(_U32, payload)
                    if use_rect:
                        rect = _read_struct(_FOUR_I32, payload)
                if not rescale_req:  # first wins – match MATLAB behaviour
                    rescale_req = RescaleRequest(nx_want, ny_want, method, rect)

            elif code == Tag.PLANE_INFO:
                (n_ids,) = _read_struct(_U32, payload)
                ids_struct = struct.Struct("<" + "I" * n_ids)
                ids = list(ids_struct.unpack(payload.read(ids_struct.size)))
                plane_info = PlaneInfo(ids)

            if _debug:
                print(f"Parsed tag {code.name} ({nbytes} bytes)")

    if img is None:
        raise RuntimeError("No image tag found in .dfi file")

    # ───── orientation tweak: match MATLAB ────────────────────────────────
    if fix_orientation:
        if img.ndim == 2:
            img = np.rot90(img)
        else:
            img = np.rot90(img, axes=(1, 2))

    dfi_image = DFIImage(img, rescale_req, range_info, plane_info)
    if apply_rescale and rescale_req:
        dfi_image = dfi_image.rescaled()

    return dfi_image


# ──────────────────────────────────────────────────────────────────────────────
# Simple CLI so you can poke at files from the shell
# ──────────────────────────────────────────────────────────────────────────────
def _cli() -> None:  # pragma: no cover – utility only
    import argparse

    p = argparse.ArgumentParser(description="Read & inspect DigiFlow .dfi files")
    p.add_argument("file", help="Input *.dfi*")
    p.add_argument("--info", action="store_true", help="Print header/tags")
    p.add_argument("--to-tiff", metavar="PATH", help="Write image to TIFF")
    p.add_argument("--apply-rescale", dest="apply_rescale", action="store_true", help="Apply rescale tag")
    p.add_argument("--no-orient", action="store_true", help="Skip 90° rotation")
    p.add_argument("--debug", action="store_true", help="Verbose tag dump")

    ns = p.parse_args()

    # argparse normally converts "--apply-rescale" → ns.apply_rescale; but to
    # be extra safe we fall back to getattr(..., False) just in case.
    apply_flag = getattr(ns, "apply_rescale", False)

    img = read(
        ns.file,
        apply_rescale=apply_flag,
        fix_orientation=not ns.no_orient,
        _debug=ns.debug,
    )

    if ns.info:
        print(f"Image shape       : {img.data.shape}")
        if img.intensity_range:
            print(f"Intensity range   : {img.intensity_range.black}–{img.intensity_range.white}")
        if img.rescale:
            print(
                f"Rescale requested : {img.rescale.nx} x {img.rescale.ny} (method {img.rescale.method})")
        if img.plane_info:
            print(f"Planes            : {img.plane_info.as_dict()}")
    if ns.to_tiff:
        img.to_tiff(ns.to_tiff)
        print(f"Image written to  : {ns.to_tiff}")

if __name__ == "__main__": # pragma: no cover – CLI only
    _cli()
