"""digiflowio.py
Enhanced DigiFlow *.dfi* reader.

Key upgrades in this revision
─────────────────────────────
1. **Robust plane‑info parser** – understands both classic *numeric IDs* and
   the newer *4‑byte ASCII labels* DigiFlow writes (e.g. ``b'u   '``).
2. **Cleaner orientation handling** – the default still matches MATLAB, but
   you can now ask for a clockwise layout in one call.
3. **Helper utilities** – `DFIImage.display()` plots an arbitrary subset of
   planes with colour‑bars using Matplotlib.

Only NumPy is compulsory; SciPy and ImageIO are optional add‑ons.
"""

from __future__ import annotations

import enum
import io
import struct
import warnings
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import scipy.ndimage as ndi  # type: ignore

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover – optional dependency
    _HAS_SCIPY = False

# ──────────────────────────────────────────────────────────────────────────────
# DigiFlow tag codes (subset)
# ──────────────────────────────────────────────────────────────────────────────
class Tag(enum.IntEnum):
    IMG32 = 0x01004        #  40964 – 32‑bit single‑plane image
    IMG32_Z = 0x12004      #  73732 – zlib‑compressed 32‑bit single‑plane
    IMG32_MP = 0x11004     #  69636 – 32‑bit multi‑plane image

    RANGE32 = 0x01014      #   4116 – display range (float32 x2)

    RESCALE = 0x01100      #   4352 – rescale whole frame
    RESCALE_RECT = 0x01101 #   4353 – rescale + rectangle

    PLANE_INFO = 0x04108   #  16648 – semantics per plane


_dtype_to_numpy = {
    Tag.IMG32:    (np.dtype("<f4"), False, False),
    Tag.IMG32_Z:  (np.dtype("<f4"), True,  False),
    Tag.IMG32_MP: (np.dtype("<f4"), False, True ),
}

_INTERP = {0: 0, 1: 1, 2: 3, 3: 3, 4: 3, 5: 5}  # DigiFlow → SciPy orders

_U32 = struct.Struct("<I")
_I32x3 = struct.Struct("<III")
_F32x2 = struct.Struct("<ff")

# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RescaleRequest:
    nx: int
    ny: int
    method: int
    rectangle: Optional[Tuple[int, int, int, int]] = None

    def __post_init__(self) -> None:
        if self.method not in _INTERP:
            raise ValueError(f"Unknown rescale code {self.method}")


@dataclass
class RangeInfo:
    black: float
    white: float


@dataclass
class PlaneInfo:
    labels: List[str]  # always human‑readable strings

    def as_dict(self) -> dict[int, str]:
        """Return {plane‑index: name}."""

        # Provide fall‑back names for classic numeric IDs
        legacy = {
            "1": "grey", "101": "x", "102": "y", "103": "z",
            "201": "u", "202": "v", "203": "w", "301": "scalar",
        }
        out = {}
        for i, lab in enumerate(self.labels):
            key = legacy.get(lab, lab.strip())  # strip pads if ascii
            out[i] = key
        return out


@dataclass
class DFIImage:
    data: np.ndarray
    rescale: Optional[RescaleRequest] = None
    intensity_range: Optional[RangeInfo] = None
    plane_info: Optional[PlaneInfo] = None

    # ─────────────── helpers ───────────────
    def rescaled(self) -> "DFIImage":
        if not self.rescale:
            return self
        if not _HAS_SCIPY:
            raise RuntimeError("SciPy is required for rescaling.")

        order = _INTERP[self.rescale.method]
        ny_new, nx_new = self.rescale.ny, self.rescale.nx
        arr = self.data
        if arr.ndim == 2:
            factors = (ny_new / arr.shape[0], nx_new / arr.shape[1])
            arr = ndi.zoom(arr, factors, order=order)
        else:  # (nz, ny, nx)
            factors = (1, ny_new / arr.shape[1], nx_new / arr.shape[2])
            arr = ndi.zoom(arr, factors, order=order)
        return replace(self, data=arr, rescale=None)

    def rotated(self, clockwise: bool = True) -> "DFIImage":
        """Return a copy rotated 90° CW (default) or CCW."""
        k = -1 if clockwise else 1
        axes = (1, 2) if self.data.ndim == 3 else (0, 1)
        arr = np.rot90(self.data, k=k, axes=axes)
        return replace(self, data=arr)

    # quick plotting utility
    def display(self, planes: Optional[Tuple[int, ...]] = None, *, vmin=None, vmax=None) -> None:  # noqa: D401,E501
        """Show selected planes with colour‑bars using Matplotlib."""

        import matplotlib.pyplot as plt

        if planes is None:
            planes = tuple(range(self.data.shape[0] if self.data.ndim == 3 else 1))
        if not isinstance(planes, (list, tuple)):
            planes = (planes,)

        n = len(planes)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        labels = self.plane_info.as_dict() if self.plane_info else {}
        for ax, idx in zip(axes, planes):
            plane = self.data[idx] if self.data.ndim == 3 else self.data
            im = ax.imshow(plane, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
            ax.set_title(labels.get(idx, f"plane {idx}"))
            ax.axis("off")
        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Core reader
# ──────────────────────────────────────────────────────────────────────────────

def _read_exact(fmt: struct.Struct, fh: io.BufferedReader):
    data = fh.read(fmt.size)
    if len(data) != fmt.size:
        raise EOFError("Unexpected EOF")
    return fmt.unpack(data)


def read(
    file: Union[str, Path],
    *,
    apply_rescale: bool = False,
    fix_orientation: str = "none",  # 'matlab', 'cw', 'none'
    _debug: bool = False,
) -> DFIImage:
    """Parse *file* and return a :class:`DFIImage`.

    *fix_orientation*
        'matlab' (default) – rotate counter‑clockwise to match MATLAB.
        'cw'     – clockwise; convenient when overlaying quivers.
        'none'   – leave pixels exactly as stored in DigiFlow.
    """

    path = Path(file)
    with path.open("rb") as fh:
        magic = fh.read(32).rstrip(b"\0").decode("ascii", "replace")
        (version,) = _read_exact(_U32, fh)
        if magic != "Tagged floating point image file" or version != 0:
            raise ValueError("Not a valid DigiFlow *.dfi* file")

        img = None
        rescale_req = None
        range_info = None
        plane_info = None

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
                    print(f"Unknown tag #{code_int} – skipped")
                continue

            if code in _dtype_to_numpy:
                dtype, compressed, multi_plane = _dtype_to_numpy[code]
                nx, ny, nz = _read_exact(_I32x3, payload)

                if compressed:
                    (sz_comp,) = _read_exact(_U32, payload)
                    raw = zlib.decompress(payload.read(sz_comp))
                else:
                    raw = payload.read()

                expected = nx * ny * nz * dtype.itemsize
                if len(raw) != expected:
                    raise ValueError("Unexpected image payload length")
                arr = np.frombuffer(raw, dtype).reshape((nz, ny, nx) if nz > 1 or multi_plane else (ny, nx))
                img = arr

            elif code == Tag.RANGE32:
                black, white = _read_exact(_F32x2, payload)
                range_info = RangeInfo(float(black), float(white))

            elif code in (Tag.RESCALE, Tag.RESCALE_RECT):
                nx_want, ny_want, method = _read_exact(_I32x3, payload)
                rectangle = None
                if code == Tag.RESCALE_RECT:
                    (use_rect,) = _read_exact(_U32, payload)
                    if use_rect:
                        rectangle = _read_exact(struct.Struct("<IIII"), payload)
                if rescale_req is None:  # first wins
                    rescale_req = RescaleRequest(nx_want, ny_want, method, rectangle)
                else:
                    warnings.warn("Multiple rescale tags – ignoring extras")

            elif code == Tag.PLANE_INFO:
                raw = payload.read()
                if len(raw) % 4:
                    warnings.warn("Plane‑info payload not multiple of 4 bytes – skipped")
                    continue
                labels = []
                for i in range(0, len(raw), 4):
                    chunk = raw[i : i + 4]
                    # heuristic: printable ASCII → treat as label, else numeric
                    if all(32 <= b <= 122 for b in chunk.rstrip(b" \0")):
                        labels.append(chunk.rstrip(b" \0").decode("ascii"))
                    else:
                        labels.append(str(int.from_bytes(chunk, "little")))
                plane_info = PlaneInfo(labels)

            if _debug:
                print(f"Tag {code.name} parsed")

        if img is None:
            raise RuntimeError("No image data found in file")

        # orientation handling
        if fix_orientation.lower() == "matlab":
            axes = (1, 2) if img.ndim == 3 else (0, 1)
            img = np.rot90(img, k=1, axes=axes)  # CCW to match MATLAB
        elif fix_orientation.lower() == "cw":
            axes = (1, 2) if img.ndim == 3 else (0, 1)
            img = np.rot90(img, k=-1, axes=axes)  # clockwise

    dfi_img = DFIImage(img, rescale_req, range_info, plane_info)
    if apply_rescale and rescale_req:
        dfi_img = dfi_img.rescaled()
    return dfi_img


# ──────────────────────────────────────────────────────────────────────────────
# Simple CLI (python digiflowio.py sample.dfi --info)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Inspect DigiFlow *.dfi* files")
    p.add_argument("file")
    p.add_argument("--info", action="store_true", help="Print header summary")
    p.add_argument("--cw", action="store_true", help="Rotate clockwise instead of MATLAB orientation")
    p.add_argument("--apply-rescale", action="store_true")
    p.add_argument("--debug", action="store_true", help="Verbose tag dump")

    ns = p.parse_args()

    img = read(
        ns.file,
        apply_rescale=ns.apply_rescale,
        fix_orientation="cw" if ns.cw else "matlab",
        _debug=ns.debug,
    )

    if ns.info:
        print(f"Image shape     : {img.data.shape}")
        print("Intensity range : ", end="")
        if img.intensity_range:
            print(f"{img.intensity_range.black}–{img.intensity_range.white}")
        else:
            print("None")
        print("Planes          : ", end="")
        if img.plane_info:
            print(img.plane_info.as_dict())
        else:
            print("None")
