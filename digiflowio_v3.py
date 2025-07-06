# -*- coding: utf-8 -*-
"""digiflowio.py  –  DigiFlow *.dfi* reader  (2025‑07‑05)
========================================================
This release fixes two critical bugs:

1. **Unknown‑tag handling** – we were accidentally skipping payload bytes *twice*,
   which broke file alignment and could end with “No image data found”.
   The reader now simply *reads* the unknown payload (already done) and moves on.

2. **Invisible NULL in `rstrip`** – the previous code contained a literal NUL
   character (`" \0"`) in the `rstrip` call, which was not intended.  It’s replaced by an explicit escape sequence
   (`b" \x00"`) so the file stays UTF‑8 clean.

Plane‑info parsing remains the safe heuristic (4‑byte chunks ⇒ ASCII label or
numeric ID), so you should finally see friendly plane names like “u, v, scalar”.

Dependencies: NumPy (core) • SciPy (rescale) • ImageIO (TIFF export)
"""
from __future__ import annotations

import enum, io, struct, warnings, zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:                                    # SciPy is optional – only for rescaling
    import scipy.ndimage as ndi          # type: ignore
    _HAS_SCIPY = True
except ImportError:                      # pragma: no cover
    _HAS_SCIPY = False

# ───────────────────────────── DigiFlow tag codes ────────────────────────────
class Tag(enum.IntEnum):
    IMG32        = 0x01004   # 32‑bit single‑plane
    IMG32_Z      = 0x12004   # zlib‑compressed 32‑bit single‑plane
    IMG32_MP     = 0x11004   # 32‑bit multi‑plane
    RANGE32      = 0x01014   # display black / white
    RESCALE      = 0x01100
    RESCALE_RECT = 0x01101
    PLANE_INFO   = 0x04108   # per‑plane semantics

_img_tag = {
    Tag.IMG32   : (np.dtype('<f4'), False),
    Tag.IMG32_Z : (np.dtype('<f4'), True ),
    Tag.IMG32_MP: (np.dtype('<f4'), False),  # multi‑plane flag handled later
}
_interp = {0:0, 1:1, 2:3, 3:3, 4:3, 5:5}

# struct shortcuts
_U32   = struct.Struct('<I')
_I32x3 = struct.Struct('<III')
_F32x2 = struct.Struct('<ff')

# ───────────────────────────────── data classes ──────────────────────────────
@dataclass
class RescaleRequest:
    nx: int; ny: int; method: int
    rectangle: Optional[Tuple[int,int,int,int]] = None
    def __post_init__(self):
        if self.method not in _interp:
            raise ValueError(f'Unknown rescale method {self.method}')

@dataclass
class RangeInfo:
    black: float; white: float

@dataclass
class PlaneInfo:
    labels: List[str]
    def as_dict(self):
        default = {'1':'grey','101':'x','102':'y','103':'z',
                   '201':'u','202':'v','203':'w','301':'scalar'}
        return {i: default.get(lbl, lbl) for i, lbl in enumerate(self.labels)}

@dataclass
class DFIImage:
    data: np.ndarray
    rescale: Optional[RescaleRequest] = None
    intensity_range: Optional[RangeInfo] = None
    plane_info: Optional[PlaneInfo] = None

    # helpers -------------------------------------------------------
    def rescaled(self):
        if not self.rescale:
            return self
        if not _HAS_SCIPY:
            raise RuntimeError('SciPy needed for rescaling')
        order = _interp[self.rescale.method]
        ny, nx = self.rescale.ny, self.rescale.nx
        arr = self.data
        if arr.ndim==2:
            arr = ndi.zoom(arr, (ny/arr.shape[0], nx/arr.shape[1]), order=order)
        else:
            arr = ndi.zoom(arr, (1, ny/arr.shape[1], nx/arr.shape[2]), order=order)
        return replace(self, data=arr, rescale=None)

    def rotated(self, clockwise: bool = True):
        k = -1 if clockwise else 1
        axes = (1,2) if self.data.ndim==3 else (0,1)
        return replace(self, data=np.rot90(self.data, k=k, axes=axes))

    def display(self, planes=None, *, vmin=None, vmax=None):
        import matplotlib.pyplot as plt
        if self.data.ndim==2:
            planes = (0,)
        elif planes is None:
            planes = range(self.data.shape[0])
        fig, axs = plt.subplots(1, len(planes), figsize=(5*len(planes),4))
        if len(planes)==1:
            axs=[axs]
        names = self.plane_info.as_dict() if self.plane_info else {}
        for ax, idx in zip(axs, planes):
            plane = self.data[idx] if self.data.ndim==3 else self.data

            # need to flip the image vertically as different conventions in digiflow and matplotlib
            # flipping in_place
            plane_flipped = np.flipud(plane)

            # If we're plotting the vorticity plane (idx == 2) and the caller
            # didn’t specify vmin/vmax, use the IntensityRange from the file.
            if vmin is None and vmax is None and self.intensity_range and idx == 2:
                vmin = self.intensity_range.black
                vmax = self.intensity_range.white

            im = ax.imshow(plane_flipped, cmap='seismic', vmin=vmin, vmax=vmax)
            
            plt.colorbar(im, ax=ax); ax.axis('off')
            ax.set_title(names.get(idx, f'plane {idx}'))
        plt.tight_layout(); plt.show()

# ───────────────────────────────── reader ────────────────────────────────────

def _read(fmt: struct.Struct, fh):
    buf = fh.read(fmt.size)
    if len(buf)!=fmt.size:
        raise EOFError('Unexpected EOF')
    return fmt.unpack(buf)

def read(file: Union[str, Path], *, apply_rescale=False, orientation='none', _debug=False):
    path = Path(file)
    with path.open('rb') as fh:
        if fh.read(32).rstrip(b'\0') != b'Tagged floating point image file' or _read(_U32,fh)[0]!=0:
            raise ValueError('Not a DigiFlow .dfi')

        img = rescale = irange = pinfo = None

        while True:
            hdr = fh.read(8)
            if not hdr:
                break
            tag_id, nbytes = struct.unpack('<II', hdr)
            payload_bytes = fh.read(nbytes)
            payload = io.BytesIO(payload_bytes)

            try:
                tag = Tag(tag_id)
            except ValueError:       # unknown tag – payload already consumed
                if _debug:
                    print(f'Unknown tag 0x{tag_id:X} skipped')
                continue

            # ─── image -----------------------------------------------------
            if tag in _img_tag:
                dtype, compressed = _img_tag[tag]
                nx, ny, nz = _read(_I32x3, payload)
                if compressed:
                    comp_len, = _read(_U32, payload)
                    raw = zlib.decompress(payload.read(comp_len))
                else:
                    raw = payload.read()
                arr = np.frombuffer(raw, dtype)
                shape = (nz, ny, nx) if (tag==Tag.IMG32_MP or nz>1) else (ny, nx)
                img = arr.reshape(shape)

            # ─── intensity range ------------------------------------------
            elif tag == Tag.RANGE32:
                b, w = _read(_F32x2, payload); irange = RangeInfo(b, w)

            # ─── rescale request ------------------------------------------
            elif tag in (Tag.RESCALE, Tag.RESCALE_RECT):
                nx, ny, method = _read(_I32x3, payload); rect = None
                if tag == Tag.RESCALE_RECT and _read(_U32, payload)[0]:
                    rect = _read(struct.Struct('<IIII'), payload)
                if not rescale:
                    rescale = RescaleRequest(nx, ny, method, rect)

            # ─── plane‑info ----------------------------------------------
            elif tag == Tag.PLANE_INFO:
                # Follow DigiFlow spec: first int32 = nPlanes, then for each
                # plane: Contains(int32) + Descr[32] + four doubles + Unknown[32]
                (n_planes,) = _read(_U32, payload)
                labels = []
                for _ in range(n_planes):
                    contains, = _read(_U32, payload)
                    descr_bytes = payload.read(32)
                    descr = descr_bytes.split(b'\x00', 1)[0].decode('ascii', 'replace').strip()
                    payload.seek(32, io.SEEK_CUR)  # skip four doubles (32 bytes)
                    payload.seek(32, io.SEEK_CUR)  # skip Unknown 32‑byte string
                    # Prefer descriptor; fall back to numeric Contains code
                    label = descr if descr else str(contains)
                    labels.append(label)
                pinfo = PlaneInfo(labels)



            if _debug:
                print(f'Parsed tag {tag.name}')

        if img is None:
            raise RuntimeError('No image data found')

        # orientation
        if orientation != 'none':
            k = 1 if orientation == 'matlab' else -1
            axes = (1,2) if img.ndim==3 else (0,1)
            img = np.rot90(img, k=k, axes=axes)

        out = DFIImage(img, rescale, irange, pinfo)
        if apply_rescale and rescale:
            out = out.rescaled()
        return out


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
        orientation="cw" if ns.cw else "matlab",
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
