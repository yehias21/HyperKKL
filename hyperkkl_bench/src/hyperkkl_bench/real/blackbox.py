"""Minimal reader of Betaflight blackbox logs (.BFL, data version 2), pure Python.

Written from the published format (betaflight/blackbox-tools, parser.c, decoders.c); it reads bytes as DATA only.
It decodes the I / P main frames of the FIRST log in a file into integers; S, G, H and E frames are skipped. A frame
that does not parse, or that is not followed by a frame marker, is dropped and the reader resynchronises on the
next I frame (as blackbox_decode does). Validated against the providers' processed CSVs, which were produced with the
official blackbox_decode: see docs/cards/neurobem_datacheck.md.

    log = read_bfl(path, fields=("time", "gyroADC[0]", ...))   ->  {field: int64 array}, log["_header"]: dict
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_MARKERS = {ord(c) for c in "IPSEGH"}


class _Eof(Exception):
    pass


class _Reader:
    __slots__ = ("b", "p", "n")

    def __init__(self, b, p):
        self.b, self.p, self.n = b, p, len(b)

    def byte(self):
        if self.p >= self.n:
            raise _Eof
        v = self.b[self.p]
        self.p += 1
        return v

    def uvb(self):
        r, s = 0, 0
        for _ in range(5):
            c = self.byte()
            r |= (c & 0x7F) << s
            s += 7
            if c < 128:
                return r & 0xFFFFFFFF
        raise ValueError("bad varint")

    def svb(self):
        u = self.uvb()
        return (u >> 1) ^ -(u & 1)


def _sx(v, bits):
    m = 1 << (bits - 1)
    return (v & (m - 1)) - (v & m)


def _tag2_3s32(r):
    lead = r.byte()
    sel = lead >> 6
    if sel == 0:
        return [_sx(lead >> 4, 2), _sx(lead >> 2, 2), _sx(lead, 2)]
    if sel == 1:
        a = _sx(lead, 4)
        l2 = r.byte()
        return [a, _sx(l2 >> 4, 4), _sx(l2, 4)]
    if sel == 2:
        a = _sx(lead, 6)
        return [a, _sx(r.byte(), 6), _sx(r.byte(), 6)]
    out, s = [], lead & 0x3F
    for _ in range(3):
        k = s & 3
        v = 0
        for j in range(k + 1):
            v |= r.byte() << (8 * j)
        out.append(_sx(v, 8 * (k + 1)))
        s >>= 2
    return out


def _tag8_4s16(r):
    sel = r.byte()
    out, nib, buf = [], 0, 0
    for _ in range(4):
        k = sel & 3
        if k == 0:
            out.append(0)
        elif k == 1:
            if nib == 0:
                buf = r.byte()
                out.append(_sx(buf >> 4, 4)); nib = 1
            else:
                out.append(_sx(buf, 4)); nib = 0
        elif k == 2:
            if nib == 0:
                out.append(_sx(r.byte(), 8))
            else:
                c = (buf << 4) & 0xFF
                buf = r.byte()
                out.append(_sx(c | (buf >> 4), 8))
        else:
            if nib == 0:
                c1 = r.byte(); c2 = r.byte()
                out.append(_sx((c1 << 8) | c2, 16))
            else:
                c1 = r.byte(); c2 = r.byte()
                out.append(_sx((buf << 12) | (c1 << 4) | (c2 >> 4), 16))
                buf = c2
        sel >>= 2
    return out


def _tag8_8svb(r, n):
    if n == 1:
        return [r.svb()]
    h = r.byte()
    out = []
    for _ in range(n):
        out.append(r.svb() if h & 1 else 0)
        h >>= 1
    return out


def _read_values(r, enc, signed=None):
    n, i, out = len(enc), 0, []
    while i < n:
        e = enc[i]
        if e == 0:
            out.append(r.svb()); i += 1
        elif e == 1:
            out.append(r.uvb()); i += 1
        elif e == 3:
            out.append(-_sx(r.uvb(), 14)); i += 1
        elif e == 9:
            out.append(0); i += 1
        elif e == 7:
            out += _tag2_3s32(r); i += 3
        elif e == 8:
            out += _tag8_4s16(r); i += 4
        elif e == 6:
            k = 1
            while i + k < n and enc[i + k] == 6 and k < 8:
                k += 1
            out += _tag8_8svb(r, k); i += k
        else:
            raise ValueError(f"encoding {e} not implemented")
    return out


def _header(b):
    h, p = {}, 0
    while b[p:p + 2] == b"H ":
        e = b.index(b"\n", p)
        k, _, v = b[p + 2:e].decode("latin-1").partition(":")
        h[k] = v
        p = e + 1
    return h, p


def _ints(s):
    return [int(t) for t in s.split(",")]


def read_bfl(path, fields=None) -> dict:
    b = Path(path).read_bytes()
    h, p = _header(b)
    assert h.get("Data version") == "2", h.get("Data version")
    names = h["Field I name"].split(",")
    ienc, ipred = _ints(h["Field I encoding"]), _ints(h["Field I predictor"])
    penc, ppred = _ints(h["Field P encoding"]), _ints(h["Field P predictor"])
    sn = len(h["Field S name"].split(",")) if "Field S name" in h else 0
    senc = _ints(h["Field S encoding"]) if sn else []
    gn = len(h["Field G name"].split(",")) if "Field G name" in h else 0
    genc = _ints(h["Field G encoding"]) if gn else []
    henc = _ints(h["Field H encoding"]) if "Field H encoding" in h else []
    nf = len(names)
    im0 = names.index("motor[0]") if "motor[0]" in names else None
    minthr = int(h.get("minthrottle", 0))
    vref = int(h.get("vbatref", 0))
    minmot = int(h["motorOutput"].split(",")[0]) if "motorOutput" in h else 0
    r = _Reader(b, p)
    rows, prev1, prev2 = [], None, None
    stats = {"I": 0, "P": 0, "S": 0, "E": 0, "dropped": 0}

    def i_frame():
        v = _read_values(r, ienc)
        for j in range(nf):
            q = ipred[j]
            if q == 4:
                v[j] += minthr
            elif q == 5:
                v[j] += v[im0]
            elif q == 9:
                v[j] += vref
            elif q == 11:
                v[j] += minmot
            elif q != 0:
                raise ValueError(f"I predictor {q}")
        return v

    def p_frame():
        v = _read_values(r, penc)
        for j in range(nf):
            q = ppred[j]
            if q == 1:
                v[j] += prev1[j]
            elif q == 2:
                v[j] += 2 * prev1[j] - prev2[j]
            elif q == 3:
                v[j] += int((prev1[j] + prev2[j]) / 2)
            elif q == 6:
                v[j] = prev1[j] + 1            # the iteration counter: only ever used as an index
            elif q == 5:
                v[j] += v[im0]
            elif q != 0:
                raise ValueError(f"P predictor {q}")
        return v

    def event():
        t = r.byte()
        if t == 255:
            raise _Eof
        if t == 0:
            r.uvb()
        elif t == 13:
            f = r.byte(); r.byte() if f < 128 else [r.byte() for _ in range(4)]
        elif t == 14:
            r.uvb(); r.uvb()
        elif t == 15:
            r.uvb()
        elif t == 30:
            r.uvb(); r.uvb()
        else:
            raise ValueError("unknown event")

    def resync():
        q = b.find(b"I", r.p)
        if q < 0:
            raise _Eof
        r.p = q

    try:
        while True:
            start = r.p
            m = chr(r.byte())
            try:
                if m == "I":
                    v = i_frame(); kind = "I"
                elif m == "P" and prev1 is not None:
                    v = p_frame(); kind = "P"
                elif m == "S":
                    _read_values(r, senc); stats["S"] += 1; continue
                elif m == "G":
                    _read_values(r, genc); continue
                elif m == "H":
                    _read_values(r, henc); continue
                elif m == "E":
                    event(); stats["E"] += 1; continue
                else:
                    stats["dropped"] += 1; prev1 = None; resync(); continue
                if r.p < r.n and r.b[r.p] not in _MARKERS:        # torn frame: not followed by a marker
                    raise ValueError("no marker")
                if kind == "I" and rows and (v[1] < rows[-1][1] or v[1] - rows[-1][1] > 10_000_000):
                    raise ValueError("time jump")
            except ValueError:
                stats["dropped"] += 1
                r.p = start + 1
                prev1 = None
                resync()
                continue
            rows.append(v)
            stats[kind] += 1
            if kind == "I":
                prev1 = prev2 = v
            else:
                prev2, prev1 = prev1, v
    except _Eof:
        pass
    a = np.asarray(rows, np.int64)
    out = {n: a[:, j] for j, n in enumerate(names) if fields is None or n in fields}
    out["_header"] = h
    out["_stats"] = stats
    return out
