"""Parse a local copy of the public Nmap OS-fingerprint corpus into the schema.

The script reads Nmap's ``nmap-os-db`` (shipped with every Nmap installation
and also published at https://github.com/nmap/nmap/blob/master/nmap-os-db), a
plain-text database whose ``Fingerprint`` blocks contain the probe replies that
Nmap uses for OS detection.  Those replies carry network- and transport-layer
evidence of the same kinds the 36-field schema covers, so they can be projected
onto the schema without any measurement of our own.

Two properties of the projection are worth stating, because they explain why
this script is a *path* to a real corpus rather than a drop-in replacement for
the benchmark:

* ``nmap-os-db`` contains no application-layer evidence, so the nine L7 fields
  (and the six device-specific ones) are returned as ``NaN`` with the mask set
  to ``False``.  This is precisely the block-missingness regime of Section 5.2,
  not a data-quality defect, and the model consumes it unchanged.
* The database annotates each fingerprint with an ``osclass`` whose device-type
  vocabulary (``general purpose``, ``router``, ``firewall``, ``webcam``, ...)
  does not coincide with the eleven classes of Section 3.1.  The mapping below
  is therefore explicit and deliberately conservative: fingerprints whose type
  is ambiguous are dropped instead of being forced into a class, and the number
  of dropped records is reported.

Usage::

    python parse_nmap_osdb.py --db /usr/share/nmap/nmap-os-db --out nmap_os.npz
    python parse_nmap_osdb.py --db nmap-os-db --report        # counts only

The output ``.npz`` has the same keys as ``benchmark.npz`` (``X``, ``y``,
``y_coarse``, ``mask``, ``names``, ``kind``, ``ncat``), so every experiment in
``src/`` runs on it without modification.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from benchmark import CLASSES, FEATURES, SUPER_CLASSES      # noqa: E402

NAMES = [f["name"] for f in FEATURES]
KIND = [f["kind"] for f in FEATURES]
NCAT = [f.get("n", 0) for f in FEATURES]
LAYER = [f["layer"] for f in FEATURES]

# --------------------------------------------------------------------------- #
# osclass device type -> fine-grained class of Section 3.1
# --------------------------------------------------------------------------- #
# Keys are the lower-cased device types that Nmap actually uses in nmap-os-db
# (``Device type:`` line of each osclass).  Values are our class names or None
# for "ambiguous, drop".  The mapping is one-to-many only where the physical
# device is the same thing under two names (a switch and a bridge are both
# forwarding boxes and are represented by our 'router' class, which is the only
# forwarding class in the taxonomy).
DEVICE_MAP = {
    "general purpose": None,          # resolved from the OS family, see below
    "router": "router",
    "broadband router": "router",
    "switch": "router",
    "hub": "router",
    "bridge": "router",
    "wap": "router",
    "firewall": "firewall",
    "webcam": "ip_camera",
    "phone": "smartphone",
    "media device": "smart_hub",
    "game console": "smart_hub",
    "specialized": "plc",
    "print server": "smart_hub",
    "power controller": "plc",
    "load balancer": "smart_hub",
    "voip phone": "smart_hub",
    "voip adapter": "smart_hub",
    "storage-misc": "dns_db_server",
    "terminal server": "dns_db_server",
}

# OS family -> class, used only when the device type is 'general purpose'.
# The three families that dominate general-purpose hosts map onto the three
# server-like classes of the taxonomy; everything else is an end host.
FAMILY_MAP = {
    "windows": "pc",
    "mac os x": "pc",
    "linux": "web_server",
    "freebsd": "web_server",
    "openbsd": "web_server",
    "netbsd": "web_server",
    "solaris": "web_server",
    "vmware": "vm_container",
    "esxi": "vm_container",
    "mikrotik": "router",
    "ios": "router",
    "junos": "router",
    "embedded": "smart_hub",
}

# --------------------------------------------------------------------------- #
# fingerprint fields
# --------------------------------------------------------------------------- #
# Nmap writes probe sections as ``T1(R=Y%DF=Y%T=40%TG=40%W=2000%S=A%F=AS%...``.
_SECTION = re.compile(r"^([A-Z]{1,3}[0-9]?)\((.*)\)\s*$")
_KV = re.compile(r"([A-Z]{1,3}[0-9]?)=([^%]*)")

_HEX = re.compile(r"^0?[xX]?[0-9A-Fa-f]{1,8}$")

# Tokens that are probe *verdicts* rather than numbers.  Nmap prints every
# numeric field of a fingerprint in hexadecimal without a prefix, so a token
# that parses as hex is a number and anything else is a verdict.
_VERDICT = {"Y", "N", "S", "O", "A", "Z", "U", "E", "R", "F", "H", "T", "P", "G"}


def _num(tok):
    """Nmap token -> float.  Hex by convention; NaN for verdicts and gaps."""
    tok = tok.strip()
    if not tok or tok in _VERDICT or len(tok) > 8:
        return np.nan
    if tok.startswith(("0x", "0X")):
        try:
            return float(int(tok, 16))
        except ValueError:
            return np.nan
    if _HEX.match(tok):
        return float(int(tok, 16))
    return np.nan


def _yn(tok):
    """Verdict token -> 1.0 / 0.0 / NaN."""
    t = tok.strip().upper()
    if t == "Y":
        return 1.0
    if t == "N":
        return 0.0
    return np.nan


def _first(sections, sec, keys):
    """Value of the first present key of section ``sec``."""
    kv = sections.get(sec)
    if not kv:
        return np.nan
    for k in keys:
        if k in kv:
            v = _num(kv[k])
            if not np.isnan(v):
                return v
    return np.nan


CLS_INDEX = {name: i for i, (name, _) in enumerate(CLASSES)}


def parse(db_path):
    """Yield one dict per fingerprint block."""
    cur = None
    sections = {}
    classes = []
    with open(db_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("Fingerprint "):
                if cur is not None:
                    cur["sections"] = sections
                    cur["classes"] = classes
                    yield cur
                cur = {"name": line[len("Fingerprint "):].strip()}
                sections, classes = {}, []
            elif cur is None:
                continue
            elif line.startswith("Class "):
                parts = [p.strip() for p in line[len("Class "):].split("|")]
                if len(parts) >= 4:
                    classes.append({"vendor": parts[0], "family": parts[1].lower(),
                                    "generation": parts[2], "device": parts[3].lower()})
            elif line.startswith("CPE "):
                cur.setdefault("cpe", []).append(line[4:].strip())
            else:
                m = _SECTION.match(line.strip())
                if m:
                    sections[m.group(1)] = dict(_KV.findall(m.group(2)))
    if cur is not None:
        cur["sections"] = sections
        cur["classes"] = classes
        yield cur


def _class_of(rec):
    """Map one fingerprint to a fine-grained class name, or None."""
    for c in rec["classes"]:
        dev = c["device"]
        if dev == "general purpose":
            fam = c["family"]
            for key, cls in FAMILY_MAP.items():
                if key in fam:
                    return cls
            continue
        if dev in DEVICE_MAP:
            return DEVICE_MAP[dev]
    # fall back to the OS family alone when the device type was ambiguous
    for c in rec["classes"]:
        for key, cls in FAMILY_MAP.items():
            if key in c["family"]:
                return cls
    return None


def project(rec, opt_vocab, opt_other, flag_vocab, flag_other):
    """One fingerprint -> a dense row of the 36-field schema (NaN = absent).

    Only the fields whose semantics coincide with a probe reply are filled in;
    every other field stays ``NaN`` and is marked unobserved, which the model
    consumes natively.  The correspondence used here is:

    ======================  ==========================================
    field                   source
    ======================  ==========================================
    ``hop_limit``           ``T1.T``      (initial TTL guess)
    ``mtu``                 ``T1.TG``     (MTU guess)
    ``frag_supported``      ``T1.DF``     (don't-fragment echo)
    ``tcp_window``          ``T1.W`` / ``WIN.W1``
    ``tcp_flag_profile``    ``T1.F``      (flags of the probe reply)
    ``tcp_option_profile``  ``OPS.O1..O6``
    ``isn_gcd``             ``SEQ.GCD``   (ISN generation)
    ``udp_open_count``      ``U1.R``      (closed port answered)
    ``icmpv6_resp_code``    ``IE.R``      (echo reply present)
    ======================  ==========================================
    """
    s = rec["sections"]
    row = np.full(len(FEATURES), np.nan)

    def put(name, v):
        if name in NAMES and v is not None and not (isinstance(v, float) and np.isnan(v)):
            row[NAMES.index(name)] = float(v)

    put("hop_limit", _first(s, "T1", ["T"]))
    mtu = _first(s, "T1", ["TG"])
    if np.isnan(mtu):
        mtu = _first(s, "ECN", ["TG"])
    put("mtu", mtu)
    put("isn_gcd", _first(s, "SEQ", ["GCD"]))
    win = _first(s, "T1", ["W"])
    for sec in ("ECN", "WIN"):
        if np.isnan(win):
            win = _first(s, sec, ["W", "W1"])
    put("tcp_window", win)
    if s.get("T1") and "DF" in s["T1"]:
        put("frag_supported", _yn(s["T1"]["DF"]))
    if s.get("U1") and "R" in s["U1"]:
        put("udp_open_count", _yn(s["U1"]["R"]))
    if s.get("IE") and "R" in s["IE"]:
        put("icmpv6_resp_code", _yn(s["IE"]["R"]) * 3.0)   # 0 = no reply, 3 = echo reply

    if s.get("T1") and s["T1"].get("F"):
        put("tcp_flag_profile", flag_vocab.get(s["T1"]["F"], flag_other))
    if s.get("OPS"):
        opt = "".join(s["OPS"].get(f"O{i}", "") for i in range(1, 7))
        put("tcp_option_profile", opt_vocab.get(opt, opt_other))
    # The nine L7 and six DV fields stay NaN: nmap-os-db carries no such probe.
    return row


def build(db_path, max_per_class=None, seed=0):
    recs = list(parse(db_path))
    # vocabulary of TCP option styles and of TCP flag strings: the six most
    # frequent styles keep their own code and everything else is pooled, which
    # is how the benchmark represents these two fingerprint fields.
    cnt = collections.Counter()
    fcnt = collections.Counter()
    for r in recs:
        if r["sections"].get("OPS"):
            cnt["".join(r["sections"]["OPS"].get(f"O{i}", "") for i in range(1, 7))] += 1
        f = r["sections"].get("T1", {}).get("F")
        if f:
            fcnt[f] += 1
    opt_vocab = {k: i for i, (k, _) in enumerate(cnt.most_common(6))}
    other = len(opt_vocab)
    flag_vocab = {k: i for i, (k, _) in enumerate(fcnt.most_common(6))}
    flag_other = len(flag_vocab)
    opt_vocab = {k: i for i, (k, _) in enumerate(cnt.most_common(6))}
    other = len(opt_vocab)

    by_class = collections.defaultdict(list)
    dropped = collections.Counter()
    for r in recs:
        cls = _class_of(r)
        if cls is None:
            dropped["ambiguous"] += 1
            continue
        by_class[cls].append(r)
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for cls, items in sorted(by_class.items()):
        idx = rng.permutation(len(items))
        if max_per_class:
            idx = idx[:max_per_class]
        for i in idx:
            r = items[i]
            rows.append(project(r, opt_vocab, other, flag_vocab, flag_other))
            labels.append(CLS_INDEX[cls])
    X = np.asarray(rows) if rows else np.zeros((0, len(FEATURES)))
    y = np.asarray(labels, dtype=np.int64)
    mask = ~np.isnan(X)
    y_coarse = np.array([SUPER_CLASSES.index(CLASSES[c][1]) for c in y], dtype=np.int64)
    return dict(X=X, y=y, y_coarse=y_coarse, mask=mask,
                names=np.array(NAMES), kind=np.array(KIND), ncat=np.array(NCAT)), \
        dict(by_class={k: len(v) for k, v in by_class.items()}, dropped=dict(dropped),
             fingerprints=len(recs), opt_vocab=len(opt_vocab), flag_vocab=len(flag_vocab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="nmap-os-db", help="path to a local nmap-os-db copy")
    ap.add_argument("--out", default="nmap_os.npz")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="print counts and exit")
    args = ap.parse_args()
    if not os.path.exists(args.db):
        sys.exit(f"nmap-os-db not found at {args.db!r}; pass --db <path>")

    data, info = build(args.db, args.max_per_class)
    print(f"fingerprints: {info['fingerprints']}   with a usable label: {len(data['y'])}")
    for k, n in sorted(info["by_class"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {n}")
    if info["dropped"]:
        print(f"dropped (ambiguous device type): {info['dropped'].get('ambiguous', 0)}")
    if len(data["y"]):
        print(f"observed-cell ratio: {data['mask'].mean():.3f} "
              f"(L7 and DV are unobservable by construction)")
    if args.report:
        return
    np.savez_compressed(args.out, difficulty_idx=NAMES.index("response_delay_ratio"),
                        **data)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
