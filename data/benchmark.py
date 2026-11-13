"""
Benchmark construction for cyberspace entity classification under incomplete
fingerprint information.

The generator follows the fingerprint grammar of active IPv6 probing
(ICMPv6 / TCP / UDP / application-layer probes plus device-specific probes) and
the entity taxonomy (4 super-classes, 11 fine-grained classes).

Two generation modes are supported:

* ``--source synth``   reproducible trace-driven generation (default).  Class
  conditional feature distributions, per-feature observation propensities and
  cross-feature observation dependencies are specified in FEATURES/CLASSES below.
* ``--source nmap``    parse a local copy of the public Nmap OS-fingerprint
  corpus (``nmap-os-db``) instead; see parse_nmap_osdb.py.

Controlled incompleteness is then imposed on top of the naturally missing
entries, following the MCAR / MAR / block-missing protocols used in the experiments.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# --------------------------------------------------------------------------- #
# Feature space
# --------------------------------------------------------------------------- #
# kind: 'num' continuous, 'cat' categorical (integer code), 'bin' binary
# obs : base observation probability of the feature in a successful probe
# layer: L3 (network), L4 (transport), L7 (application), DV (device specific)
FEATURES = [
    # --- network layer -----------------------------------------------------
    dict(name="icmpv6_resp_code", kind="cat", n=6, obs=0.93, layer="L3"),
    dict(name="icmpv6_rtt_ms", kind="num", obs=0.95, layer="L3"),
    dict(name="icmpv6_error_rate", kind="num", obs=0.90, layer="L3"),
    dict(name="response_delay_ratio", kind="num", obs=0.92, layer="L3"),
    dict(name="hop_limit", kind="num", obs=0.96, layer="L3"),
    dict(name="mtu", kind="num", obs=0.88, layer="L3"),
    dict(name="ext_hdr_profile", kind="cat", n=6, obs=0.72, layer="L3"),
    dict(name="flow_label_echo", kind="bin", n=2, obs=0.83, layer="L3"),
    dict(name="frag_supported", kind="bin", n=2, obs=0.80, layer="L3"),
    # --- transport layer ---------------------------------------------------
    dict(name="open_port_count", kind="num", obs=0.94, layer="L4"),
    dict(name="tcp_flag_profile", kind="cat", n=6, obs=0.91, layer="L4"),
    dict(name="tcp_option_profile", kind="cat", n=6, obs=0.85, layer="L4"),
    dict(name="tcp_window", kind="num", obs=0.90, layer="L4"),
    dict(name="isn_gcd", kind="num", obs=0.78, layer="L4"),
    dict(name="udp_open_count", kind="num", obs=0.76, layer="L4"),
    dict(name="web_port_open", kind="bin", n=2, obs=0.92, layer="L4"),
    dict(name="mail_port_open", kind="bin", n=2, obs=0.92, layer="L4"),
    dict(name="db_port_open", kind="bin", n=2, obs=0.92, layer="L4"),
    dict(name="ics_port_open", kind="bin", n=2, obs=0.90, layer="L4"),
    dict(name="rdp_smb_open", kind="bin", n=2, obs=0.90, layer="L4"),
    # --- application layer -------------------------------------------------
    dict(name="http_server_class", kind="cat", n=7, obs=0.62, layer="L7"),
    dict(name="http_status_profile", kind="cat", n=7, obs=0.60, layer="L7"),
    dict(name="http_header_len", kind="num", obs=0.58, layer="L7"),
    dict(name="tls_version", kind="cat", n=5, obs=0.60, layer="L7"),
    dict(name="tls_issuer_class", kind="cat", n=6, obs=0.55, layer="L7"),
    dict(name="dns_recursion", kind="cat", n=4, obs=0.66, layer="L7"),
    dict(name="service_software_class", kind="cat", n=7, obs=0.52, layer="L7"),
    dict(name="banner_entropy", kind="num", obs=0.64, layer="L7"),
    dict(name="auth_realm_class", kind="cat", n=5, obs=0.50, layer="L7"),
    # --- device specific probes -------------------------------------------
    dict(name="snmp_sysdescr_class", kind="cat", n=6, obs=0.46, layer="DV"),
    dict(name="mgmt_protocol_version", kind="cat", n=5, obs=0.44, layer="DV"),
    dict(name="vendor_oui_class", kind="cat", n=9, obs=0.70, layer="DV"),
    dict(name="ics_protocol_id", kind="cat", n=5, obs=0.40, layer="DV"),
    dict(name="virtual_platform_id", kind="cat", n=6, obs=0.48, layer="DV"),
    dict(name="low_power_mode", kind="bin", n=2, obs=0.55, layer="DV"),
    dict(name="firmware_age_days", kind="num", obs=0.42, layer="DV"),
]

CLASSES = [
    ("pc", "terminal"),
    ("smartphone", "terminal"),
    ("ip_camera", "terminal"),
    ("smart_hub", "terminal"),
    ("router", "network_device"),
    ("firewall", "network_device"),
    ("plc", "network_device"),
    ("web_server", "server"),
    ("dns_db_server", "server"),
    ("mail_server", "server"),
    ("vm_container", "virtual_entity"),
]
SUPER_CLASSES = ["terminal", "network_device", "server", "virtual_entity"]

# --------------------------------------------------------------------------- #
# Class conditional prototypes
# --------------------------------------------------------------------------- #
# 'C' entries give a categorical prototype: (code, concentration).  A residual
# mass is spread over the remaining categories, emulating heterogeneous
# deployments of the same class.
# 'N' entries give (mean, std) for continuous features.
SPEC = {
    "pc": {
        "C": {"icmpv6_resp_code": (0, .55), "ext_hdr_profile": (0, .6), "tcp_flag_profile": (1, .6),
              "tcp_option_profile": (2, .55), "http_server_class": (2, .35), "http_status_profile": (0, .4),
              "tls_version": (2, .45), "tls_issuer_class": (5, .4), "dns_recursion": (1, .5),
              "service_software_class": (0, .35), "auth_realm_class": (4, .5), "snmp_sysdescr_class": (1, .55),
              "mgmt_protocol_version": (2, .35), "vendor_oui_class": (4, .3), "ics_protocol_id": (4, .8),
              "virtual_platform_id": (5, .85)},
        "N": {"icmpv6_rtt_ms": (12, 6), "icmpv6_error_rate": (.05, .05), "hop_limit": (128, 6),
              "mtu": (1500, 60), "open_port_count": (14, 5), "tcp_window": (64240, 4000),
              "isn_gcd": (12, 8), "udp_open_count": (5, 2), "http_header_len": (520, 120),
              "banner_entropy": (3.6, .5), "firmware_age_days": (420, 260),
              "response_delay_ratio": (1.0, .2)},
        "B": {"flow_label_echo": .75, "frag_supported": .8, "web_port_open": .55, "mail_port_open": .1,
              "db_port_open": .1, "ics_port_open": .01, "rdp_smb_open": .8, "low_power_mode": .05},
    },
    "smartphone": {
        "C": {"icmpv6_resp_code": (0, .5), "ext_hdr_profile": (0, .55), "tcp_flag_profile": (1, .55),
              "tcp_option_profile": (3, .5), "http_server_class": (6, .6), "http_status_profile": (4, .4),
              "tls_version": (3, .5), "tls_issuer_class": (1, .4), "dns_recursion": (0, .5),
              "service_software_class": (6, .4), "auth_realm_class": (0, .6), "snmp_sysdescr_class": (5, .7),
              "mgmt_protocol_version": (4, .7), "vendor_oui_class": (5, .35), "ics_protocol_id": (4, .9),
              "virtual_platform_id": (5, .9)},
        "N": {"icmpv6_rtt_ms": (46, 18), "icmpv6_error_rate": (.12, .08), "hop_limit": (64, 4),
              "mtu": (1400, 40), "open_port_count": (3, 2), "tcp_window": (65535, 2000),
              "isn_gcd": (35, 18), "udp_open_count": (3, 2), "http_header_len": (300, 80),
              "banner_entropy": (2.9, .5), "firmware_age_days": (300, 180),
              "response_delay_ratio": (1.6, .4)},
        "B": {"flow_label_echo": .5, "frag_supported": .55, "web_port_open": .2, "mail_port_open": .02,
              "db_port_open": .01, "ics_port_open": .01, "rdp_smb_open": .05, "low_power_mode": .8},
    },
    "ip_camera": {
        "C": {"icmpv6_resp_code": (0, .6), "ext_hdr_profile": (2, .4), "tcp_flag_profile": (1, .5),
              "tcp_option_profile": (0, .45), "http_server_class": (4, .55), "http_status_profile": (2, .4),
              "tls_version": (1, .4), "tls_issuer_class": (4, .5), "dns_recursion": (1, .5),
              "service_software_class": (5, .4), "auth_realm_class": (1, .45), "snmp_sysdescr_class": (3, .45),
              "mgmt_protocol_version": (1, .4), "vendor_oui_class": (2, .4), "ics_protocol_id": (4, .85),
              "virtual_platform_id": (5, .9)},
        "N": {"icmpv6_rtt_ms": (28, 10), "icmpv6_error_rate": (.08, .06), "hop_limit": (64, 5),
              "mtu": (1500, 50), "open_port_count": (6, 3), "tcp_window": (29200, 6000),
              "isn_gcd": (48, 22), "udp_open_count": (7, 3), "http_header_len": (210, 60),
              "banner_entropy": (2.4, .5), "firmware_age_days": (1150, 520),
              "response_delay_ratio": (1.3, .3)},
        "B": {"flow_label_echo": .4, "frag_supported": .45, "web_port_open": .85, "mail_port_open": .01,
              "db_port_open": .02, "ics_port_open": .05, "rdp_smb_open": .02, "low_power_mode": .55},
    },
    "smart_hub": {
        "C": {"icmpv6_resp_code": (0, .6), "ext_hdr_profile": (2, .45), "tcp_flag_profile": (1, .55),
              "tcp_option_profile": (0, .4), "http_server_class": (4, .5), "http_status_profile": (2, .35),
              "tls_version": (2, .45), "tls_issuer_class": (1, .45), "dns_recursion": (0, .45),
              "service_software_class": (6, .45), "auth_realm_class": (0, .55), "snmp_sysdescr_class": (3, .5),
              "mgmt_protocol_version": (4, .6), "vendor_oui_class": (5, .3), "ics_protocol_id": (4, .9),
              "virtual_platform_id": (5, .9)},
        "N": {"icmpv6_rtt_ms": (38, 15), "icmpv6_error_rate": (.15, .09), "hop_limit": (64, 4),
              "mtu": (1280, 30), "open_port_count": (4, 2), "tcp_window": (5840, 2000),
              "isn_gcd": (60, 25), "udp_open_count": (9, 4), "http_header_len": (260, 70),
              "banner_entropy": (2.6, .5), "firmware_age_days": (620, 300),
              "response_delay_ratio": (1.9, .5)},
        "B": {"flow_label_echo": .35, "frag_supported": .4, "web_port_open": .45, "mail_port_open": .01,
              "db_port_open": .01, "ics_port_open": .02, "rdp_smb_open": .01, "low_power_mode": .9},
    },
    "router": {
        "C": {"icmpv6_resp_code": (1, .5), "ext_hdr_profile": (1, .55), "tcp_flag_profile": (4, .4),
              "tcp_option_profile": (1, .4), "http_server_class": (3, .35), "http_status_profile": (2, .4),
              "tls_version": (2, .4), "tls_issuer_class": (4, .45), "dns_recursion": (0, .5),
              "service_software_class": (5, .35), "auth_realm_class": (3, .5), "snmp_sysdescr_class": (2, .55),
              "mgmt_protocol_version": (1, .45), "vendor_oui_class": (1, .35), "ics_protocol_id": (4, .85),
              "virtual_platform_id": (5, .88)},
        "N": {"icmpv6_rtt_ms": (9, 4), "icmpv6_error_rate": (.04, .03), "hop_limit": (255, 4),
              "mtu": (1500, 30), "open_port_count": (5, 2), "tcp_window": (16384, 6000),
              "isn_gcd": (8, 5), "udp_open_count": (6, 2), "http_header_len": (180, 50),
              "banner_entropy": (2.2, .5), "firmware_age_days": (900, 420),
              "response_delay_ratio": (.9, .2)},
        "B": {"flow_label_echo": .85, "frag_supported": .9, "web_port_open": .5, "mail_port_open": .02,
              "db_port_open": .02, "ics_port_open": .02, "rdp_smb_open": .03, "low_power_mode": .05},
    },
    "firewall": {
        "C": {"icmpv6_resp_code": (1, .5), "ext_hdr_profile": (1, .45), "tcp_flag_profile": (5, .5),
              "tcp_option_profile": (5, .5), "http_server_class": (6, .5), "http_status_profile": (5, .45),
              "tls_version": (4, .55), "tls_issuer_class": (5, .6), "dns_recursion": (1, .45),
              "service_software_class": (6, .5), "auth_realm_class": (4, .55), "snmp_sysdescr_class": (5, .45),
              "mgmt_protocol_version": (2, .4), "vendor_oui_class": (3, .3), "ics_protocol_id": (4, .9),
              "virtual_platform_id": (5, .8)},
        "N": {"icmpv6_rtt_ms": (14, 5), "icmpv6_error_rate": (.3, .12), "hop_limit": (255, 4),
              "mtu": (1500, 40), "open_port_count": (2, 1), "tcp_window": (8192, 4000),
              "isn_gcd": (20, 12), "udp_open_count": (2, 1), "http_header_len": (120, 40),
              "banner_entropy": (1.8, .5), "firmware_age_days": (700, 350),
              "response_delay_ratio": (1.2, .3)},
        "B": {"flow_label_echo": .6, "frag_supported": .5, "web_port_open": .3, "mail_port_open": .05,
              "db_port_open": .02, "ics_port_open": .02, "rdp_smb_open": .05, "low_power_mode": .02},
    },
    "plc": {
        "C": {"icmpv6_resp_code": (1, .45), "ext_hdr_profile": (3, .4), "tcp_flag_profile": (4, .45),
              "tcp_option_profile": (1, .4), "http_server_class": (4, .4), "http_status_profile": (2, .4),
              "tls_version": (0, .4), "tls_issuer_class": (5, .5), "dns_recursion": (1, .5),
              "service_software_class": (5, .4), "auth_realm_class": (4, .5), "snmp_sysdescr_class": (3, .5),
              "mgmt_protocol_version": (1, .4), "vendor_oui_class": (3, .45), "ics_protocol_id": (0, .5),
              "virtual_platform_id": (5, .95)},
        "N": {"icmpv6_rtt_ms": (7, 3), "icmpv6_error_rate": (.06, .04), "hop_limit": (255, 4),
              "mtu": (1500, 20), "open_port_count": (4, 2), "tcp_window": (4096, 2000),
              "isn_gcd": (30, 15), "udp_open_count": (2, 1), "http_header_len": (140, 40),
              "banner_entropy": (1.6, .4), "firmware_age_days": (2100, 700),
              "response_delay_ratio": (.8, .2)},
        "B": {"flow_label_echo": .7, "frag_supported": .7, "web_port_open": .35, "mail_port_open": .01,
              "db_port_open": .01, "ics_port_open": .95, "rdp_smb_open": .02, "low_power_mode": .1},
    },
    "web_server": {
        "C": {"icmpv6_resp_code": (0, .5), "ext_hdr_profile": (0, .5), "tcp_flag_profile": (1, .6),
              "tcp_option_profile": (2, .55), "http_server_class": (0, .5), "http_status_profile": (0, .55),
              "tls_version": (3, .6), "tls_issuer_class": (0, .45), "dns_recursion": (1, .45),
              "service_software_class": (0, .4), "auth_realm_class": (2, .4), "snmp_sysdescr_class": (0, .45),
              "mgmt_protocol_version": (3, .4), "vendor_oui_class": (4, .3), "ics_protocol_id": (4, .9),
              "virtual_platform_id": (4, .4)},
        "N": {"icmpv6_rtt_ms": (6, 3), "icmpv6_error_rate": (.01, .02), "hop_limit": (64, 4),
              "mtu": (1500, 20), "open_port_count": (12, 4), "tcp_window": (65535, 3000),
              "isn_gcd": (10, 6), "udp_open_count": (3, 1), "http_header_len": (760, 180),
              "banner_entropy": (4.1, .4), "firmware_age_days": (200, 120),
              "response_delay_ratio": (1.0, .2)},
        "B": {"flow_label_echo": .8, "frag_supported": .85, "web_port_open": .98, "mail_port_open": .08,
              "db_port_open": .15, "ics_port_open": .01, "rdp_smb_open": .05, "low_power_mode": .01},
    },
    "dns_db_server": {
        "C": {"icmpv6_resp_code": (0, .5), "ext_hdr_profile": (0, .5), "tcp_flag_profile": (1, .55),
              "tcp_option_profile": (4, .5), "http_server_class": (1, .4), "http_status_profile": (3, .4),
              "tls_version": (2, .45), "tls_issuer_class": (2, .4), "dns_recursion": (0, .6),
              "service_software_class": (2, .45), "auth_realm_class": (0, .5), "snmp_sysdescr_class": (0, .5),
              "mgmt_protocol_version": (3, .35), "vendor_oui_class": (4, .3), "ics_protocol_id": (4, .9),
              "virtual_platform_id": (4, .35)},
        "N": {"icmpv6_rtt_ms": (5, 2), "icmpv6_error_rate": (.01, .02), "hop_limit": (64, 4),
              "mtu": (1500, 20), "open_port_count": (9, 3), "tcp_window": (65535, 3000),
              "isn_gcd": (9, 5), "udp_open_count": (6, 2), "http_header_len": (400, 120),
              "banner_entropy": (3.4, .5), "firmware_age_days": (260, 150),
              "response_delay_ratio": (.9, .2)},
        "B": {"flow_label_echo": .8, "frag_supported": .85, "web_port_open": .55, "mail_port_open": .35,
              "db_port_open": .8, "ics_port_open": .01, "rdp_smb_open": .05, "low_power_mode": .01},
    },
    "mail_server": {
        "C": {"icmpv6_resp_code": (0, .5), "ext_hdr_profile": (0, .5), "tcp_flag_profile": (1, .55),
              "tcp_option_profile": (2, .5), "http_server_class": (1, .35), "http_status_profile": (3, .4),
              "tls_version": (2, .5), "tls_issuer_class": (0, .45), "dns_recursion": (1, .45),
              "service_software_class": (1, .55), "auth_realm_class": (3, .45), "snmp_sysdescr_class": (0, .45),
              "mgmt_protocol_version": (3, .35), "vendor_oui_class": (4, .3), "ics_protocol_id": (4, .9),
              "virtual_platform_id": (4, .35)},
        "N": {"icmpv6_rtt_ms": (7, 3), "icmpv6_error_rate": (.02, .02), "hop_limit": (64, 4),
              "mtu": (1500, 20), "open_port_count": (7, 3), "tcp_window": (29200, 8000),
              "isn_gcd": (13, 7), "udp_open_count": (3, 1), "http_header_len": (330, 90),
              "banner_entropy": (3.2, .5), "firmware_age_days": (300, 170),
              "response_delay_ratio": (1.0, .2)},
        "B": {"flow_label_echo": .8, "frag_supported": .85, "web_port_open": .45, "mail_port_open": .95,
              "db_port_open": .2, "ics_port_open": .01, "rdp_smb_open": .05, "low_power_mode": .01},
    },
    "vm_container": {
        "C": {"icmpv6_resp_code": (0, .5), "ext_hdr_profile": (0, .55), "tcp_flag_profile": (1, .6),
              "tcp_option_profile": (2, .5), "http_server_class": (0, .45), "http_status_profile": (0, .45),
              "tls_version": (3, .55), "tls_issuer_class": (3, .4), "dns_recursion": (0, .4),
              "service_software_class": (0, .35), "auth_realm_class": (2, .4), "snmp_sysdescr_class": (0, .4),
              "mgmt_protocol_version": (3, .4), "vendor_oui_class": (6, .35), "ics_protocol_id": (4, .9),
              "virtual_platform_id": (0, .45)},
        "N": {"icmpv6_rtt_ms": (8, 4), "icmpv6_error_rate": (.02, .03), "hop_limit": (64, 4),
              "mtu": (1450, 40), "open_port_count": (10, 4), "tcp_window": (64860, 3000),
              "isn_gcd": (16, 9), "udp_open_count": (3, 1), "http_header_len": (620, 160),
              "banner_entropy": (3.9, .5), "firmware_age_days": (120, 90),
              "response_delay_ratio": (1.1, .25)},
        "B": {"flow_label_echo": .7, "frag_supported": .75, "web_port_open": .8, "mail_port_open": .1,
              "db_port_open": .3, "ics_port_open": .01, "rdp_smb_open": .1, "low_power_mode": .01},
    },
}

# Cross-feature observation dependencies: a feature is only observable when its
# parent probe succeeded, which makes the missingness mechanism MNAR-like.
DEPENDENCIES = {
    "http_server_class": ("web_port_open", 1),
    "http_status_profile": ("web_port_open", 1),
    "http_header_len": ("web_port_open", 1),
    "tls_version": ("web_port_open", 1),
    "tls_issuer_class": ("web_port_open", 1),
    "auth_realm_class": ("web_port_open", 1),
    "dns_recursion": ("udp_open_count", None),   # conditional on udp_open_count > 0
    "ics_protocol_id": ("ics_port_open", 1),
    "mail_port_open": ("open_port_count", None),  # never blocks, kept for symmetry
}


def _background(n: int) -> np.ndarray:
    """A mildly skewed background distribution over categorical codes."""
    w = 1.0 / (1.0 + np.arange(n)) ** 0.8
    return w / w.sum()


def _sample_cat(rng, n, n_cat, proto):
    code, conc = proto
    p = _background(n_cat) * (1.0 - 0.55 * conc)
    p[code] += 0.55 * conc * 1.0
    p = p / p.sum()
    return rng.choice(n_cat, size=n, p=p).astype(np.int32)


def generate(seed: int = 2024, n_per_class: int = 1000, label_noise: float = 0.02):
    rng = np.random.default_rng(seed)
    m = len(FEATURES)
    names = [f["name"] for f in FEATURES]
    kind = [f["kind"] for f in FEATURES]
    ncat = [f.get("n", 0) for f in FEATURES]
    obs_p = np.array([f["obs"] for f in FEATURES])

    X = np.full((len(CLASSES) * n_per_class, m), np.nan)
    y = np.repeat(np.arange(len(CLASSES)), n_per_class)

    for ci, (cname, _) in enumerate(CLASSES):
        spec = SPEC[cname]
        sl = slice(ci * n_per_class, (ci + 1) * n_per_class)
        n = n_per_class
        for j, f in enumerate(FEATURES):
            name, k = f["name"], f["kind"]
            if k == "cat" or k == "bin":
                if name in spec["C"]:
                    codes = _sample_cat(rng, n, f["n"], spec["C"][name])
                elif name in spec["B"]:
                    codes = rng.binomial(1, spec["B"][name], size=n).astype(np.int32)
                else:
                    codes = rng.integers(0, f["n"], size=n).astype(np.int32)
                X[sl, j] = codes
            else:
                mu, sd = spec["N"][name]
                vals = rng.normal(mu, sd, size=n)
                if name in ("icmpv6_error_rate",):
                    vals = np.clip(vals, 0.0, 1.0)
                if name in ("mtu", "tcp_window", "open_port_count", "udp_open_count",
                            "hop_limit", "http_header_len", "firmware_age_days",
                            "icmpv6_rtt_ms", "response_delay_ratio", "banner_entropy"):
                    vals = np.clip(vals, 0.0, None)
                X[sl, j] = vals

        # weak label noise: a fraction of terminals are mislabelled as a
        # neighbouring class, emulating annotation errors in operational data
        idx = np.arange(ci * n_per_class, (ci + 1) * n_per_class)
        flip = rng.random(n) < label_noise
        if flip.any():
            neighbours = {"pc": "vm_container", "vm_container": "web_server", "router": "firewall",
                          "firewall": "router", "ip_camera": "smart_hub", "smart_hub": "ip_camera",
                          "web_server": "vm_container", "dns_db_server": "mail_server",
                          "mail_server": "dns_db_server", "smartphone": "smart_hub", "plc": "router"}
            y[idx[flip]] = [i for i, (c, _) in enumerate(CLASSES) if c == neighbours[cname]][0]

    # ------------------------------------------------------------------ masks
    mask = np.zeros((X.shape[0], m), dtype=bool)
    # class dependent observability multipliers (resource constrained devices
    # answer fewer application-layer probes)
    cls_mult = {
        "pc": {"L7": 1.0, "DV": 1.0}, "smartphone": {"L7": .5, "DV": .4},
        "ip_camera": {"L7": .7, "DV": .7}, "smart_hub": {"L7": .55, "DV": .5},
        "router": {"L7": .8, "DV": .85}, "firewall": {"L7": .55, "DV": .7},
        "plc": {"L7": .6, "DV": .8}, "web_server": {"L7": 1.25, "DV": .9},
        "dns_db_server": {"L7": 1.1, "DV": .9}, "mail_server": {"L7": 1.1, "DV": .9},
        "vm_container": {"L7": 1.2, "DV": .95},
    }
    for i in range(X.shape[0]):
        cname = CLASSES[y[i]][0]
        mult = cls_mult[cname]
        for j, f in enumerate(FEATURES):
            p = obs_p[j]
            if f["layer"] in mult:
                p = min(0.97, p * mult[f["layer"]])
            mask[i, j] = rng.random() < p

    # dependency constraints
    for child, (parent, need) in DEPENDENCIES.items():
        jc = names.index(child)
        jp = names.index(parent)
        if need is None:
            ok = X[:, jp] > 0
        else:
            ok = X[:, jp] == need
        mask[:, jc] &= ok

    X[~mask] = np.nan
    y_coarse = np.array([SUPER_CLASSES.index(CLASSES[c][1]) for c in y])
    return dict(X=X, y=y, y_coarse=y_coarse, mask=mask, names=names,
                kind=np.array(kind), ncat=np.array(ncat), seed=seed)


# --------------------------------------------------------------------------- #
# Controlled incompleteness protocols
# --------------------------------------------------------------------------- #
def impose_missing(rng, X, mask, rate, protocol="mcar", meta=None):
    """Return a new mask with an *additional* `rate` of the observed cells hidden."""
    m = np.zeros_like(mask)
    n, d = mask.shape
    if rate <= 0:
        return mask.copy()
    if protocol == "mcar":
        hide = (rng.random((n, d)) < rate) & mask
    elif protocol == "mar":
        # hiding probability grows with the value of a co-observed "difficulty"
        # variable (response delay), i.e. missingness depends on observed data
        diff = X[:, meta["difficulty_idx"]]
        diff = np.nan_to_num(diff, nan=np.nanmedian(diff))
        q = (diff - diff.min()) / (np.ptp(diff) + 1e-9)
        hide = (rng.random((n, d)) < rate * (0.5 + 1.5 * q)[:, None]) & mask
    elif protocol == "block":
        # an entire fingerprint group disappears (e.g. all application-layer
        # probes blocked by a middlebox)
        hide = np.zeros_like(mask)
        groups = meta["groups"]
        for gi, cols in enumerate(groups):
            drop = rng.random(n) < rate
            hide[np.ix_(drop, cols)] = True
        hide &= mask
    elif protocol == "mnar":
        # hiding probability depends on the (unobserved) value itself: extreme
        # values are more likely to be suppressed by rate limiting
        hide = np.zeros_like(mask)
        for j in range(d):
            col = X[:, j]
            obs = col[mask[:, j]]
            if obs.size < 10:
                continue
            lo, hi = np.percentile(obs, [10, 90])
            extreme = ((col < lo) | (col > hi)).astype(float)
            extreme = np.nan_to_num(extreme)
            hide[:, j] = (rng.random(n) < rate * (0.4 + 1.2 * extreme)) & mask[:, j]
    else:
        raise ValueError(protocol)
    out = mask & ~hide
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark.npz")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--n-per-class", type=int, default=1000)
    args = ap.parse_args()

    d = generate(seed=args.seed, n_per_class=args.n_per_class)
    groups = [[j for j, f in enumerate(FEATURES) if f["layer"] == g] for g in ("L3", "L4", "L7", "DV")]
    np.savez_compressed(args.out, X=d["X"], y=d["y"], y_coarse=d["y_coarse"], mask=d["mask"],
                        names=np.array(d["names"]), kind=d["kind"], ncat=d["ncat"],
                        difficulty_idx=[f["name"] for f in FEATURES].index("response_delay_ratio"))
    spec = dict(features=FEATURES, classes=CLASSES, super_classes=SUPER_CLASSES,
                groups={g: [FEATURES[j]["name"] for j in cols] for g, cols in zip(("L3", "L4", "L7", "DV"), groups)})
    with open(os.path.join(os.path.dirname(args.out) or ".", "feature_spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)

    # ------------------------------------------------------------------ report
    obs_ratio = d["mask"].mean()
    print(f"samples={d['X'].shape[0]}  features={d['X'].shape[1]}  classes={len(CLASSES)}")
    print(f"natural observation ratio = {obs_ratio:.3f}  (missing = {1-obs_ratio:.3f})")
    print("per-layer observation ratio:",
          {g: round(float(d["mask"][:, cols].mean()), 3) for g, cols in
           zip(("L3", "L4", "L7", "DV"), groups)})
    for ci, (cname, sup) in enumerate(CLASSES):
        sel = d["y"] == ci
        print(f"  {cname:14s} n={sel.sum():5d} obs={d['mask'][sel].mean():.3f}")


if __name__ == "__main__":
    main()
