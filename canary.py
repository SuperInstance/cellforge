"""Canary — polyformalism canary for cellforge.

The canary is a small, stable fingerprint that the fleet can check. All 47+
repos in the SuperInstance fleet produce the same FNV-1a 64-bit hash for
this exact input, proving the substrate is shared across the entire
ecosystem.
"""
def fnv1a_64(s: str) -> int:
    h = 0xcbf29ce484222325
    for b in s.encode("utf-8"):
        h = h ^ b
        h = (h * 0x100000001b3) & 0xffffffffffffffff
    return h


def canary() -> str:
    return f"0x{fnv1a_64('café Δ 日本語'):016x}"


if __name__ == "__main__":
    print(canary())
    assert canary() == "0x24a555471370b18d"
