"""Build the 25M-token multi-epoch sub-pool: a stratified slice of the 05
pool at the same ~84.5/15.5 FineWeb-Edu/Cosmopedia blend (a naive prefix
slice would be 100% FineWeb-Edu and confound the epoch test with a mix
change). 100M-token training budget / 25M pool = each token seen ~4x.
"""

import os

import numpy as np

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
POOL = os.path.join(ROOT, "..", "..", "05-data-frontier", "data", "train.bin")

# 05 train.bin layout (tokens): fineweb-edu [0, 1_501_640_104), cosmopedia after
COSMO_START = 1_501_640_104
FWE_TAKE = 21_125_000
COSMO_TAKE = 3_875_000  # 15.5% of 25M


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "subpool.bin")
    if os.path.exists(out):
        print(f"already built: {os.path.getsize(out) // 2:,} tokens")
        return
    data = np.memmap(POOL, dtype=np.uint16, mode="r")
    assert len(data) > COSMO_START + COSMO_TAKE
    with open(out + ".part", "wb") as f:
        f.write(data[:FWE_TAKE].tobytes())
        f.write(data[COSMO_START:COSMO_START + COSMO_TAKE].tobytes())
    os.replace(out + ".part", out)
    print(f"subpool: {(FWE_TAKE + COSMO_TAKE):,} tokens ({FWE_TAKE:,} fineweb + {COSMO_TAKE:,} cosmo)")


if __name__ == "__main__":
    main()
