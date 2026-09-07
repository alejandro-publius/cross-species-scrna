"""
Phase 0/1: fetch raw inputs into data/raw/ (idempotent -- skips files already present).
  - Baron GSE84133 human+mouse islet counts (tarball of per-sample CSVs)
  - MGI mouse<->human homology table (static ortholog source; the biomart fallback)
"""
import shutil
import socket
import tarfile
import urllib.error
import urllib.request

import config as C

BARON_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84133/suppl/GSE84133_RAW.tar"
MGI_URL = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
TIMEOUT_S = 30


def fetch(url, dest):
    if dest.exists():
        print(f"  have {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except (urllib.error.URLError, OSError, socket.timeout) as e:
        dest.unlink(missing_ok=True)  # don't leave a truncated file that looks "have"-able
        raise SystemExit(
            f"Failed to download {url}\n"
            f"  -> {e}\n"
            f"  -> Check network access, or download the file manually to {dest}."
        ) from e


def main():
    C.RAW.mkdir(parents=True, exist_ok=True)
    tar = C.RAW / "GSE84133_RAW.tar"
    fetch(BARON_URL, tar)
    # extract the per-sample CSVs if missing
    if not (C.RAW / C.HUMAN_FILES[0]).exists():
        print("  extracting Baron CSVs ...")
        with tarfile.open(tar) as t:
            t.extractall(C.RAW)
    fetch(MGI_URL, C.RAW / "HOM_MouseHumanSequence.rpt")
    print("Raw inputs ready in", C.RAW)


if __name__ == "__main__":
    main()
