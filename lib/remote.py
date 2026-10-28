"""HTTP helpers for reference products: download whole files, or read single members of remote zip archives."""
import io
import subprocess
from pathlib import Path

import requests


def download(url: str, dest: Path) -> Path:
    """Download url to dest with curl, through a .part file. A dest that already exists and is not empty is kept."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    have {dest.name}")
        return dest
    print(f"    get  {dest.name}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl", "-sSL", "--fail", "-o", str(tmp), url], check=True)
    tmp.rename(dest)
    return dest


class HTTPRange(io.RawIOBase):
    """A read-only, seekable view of a remote file, using HTTP range requests.

    Wrap it in io.BufferedReader and pass it to zipfile. Then you can read single members of a large remote
    archive without downloading the rest.
    """

    def __init__(self, url):
        self.s = requests.Session()
        r = self.s.get(url, stream=True, allow_redirects=True, timeout=(15, 60))
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0
        r.close()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=(15, 120))
        n = len(r.content)
        b[:n] = r.content
        self.pos += n
        return n


def extract_members(url: str, names, into: Path, buffer=2**23):
    """Copy chosen members of a remote zip archive into `into`, reading only those members.

    A member is chosen when its basename is in `names`, or when `names` is a function that returns True for it."""
    import shutil
    import zipfile
    into.mkdir(parents=True, exist_ok=True)
    want = names if callable(names) else (lambda n, s=set(names): n in s)
    out = []
    with io.BufferedReader(HTTPRange(url), buffer_size=buffer) as f:
        z = zipfile.ZipFile(f)
        for info in z.infolist():
            base = info.filename.rsplit("/", 1)[-1]
            if base and want(base):
                dest = into / base
                if not dest.exists():
                    with z.open(info) as src, open(dest.with_suffix(dest.suffix + ".part"), "wb") as o:
                        shutil.copyfileobj(src, o, 2**22)
                    dest.with_suffix(dest.suffix + ".part").rename(dest)
                out.append(dest)
    return out
