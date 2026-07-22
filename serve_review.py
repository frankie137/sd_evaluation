#!/usr/bin/env python3
"""Serve out/curation/ with HTTP Range support for the review pages.

`python -m http.server` ignores Range requests, which breaks audio seeking in
browsers. Run this instead and open http://<host>:8000/<dataset>/review.html.
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = "/workspace/sd_evaluation/out/curation"


class RangeHandler(SimpleHTTPRequestHandler):
    # HTTP/1.1 keep-alive: proxies (e.g. VSCode port forwarding) handle
    # persistent connections better than HTTP/1.0 close-per-request.
    protocol_version = "HTTP/1.1"

    def send_head(self):
        path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d+)-(\d*)$", self.headers.get("Range") or "")
        if not (m and os.path.isfile(path)):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start >= size:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        # copyfile streams to EOF; wrap so only the requested span is sent.
        self._span_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_span_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        self._span_left = None
        while left > 0:
            buf = source.read(min(64 * 1024, left))
            if not buf:
                break
            outputfile.write(buf)
            left -= len(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    handler = partial(RangeHandler, directory=args.root)
    print(f"serving {args.root} at http://0.0.0.0:{args.port}/ "
          f"(open /<dataset>/review.html)")
    ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
