"""Isolated HTTP worker; input and output are bounded and errors are redacted.

The parent kills this process at its elapsed subprocess timeout, including while
urllib waits for DNS, headers or a trickled body. Credentials arrive on stdin.
"""
from __future__ import annotations

import base64
import json
import sys
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect rejected")


def main():
    raw = sys.stdin.buffer.read(524289)
    if len(raw) > 524288:
        raise ValueError("worker input too large")
    data = json.loads(raw)
    body = base64.b64decode(data["body_base64"], validate=True)
    request = Request(data["endpoint"], data=body, headers=data["headers"], method="POST")
    with build_opener(NoRedirect()).open(request, timeout=data["timeout_seconds"]) as response:
        if response.status != 200 or response.headers.get_content_type() != "application/json":
            raise ValueError("invalid HTTP response")
        payload = response.read(data["max_response_bytes"] + 1)
        if len(payload) > data["max_response_bytes"]:
            raise ValueError("worker response too large")
        sys.stdout.buffer.write(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # HTTP exception strings and tracebacks can contain endpoint credentials.
        raise SystemExit(2) from None
