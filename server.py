#!/usr/bin/env python3
"""HTTP server with Cross-Origin-Isolation headers for SharedArrayBuffer support."""
import http.server
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8100

class COOPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

print(f"Serving on http://localhost:{PORT} with Cross-Origin-Isolation headers")
http.server.HTTPServer(("", PORT), COOPHandler).serve_forever()
