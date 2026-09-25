import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        # Suppress spamming terminal with 10 requests per second
        pass

if __name__ == "__main__":
    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
    os.chdir(web_dir)
    server = HTTPServer(("0.0.0.0", 8000), NoCacheHandler)
    print("Web dashboard server running on http://127.0.0.1:8000 (serving web/)")
    server.serve_forever()
