import webview
import threading
from pathlib import Path

def launch_gui(port: int = 5000, block: bool = True):
    """
    Open the desktop GUI window.
 
    Parameters:
    port: Port your FastAPI server is running on (default 5000).
    block: bool  If True (default) this call blocks until the window closes.
    Pass False to open the window in a background thread.
    """
    file_url = (Path(__file__).parent / "index.html").resolve().as_uri()
    def _open():
        window = webview.create_window(
            title="RAG Coding Assistant",
            url=file_url,
            width=1080,
            height=700,
            min_size=(720, 480),
            frameless=False,
            easy_drag=False,
        )
        webview.start(debug=False, http_server=True)
 
    if block:
        _open()
    else:
        t = threading.Thread(target=_open, daemon=True)
        t.start()
 
if __name__ == "__main__":
    launch_gui()