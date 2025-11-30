# app.py
from flask import Flask, Response, render_template
import threading

from event import sse_stream
from audio_core import audio_worker

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stream")
def stream():
    return Response(sse_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    th = threading.Thread(target=audio_worker, daemon=True)
    th.start()
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
