# events_bus.py
import json
import queue

# cola global de eventos para el frontend
events_q: "queue.Queue[str]" = queue.Queue()

def push_event(kind: str, payload: str):
    data = json.dumps({"type": kind, "data": payload})
    events_q.put(data)

def sse_stream():
    while True:
        msg = events_q.get()
        yield f"data: {msg}\n\n"

def log(msg: str):
    print(msg)
    push_event("log", msg)

def update_state(msg: str):
    push_event("state", msg)

def show_transcript(msg: str):
    push_event("transcript", msg)

def show_answer(msg: str):
    push_event("answer", msg)
