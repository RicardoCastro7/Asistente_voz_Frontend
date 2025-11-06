# audio_core.py
import json, time, io, wave, threading, queue
from pathlib import Path
import requests
import numpy as np
import sounddevice as sd
import webrtcvad
import vosk
from faster_whisper import WhisperModel
from piper import PiperVoice

from event import log, update_state, show_transcript, show_answer

# ========== CONFIG ==========
PIPER_MODEL_PATH = "C:/Users/r-ica/OneDrive/Desktop/PRUEBAS/piper_voices/es_MX-claude-high.onnx"
API_URL = "http://192.168.100.37:7890/rag"
WAKE_WORD = "alexa"
VOSK_MODEL_PATH = r"C:/Users/r-ica/OneDrive/Desktop/PRUEBAS/vosk-model-small-es-0.42"

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20
VAD_AGGRESSIVENESS = 2
START_SPEECH_FRAMES = 8
END_SILENCE_FRAMES = 25
MAX_UTTER_SILENCE = 8.0

MODEL_NAME = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LANG = "es"

TMP_WAV = Path("utter_tmp.wav")

STREAM_ACTIVE = True
audio_q: queue.Queue | None = None

voice = PiperVoice.load(PIPER_MODEL_PATH)
# ============================


def clear_audio_queue(qobj: queue.Queue):
    try:
        while not qobj.empty():
            qobj.get_nowait()
    except queue.Empty:
        pass


def tts_say(text: str):
    if not text:
        return

    def _worker(msg: str):
        global STREAM_ACTIVE, audio_q
        try:
            STREAM_ACTIVE = False
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_out:
                voice.synthesize_wav(msg, wav_out)
            buf.seek(0)
            with wave.open(buf, "rb") as wav_in:
                nframes = wav_in.getnframes()
                framerate = wav_in.getframerate()
                nchannels = wav_in.getnchannels()
                sampwidth = wav_in.getsampwidth()
                frames = wav_in.readframes(nframes)

            dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
            if sampwidth not in dtype_map:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio = np.frombuffer(frames, dtype=dtype_map[sampwidth])
                if sampwidth == 1:
                    audio = (audio.astype(np.float32) - 128) / 128.0
                elif sampwidth == 2:
                    audio = audio.astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    audio = audio.astype(np.float32) / 2147483648.0

            if nchannels > 1:
                audio = audio.reshape(-1, nchannels)

            sd.play(audio, framerate)
            sd.wait()
        finally:
            time.sleep(0.8)
            if audio_q is not None:
                clear_audio_queue(audio_q)
            STREAM_ACTIVE = True

    threading.Thread(target=_worker, args=(text,), daemon=True).start()


def float32_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()


def pcm16_to_wav(pcm_bytes: bytes, path: Path, sample_rate=SAMPLE_RATE, channels=1):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def create_vosk_recognizer():
    model = vosk.Model(VOSK_MODEL_PATH)
    grammar = f'["{WAKE_WORD}", "[unk]"]'
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE, grammar)
    rec.SetWords(False)
    return rec


def send_to_api(query_text: str):
    """
    query_text: lo que ya transcribiste con Whisper.
    MOSTRAMOS solo la respuesta de la API, no la pregunta que viene en el JSON
    para evitar el duplicado en el chat.
    """
    try:
        r = requests.get(API_URL, params={"q": query_text}, timeout=15)
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            data = r.text

        log(f"Respuesta API cruda: {data!r}")

        # --- caso JSON {pregunta: ..., respuesta: ...} ---
        if isinstance(data, dict):
            # la API te la manda, pero NO la volvemos a mostrar
            # pregunta_api = (data.get("pregunta") or "").strip()

            respuesta = (
                data.get("respuesta")
                or data.get("answer")
                or data.get("message")
                or ""
            ).strip()

            if not respuesta:
                respuesta = "La API no devolvió respuesta."

            # aquí sí mostramos SOLO la respuesta
            show_answer(respuesta)
            tts_say(respuesta)
            return respuesta

        # --- caso texto plano ---
        elif isinstance(data, str):
            respuesta = data.strip()
            show_answer(respuesta)
            tts_say(respuesta)
            return respuesta

        # --- caso raro ---
        else:
            respuesta = str(data)
            show_answer(respuesta)
            tts_say(respuesta)
            return respuesta

    except requests.RequestException as e:
        msg = f"Error al contactar la API: {e}"
        log(msg)
        tts_say("Hubo un error al contactar la API.")
        return None


def audio_worker():
    log("Cargando modelos…")
    rec = create_vosk_recognizer()
    whisper = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    global audio_q, STREAM_ACTIVE
    audio_q = queue.Queue()
    frame_len = int(SAMPLE_RATE * (FRAME_MS / 1000.0))

    def audio_cb(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    update_state("Esperando wake word: 'alexa'")
    state = "waiting"
    ring = [0] * END_SILENCE_FRAMES
    voice_frames = 0
    utter_pcm = bytearray()
    last_voice_t = time.time()

    import sounddevice as sd
    with sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        blocksize=frame_len,
        callback=audio_cb,
        latency="low",
    ):
        while True:
            try:
                buf = audio_q.get(timeout=2.0)
            except queue.Empty:
                continue

            if not STREAM_ACTIVE:
                continue

            mono = buf[:, 0]
            pcm16 = float32_to_pcm16(mono)

            if state == "waiting":
                if rec.AcceptWaveform(pcm16):
                    res = json.loads(rec.Result())
                    text = (res.get("text") or "").strip().lower()
                    if WAKE_WORD in text.split():
                        log("Wake word detectada (final).")
                        update_state("Grabando…")
                        state = "recording"
                        utter_pcm.clear()
                        ring = [0] * END_SILENCE_FRAMES
                        voice_frames = 0
                        last_voice_t = time.time()
                else:
                    part = json.loads(rec.PartialResult()).get("partial", "").lower()
                    if WAKE_WORD in part.split():
                        log("Wake word detectada (parcial).")
                        update_state("Grabando…")
                        state = "recording"
                        utter_pcm.clear()
                        ring = [0] * END_SILENCE_FRAMES
                        voice_frames = 0
                        last_voice_t = time.time()

            elif state == "recording":
                is_speech = vad.is_speech(pcm16, SAMPLE_RATE)
                ring = (ring + [1 if is_speech else 0])[-END_SILENCE_FRAMES:]

                if voice_frames < START_SPEECH_FRAMES:
                    if is_speech:
                        voice_frames += 1
                    utter_pcm.extend(pcm16)
                    continue

                utter_pcm.extend(pcm16)
                if is_speech:
                    last_voice_t = time.time()

                no_voice_window = all(v == 0 for v in ring)
                if no_voice_window or (time.time() - last_voice_t) > MAX_UTTER_SILENCE:
                    update_state("Transcribiendo…")
                    log("Silencio detectado, transcribiendo…")
                    pcm16_to_wav(bytes(utter_pcm), TMP_WAV)
                    try:
                        segments, info = whisper.transcribe(str(TMP_WAV), language=LANG, task="transcribe")
                        text = " ".join(s.text.strip() for s in segments if s.text.strip())
                        show_transcript(text)
                       # log(f"Transcripción: {text}")
                        if text.strip():
                            send_to_api(text)
                    except Exception as e:
                        log(f"Error transcribiendo: {e}")
                    finally:
                        try:
                            TMP_WAV.unlink(missing_ok=True)
                        except Exception:
                            pass

                    update_state("Esperando wake word: 'ALEXA'")
                    rec = create_vosk_recognizer()
                    state = "waiting"
