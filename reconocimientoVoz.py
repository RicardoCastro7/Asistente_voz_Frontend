import json
import queue
import time
import io, wave
from pathlib import Path
import requests
from piper import PiperVoice
import numpy as np
import sounddevice as sd
import webrtcvad
import vosk
from faster_whisper import WhisperModel

# ========= AÑADIDOS =========
import threading

# flag global: si es False, ignoramos frames del micrófono (mute durante TTS)
STREAM_ACTIVE = True

# haremos que tts_say limpie la cola 'q' tras hablar; la declaramos global
q: queue.Queue | None = None

def clear_audio_queue(qobj: queue.Queue):
    """Vacía la cola de audio para descartar frames residuales (eco del TTS)."""
    try:
        while not qobj.empty():
            qobj.get_nowait()
    except queue.Empty:
        pass
# ============================


# Configura la ruta de la voz
PIPER_MODEL_PATH ="C:/Users/r-ica/OneDrive/Desktop/PRUEBAS/piper_voices/es_MX-claude-high.onnx"
voice = PiperVoice.load(PIPER_MODEL_PATH)

API_URL = "http://192.168.100.37:7890/rag"

# =================== CONFIG ===================
WAKE_WORD = "alexa"  # en minúsculas
VOSK_MODEL_PATH = r"C:/Users/r-ica/OneDrive/Desktop/PRUEBAS/vosk-model-small-es-0.42"  # <-- AJUSTA ESTA RUTA

# Audio / VAD
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20                 # 10/20/30 ms para VAD
VAD_AGGRESSIVENESS = 2        # 0..3 (3 más agresivo)
START_SPEECH_FRAMES = 8       # ~160 ms de voz para arrancar
END_SILENCE_FRAMES = 25       # ~500 ms de silencio para cortar
MAX_UTTER_SILENCE = 8.0       # corte por inactividad (seguridad)

# Transcripción
MODEL_NAME = "small"          # tiny/base/small/medium/large
DEVICE = "cpu"                # "cpu" o "cuda"
COMPUTE_TYPE = "int8"         # "int8" (CPU) | "float16" (GPU)
LANG = "es"

TMP_WAV = Path("utter_tmp.wav")
# =============================================


def send_to_api(query_text: str):
    try:
        r = requests.get(API_URL, params={"q": query_text}, timeout=15)
        r.raise_for_status()

        # Intenta JSON, si no, usa texto
        try:
            data = r.json()
        except ValueError:
            data = r.text

        print(f"🤖 Respuesta API cruda: {data!r}\n")

        if isinstance(data, str):
            respuesta = data.strip()
        elif isinstance(data, dict):
            respuesta = (data.get("respuesta")
                         or data.get("answer")
                         or data.get("message")
                         or str(data))
        else:
            respuesta = str(data)

        if not respuesta:
            respuesta = "La API respondió vacío."

        print(f"📝 Texto a decir: {respuesta}\n")
        tts_say(respuesta)      # 🗣️ Reproducir con Piper
        return respuesta

    except requests.RequestException as e:
        msg = f"Hubo un error al contactar la API: {e}"
        print(f"⚠️ {msg}")
        tts_say("Hubo un error al contactar la API.")
        return None


def tts_say(text: str):
    """Sintetiza con Piper a un WAV en memoria y lo reproduce con sounddevice.
       Se ejecuta en un hilo, silenciando el micro y limpiando la cola al final.
    """
    if not text:
        return

    def _worker(msg: str):
        global STREAM_ACTIVE, q
        try:
            STREAM_ACTIVE = False   # << mute micrófono para evitar eco

            # 1) Sintetiza a un buffer WAV en memoria
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_out:
                # Esto escribe cabeceras WAV (sample rate correcto del modelo, canales, etc.)
                voice.synthesize_wav(msg, wav_out)

            # 2) Vuelve a abrir el WAV para leer parámetros y frames
            buf.seek(0)
            with wave.open(buf, "rb") as wav_in:
                nframes      = wav_in.getnframes()
                framerate    = wav_in.getframerate()   # **Usa el sample rate real del modelo**
                nchannels    = wav_in.getnchannels()
                sampwidth    = wav_in.getsampwidth()   # bytes por muestra (normalmente 2 = int16)
                frames       = wav_in.readframes(nframes)

            # 3) Convierte a ndarray con el dtype correcto -> float32
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

            # 4) Reproduce (bloquea dentro del hilo)
            sd.play(audio, framerate)
            sd.wait()
        finally:
            # 5) Espera un poco para que se disipe la reverberación/eco del TTS
            time.sleep(0.8)  # si aún re-transcribe, sube a 1.0 s

            # 6) Limpia cualquier frame residual que haya quedado en la cola
            try:
                if q is not None:
                    clear_audio_queue(q)
            except Exception:
                pass

            # 7) Reactiva el micrófono
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
    # Gramática mínima para acelerar detección del wake word
    grammar = f'["{WAKE_WORD}", "[unk]"]'
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE, grammar)
    rec.SetWords(False)
    return rec

def main():
    print("Cargando modelos…")
    rec = create_vosk_recognizer()
    whisper = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    global q
    q = queue.Queue()  # << hacemos 'q' accesible a tts_say para poder limpiarla
    frame_len = int(SAMPLE_RATE * (FRAME_MS / 1000.0))  # muestras por frame

    def audio_cb(indata, frames, time_info, status):
        if status:
            # Puedes imprimir status para depurar
            pass
        q.put(indata.copy())

    print("⏳ Esperando wake word: 'Alexa' (Ctrl+C para salir)")
    state = "waiting"          # waiting | recording
    ring = [0] * END_SILENCE_FRAMES
    voice_frames = 0
    utter_pcm = bytearray()
    last_voice_t = time.time()

    with sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        blocksize=frame_len,
        callback=audio_cb,
        latency="low",
    ):
        try:
            while True:
                try:
                    buf = q.get(timeout=2.0)  # np.float32 shape (N,1)
                except queue.Empty:
                    continue

                # << Si el TTS está hablando, ignoramos frames para no
                #    transcribir la respuesta del propio sistema.
                if not STREAM_ACTIVE:
                    continue

                mono = buf[:, 0]
                pcm16 = float32_to_pcm16(mono)

                if state == "waiting":
                    # Alimenta Vosk para detectar "alexa"
                    if rec.AcceptWaveform(pcm16):
                        res = json.loads(rec.Result())
                        text = (res.get("text") or "").strip().lower()
                        if WAKE_WORD in text.split():
                            print("✅ Wake word detectada: ALEXA")
                            # reinicia buffers
                            state = "recording"
                            utter_pcm.clear()
                            ring = [0] * END_SILENCE_FRAMES
                            voice_frames = 0
                            last_voice_t = time.time()
                    else:
                        # también podemos mirar parciales
                        part = json.loads(rec.PartialResult()).get("partial", "").lower()
                        if WAKE_WORD in part.split():
                            print("✅ Wake word detectada (parcial): ALEXA")
                            state = "recording"
                            utter_pcm.clear()
                            ring = [0] * END_SILENCE_FRAMES
                            voice_frames = 0
                            last_voice_t = time.time()

                elif state == "recording":
                    # VAD por frames exactos
                    is_speech = vad.is_speech(pcm16, SAMPLE_RATE)
                    ring = (ring + [1 if is_speech else 0])[-END_SILENCE_FRAMES:]

                    # Arranque: espera unos frames de voz reales
                    if voice_frames < START_SPEECH_FRAMES:
                        if is_speech:
                            voice_frames += 1
                        # guarda audio igual para no perder inicio
                        utter_pcm.extend(pcm16)
                        continue

                    # Grabación activa
                    utter_pcm.extend(pcm16)
                    if is_speech:
                        last_voice_t = time.time()

                    # Silencio sostenido o timeout
                    no_voice_window = all(v == 0 for v in ring)
                    if no_voice_window or (time.time() - last_voice_t) > MAX_UTTER_SILENCE:
                        print("🟡 Silencio detectado, transcribiendo…")
                        # Transcribir con Whisper
                        pcm16_to_wav(bytes(utter_pcm), TMP_WAV)
                        try:
                            segments, info = whisper.transcribe(str(TMP_WAV), language=LANG, task="transcribe")
                            text = " ".join(s.text.strip() for s in segments if s.text.strip())
                            print(f"📝 Transcripción: {text}\n")
                            if text and text.strip():
                                send_to_api(text)
                        except Exception as e:
                            print(f"Error transcribiendo: {e}")
                        finally:
                            try:
                                TMP_WAV.unlink(missing_ok=True)
                            except Exception:
                                pass

                        print("🔁 Di “Alexa” para otra transcripción.")
                        # Volver a esperar wake word (reiniciar recognizer)
                        rec = create_vosk_recognizer()
                        state = "waiting"
        except KeyboardInterrupt:
            print("\nSaliendo…")

if __name__ == "__main__":
    main()