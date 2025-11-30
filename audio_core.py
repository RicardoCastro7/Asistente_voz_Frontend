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
PIPER_MODEL_PATH = "C:/Users/r-ica/OneDrive/Desktop/frontend/Asistente_voz/piper_voices/es_MX-claude-high.onnx"
API_URL = "http://192.168.100.37:5000/rag"
WAKE_WORD = "alexa"
VOSK_MODEL_PATH = r"C:/Users/r-ica/OneDrive/Desktop/frontend/Asistente_voz/vosk-model-small-es-0.42"

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20

# Más agresivo para reducir ruido como voz
VAD_AGGRESSIVENESS = 3

START_SPEECH_FRAMES = 8
END_SILENCE_FRAMES = 25
MAX_UTTER_SILENCE = 8.0

# Tiempo mínimo de escucha después de detectar la wake word
MIN_RECORDING_TIME = 4.0

MODEL_NAME = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LANG = "es"

TMP_WAV = Path("utter_tmp.wav")

STREAM_ACTIVE = True
audio_q: queue.Queue | None = None

# Flags para TTS
TTS_PLAYING = False

voice = PiperVoice.load(PIPER_MODEL_PATH)
# ============================


def clear_audio_queue(qobj: queue.Queue):
    try:
        while not qobj.empty():
            qobj.get_nowait()
    except queue.Empty:
        pass


def tts_say(text: str):
    """
    Reproduce el texto con Piper.
    Mientras habla, se PAUSA la captura de audio (STREAM_ACTIVE = False)
    para evitar que el TTS dispare la wake word.
    """
    if not text:
        return

    def _worker(msg: str):
        global audio_q, TTS_PLAYING, STREAM_ACTIVE

        try:
            TTS_PLAYING = True

            # Pausar captura de audio
            STREAM_ACTIVE = False

            # Sintetizar a WAV en memoria
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

            # Aseguramos forma (frames, channels)
            if nchannels > 1:
                audio = audio.reshape(-1, nchannels)
            else:
                audio = audio.reshape(-1, 1)

            # Reproducimos de una
            sd.play(audio, framerate)
            sd.wait()

        finally:
            TTS_PLAYING = False
            # Reanudar escucha
            STREAM_ACTIVE = True
            time.sleep(0.2)
            if audio_q is not None:
                clear_audio_queue(audio_q)


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
            respuesta = (
                data.get("respuesta")
                or data.get("answer")
                or data.get("message")
                or ""
            ).strip()

            if not respuesta:
                respuesta = "La API no devolvió respuesta."

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

    global audio_q, STREAM_ACTIVE, TTS_PLAYING
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
    recording_start_t = None

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

            # Si el stream está "pausado" (TTS), no procesar
            if not STREAM_ACTIVE:
                continue

            mono = buf[:, 0]
            pcm16 = float32_to_pcm16(mono)

            # ==================== ESTADO: WAITING (wake word) ====================
            if state == "waiting":
                # Si está hablando (por seguridad extra), no buscamos wake word
                if TTS_PLAYING:
                    continue

                # Solo usamos el resultado FINAL de Vosk
                if rec.AcceptWaveform(pcm16):
                    res = json.loads(rec.Result())
                    text = (res.get("text") or "").strip().lower()
                    log(f"Vosk final (wake): {text!r}")

                    # Solo activamos si es EXACTAMENTE "alexa"
                    if text == WAKE_WORD:
                        log("Wake word detectada (final, match exacto).")

                        update_state("Grabando…")
                        state = "recording"
                        utter_pcm.clear()
                        ring = [0] * END_SILENCE_FRAMES
                        voice_frames = 0
                        last_voice_t = time.time()
                        recording_start_t = time.time()
                # No usamos PartialResult para evitar falsos positivos

            # ==================== ESTADO: RECORDING (captura pregunta) ====================
            elif state == "recording":
                # VAD para detectar voz/silencio
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
                elapsed_recording = (
                    time.time() - recording_start_t if recording_start_t else 0.0
                )

                # Solo cerramos si:
                # - hay silencio o se pasó el límite de utterance
                # - y ya escuchamos al menos MIN_RECORDING_TIME
                if (no_voice_window or (time.time() - last_voice_t) > MAX_UTTER_SILENCE) and (
                    elapsed_recording >= MIN_RECORDING_TIME
                ):
                    update_state("Transcribiendo…")
                    log("Silencio detectado, evaluando utterance…")

                    # Si casi no hubo voz real, no enviamos nada
                    if voice_frames < START_SPEECH_FRAMES:
                        log(
                            "No se detectó voz suficiente tras la wake word. "
                            "Se cierra sin enviar nada a la API."
                        )
                    else:
                        # ---- FILTRO POR DURACIÓN (para evitar alucinaciones) ----
                        duration_sec = len(utter_pcm) / (SAMPLE_RATE * 2)
                        log(f"Duración utterance: {duration_sec:.2f} s")

                        if duration_sec < 0.7:
                            log("Utterance demasiado corta, se ignora (posible ruido/silencio).")
                        else:
                            pcm16_to_wav(bytes(utter_pcm), TMP_WAV)
                            try:
                                segments, info = whisper.transcribe(
                                    str(TMP_WAV), language=LANG, task="transcribe"
                                )
                                text = " ".join(
                                    s.text.strip() for s in segments if s.text.strip()
                                )
                                show_transcript(text)

                                # ---- FILTRO POR LONGITUD DE TEXTO ----
                                if text.strip() and len(text.split()) >= 3:
                                    log(f"Transcripción válida: {text!r}")
                                    send_to_api(text)
                                else:
                                    log(f"Texto muy corto/ruidoso, se ignora: {text!r}")

                            except Exception as e:
                                log(f"Error transcribiendo: {e}")
                            finally:
                                try:
                                    TMP_WAV.unlink(missing_ok=True)
                                except Exception:
                                    pass

                    # Resetear para esperar nueva wake word
                    update_state("Esperando wake word: 'alexa'")
                    rec = create_vosk_recognizer()
                    state = "waiting"
                    utter_pcm.clear()
                    ring = [0] * END_SILENCE_FRAMES
                    voice_frames = 0
                    last_voice_t = time.time()
                    recording_start_t = None
