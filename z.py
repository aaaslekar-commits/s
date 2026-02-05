import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import font as tkfont
import traceback

import numpy as np
import sounddevice as sd
from sounddevice import PortAudioError
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

try:
    import torch  # type: ignore
except Exception:
    torch = None

TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 3
BLOCK_SIZE = 1024
SILENCE_RMS_THRESHOLD = 0.01

LANGUAGES = {
    "Auto (English/Persian)": None,
    "English": "en",
    "Persian (Farsi)": "fa",
}

DISPLAY_LANGUAGE_NAMES = {
    "en": "English",
    "fa": "Persian (Farsi)",
}

TARGET_BY_SOURCE = {
    "en": "fa",
    "fa": "en",
}

LOCAL_WHISPER_MODELS = [
    r"C:\whisper-models\turbo-ct2",
    r"C:\whisper-models\persian-v4-ct2",
]


def simple_resample(audio: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    if orig_sr == target_sr:
        return audio
    if audio.size < 2:
        return audio
    ratio = float(target_sr) / float(orig_sr)
    new_len = int(round(audio.shape[0] * ratio))
    if new_len <= 1:
        return audio[:1]
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def prettify_text(text: str) -> str:
    t = " ".join(text.split())
    for ch in ["؟", "!", "؛", ".", "…"]:
        t = t.replace(ch + " ", ch + "\n")
    return t.strip()


def validate_ct2_folder(path_str: str) -> tuple[bool, str]:
    p = Path(path_str)
    if not p.is_dir():
        return False, f"Folder not found: {p}"
    if not (p / "model.bin").exists():
        return False, f"Missing model.bin in: {p}"
    # ✅ important: ensure tokenizer is local to avoid HF hub
    if not (p / "tokenizer.json").exists():
        return False, (
            f"Missing tokenizer.json in: {p}\n\n"
            "Fix:\n"
            "1) Download tokenizer.json once (openai/whisper-tiny)\n"
            "2) Copy it into this model folder.\n"
            "Example:\n"
            "Copy-Item <cache>\\tokenizer.json "
            f"\"{p}\\tokenizer.json\" -Force"
        )
    return True, "OK"


class LiveWhisperApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Whisper Live Transcriber + EN↔FA Translator (LOCAL + GPU-ready)")
        self.root.geometry("1250x780")
        self.root.minsize(1100, 680)

        self.is_running = False
        self._ui_ready = False

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.ui_transcript_queue: queue.Queue[str] = queue.Queue()
        self.ui_translation_queue: queue.Queue[str] = queue.Queue()
        self.translate_work_queue: queue.Queue[tuple[str, str, str]] = queue.Queue()

        self.worker_thread = None
        self.translate_thread = None

        self.model_name_var = tk.StringVar(value=LOCAL_WHISPER_MODELS[0])
        self.language_var = tk.StringVar(value="Auto (English/Persian)")
        self.status_var = tk.StringVar(value="Idle")

        self.mic_var = tk.StringVar(value="")
        self.mic_map: dict[str, int] = {}

        self.device_mode_var = tk.StringVar(value="Auto (GPU if available)")
        self.compute_var = tk.StringVar(value="Auto")

        self.font_family_var = tk.StringVar(value="Segoe UI")
        self.font_size_var = tk.IntVar(value=13)

        self.whisper_model = None
        self.loaded_model_name = None
        self.translators: dict[tuple[str, str], GoogleTranslator] = {}

        self.input_device: int | None = None
        self.input_samplerate: float | None = None

        self._apply_theme()
        self._build_ui()
        self._refresh_mics()
        self._refresh_fonts()

        self._ui_ready = True
        self._apply_text_font()
        self._set_font_slider_without_trigger(self.font_size_var.get())

        self.root.after(80, self._drain_ui_queues)

    def _apply_theme(self):
        style = ttk.Style(self.root)
        try:
            if "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TCombobox", padding=4)
        style.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"))

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        top = ttk.Labelframe(container, text="Settings", padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Whisper model (local CT2):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.model_name_var,
            values=LOCAL_WHISPER_MODELS,
            width=60,
            state="readonly",
        ).grid(row=0, column=1, padx=8, pady=4, sticky="w")

        ttk.Label(top, text="Input language:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.language_var,
            values=list(LANGUAGES.keys()),
            width=22,
            state="readonly",
        ).grid(row=0, column=3, padx=8, pady=4, sticky="w")

        ttk.Label(top, text="Microphone:").grid(row=0, column=4, sticky="w")
        self.mic_combo = ttk.Combobox(top, textvariable=self.mic_var, width=38, state="readonly")
        self.mic_combo.grid(row=0, column=5, padx=8, pady=4, sticky="w")
        ttk.Button(top, text="Refresh", command=self._refresh_mics).grid(row=0, column=6, padx=4, pady=4)

        ttk.Label(top, text="Device:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.device_mode_var,
            values=["Auto (GPU if available)", "Force GPU (CUDA)", "Force CPU"],
            width=22,
            state="readonly",
        ).grid(row=1, column=1, padx=8, pady=4, sticky="w")

        ttk.Label(top, text="Compute:").grid(row=1, column=2, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.compute_var,
            values=["Auto", "float16 (GPU fast)", "int8_float16 (GPU low VRAM)", "int8 (CPU)"],
            width=26,
            state="readonly",
        ).grid(row=1, column=3, padx=8, pady=4, sticky="w")

        ttk.Label(top, text="Font:").grid(row=1, column=4, sticky="w")
        self.font_combo = ttk.Combobox(top, textvariable=self.font_family_var, values=[], width=22, state="readonly")
        self.font_combo.grid(row=1, column=5, padx=8, pady=4, sticky="w")
        ttk.Button(top, text="Apply", command=self._apply_text_font).grid(row=1, column=6, padx=4, pady=4)

        ttk.Label(top, text="Font size:").grid(row=2, column=0, sticky="w")
        self.font_scale = ttk.Scale(top, from_=10, to=26, orient=tk.HORIZONTAL, command=self._on_font_scale)
        self.font_scale.grid(row=2, column=1, padx=8, pady=6, sticky="we")
        self.font_size_label = ttk.Label(top, text=str(self.font_size_var.get()))
        self.font_size_label.grid(row=2, column=2, sticky="w")

        top.grid_columnconfigure(1, weight=1)

        btns = ttk.Frame(container, padding=(0, 10))
        btns.pack(fill="x")

        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        ttk.Button(btns, text="Clear", command=self.clear_output).pack(side="left", padx=5)
        ttk.Label(btns, textvariable=self.status_var).pack(side="right")

        body = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True)

        left_frame = ttk.Labelframe(body, text="Transcription", padding=8)
        right_frame = ttk.Labelframe(body, text="Translation (EN↔FA)", padding=8)

        self.transcription_box = ScrolledText(left_frame, wrap=tk.WORD, height=30, padx=8, pady=8)
        self.translation_box = ScrolledText(right_frame, wrap=tk.WORD, height=30, padx=8, pady=8)

        self.transcription_box.pack(fill="both", expand=True)
        self.translation_box.pack(fill="both", expand=True)

        body.add(left_frame, weight=1)
        body.add(right_frame, weight=1)

    def _set_font_slider_without_trigger(self, value: int):
        prev = self._ui_ready
        self._ui_ready = False
        try:
            self.font_scale.set(value)
        finally:
            self._ui_ready = prev

    def _on_font_scale(self, value):
        if not self._ui_ready:
            return
        v = int(float(value))
        self.font_size_var.set(v)
        self.font_size_label.config(text=str(v))
        self._apply_text_font()

    def _refresh_fonts(self):
        families = sorted(set(tkfont.families()))
        preferred = ["Segoe UI", "Tahoma", "Arial", "Calibri", "Consolas"]
        for p in preferred:
            if p in families:
                self.font_family_var.set(p)
                break
        self.font_combo["values"] = families

    def _apply_text_font(self):
        family = self.font_family_var.get() or "Segoe UI"
        size = int(self.font_size_var.get())
        f = (family, size)
        self.transcription_box.configure(font=f)
        self.translation_box.configure(font=f)

    def clear_output(self):
        self.transcription_box.delete("1.0", tk.END)
        self.translation_box.delete("1.0", tk.END)

    def set_status(self, message: str):
        self.root.after(0, lambda: self.status_var.set(message))

    def _drain_ui_queues(self):
        try:
            while True:
                line = self.ui_transcript_queue.get_nowait()
                self.transcription_box.insert(tk.END, line + "\n\n")
                self.transcription_box.see(tk.END)
        except queue.Empty:
            pass
        try:
            while True:
                line = self.ui_translation_queue.get_nowait()
                self.translation_box.insert(tk.END, line + "\n\n")
                self.translation_box.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_ui_queues)

    def _refresh_mics(self):
        self.mic_map.clear()
        devices = sd.query_devices()
        labels = []
        for i, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) > 0:
                name = dev.get("name", f"Input #{i}")
                label = f"{i}: {name}"
                self.mic_map[label] = i
                labels.append(label)
        self.mic_combo["values"] = labels
        if labels and self.mic_var.get() not in self.mic_map:
            self.mic_var.set(labels[0])

    def _resolve_input_device(self) -> int:
        selected = self.mic_var.get().strip()
        if selected in self.mic_map:
            return self.mic_map[selected]
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) > 0:
                return i
        raise RuntimeError("No input audio devices found.")

    def _gpu_available(self) -> bool:
        if torch is None:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _pick_whisper_backend(self) -> tuple[str, str]:
        device_mode = self.device_mode_var.get()
        compute_mode = self.compute_var.get()

        if device_mode == "Force CPU":
            device = "cpu"
        elif device_mode == "Force GPU (CUDA)":
            device = "cuda"
        else:
            device = "cuda" if self._gpu_available() else "cpu"

        if compute_mode == "float16 (GPU fast)":
            compute = "float16"
        elif compute_mode == "int8_float16 (GPU low VRAM)":
            compute = "int8_float16"
        elif compute_mode == "int8 (CPU)":
            compute = "int8"
        else:
            compute = "float16" if device == "cuda" else "int8"

        if device == "cpu" and compute in {"float16", "int8_float16"}:
            compute = "int8"
        return device, compute

    def _load_model(self):
        selected = self.model_name_var.get().strip()
        ok, msg = validate_ct2_folder(selected)
        if not ok:
            raise RuntimeError(msg)

        device, compute = self._pick_whisper_backend()

        if self.whisper_model is None or self.loaded_model_name != selected:
            self.loaded_model_name = selected
            self.set_status(f"Loading LOCAL model: {selected} | {device} ({compute}) ...")
            self.whisper_model = WhisperModel(
                selected,
                device=device,
                compute_type=compute,
                local_files_only=True,
            )

    def _get_translator(self, src: str, dst: str) -> GoogleTranslator:
        key = (src, dst)
        if key not in self.translators:
            self.translators[key] = GoogleTranslator(source=src, target=dst)
        return self.translators[key]

    def _normalize_lang(self, lang: str | None) -> str | None:
        if not lang:
            return None
        lang = lang.lower().strip()
        if "-" in lang:
            lang = lang.split("-", 1)[0]
        return lang

    def _choose_translation_target(self, source_lang: str | None) -> str | None:
        if not source_lang:
            return None
        return TARGET_BY_SOURCE.get(source_lang)

    def _run_translator_worker(self):
        while self.is_running:
            try:
                ts, src, text = self.translate_work_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            dst = self._choose_translation_target(src)
            if not dst:
                continue
            try:
                translator = self._get_translator(src, dst)
                translated = translator.translate(text)
                self.ui_translation_queue.put(f"[{ts}] ({src} → {dst})\n{prettify_text(translated)}")
            except Exception as exc:
                self.ui_translation_queue.put(f"[{ts}] ({src} → {dst}) [translation failed]\n{exc}")

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_running:
            self.audio_queue.put(indata.copy())

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.clear_output()

        for q in (self.audio_queue, self.ui_transcript_queue, self.ui_translation_queue, self.translate_work_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self.translate_thread = threading.Thread(target=self._run_translator_worker, daemon=True)
        self.worker_thread.start()
        self.translate_thread.start()

    def stop(self):
        self.is_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.set_status("Stopped")

    def _run_worker(self):
        try:
            self._load_model()

            self.input_device = self._resolve_input_device()
            dev = sd.query_devices(self.input_device)
            self.input_samplerate = float(dev.get("default_samplerate", 44100))
            mic_name = dev.get("name", f"#{self.input_device}")

            device, compute = self._pick_whisper_backend()
            self.set_status(f"Listening: {mic_name} | Whisper: {device} ({compute})")

            chunk_samples_in = int(round(self.input_samplerate * CHUNK_SECONDS))
            rolling = np.empty((0, CHANNELS), dtype=np.float32)

            with sd.InputStream(
                device=self.input_device,
                channels=CHANNELS,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=BLOCK_SIZE,
                samplerate=self.input_samplerate,
            ):
                while self.is_running:
                    try:
                        data = self.audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    rolling = np.concatenate([rolling, data], axis=0)
                    if len(rolling) < chunk_samples_in:
                        continue

                    audio_in = rolling[:chunk_samples_in, 0]
                    rolling = rolling[chunk_samples_in:]

                    if float(np.sqrt(np.mean(audio_in ** 2))) < SILENCE_RMS_THRESHOLD:
                        continue

                    audio = simple_resample(audio_in, self.input_samplerate, TARGET_SAMPLE_RATE)

                    preferred_lang = LANGUAGES[self.language_var.get()]
                    segments, info = self.whisper_model.transcribe(
                        audio,
                        language=preferred_lang,
                        beam_size=1,
                        vad_filter=True,
                    )

                    text = " ".join(seg.text.strip() for seg in segments).strip()
                    if not text:
                        continue

                    text = prettify_text(text)

                    detected = self._normalize_lang(getattr(info, "language", None))
                    preferred_norm = self._normalize_lang(preferred_lang)
                    lang = detected or preferred_norm or "unknown"

                    ts = datetime.now().strftime("%H:%M:%S")
                    display_lang = DISPLAY_LANGUAGE_NAMES.get(lang, lang)
                    self.ui_transcript_queue.put(f"[{ts}] ({display_lang})\n{text}")

                    if lang in {"en", "fa"}:
                        self.translate_work_queue.put((ts, lang, text))

        except PortAudioError as exc:
            self.set_status(f"Audio device error: {exc}")
            self.ui_transcript_queue.put("[AUDIO ERROR]\n" + str(exc))
        except Exception as exc:
            self.set_status(f"Error: {exc}")
            self.ui_transcript_queue.put("[ERROR DETAILS]\n" + traceback.format_exc())
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_btn.config(state="normal"))
            self.root.after(0, lambda: self.stop_btn.config(state="disabled"))


def main():
    root = tk.Tk()
    LiveWhisperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
