"""Camada multimodal 1 (100% offline, TAPI §6.2): vídeo + áudio → colunas para o GBM. Tudo roda em CPU.

Modelos carregados uma vez (`@cache`). Mesmas funções no treino e na inferência (PLANO regra 6).
"""

from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from parallax import config

T_GANCHO = (0.0, 0.5, 1.5, 2.5)  # segundos: o gancho decide se a pessoa continua assistindo
N_RESTO = 4
FACE_MODEL = config.DATA / "cache" / "blaze_face_short_range.tflite"
IMG_MODEL = "google/siglip2-base-patch16-224"
DET_MODEL = "PekingU/rtdetr_r18vd_coco_o365"
VEICULO, BOLSA = {"car", "truck", "bus", "motorcycle"}, {"handbag", "backpack", "suitcase"}


# ---------- modelos ----------
@cache
def _whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)  # tiny: 1,2 s vs 5 s do base; 4 threads para não disputar CPU (D23)


@cache
def _face():
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
                                   min_detection_confidence=0.5))


@cache
def _ocr():
    import easyocr
    return easyocr.Reader(["pt", "en"], gpu=False, verbose=False)


@cache
def _detector():
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    return AutoImageProcessor.from_pretrained(DET_MODEL), AutoModelForObjectDetection.from_pretrained(DET_MODEL).eval()


@cache
def _siglip():
    from transformers import AutoModel, AutoProcessor
    return AutoProcessor.from_pretrained(IMG_MODEL), AutoModel.from_pretrained(IMG_MODEL).eval()


def image_embeddings(imgs: list[np.ndarray]) -> np.ndarray:
    """SigLIP2, normalizado. imgs em RGB."""
    import torch
    proc, model = _siglip()
    with torch.no_grad():
        out = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
    e = (out.pooler_output if hasattr(out, "pooler_output") else out).numpy()  # transformers 5 devolve objeto
    return e / np.linalg.norm(e, axis=1, keepdims=True)


# ---------- vídeo ----------
def ler_frames(video: Path) -> tuple[list[np.ndarray], list[float], dict]:
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = n / fps
    ts = sorted({t for t in T_GANCHO if t < dur} | set(np.linspace(3.0, max(dur - 0.5, 3.0), N_RESTO).round(2)))
    frames, tempos = [], []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            tempos.append(t)
    h, w = (frames[0].shape[:2] if frames else (0, 0))
    meta = {"vd_dur_s": dur, "vd_w": w, "vd_h": h, "vd_fps": fps}
    # movimento: optical flow a 2 fps em escala reduzida
    mags, prev = [], None
    for t in np.arange(0, dur, 0.5):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (90, 160)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            mags.append(np.linalg.norm(flow, axis=2).mean())
        prev = g
    cap.release()
    meta.update(vd_motion_mean=float(np.mean(mags)) if mags else np.nan,
                vd_motion_std=float(np.std(mags)) if mags else np.nan)
    return frames, tempos, meta


def cor(frames: list[np.ndarray]) -> dict:
    hsv = [cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in frames]
    s = np.concatenate([h[..., 1].ravel() for h in hsv])
    hue = np.concatenate([h[..., 0][h[..., 1] > 60].ravel() for h in hsv])
    rg = [f[..., 0].astype(float) - f[..., 1] for f in frames]
    yb = [0.5 * (f[..., 0].astype(float) + f[..., 1]) - f[..., 2] for f in frames]
    return {
        "vd_brilho": float(np.mean([h[..., 2].mean() for h in hsv])),
        "vd_contraste": float(np.mean([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).std() for f in frames])),
        "vd_saturacao": float(s.mean()),
        "vd_colorfulness": float(np.mean([np.sqrt(a.std() ** 2 + b.std() ** 2) + 0.3 * np.sqrt(a.mean() ** 2 + b.mean() ** 2)
                                         for a, b in zip(rg, yb)])),
        "vd_hue_dom": int(np.bincount(hue // 15, minlength=12).argmax()) if len(hue) else -1,  # 12 faixas de 30°
    }


def cortes(video: Path, dur: float) -> dict:
    from scenedetect import ContentDetector, detect
    n = len(detect(str(video), ContentDetector(), show_progress=False))
    return {"vd_cortes": n, "vd_cortes_por_s": n / dur if dur else np.nan}


def rostos(frames: list[np.ndarray], tempos: list[float]) -> dict:
    import mediapipe as mp
    det = _face()
    n, area = [], []
    for f in frames:
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f))).detections
        n.append(len(r))
        area.append(max((d.bounding_box.width * d.bounding_box.height for d in r), default=0) / (f.shape[0] * f.shape[1]))
    gancho = [k for k, t in zip(n, tempos) if t < 3]
    return {"fc_max": max(n, default=0), "fc_frac_frames": float(np.mean([k > 0 for k in n])) if n else 0.0,
            "fc_area_max": float(max(area, default=0)), "fc_no_gancho": float(any(gancho))}


def texto_tela(frames: list[np.ndarray]) -> dict:
    chars = area = 0.0
    for f in frames:
        for caixa, txt, conf in _ocr().readtext(f):
            if conf < 0.4:
                continue
            chars += len(txt)
            area += cv2.contourArea(np.array(caixa, dtype=np.float32))
    tot = sum(f.shape[0] * f.shape[1] for f in frames) or 1
    return {"oc_chars": chars / max(len(frames), 1), "oc_area": area / tot, "oc_tem_texto": float(chars > 0)}


def objetos(frames: list[np.ndarray]) -> dict:
    import torch
    proc, model = _detector()
    with torch.no_grad():
        out = model(**proc(images=frames, return_tensors="pt"))
    res = proc.post_process_object_detection(out, threshold=0.5, target_sizes=[f.shape[:2] for f in frames])
    labels = [[model.config.id2label[int(i)] for i in r["labels"]] for r in res]
    return {"ob_pessoas": float(np.mean([l.count("person") for l in labels])),
            "ob_veiculo": float(any(set(l) & VEICULO for l in labels)),
            "ob_bolsa": float(any(set(l) & BOLSA for l in labels)),
            "ob_n": float(np.mean([len(l) for l in labels]))}


# ---------- áudio ----------
def _wav16k(path: Path) -> np.ndarray:
    """Áudio mono 16 kHz. Não usa faster_whisper.audio.decode_audio porque ela passa um argumento que o PyAV ≥15
    removeu (`metadata_errors`), e outra dependência exige PyAV ≥15."""
    import av
    rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
    with av.open(str(path)) as c:
        pcm = [f.to_ndarray() for fr in c.decode(audio=0) for f in rs.resample(fr)]
    pcm += [f.to_ndarray() for f in rs.resample(None)]
    return np.concatenate(pcm, axis=1).ravel().astype(np.float32) / 32768.0


def audio(path: Path, dur: float) -> tuple[dict, str]:
    wav = _wav16k(path)
    jan = wav[: len(wav) // 800 * 800].reshape(-1, 800)  # janelas de 50 ms
    rms = np.sqrt((jan ** 2).mean(axis=1)) + 1e-9
    espec = np.abs(np.fft.rfft(jan, axis=1)) + 1e-9
    flat = np.exp(np.log(espec).mean(axis=1)) / espec.mean(axis=1)  # planicidade espectral: ~0 tonal, ~1 ruído
    segs, info = _whisper().transcribe(wav, vad_filter=True, beam_size=1)
    segs = list(segs)
    txt = " ".join(s.text.strip() for s in segs)
    fala = sum(s.end - s.start for s in segs)
    return {
        "au_rms_mean": float(rms.mean()), "au_rms_std": float(rms.std()),
        "au_silencio": float((rms < 0.01).mean()), "au_flatness": float(flat.mean()),
        "au_fala_frac": fala / dur if dur else np.nan, "au_palavras": float(len(txt.split())),
        "au_palavras_por_s": len(txt.split()) / dur if dur else np.nan,
        "au_lang_pt": float(info.language == "pt"), "au_lang_en": float(info.language == "en"),
    }, txt


def extrair(video: Path, som: Path | None) -> tuple[dict, np.ndarray, str]:
    """Features escalares + embedding médio dos frames + transcrição de um vídeo baixado."""
    frames, tempos, f = ler_frames(video)
    if not frames:
        raise ValueError(f"nenhum frame legível em {video}")
    f |= cor(frames) | cortes(video, f["vd_dur_s"]) | rostos(frames, tempos)
    meio = [frames[1 if len(frames) > 1 else 0], frames[len(frames) // 2]]  # gancho + meio
    f |= texto_tela([frames[min(2, len(frames) - 1)]]) | objetos(meio)  # OCR só no gancho: é a etapa mais cara
    txt = ""
    if som is not None and som.exists():
        a, txt = audio(som, f["vd_dur_s"])
        f |= a
    return f, image_embeddings(frames).mean(axis=0), txt


def carregar(post_ids: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    """Features escalares (vd_/fc_/oc_/ob_/au_) e embedding médio dos frames; NaN onde não houve extração."""
    import json
    linhas, embs = [], []
    for v in post_ids:
        p = config.MEDIA / v
        if (p / "media.json").exists():
            d = json.loads((p / "media.json").read_text(encoding="utf-8"))
            d.pop("transcricao", None)
            linhas.append(d)
            embs.append(np.load(p / "frames_emb.npy"))
        else:
            linhas.append({})
            embs.append(np.full(768, np.nan, dtype=np.float32))
    return pd.DataFrame(linhas, index=post_ids.index), np.vstack(embs)


def atributos_do_criativo(creative: str) -> tuple[dict, np.ndarray, str] | None:
    """Camada 1 para um criativo novo (T1 da tool), com o MESMO `extrair` usado na base (paridade).

    Aceita URL do YouTube (baixa vídeo e áudio em 360p, como na base) ou caminho local de vídeo.
    Devolve None para imagem ou texto puro. A mídia baixada é apagada no fim.
    """
    import re
    import shutil
    import tempfile

    m = re.search(r"(?:shorts/|v=|youtu\.be/)([\w-]{11})", creative)
    if m:
        import yt_dlp
        pasta = Path(tempfile.mkdtemp(prefix="parallax_"))
        try:
            opts = {"format": "bv*,ba", "format_sort": ["res:360", "vcodec:h264", "abr"], "quiet": True,
                    "no_warnings": True, "noprogress": True, "outtmpl": "%(id)s.%(acodec.0:4)s.%(ext)s",
                    "paths": {"home": str(pasta)}}
            with yt_dlp.YoutubeDL(opts) as y:
                y.download([f"https://www.youtube.com/shorts/{m.group(1)}"])
            video = next(pasta.glob("*.none.*"), None)
            som = next(pasta.glob("*.mp4a.*"), None) or next(pasta.glob("*.opus.*"), None)
            return extrair(video, som) if video else None
        finally:
            shutil.rmtree(pasta, ignore_errors=True)
    p = Path(creative)
    if p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"} and p.exists():
        return extrair(p, p)  # o áudio sai do próprio arquivo
    return None
