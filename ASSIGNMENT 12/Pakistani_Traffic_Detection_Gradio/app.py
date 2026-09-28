"""
=============================================================================
 🚗 Pakistani Traffic Object Detection & Analytics System
-----------------------------------------------------------------------------
 Custom Project — Assignment 12 | SMIT AIDS 2026
 Built with: Ultralytics YOLOv8, Gradio Blocks, OpenCV, Pandas, Matplotlib
 Model: Pakistani_Trafic_V2.pt (custom-trained)
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Prevent OpenMP collision on Windows

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import gradio as gr
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
MODEL_PATHS = [
    PROJECT_ROOT / "Pakistani_Trafic_V2.pt",
    PROJECT_ROOT / "Pakistan_Traffic_Model.pt",
]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(REPORTS_DIR / "system.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("PakTraffic")

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
PALETTE = [
    (255,  59,  48), ( 52, 199,  89), (  0, 122, 255), (255, 149,   0),
    (175,  82, 222), (255,  45,  85), ( 90, 200, 250), (255, 214,  10),
    ( 48, 209,  88), (100, 210, 255), (255, 100, 130), (200, 100, 220),
    (  0, 180, 150), (255, 180,   0), (140, 100, 255), (255,  70,  70),
]

def get_color(class_id: int) -> Tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]

# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------
_model_cache: Dict[str, YOLO] = {}

def load_model(model_path: Optional[str] = None) -> YOLO:
    if model_path and model_path != "auto":
        target = Path(model_path)
    else:
        target = None
        for p in MODEL_PATHS:
            if p.exists():
                target = p
                break
        if target is None:
            logger.warning("Custom .pt not found — falling back to yolov8s.")
            target = Path("yolov8s.pt")

    key = str(target)
    if key not in _model_cache:
        logger.info(f"Loading model: {key}")
        _model_cache[key] = YOLO(key)
        logger.info(f"Model loaded. Classes: {list(_model_cache[key].names.values())[:10]}")
    return _model_cache[key]

try:
    DEFAULT_MODEL = load_model()
    logger.info("Default model ready.")
except Exception as e:
    DEFAULT_MODEL = None
    logger.error(f"Startup model load failed: {e}")

# ---------------------------------------------------------------------------
# Frame Annotation
# ---------------------------------------------------------------------------
def draw_fancy_box(frame, x1, y1, x2, y2, label, color, conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
    for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (sx, sy), (sx + dx * corner_len, sy), color, 3)
        cv2.line(frame, (sx, sy), (sx, sy + dy * corner_len), color, 3)
    text = f"{label}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    pad = 4
    bx1, by1 = x1, max(0, y1 - th - 2 * pad)
    bx2, by2 = x1 + tw + 2 * pad, y1
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, counts, total, fps, frame_no):
    panel_w = 240
    lines = [f"FPS  {fps:.1f}", f"Frame {frame_no}", f"Objects {total}", ""] + \
            [f"  {cls:<18}{cnt}" for cls, cnt in sorted(counts.items())]
    row_h = 22
    panel_h = max(80, len(lines) * row_h + 20)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 120), 1)
    for i, line in enumerate(lines):
        if i == 3:
            continue
        color = (120, 200, 255) if i < 3 else (200, 200, 200)
        cv2.putText(frame, line, (18, 10 + (i + 1) * row_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame

# ---------------------------------------------------------------------------
# Analytics Chart
# ---------------------------------------------------------------------------
def build_bar_chart(counts: dict, title: str = "Detected Objects") -> np.ndarray:
    if not counts:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0d1117")
        ax.text(0.5, 0.5, "No objects detected", ha="center", va="center",
                color="white", fontsize=13, transform=ax.transAxes)
        ax.set_facecolor("#0d1117"); ax.axis("off")
    else:
        classes = list(counts.keys())
        values  = list(counts.values())
        colors  = [f"#{PALETTE[i%len(PALETTE)][0]:02x}{PALETTE[i%len(PALETTE)][1]:02x}{PALETTE[i%len(PALETTE)][2]:02x}"
                   for i in range(len(classes))]
        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.9), 4), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        bars = ax.bar(classes, values, color=colors, edgecolor="#30363d", linewidth=0.8, zorder=3)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Count", color="#8b949e", fontsize=10)
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.spines[:].set_color("#30363d")
        ax.yaxis.grid(True, color="#30363d", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(val), ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
        plt.xticks(rotation=30, ha="right", color="#c9d1d9")
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf

# ---------------------------------------------------------------------------
# Core Detection Functions
# ---------------------------------------------------------------------------
def run_detection_image(pil_image, conf_thresh, iou_thresh, model_choice):
    if pil_image is None:
        return None, pd.DataFrame(), build_bar_chart({}), "Upload an image first."
    try:
        model = load_model(model_choice)
    except Exception as e:
        return None, pd.DataFrame(), build_bar_chart({}), f"Model error: {e}"

    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    results = model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
    elapsed = time.perf_counter() - t0

    counts, rows = {}, []
    for r in results:
        names = r.names
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw_fancy_box(frame, x1, y1, x2, y2, name, get_color(cls_id), conf)
            counts[name] = counts.get(name, 0) + 1
            rows.append({"Class": name, "Confidence": f"{conf:.2%}",
                         "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                         "Width": x2-x1, "Height": y2-y1})

    total = sum(counts.values())
    annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Class","Confidence","X1","Y1","X2","Y2","Width","Height"])
    summary = (
        f"**Detection complete** in `{elapsed:.3f}s`\n\n"
        f"- **Total detected:** {total}\n"
        f"- **Unique classes:** {len(counts)}\n"
        + "".join(f"- **{k}:** {v}\n" for k,v in sorted(counts.items()))
    )
    return annotated, df, build_bar_chart(counts, "Detection Results"), summary


def run_detection_video(video_path, conf_thresh, iou_thresh, model_choice, enable_tracking, max_frames):
    if not video_path:
        return None, pd.DataFrame(), build_bar_chart({}), "Upload a video first."
    try:
        model = load_model(model_choice)
    except Exception as e:
        return None, pd.DataFrame(), build_bar_chart({}), f"Model error: {e}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, pd.DataFrame(), build_bar_chart({}), "Could not open video."

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = str(OUTPUTS_DIR / f"output_{datetime.now():%Y%m%d_%H%M%S}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), orig_fps, (orig_w, orig_h))

    global_counts, total_detections, frame_no = {}, 0, 0
    fps_buf: List[float] = []
    process_limit = min(int(max_frames), total_frames) if max_frames > 0 else total_frames

    while frame_no < process_limit:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        if enable_tracking:
            results = model.track(source=frame, conf=conf_thresh, iou=iou_thresh, persist=True, verbose=False)
        else:
            results = model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
        elapsed = time.perf_counter() - t0
        fps_buf.append(1.0 / max(elapsed, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)

        frame_counts: Dict[str, int] = {}
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_fancy_box(frame, x1, y1, x2, y2, name, get_color(cls_id), conf)
                frame_counts[name] = frame_counts.get(name, 0) + 1
                global_counts[name] = global_counts.get(name, 0) + 1
                total_detections += 1

        draw_hud(frame, frame_counts, sum(frame_counts.values()), sum(fps_buf)/len(fps_buf), frame_no)
        writer.write(frame)
        frame_no += 1

    cap.release()
    writer.release()

    df_rows = [{"Class": k, "Total Detections": v, "% of Total": f"{v/max(total_detections,1):.1%}"}
               for k, v in sorted(global_counts.items(), key=lambda x: -x[1])]
    df = pd.DataFrame(df_rows) if df_rows else pd.DataFrame()
    summary = (
        f"**Video processing complete!**\n\n"
        f"- **Frames processed:** {frame_no}\n"
        f"- **Total detections:** {total_detections}\n"
        f"- **Unique classes:** {len(global_counts)}\n"
        + "".join(f"- **{k}:** {v}\n" for k,v in sorted(global_counts.items(), key=lambda x:-x[1]))
    )
    return out_path, df, build_bar_chart(global_counts, "Overall Detection Summary"), summary

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
body, .gradio-container {
    background: #0d1117 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #c9d1d9 !important;
}
.header-banner {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #161b22 100%) !important;
    border: 1px solid #30363d !important;
    border-radius: 16px !important;
    padding: 32px 40px !important;
    margin-bottom: 24px !important;
}
.stat-badge {
    display: inline-block !important;
    background: rgba(88,166,255,0.1) !important;
    border: 1px solid rgba(88,166,255,0.3) !important;
    border-radius: 20px !important;
    padding: 4px 14px !important;
    font-size: 12px !important;
    color: #58a6ff !important;
    font-weight: 600 !important;
    margin: 4px !important;
}
.section-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    color: #58a6ff !important;
    margin-bottom: 8px !important;
}
.card {
    background: rgba(22,27,34,0.85) !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.tab-nav button { background: transparent !important; color: #8b949e !important; }
.tab-nav button.selected { color: #58a6ff !important; }
button.primary {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; letter-spacing: 0.3px !important;
}
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Pakistani Traffic Object Detection",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    ) as demo:

        gr.HTML("""
        <div class="header-banner">
            <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
                <div style="font-size:52px;line-height:1;">🚗</div>
                <div>
                    <div style="font-size:26px;font-weight:800;color:#f0f6fc;letter-spacing:-0.5px;">
                        Pakistani Traffic Detection System
                    </div>
                    <div style="font-size:14px;color:#8b949e;margin-top:6px;">
                        Real-time Object Detection &amp; Analytics · YOLOv8 · OpenCV · Gradio
                    </div>
                    <div style="margin-top:12px;">
                        <span class="stat-badge">⚡ YOLOv8 Powered</span>
                        <span class="stat-badge">📷 Image &amp; Video</span>
                        <span class="stat-badge">🎯 Object Tracking</span>
                        <span class="stat-badge">📊 Live Analytics</span>
                        <span class="stat-badge">🚀 SMIT AIDS 2026</span>
                    </div>
                </div>
            </div>
        </div>
        """)

        with gr.Accordion("⚙️  Detection Settings", open=True):
            with gr.Row():
                conf_slider = gr.Slider(0.1, 0.95, value=0.35, step=0.05,
                                        label="🎯 Confidence Threshold")
                iou_slider  = gr.Slider(0.1, 0.95, value=0.45, step=0.05,
                                        label="📐 IoU Threshold (NMS)")
                available_models = ["auto"] + [str(p) for p in MODEL_PATHS if p.exists()]
                model_choice = gr.Dropdown(choices=available_models, value="auto",
                                           label="🤖 Model")

        with gr.Tabs():

            # IMAGE TAB
            with gr.Tab("🖼️  Image Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-label">Input</div>')
                        img_input = gr.Image(type="pil", label="Upload Image",
                                             height=300, sources=["upload","clipboard"])
                        detect_img_btn = gr.Button("🔍 Detect Objects", variant="primary")
                        img_summary = gr.Markdown("_Upload an image and click Detect._")

                    with gr.Column(scale=2):
                        gr.HTML('<div class="section-label">Annotated Output</div>')
                        img_output = gr.Image(type="numpy", label="Result",
                                              height=350, show_download_button=True)
                        gr.HTML('<div class="section-label" style="margin-top:12px;">Distribution Chart</div>')
                        img_chart  = gr.Image(type="numpy", label="Chart",
                                              height=220, show_download_button=True)

                gr.HTML('<div class="section-label" style="margin-top:12px;">Detections Table</div>')
                img_table = gr.Dataframe(
                    headers=["Class","Confidence","X1","Y1","X2","Y2","Width","Height"],
                    interactive=False, wrap=True)

                detect_img_btn.click(
                    fn=run_detection_image,
                    inputs=[img_input, conf_slider, iou_slider, model_choice],
                    outputs=[img_output, img_table, img_chart, img_summary],
                )

            # VIDEO TAB
            with gr.Tab("🎬  Video Detection & Tracking"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-label">Input</div>')
                        vid_input = gr.Video(label="Upload Video", height=280)
                        with gr.Row():
                            enable_tracking = gr.Checkbox(value=True, label="🎯 Enable Tracking")
                            max_frames_slider = gr.Slider(50, 1000, value=300, step=50,
                                                          label="⏱️ Max Frames")
                        detect_vid_btn = gr.Button("▶️  Process Video", variant="primary")
                        vid_summary = gr.Markdown("_Upload a video and click Process._")

                    with gr.Column(scale=2):
                        gr.HTML('<div class="section-label">Processed Output</div>')
                        vid_output = gr.Video(label="Output Video",
                                              height=340, show_download_button=True)
                        gr.HTML('<div class="section-label" style="margin-top:12px;">Stats Chart</div>')
                        vid_chart  = gr.Image(type="numpy", label="Detection Distribution",
                                              height=220, show_download_button=True)

                gr.HTML('<div class="section-label" style="margin-top:12px;">Class Summary Table</div>')
                vid_table = gr.Dataframe(
                    headers=["Class","Total Detections","% of Total"],
                    interactive=False, wrap=True)

                detect_vid_btn.click(
                    fn=run_detection_video,
                    inputs=[vid_input, conf_slider, iou_slider, model_choice,
                            enable_tracking, max_frames_slider],
                    outputs=[vid_output, vid_table, vid_chart, vid_summary],
                )

            # INFO TAB
            with gr.Tab("ℹ️  About"):
                gr.HTML("""
                <div style="max-width:760px;margin:0 auto;">
                  <div class="card" style="margin-bottom:20px;">
                    <h2 style="color:#58a6ff;margin-top:0;">🚗 Pakistani Traffic Object Detection</h2>
                    <p style="color:#8b949e;line-height:1.7;">
                      A custom YOLOv8 computer vision system trained on Pakistani road traffic.
                      Detects and classifies vehicles and road entities common in Pakistan.
                    </p>
                    <h3 style="color:#f0f6fc;">📦 Stack</h3>
                    <ul style="color:#8b949e;line-height:2;">
                      <li><strong style="color:#c9d1d9;">YOLOv8</strong> — Ultralytics detection</li>
                      <li><strong style="color:#c9d1d9;">Gradio Blocks</strong> — Web UI</li>
                      <li><strong style="color:#c9d1d9;">OpenCV</strong> — Frame annotation &amp; video I/O</li>
                      <li><strong style="color:#c9d1d9;">Pandas + Matplotlib</strong> — Analytics</li>
                    </ul>
                  </div>
                  <div class="card">
                    <h3 style="color:#f0f6fc;margin-top:0;">🎓 Assignment 12 · SMIT AIDS 2026</h3>
                    <p style="color:#8b949e;line-height:1.7;">
                      Custom project built on top of the reference by
                      <strong style="color:#58a6ff;">murtazakhanpro</strong>.
                      Extended with custom Gradio Blocks UI, object tracking, HUD overlay,
                      analytics charts, and downloadable outputs.
                    </p>
                    <div style="margin-top:16px;padding:12px 16px;background:#161b22;
                                border-left:3px solid #58a6ff;border-radius:4px;">
                      <code style="color:#79c0ff;font-size:13px;">
                        git clone https://github.com/murtazakhanpro/Assignment_SMIT_AIDS_2026
                      </code>
                    </div>
                  </div>
                </div>
                """)

        gr.HTML("""
        <div style="text-align:center;color:#484f58;font-size:12px;margin-top:32px;
                    padding:16px;border-top:1px solid #21262d;">
            Pakistani Traffic Detection System · SMIT AIDS 2026 Assignment 12 ·
            Built with YOLOv8 &amp; Gradio
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860,
                share=False, inbrowser=True, show_error=True)
