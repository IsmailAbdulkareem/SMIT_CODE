# 🚗 Pakistani Traffic Object Detection System

**Assignment 12 — SMIT AIDS 2026**  
Custom YOLOv8-powered traffic detection & analytics app built with Gradio.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 📷 **Image Detection** | Upload any image, detect & annotate Pakistani traffic objects |
| 🎬 **Video Detection** | Process video files with frame-by-frame detection |
| 🎯 **Object Tracking** | Multi-object tracking with persistent IDs via `model.track()` |
| 📊 **Analytics Charts** | Real-time Matplotlib bar charts for class distribution |
| 🖥️ **HUD Overlay** | On-frame info panel showing FPS, frame count, and object counts |
| 💾 **Download Outputs** | Save annotated images and processed videos |
| ⚙️ **Configurable** | Confidence & IoU sliders, model selector |

---

## 🚀 Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your model file in this directory
# (Pakistani_Trafic_V2.pt or Pakistan_Traffic_Model.pt)
# Or it will auto-download yolov8s.pt as fallback

# 3. Run the Gradio app
python app.py
```

Open browser at: **http://localhost:7860**

---

## 📁 Project Structure

```
Pakistani_Traffic_Detection_Gradio/
├── app.py                    # Main Gradio application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Pakistani_Trafic_V2.pt    # Custom model (place here)
├── Pakistan_Traffic_Model.pt # Fallback model (place here)
├── outputs/                  # Auto-created: processed videos
└── reports/                  # Auto-created: system logs
```

---

## 🛠️ Tech Stack

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — Object detection & tracking
- **[Gradio](https://gradio.app/)** — Interactive web UI
- **[OpenCV](https://opencv.org/)** — Frame annotation & video I/O
- **[Pandas](https://pandas.pydata.org/)** — Detection result tables
- **[Matplotlib](https://matplotlib.org/)** — Analytics charts

---

## 📖 Reference

Based on: [murtazakhanpro/Assignment_SMIT_AIDS_2026](https://github.com/murtazakhanpro/Assignment_SMIT_AIDS_2026/tree/main/Assignment_12_ObjectDetection_CV)
