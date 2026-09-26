"""
ml_server.py
============
Production ML Model Inference Server for Hoichoi Problem 2.
Provides REST API endpoints for:
  - GET  /health          : Status, device (CUDA/CPU), loaded models
  - POST /pipeline        : Run end-to-end subtitling, diarization, translation & QC
  - GET  /deliverables/:f : Fetch/download output files
  - POST /upload          : Upload custom video and process

Can be executed:
  1. Locally in Docker on port 8000:
     docker run -p 8000:8000 ... hoichoi-p2-full python ml_server.py
  2. In Google Colab with GPU + Ngrok/Localtunnel
"""

import sys
import os
import json
import time
import shutil
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = int(os.environ.get("PORT", os.environ.get("ML_PORT", 8000)))
DRIVE_ROOT = os.environ.get("DRIVE_ROOT", str(PROJECT_ROOT))
OUT_DIR = os.environ.get("OUT_DIR", str(PROJECT_ROOT / "deliverables"))

print("=" * 68)
print("  Hoichoi Problem 2 — ML Model Inference Server")
print(f"  Device     : {DEVICE}")
print(f"  Drive Root : {DRIVE_ROOT}")
print(f"  Out Dir    : {OUT_DIR}")
print(f"  Port       : {PORT}")
print("=" * 68)

# Lazy-load pipeline on demand to avoid memory spike (>512MB) and port scan delay on cloud free tiers
PIPELINE = None

def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        print("[ML Server] Lazy-loading BengaliSubtitlePipeline on first request...")
        from problem_2_caption_diarization.src.pipeline import BengaliSubtitlePipeline
        PIPELINE = BengaliSubtitlePipeline(
            drive_root=DRIVE_ROOT,
            output_dir=OUT_DIR,
            device=DEVICE,
            base_model_id="SayedShaun/bengali-whisper-medium",
            use_pyannote=False,
        )
        print("[ML Server] Pipeline loaded and ready.")
    return PIPELINE


class MLRequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/health" or path == "/api/health":
            resp = {
                "status": "online",
                "device": DEVICE,
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "asr_backbone": "SayedShaun/bengali-whisper-medium",
                "adapters": "TIES-Merged PolyWhisper (High+Mid+Base)",
                "vad": "Silero-VAD (snakers4/silero-vad)",
                "translation": "Helsinki-NLP/opus-mt-bn-en",
                "qc_engine": "10-point broadcast timed-text auditor",
            }
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if path.startswith("/deliverables/"):
            filename = os.path.basename(path)
            filepath = Path(OUT_DIR) / filename
            if not filepath.exists():
                filepath = PROJECT_ROOT / "docker_out" / filename

            if filepath.exists() and filepath.is_file():
                self.send_response(200)
                self._send_cors_headers()
                content_type = "text/vtt" if filename.endswith(".vtt") else ("application/x-subrip" if filename.endswith(".srt") else "application/json")
                self.send_header("Content-Type", content_type)
                self.end_headers()
                with open(filepath, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
            else:
                self.send_response(404)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(b"File not found")
                return

        self.send_response(404)
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/pipeline" or path == "/api/pipeline" or path == "/api/run-pipeline":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {e}"}).encode("utf-8"))
                return

            video_name = data.get("video_name") or data.get("videoName") or "mohanagar.mp4"
            max_duration = float(data.get("max_duration") or data.get("maxDuration") or 60.0)
            video_base64 = data.get("video_base64")

            # If video bytes were passed directly (e.g. from cloud web app upload)
            video_path = None
            if video_base64:
                import base64
                tmp_dir = Path(OUT_DIR) / "_tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                clean_name = Path(video_name).name
                video_path = tmp_dir / f"up_{int(time.time())}_{clean_name}"
                with open(video_path, "wb") as f:
                    f.write(base64.b64decode(video_base64))
                print(f"[ML Server] Decoded uploaded video payload: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")

            if not video_path or not video_path.exists():
                direct_path = Path(video_name)
                if direct_path.exists():
                    video_path = direct_path
                else:
                    candidates = [
                        PROJECT_ROOT / video_name,
                        PROJECT_ROOT / "web_app" / "uploads" / video_name,
                        PROJECT_ROOT / "uploads" / video_name,
                        Path("/video") / video_name,
                        Path("/video") / "uploads" / video_name,
                        Path("/app") / "uploads" / video_name,
                        PROJECT_ROOT / "web_app" / "public" / "previews" / video_name,
                    ]
                    for cand in candidates:
                        if cand.exists():
                            video_path = cand
                            break

            if not video_path or not video_path.exists():
                self.send_response(404)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Video file not found: {video_name}"}).encode("utf-8"))
                return

            print(f"[ML Server] Starting pipeline run for {video_path} (max_duration={max_duration}s)...")
            start_t = time.time()
            try:
                pipeline = get_pipeline()
                results = pipeline.run(
                    video_path=str(video_path),
                    max_duration_sec=max_duration,
                )
                elapsed = time.time() - start_t
                results["elapsed_seconds"] = round(elapsed, 2)

                # Read deliverable cue contents for instant web frontend consumption
                base_name = Path(video_name).stem
                vtt_path = Path(OUT_DIR) / f"{base_name}_bn_cc.vtt"
                en_path = Path(OUT_DIR) / f"{base_name}_en.srt"
                hi_path = Path(OUT_DIR) / f"{base_name}_hi.srt"
                qc_path = Path(OUT_DIR) / f"{base_name}_qc_report.json"

                if vtt_path.exists():
                    with open(vtt_path, "r", encoding="utf-8") as f:
                        results["vtt_bn_raw"] = f.read()
                if en_path.exists():
                    with open(en_path, "r", encoding="utf-8") as f:
                        results["srt_en_raw"] = f.read()
                if hi_path.exists():
                    with open(hi_path, "r", encoding="utf-8") as f:
                        results["srt_hi_raw"] = f.read()
                if qc_path.exists():
                    with open(qc_path, "r", encoding="utf-8") as f:
                        results["qc_report_json"] = json.load(f)

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(results, ensure_ascii=False).encode("utf-8"))
                print(f"[ML Server] Pipeline finished successfully in {elapsed:.1f}s.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Pipeline processing error: {e}"}).encode("utf-8"))
            return

        self.send_response(404)
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(b"Not Found")


def run_server():
    server_address = ("0.0.0.0", PORT)
    httpd = ThreadingHTTPServer(server_address, MLRequestHandler)
    print(f"\n🚀 ML Model Server listening on http://0.0.0.0:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ML Server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
