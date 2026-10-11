import time
from typing import cast

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

IMGSZ = 640
TARGET_FPS = 30
FRAME_TIME = 1 / TARGET_FPS

if not torch.cuda.is_available():
    raise SystemExit("CUDA tidak aktif, pakai versi CPU.")

print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Warm-up
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
for _ in range(3):
    model.predict(dummy, imgsz=IMGSZ, device=0, half=True, verbose=False)


def wait_until(deadline: float) -> None:
    """Tunggu sampai deadline: sleep kasar, lalu spin untuk presisi."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.003:
            time.sleep(remaining - 0.002)


fps_avg = 0.0
max_avg = 0.0
prev_start = time.perf_counter()

while True:
    loop_start = time.perf_counter()

    ok, frame = cap.read()
    if not ok:
        break

    result = cast(
        list[Results],
        model.predict(
            frame,
            classes=[0],
            conf=0.5,
            imgsz=IMGSZ,
            device=0,
            half=True,
            verbose=False,
        ),
    )[0]

    annotated = result.plot()

    # Hitung kemampuan maksimal (sebelum ditahan pembatas)
    work_time = time.perf_counter() - loop_start
    max_fps = 1 / work_time if work_time > 0 else 0.0
    max_avg = max_fps if max_avg == 0 else 0.9 * max_avg + 0.1 * max_fps

    # FPS aktual (sudah termasuk pembatas)
    fps = 1 / (loop_start - prev_start) if loop_start != prev_start else 0.0
    prev_start = loop_start
    fps_avg = fps if fps_avg == 0 else 0.9 * fps_avg + 0.1 * fps

    cv2.putText(
        annotated,
        f"GPU | FPS: {fps_avg:.1f} | MAX: {max_avg:.0f} | infer: {result.speed['inference']:.1f} ms",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    cv2.imshow("YOLO26 - GPU (30 FPS)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Tahan sampai genap 1/30 detik
    wait_until(loop_start + FRAME_TIME)

cap.release()
cv2.destroyAllWindows()