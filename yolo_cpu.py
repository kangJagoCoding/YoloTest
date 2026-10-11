import time
from typing import cast

import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

IMGSZ = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE != "cpu"

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# warm-up supaya frame awal tidak lambat
model.predict(
    torch.zeros(480, 640, 3).numpy().astype("uint8"),
    imgsz=IMGSZ, device=DEVICE, half=USE_HALF, verbose=False,
)

fps_avg = 0.0
prev = time.perf_counter()

while True:
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
            device=DEVICE,
            half=USE_HALF,
            verbose=False,
        ),
    )[0]

    annotated = result.plot()

    # FPS dirata-rata (EMA) supaya angkanya tidak loncat-loncat
    now = time.perf_counter()
    fps = 1 / (now - prev)
    prev = now
    fps_avg = fps if fps_avg == 0 else 0.9 * fps_avg + 0.1 * fps

    infer_ms = result.speed["inference"]
    label = "GPU" if DEVICE != "cpu" else "CPU"
    cv2.putText(annotated, f"{label} | FPS: {fps_avg:.1f} | infer: {infer_ms:.1f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO26", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()