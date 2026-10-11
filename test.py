import time

import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(0)

fps = 0.0
prev = time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    t0 = time.perf_counter()
    result = model(frame, verbose=False)[0]
    latency = (time.perf_counter() - t0) * 1000  # ms: dari frame masuk sampai hasil jadi

    annotated = result.plot()

    # FPS dirata-rata supaya angkanya tidak loncat-loncat
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 * (1 / (now - prev)) if fps else 1 / (now - prev)
    prev = now

    cv2.putText(annotated, f"FPS: {fps:.1f} | Latency: {latency:.1f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO26", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()