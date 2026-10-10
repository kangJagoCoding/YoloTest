
import torch
from ultralytics import YOLO

print("CUDA aktif:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolo26n-pose.pt")

results = model.predict(
    source=0,
    show=True,
    conf=0.5,
    stream=True,
    device=0 if torch.cuda.is_available() else "cpu"
)

for result in results:
    keypoints = getattr(result, "keypoints", None)

    if keypoints is None:
        continue

    points = getattr(keypoints, "xy", None)

    if points is None:
        continue

    if hasattr(points, "cpu"):
        points = points.cpu().numpy()

    for person_index, person in enumerate(points):
        if len(person) < 7:
            continue

        nose = person[0].tolist()
        left_shoulder = person[5].tolist()
        right_shoulder = person[6].tolist()

        print(f"Person {person_index + 1}")
        print("Hidung:", nose)
        print("Bahu kiri:", left_shoulder)
        print("Bahu kanan:", right_shoulder)
