import json
from pathlib import Path

from app.image_processing import load_image
from app.detector import BoxDetector

# ==========================================================
# Initialize
# ==========================================================

detector = BoxDetector()

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

image_paths = [
    "images/Picture1.png",
    "images/Picture2.png",
    "images/Picture3.png",
    "images/Picture4.png"
]

all_results = []

# ==========================================================
# Process Images
# ==========================================================

for image_path in image_paths:

    print("=" * 60)
    print(f"Processing : {image_path}")
    print("=" * 60)

    image = load_image(image_path)

    summary = detector.detect(image)

    # tampilkan di terminal
    print(json.dumps(summary, indent=4))

    # simpan ke list
    all_results.append({

        "image": Path(image_path).name,

        "summary": summary

    })

# ==========================================================
# Save JSON
# ==========================================================

output_file = output_dir / "results.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=4,
        ensure_ascii=False
    )

print("=" * 60)
print(f"Semua hasil berhasil disimpan ke:\n{output_file}")
print("=" * 60)