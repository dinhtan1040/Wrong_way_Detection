"""Bước 4: Gom ảnh + nhãn đã duyệt thành bộ dataset chuẩn YOLO, CHIA THEO VIDEO (không chia ngẫu nhiên theo khung)
và sinh file cấu hình dataset/data.yaml để huấn luyện (fine-tune) và đánh giá (07_eval_detector.py).

Chạy:
    python 04_build_dataset.py --frames data/frames --out dataset --weights visdrone_best.pt \
        --val-videos Normal_2 --test-videos Wrong_way_1

- test-videos: video giữ lại hoàn toàn làm tập kiểm thử cuối (dùng trong 07_eval_detector.py).
- val-videos : dùng chọn checkpoint khi train. Bỏ trống -> tự lấy mỗi 5 video 1 video.
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--weights", default="yolo11m.pt", help="model dùng ở 02_auto_label.py (lấy bảng tên lớp)")
    ap.add_argument("--val-videos", nargs="*", default=[])
    ap.add_argument("--test-videos", nargs="*", default=[])
    a = ap.parse_args()

    root, out = Path(a.frames), Path(a.out)
    stems = sorted(d.name for d in root.iterdir() if d.is_dir())
    test = set(a.test_videos)
    val = set(a.val_videos)
    overlap = test & val
    if overlap:
        raise SystemExit(f"Video không được trùng giữa --val-videos và --test-videos: {sorted(overlap)}")

    pool = [s for s in stems if s not in test]
    if not val:
        if len(pool) < 2:
            raise SystemExit(f"Cần ít nhất 2 thư mục video cho train/val (hiện có {len(pool)}: {pool}).")
        val = set(pool[1::5] or [pool[-1]])

    train_stems = [s for s in stems if s not in val and s not in test]
    if not train_stems or not val:
        raise SystemExit("Tập train hoặc val đang rỗng. Kiểm tra lại --val-videos / --test-videos.")

    counts = {"train": 0, "val": 0, "test": 0}
    for stem in stems:
        split = "test" if stem in test else "val" if stem in val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted((root / stem).glob("*.jpg")):
            shutil.copy2(img, out / "images" / split / img.name)
            lab = img.with_suffix(".txt")
            if lab.exists():
                shutil.copy2(lab, out / "labels" / split / lab.name)
            counts[split] += 1

    names = YOLO(a.weights).names
    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": dict(names)}
    if counts["test"]:
        data["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(counts, "->", out / "data.yaml")
    print("Train:", train_stems)
    print("Val  :", sorted(val))
    print("Test :", sorted(test))


if __name__ == "__main__":
    main()
