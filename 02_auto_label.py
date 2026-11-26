"""Bước 2: Dán nhãn nháp tự động (.txt) bằng YOLO cho toàn bộ ảnh trong data/frames.
Chỉ ghi nhãn 2 nhóm mục tiêu: PHƯƠNG TIỆN (đa loại hình) và NGƯỜI (bỏ qua mọi lớp nền khác).

Chạy (nên dùng model VisDrone gốc để bộ lớp khớp sẵn trước khi fine-tune):
    python 02_auto_label.py --frames data/frames --weights visdrone_best.pt

Mặc định BỎ QUA ảnh đã có file .txt để không ghi đè nhãn đã sửa tay (thêm --overwrite nếu muốn ghi đè).
QUY ƯỚC GÁN NHÃN:
  - Xe máy + người ngồi trên xe = MỘT khung duy nhất, nhãn motor/motorcycle. Không vẽ khung người riêng.
  - Người đi bộ / đứng riêng lẻ = khung person/pedestrian (giữ trong tập train để model phân biệt người vs xe).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from target_classes import vehicle_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-person", action="store_true",
                    help="không ghi lớp người vào nhãn nháp (mặc định: ghi cả phương tiện đa loại hình & người)")
    ap.add_argument("--overwrite", action="store_true",
                    help="ghi đè lên file .txt đã tồn tại (mặc định bỏ qua để bảo vệ nhãn đã sửa tay)")
    a = ap.parse_args()

    model = YOLO(a.weights)
    ids = vehicle_ids(model.names, include_person=not a.no_person)
    print("Lớp ghi vào nhãn nháp (phương tiện & người):", [f"{i}:{model.names[i]}" for i in ids])
    root = Path(a.frames)
    (root / "classes.txt").write_text("\n".join(model.names[i] for i in range(len(model.names))), encoding="utf-8")
    imgs = sorted(root.rglob("*.jpg"))
    print(f"Tìm thấy {len(imgs)} ảnh")
    written = skipped = 0
    for n, img in enumerate(imgs, 1):
        txt = img.with_suffix(".txt")
        if txt.exists() and not a.overwrite:
            skipped += 1
            continue
        r = model.predict(str(img), imgsz=a.imgsz, conf=a.conf, classes=ids, verbose=False)[0]
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
        txt.write_text("\n".join(lines), encoding="utf-8")
        written += 1
        if n % 100 == 0:
            print(f"  {n}/{len(imgs)}")
    print(f"Xong: đã ghi {written} file nhãn, bỏ qua {skipped} file đã tồn tại.")


if __name__ == "__main__":
    main()
