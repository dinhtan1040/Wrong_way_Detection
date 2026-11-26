"""Bước 7: Đánh giá & đối chiếu độ chính xác nhận diện (Detection) giữa các model từ VisDrone -> Finetuned.

Tự động chuẩn hóa tên lớp về nhóm chung (motor, car, bus, truck, bicycle, tricycle, person) để so sánh
CÔNG BẰNG ngay cả khi model VisDrone gốc (10 lớp) và các bản Finetuned có số lượng hoặc tên lớp khác nhau!

Ví dụ chạy đối chiếu 3 checkpoint trên tập test (hoặc val) trong dataset/data.yaml:
    python 07_eval_detector.py --data dataset/data.yaml --split test \
        --weights VisDrone=weights/visdrone_best.pt FT_v1=runs/ft_v1/weights/best.pt FT_v2=runs/ft_v2/weights/best.pt

Xuất ra:
  - Bảng so sánh tổng hợp (Vehicles mAP50, mAP50-95, F1 | Person AP50, F1 | All mAP50, Tốc độ ms/ảnh)
  - Bảng chi tiết AP50 và F1 theo từng loại phương tiện & người
  - File CSV tổng hợp (--out runs/eval_detector_compare.csv) để đưa vào báo cáo.
"""
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from target_classes import EVAL_ALL_GROUPS, EVAL_PERSON_GROUPS, EVAL_VEHICLE_GROUPS, eval_group_of

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 -> 0.95


def imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def resolve_split_images(data_yaml, split=None):
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(cfg.get("path", Path(data_yaml).parent))
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()

    if split is None:
        split = "test" if "test" in cfg else "val"
    if split not in cfg:
        raise SystemExit(f"Không tìm thấy mục '{split}' trong {data_yaml}. Các mục có sẵn: {list(cfg.keys())}")

    img_dir = Path(cfg[split])
    if not img_dir.is_absolute():
        img_dir = (root / img_dir).resolve()
    if not img_dir.exists():
        raise SystemExit(f"Không tìm thấy thư mục ảnh: {img_dir}")

    raw_names = cfg["names"]
    names = raw_names if isinstance(raw_names, dict) else {i: n for i, n in enumerate(raw_names)}
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"Không có ảnh nào trong {img_dir}")
    return split, imgs, names


def load_ground_truth(imgs, names):
    """Trả về gt_by_group[group][img_idx] = np.ndarray shape (N, 4) xyxy."""
    gt_by_group = {g: defaultdict(list) for g in EVAL_ALL_GROUPS}
    gt_counts = defaultdict(int)
    for idx, img_p in enumerate(imgs):
        im = imread_unicode(img_p)
        if im is None:
            continue
        H, W = im.shape[:2]
        # Tìm file nhãn tương ứng trong thư mục labels/
        parts = list(img_p.parts)
        if "images" in parts:
            pos = len(parts) - 1 - parts[::-1].index("images")
            parts[pos] = "labels"
            lbl_p = Path(*parts).with_suffix(".txt")
        else:
            lbl_p = img_p.with_suffix(".txt")
        if not lbl_p.exists():
            continue
        for ln in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
            v = ln.split()
            if len(v) < 5:
                continue
            cid = int(v[0])
            grp = eval_group_of(names.get(cid, ""))
            if grp is None:
                continue
            xc, yc, bw, bh = map(float, v[1:5])
            x1, y1 = (xc - bw / 2.0) * W, (yc - bh / 2.0) * H
            x2, y2 = (xc + bw / 2.0) * W, (yc + bh / 2.0) * H
            gt_by_group[grp][idx].append([x1, y1, x2, y2])
            gt_counts[grp] += 1

    for g in EVAL_ALL_GROUPS:
        for idx in list(gt_by_group[g].keys()):
            gt_by_group[g][idx] = np.asarray(gt_by_group[g][idx], dtype=np.float64)
    return gt_by_group, gt_counts


def box_iou_matrix(a, b):
    """a: (N, 4), b: (M, 4) -> (N, M)."""
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    iw = np.maximum(0.0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    ih = np.maximum(0.0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = iw * ih
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_ap_101(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0.0, 1.0, 101)
    return float(np.mean(np.interp(x, mrec, mpre)))


def eval_group_metrics(preds, gt_dict, n_gt, conf_op=0.25):
    """preds: list of (img_idx, conf, x1, y1, x2, y2).
    Trả về dict(ap50, ap50_95, prec, rec, f1, tp, fp, fn)."""
    if n_gt == 0:
        return None
    if not preds:
        return dict(ap50=0.0, ap50_95=0.0, prec=0.0, rec=0.0, f1=0.0, tp=0, fp=0, fn=n_gt)

    preds_sorted = sorted(preds, key=lambda x: -x[1])
    n_p = len(preds_sorted)
    aps = []

    prec_op = rec_op = f1_op = 0.0
    tp_op = fp_op = 0
    fn_op = n_gt

    for thr in IOU_THRESHOLDS:
        matched = {idx: np.zeros(len(boxes), dtype=bool) for idx, boxes in gt_dict.items()}
        tp = np.zeros(n_p, dtype=np.float64)
        fp = np.zeros(n_p, dtype=np.float64)
        for k, (img_idx, cf, x1, y1, x2, y2) in enumerate(preds_sorted):
            gboxes = gt_dict.get(img_idx)
            if gboxes is None or len(gboxes) == 0:
                fp[k] = 1.0
                continue
            ious = box_iou_matrix(np.array([[x1, y1, x2, y2]]), gboxes)[0]
            best_j = int(np.argmax(ious))
            if ious[best_j] >= thr and not matched[img_idx][best_j]:
                tp[k] = 1.0
                matched[img_idx][best_j] = True
            else:
                fp[k] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / n_gt
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        aps.append(compute_ap_101(rec, prec))

        if abs(thr - 0.50) < 1e-6:
            keep = [k for k, p in enumerate(preds_sorted) if p[1] >= conf_op]
            if keep:
                tp_op = int(cum_tp[keep[-1]])
                fp_op = int(cum_fp[keep[-1]])
                fn_op = n_gt - tp_op
                prec_op = tp_op / max(1, tp_op + fp_op)
                rec_op = tp_op / max(1, n_gt)
                f1_op = (2 * prec_op * rec_op / (prec_op + rec_op)) if (prec_op + rec_op) > 0 else 0.0

    return dict(
        ap50=aps[0],
        ap50_95=float(np.mean(aps)),
        prec=prec_op,
        rec=rec_op,
        f1=f1_op,
        tp=tp_op,
        fp=fp_op,
        fn=fn_op,
    )


def parse_model_specs(specs):
    out = []
    for item in specs:
        if "=" in item and not Path(item).exists():
            label, wpath = item.split("=", 1)
        else:
            p = Path(item)
            label = p.parent.parent.name if p.parent.name == "weights" else p.stem
            wpath = item
        out.append((label, wpath))
    return out


def evaluate_model(label, weights_path, imgs, gt_by_group, gt_counts, imgsz=960, conf_op=0.25, device=None):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    cls_to_grp = {int(i): eval_group_of(n) for i, n in model.names.items() if eval_group_of(n) is not None}
    if not cls_to_grp:
        raise SystemExit(f"Model {weights_path} không có lớp phương tiện hoặc người nào.")

    preds_by_group = defaultdict(list)
    t0 = time.time()
    for idx, img_p in enumerate(imgs):
        kw = dict(source=str(img_p), imgsz=imgsz, conf=0.001, iou=0.6,
                  classes=sorted(cls_to_grp.keys()), verbose=False)
        if device is not None:
            kw["device"] = device
        r = model.predict(**kw)[0]
        if r.boxes is not None and len(r.boxes.xyxy):
            for (x1, y1, x2, y2), c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(),
                                               r.boxes.conf.tolist()):
                grp = cls_to_grp.get(c)
                if grp is not None:
                    preds_by_group[grp].append((idx, float(cf), float(x1), float(y1), float(x2), float(y2)))
    ms_per_img = (time.time() - t0) * 1000.0 / max(1, len(imgs))

    group_res = {}
    for g in EVAL_ALL_GROUPS:
        if gt_counts[g] > 0:
            group_res[g] = eval_group_metrics(preds_by_group[g], gt_by_group[g], gt_counts[g], conf_op=conf_op)

    def macro(sub_groups, key):
        vals = [group_res[g][key] for g in sub_groups if g in group_res and group_res[g] is not None]
        return float(np.mean(vals)) if vals else 0.0

    summary = dict(
        model=label,
        weights=str(weights_path),
        veh_ap50=macro(EVAL_VEHICLE_GROUPS, "ap50"),
        veh_ap50_95=macro(EVAL_VEHICLE_GROUPS, "ap50_95"),
        veh_f1=macro(EVAL_VEHICLE_GROUPS, "f1"),
        per_ap50=macro(EVAL_PERSON_GROUPS, "ap50"),
        per_f1=macro(EVAL_PERSON_GROUPS, "f1"),
        all_ap50=macro(EVAL_ALL_GROUPS, "ap50"),
        all_ap50_95=macro(EVAL_ALL_GROUPS, "ap50_95"),
        all_prec=macro(EVAL_ALL_GROUPS, "prec"),
        all_rec=macro(EVAL_ALL_GROUPS, "rec"),
        all_f1=macro(EVAL_ALL_GROUPS, "f1"),
        ms_per_img=ms_per_img,
        groups=group_res,
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="đường dẫn dataset/data.yaml")
    ap.add_argument("--split", default=None, help="tập đánh giá: test hoặc val (mặc định: ưu tiên test)")
    ap.add_argument("--weights", nargs="+", required=True,
                    help="danh sách model cần đối chiếu, vd: VisDrone=visdrone.pt FT_v1=runs/ft_v1/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng conf tính Precision / Recall / F1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/eval_detector_compare.csv", help="file CSV lưu bảng kết quả đối chiếu")
    a = ap.parse_args()

    split, imgs, names = resolve_split_images(a.data, a.split)
    gt_by_group, gt_counts = load_ground_truth(imgs, names)
    active_groups = [g for g in EVAL_ALL_GROUPS if gt_counts[g] > 0]
    if not active_groups:
        raise SystemExit("Tập đánh giá không có nhãn phương tiện hoặc người nào.")

    print(f"Đánh giá trên tập '{split}': {len(imgs)} ảnh | Số lượng nhãn chuẩn theo nhóm: "
          + ", ".join(f"{g}={gt_counts[g]}" for g in active_groups))

    specs = parse_model_specs(a.weights)
    results = [evaluate_model(lbl, wp, imgs, gt_by_group, gt_counts, imgsz=a.imgsz, conf_op=a.conf, device=a.device)
               for lbl, wp in specs]

    print("\n=== BẢNG 1: ĐỐI CHIẾU TỔNG HỢP (VisDrone -> Finetuned) ===")
    hdr = f"{'Model':<18} | {'Veh mAP50':>9} | {'Veh 50-95':>9} | {'Veh F1':>7} | {'Per AP50':>8} | {'Per F1':>7} | {'All mAP50':>9} | {'All F1':>7} | {'ms/ảnh':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['model']:<18} | {r['veh_ap50']*100:8.2f}% | {r['veh_ap50_95']*100:8.2f}% | {r['veh_f1']*100:6.2f}% | "
              f"{r['per_ap50']*100:7.2f}% | {r['per_f1']*100:6.2f}% | {r['all_ap50']*100:8.2f}% | {r['all_f1']*100:6.2f}% | {r['ms_per_img']:7.1f}")

    print(f"\n=== BẢNG 2: CHI TIẾT TỪNG NHÓM PHƯƠNG TIỆN & NGƯỜI (AP@50 / F1 @conf={a.conf}) ===")
    col_w = 15
    hdr2 = f"{'Nhóm đối tượng':<14} | {'GT':>5} | " + " | ".join(f"{r['model']:^{col_w}}" for r in results)
    print(hdr2)
    print("-" * len(hdr2))
    for g in active_groups:
        cells = []
        for r in results:
            m = r["groups"].get(g)
            cells.append(f"{m['ap50']*100:5.1f}% / {m['f1']*100:4.1f}%" if m else f"{'N/A':^{col_w}}")
        print(f"{g:<14} | {gt_counts[g]:>5} | " + " | ".join(cells))

    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "weights", "split", "n_images", "veh_ap50", "veh_ap50_95", "veh_f1",
                  "per_ap50", "per_f1", "all_ap50", "all_ap50_95", "all_prec", "all_rec", "all_f1", "ms_per_img"]
    for g in active_groups:
        fieldnames.extend([f"{g}_ap50", f"{g}_f1"])

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: round(r[k], 4) if isinstance(r[k], float) else r[k]
                   for k in fieldnames if k in r}
            row["split"] = split
            row["n_images"] = len(imgs)
            for g in active_groups:
                m = r["groups"].get(g)
                row[f"{g}_ap50"] = round(m["ap50"], 4) if m else ""
                row[f"{g}_f1"] = round(m["f1"], 4) if m else ""
            w.writerow(row)
    print(f"\nĐã lưu bảng đối chiếu detector -> {out_p}")


if __name__ == "__main__":
    main()
