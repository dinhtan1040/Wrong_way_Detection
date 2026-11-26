"""Bước 8: Đánh giá & đối chiếu hiệu quả bắt lỗi ngược chiều (cấp sự kiện) giữa các model từ VisDrone -> Finetuned.

Đối chiếu các file <ten_video>_events.csv (do 06_detect_wrong_way.py tạo ra) với bảng sự kiện thực tế (ground_truth.csv)
để tính TP, FP, FN, Precision, Recall, F1, tỷ lệ báo nhầm theo giờ (FP/giờ) và độ trễ phát hiện trung bình (giây).

1. Tạo nhanh file mẫu ground_truth.csv từ thư mục video:
     python 08_eval_wrong_way.py --make-template ground_truth.csv --video Video

2. Đối chiếu nhiều lần chạy từ VisDrone -> Finetuned:
     python 08_eval_wrong_way.py --gt ground_truth.csv \
         --runs VisDrone=runs/visdrone FT_v1=runs/ft_v1 FT_v2=runs/ft_v2

Định dạng file ground_truth.csv:
  video,t_start,t_end,lane,cls
  Wrong_way_1,12.0,18.5,lane1,motor
  Normal_1,,,,            (video không có xe ngược chiều -> để trống t_start, t_end để tính FP/giờ)
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def make_template(gt_path, video_dir):
    src = Path(video_dir)
    vids = [src] if src.is_file() else [p for p in sorted(src.iterdir()) if p.suffix.lower() in VIDEO_EXT]
    if not vids:
        raise SystemExit(f"Không tìm thấy video nào trong: {src}")
    out_p = Path(gt_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "t_start", "t_end", "lane", "cls"])
        for v in vids:
            w.writerow([v.stem, "", "", "", ""])
    print(f"Đã tạo file mẫu {out_p} với {len(vids)} video. Hãy điền khoảng giây (t_start, t_end) của các xe vi phạm thực tế.")


def load_gt(gt_path):
    """Trả về (gt_by_video, all_videos)."""
    gt_by_video = defaultdict(list)
    all_videos = set()
    with open(gt_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = Path(row.get("video", "").strip()).stem
            if not vid:
                continue
            all_videos.add(vid)
            ts_raw = (row.get("t_start") or row.get("t") or row.get("t_event") or "").strip()
            te_raw = (row.get("t_end") or ts_raw).strip()
            if not ts_raw:
                continue
            ts = float(ts_raw)
            te = max(ts, float(te_raw))
            gt_by_video[vid].append(dict(
                video=vid,
                t_start=ts,
                t_end=te,
                lane=(row.get("lane") or "").strip(),
                cls=(row.get("cls") or "").strip(),
            ))
    return gt_by_video, all_videos


def parse_run_specs(specs):
    out = []
    for item in specs:
        if "=" in item and not Path(item).exists():
            label, rdir = item.split("=", 1)
        else:
            p = Path(item)
            label = p.name
            rdir = item
        out.append((label, Path(rdir)))
    return out


def load_predictions(run_dir, videos):
    preds_by_video = defaultdict(list)
    total_dur_s = 0.0
    for vid in sorted(videos):
        meta_f = run_dir / f"{vid}_meta.json"
        if meta_f.exists():
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            total_dur_s += float(meta.get("duration_s") or 0.0)
        ev_f = run_dir / f"{vid}_events.csv"
        if not ev_f.exists():
            continue
        with open(ev_f, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                ts = float(row["t_start"])
                td = float(row.get("t_det") or ts)
                preds_by_video[vid].append(dict(
                    video=vid,
                    track_id=row.get("track_id", ""),
                    cls=row.get("cls", ""),
                    t_start=ts,
                    t_det=td,
                    lane=(row.get("lane") or "").strip(),
                ))
    return preds_by_video, total_dur_s


def evaluate_run(label, run_dir, gt_by_video, videos, tol=2.0, match_lane=False):
    preds_by_video, total_dur_s = load_predictions(run_dir, videos)
    tp = fp = fn = 0
    delays = []
    fn_list = []
    fp_list = []

    for vid in sorted(videos):
        gts = gt_by_video.get(vid, [])
        preds = preds_by_video.get(vid, [])

        pairs = []
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                if match_lane and g["lane"] and p["lane"] and g["lane"] != p["lane"]:
                    continue
                p_lo = min(p["t_start"], p["t_det"]) - tol
                p_hi = max(p["t_start"], p["t_det"]) + tol
                if p_hi >= g["t_start"] and p_lo <= g["t_end"]:
                    dist = abs(p["t_det"] - g["t_start"])
                    pairs.append((dist, gi, pi))

        pairs.sort()
        used_g, used_p = set(), set()
        for _, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            tp += 1
            delays.append(max(0.0, preds[pi]["t_det"] - gts[gi]["t_start"]))

        for gi, g in enumerate(gts):
            if gi not in used_g:
                fn += 1
                fn_list.append(g)
        for pi, p in enumerate(preds):
            if pi not in used_p:
                fp += 1
                fp_list.append(p)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    hours = total_dur_s / 3600.0 if total_dur_s > 0 else 0.0
    fp_per_hr = (fp / hours) if hours > 0 else float("nan")
    mean_delay = (sum(delays) / len(delays)) if delays else float("nan")

    return dict(
        model=label,
        run_dir=str(run_dir),
        n_videos=len(videos),
        total_gt=tp + fn,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=prec,
        recall=rec,
        f1=f1,
        fp_per_hr=fp_per_hr,
        mean_delay_s=mean_delay,
        fn_list=fn_list,
        fp_list=fp_list,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", help="đường dẫn file ground_truth.csv")
    ap.add_argument("--runs", nargs="*", help="các thư mục kết quả cần đối chiếu, vd: VisDrone=runs/visdrone FT_v1=runs/ft_v1")
    ap.add_argument("--tol", type=float, default=2.0, help="dung sai thời gian khớp sự kiện (giây, mặc định: 2.0)")
    ap.add_argument("--match-lane", action="store_true", help="bắt buộc khớp đúng tên làn đường nếu GT có ghi cột lane")
    ap.add_argument("--out", default="runs/eval_wrong_way_compare.csv", help="file CSV lưu bảng đối chiếu")
    ap.add_argument("--make-template", help="tạo file mẫu ground_truth.csv và thoát")
    ap.add_argument("--video", default="Video", help="thư mục video dùng kèm --make-template")
    a = ap.parse_args()

    if a.make_template:
        make_template(a.make_template, a.video)
        return

    if not a.gt or not a.runs:
        raise SystemExit("Cần truyền --gt <ground_truth.csv> và --runs <thư_mục_kết_quả> (hoặc dùng --make-template).")

    gt_by_video, videos = load_gt(a.gt)
    if not videos:
        raise SystemExit(f"File {a.gt} không có dòng video nào.")

    specs = parse_run_specs(a.runs)
    results = [evaluate_run(lbl, rdir, gt_by_video, videos, tol=a.tol, match_lane=a.match_lane)
               for lbl, rdir in specs]

    print(f"\n=== ĐỐI CHIẾU BẮT LỖI NGƯỢC CHIỀU (VisDrone -> Finetuned | {len(videos)} video, dung sai ±{a.tol}s) ===")
    hdr = f"{'Model':<18} | {'GT':>4} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} | {'FP/giờ':>7} | {'Trễ TB':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        fp_hr_s = f"{r['fp_per_hr']:.1f}" if r["fp_per_hr"] == r["fp_per_hr"] else "N/A"
        dly_s = f"{r['mean_delay_s']:.2f}s" if r["mean_delay_s"] == r["mean_delay_s"] else "N/A"
        print(f"{r['model']:<18} | {r['total_gt']:>4} | {r['tp']:>4} | {r['fp']:>4} | {r['fn']:>4} | "
              f"{r['precision']*100:8.2f}% | {r['recall']*100:7.2f}% | {r['f1']*100:7.2f}% | {fp_hr_s:>7} | {dly_s:>7}")

    for r in results:
        if r["fn_list"] or r["fp_list"]:
            print(f"\n--- Chi tiết lỗi của [{r['model']}] ---")
            for g in r["fn_list"]:
                print(f"  [BỎ SÓT - FN] {g['video']}: t={g['t_start']:.1f}s..{g['t_end']:.1f}s (làn={g['lane'] or '?'}, lớp={g['cls'] or '?'})")
            for p in r["fp_list"]:
                print(f"  [BÁO NHẦM - FP] {p['video']}: t_det={p['t_det']:.1f}s (id={p['track_id']}, lớp={p['cls']}, làn={p['lane']})")

    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "run_dir", "n_videos", "total_gt", "tp", "fp", "fn",
                  "precision", "recall", "f1", "fp_per_hr", "mean_delay_s"]
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) and r[k] == r[k] else ("" if isinstance(r[k], float) else r[k]))
                        for k in fieldnames})
    print(f"\nĐã lưu bảng đối chiếu sự kiện -> {out_p}")


if __name__ == "__main__":
    main()
