"""Bước 6: Phát hiện phương tiện đi ngược chiều (chỉ quan tâm PHƯƠNG TIỆN đa loại hình & NGƯỜI):
  - Phần 1 (Detection): YOLO nhận diện & tách riêng Phương tiện (car, van, truck, bus, motor/motorcycle, bicycle, tricycle...)
                        và Người (person, pedestrian, people), bỏ qua toàn bộ lớp nền khác.
  - Phần 2 (Tracking & Violation): Chỉ theo dõi phương tiện trong ROI -> quỹ đạo điểm chạm đất -> vận tốc & thời gian
                        trong vùng ROI -> cảnh báo phương tiện đi ngược chiều.

Chạy một video hoặc cả thư mục:
    python 06_detect_wrong_way.py --video Video --config configs --weights runs/ft_v1/weights/best.pt --out runs/ft_v1

Mỗi video xuất ra trong thư mục --out:
    <ten>_events.csv  danh sách sự kiện ngược chiều (dùng cho 08_eval_wrong_way.py)
    <ten>_tracks.csv  thống kê mọi quỹ đạo phương tiện
    <ten>_meta.json   thông tin video / tham số / tốc độ xử lý
    <ten>_out.mp4     video minh họa kết quả (tắt bằng --no-video)
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from target_classes import person_ids, vehicle_ids
from vehicle_tracker import LaneTracker, lane_track_stream

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
DEFAULT_PARAMS = dict(
    window_s=0.5,            # cửa sổ tính hướng & vận tốc (giây)
    min_disp_frac=0.3,       # dịch chuyển tối thiểu trong cửa sổ = tỉ lệ x chiều cao bbox
    cos_thresh=-0.5,         # cos(góc giữa hướng xe và chiều chuẩn) < ngưỡng => ngược (-0.5 ~ 120 độ)
    min_viol_s=0.3,          # phải ngược chiều liên tục tối thiểu (giây) mới báo
    max_gap_s=1.0,           # đứt quãng thời gian giữa 2 quan sát > ngưỡng này -> reset lịch sử vận tốc
    max_box_frac=0.40,       # bỏ box quá lớn > 40% khung hình (nhiễu nhận diện)
    lane_margin_frac=0.03,   # lề quanh polygon làn (tỉ lệ x chiều cao khung)
    link_gap_s=3.0,          # (--tracking ultra) xe bị che tối đa bao lâu vẫn nối lại ID
    link_radius_frac=1.2,    # (--tracking ultra) bán kính tìm xe tái xuất
    lost_s=5.0,              # (--tracking lane) giữ ID xe mất dấu tối đa bao lâu (giây)
    gate_frac=0.8,           # (--tracking lane) bán kính ghép = gate_frac x chiều cao box
    new_conf=0.4,            # (--tracking lane) chỉ cấp ID mới khi conf >= ngưỡng này
    vel_alpha=0.5,           # (--tracking lane) độ mượt vận tốc
    static_lost_s=10.0,      # (--tracking lane) xe ĐỨNG YÊN được nhớ lâu hơn (giây)
    dup_iomin=0.6,           # (--tracking lane) ngưỡng coi 2 box chồng nhau là trùng vật thể
    appearance=True,         # (--tracking lane) dùng histogram màu khi ghép ID
    coast_s=1.5,             # (--tracking lane) detector hụt thì suy đoán vị trí xe chạy tối đa N giây
    coast_static_s=4.0,      # (--tracking lane) xe ĐỨNG YÊN được suy đoán lâu hơn (giây)
)
CONF_MIN = {"bus": 0.3, "truck": 0.3, "bus_truck": 0.3}


def load_cfg(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pick_cfg(config, video):
    p = Path(config)
    if not p.is_dir():
        return load_cfg(p)
    f = p / f"{Path(video).stem}.json"
    if not f.exists():
        f = p / "default.json"
    if not f.exists():
        raise SystemExit(f"Không có {Path(video).stem}.json (hoặc default.json) trong {p}")
    return load_cfg(f)


def build_lanes(cfg, W, H):
    sx, sy = W / cfg["frame_size"][0], H / cfg["frame_size"][1]
    lanes = []
    for l in cfg["lanes"]:
        poly = (np.array(l["polygon"], np.float32) * np.array([sx, sy], np.float32)).astype(np.float32)
        d = np.array(l["direction"], np.float64)
        d = d / (np.linalg.norm(d) + 1e-9)
        lanes.append({"name": l["name"], "poly": poly, "dir": d})
    return lanes


def lane_of(pt, lanes, margin=0.0):
    best, best_d = None, -1e9
    for l in lanes:
        d = cv2.pointPolygonTest(l["poly"], (float(pt[0]), float(pt[1])), True)
        if d >= -margin and d > best_d:
            best, best_d = l, d
    return best


def run_video(video, cfg, model, out_dir, imgsz=960, conf=0.15, stride=1, save_video=True,
              params=None, tracker="botsort.yaml", include_person=False,
              detect_person=True, device=None, show=True, drop_classes=(), extra_classes=(), tracking="lane"):
    p = {**DEFAULT_PARAMS, **cfg.get("params", {}), **(params or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video).stem

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    eff_fps = fps / stride
    win = max(3, round(p["window_s"] * eff_fps))
    lanes = build_lanes(cfg, W, H)
    margin_px = p["lane_margin_frac"] * H
    sx_, sy_ = W / cfg["frame_size"][0], H / cfg["frame_size"][1]
    ignore = [(np.array(z, np.float32) * np.array([sx_, sy_], np.float32)).astype(np.float32)
              for z in cfg.get("ignore_zones", [])]

    hist = defaultdict(lambda: deque(maxlen=win))
    viol = defaultdict(int)
    first_viol, cos_acc, speed_acc = {}, defaultdict(list), defaultdict(list)
    flagged, tracks = {}, {}
    writer = None
    last_t = 0.0
    filtered = Counter()
    link_gap = 0 if tracking == "lane" else round(p["link_gap_s"] * eff_fps)
    canon = {}
    seen_raw = set()
    last_seen = {}
    trail = defaultdict(lambda: deque(maxlen=8))
    n_linked = 0
    n_person_boxes = 0

    cls_ids = vehicle_ids(model.names, include_person)
    drop_set = {n.lower() for n in drop_classes}
    cls_ids = [i for i in cls_ids if model.names[i].lower() not in drop_set]
    name2id = {n.lower(): k for k, n in model.names.items()}
    for n in extra_classes:
        k = name2id.get(n.lower())
        if k is None:
            print(f"[cảnh báo] model không có lớp '{n}'. Lớp có sẵn: {list(model.names.values())}")
        elif k not in cls_ids:
            cls_ids.append(k)

    p_ids = (set(person_ids(model.names)) - set(cls_ids)) if detect_person else set()
    infer_cls = sorted(set(cls_ids) | p_ids)

    def reset_viol(tid):
        viol[tid] = 0
        first_viol.pop(tid, None)
        cos_acc[tid].clear()
        speed_acc[tid].clear()

    def is_valid_vehicle(x1, y1, x2, y2, c, cf):
        if (y2 - y1) > p["max_box_frac"] * H or (x2 - x1) > p["max_box_frac"] * W:
            filtered["too_big"] += 1
            return False
        if cf < CONF_MIN.get(str(model.names[c]).lower(), 0.0):
            filtered["low_conf_large_cls"] += 1
            return False
        ctr = ((x1 + x2) / 2, (y1 + y2) / 2)
        if any(cv2.pointPolygonTest(z, (float(ctr[0]), float(ctr[1])), False) >= 0 for z in ignore):
            filtered["ignore_zone"] += 1
            return False
        return True

    model.predictor = None
    t0 = time.time()
    n_proc = 0

    track_kw = dict(
        source=str(video),
        stream=True,
        persist=True,
        tracker=tracker,
        imgsz=imgsz,
        conf=conf,
        iou=0.4,
        agnostic_nms=(len(p_ids) == 0),
        classes=infer_cls,
        vid_stride=stride,
        verbose=False,
    )
    if device is not None:
        track_kw["device"] = device
    tracker_obj = None
    if tracking == "lane":
        def accept(x1, y1, x2, y2, c, cf):
            if not is_valid_vehicle(x1, y1, x2, y2, c, cf):
                return False
            if lane_of(((x1 + x2) / 2, y2), lanes, margin_px) is None:
                filtered["outside_lane"] += 1
                return False
            return True

        pred_kw = {k: v for k, v in track_kw.items() if k not in ("persist", "tracker")}
        tracker_obj = LaneTracker(eff_fps=eff_fps, lost_s=p["lost_s"], static_lost_s=p["static_lost_s"],
                                  gate_frac=p["gate_frac"], new_conf=p["new_conf"], vel_alpha=p["vel_alpha"],
                                  dup_iomin=max(0.0, p["dup_iomin"] - 0.1), appearance=bool(p["appearance"]),
                                  coast_s=p["coast_s"], coast_static_s=p["coast_static_s"],
                                  coast_ok=lambda f: (0 <= f[0] < W and 0 <= f[1] < H
                                                      and lane_of(f, lanes, margin_px) is not None))
        stream = lane_track_stream(model, pred_kw, accept, tracker_obj,
                                   dup_iomin=p["dup_iomin"], person_cls_set=p_ids)
    else:
        stream = model.track(**track_kw)

    render_needed = save_video or show
    for i, r in enumerate(stream):
        n_proc += 1
        t = i * stride / fps
        last_t = t
        persons = getattr(r, "persons", [])
        n_person_boxes += len(persons)
        draw = r.orig_img.copy() if render_needed else None
        if render_needed:
            if save_video and writer is None:
                h_, w_ = draw.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / f"{stem}_out.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), eff_fps, (w_, h_))
            for l in lanes:
                cv2.polylines(draw, [l["poly"].astype(np.int32)], True, (0, 200, 0), 2)
            for (px1, py1, px2, py2, pc, pcf) in persons:
                cv2.rectangle(draw, (int(px1), int(py1)), (int(px2), int(py2)), (180, 180, 180), 1)
                cv2.putText(draw, f"{model.names[pc]} (non-veh)", (int(px1), max(15, int(py1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            boxes = r.boxes.xyxy.tolist()
            clss = r.boxes.cls.int().tolist()
            confs = r.boxes.conf.tolist()
            preds = getattr(r.boxes, "pred", None) or [False] * len(ids)

            if link_gap > 0:
                expired = [oid for oid, (fi_, _, _, _) in last_seen.items() if i - fi_ > link_gap]
                for oid in expired:
                    last_seen.pop(oid, None)
                    trail.pop(oid, None)

            for raw, (x1, y1, x2, y2), c, cf, pred in zip(ids, boxes, clss, confs, preds):
                if c in p_ids:
                    n_person_boxes += 1
                    if render_needed:
                        cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), (180, 180, 180), 1)
                        cv2.putText(draw, f"{model.names[c]} (non-veh)", (int(x1), max(15, int(y1) - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
                    continue

                if tracking != "lane" and not pred:
                    if not is_valid_vehicle(x1, y1, x2, y2, c, cf):
                        continue

                foot = ((x1 + x2) / 2, y2)
                tid = canon.get(raw, raw)
                if raw not in seen_raw:
                    seen_raw.add(raw)
                    best, best_d = None, 1e9
                    if link_gap > 0:
                        for oid, (fi_, ox, oy, oh) in last_seen.items():
                            gap = i - fi_
                            if gap < 2 or gap > link_gap:
                                continue
                            tr = trail[oid]
                            vx = vy = 0.0
                            if len(tr) >= 2:
                                (g0, ax, ay), (g1, bx, by) = tr[0], tr[-1]
                                span = max(1, g1 - g0)
                                vx, vy = (bx - ax) / span, (by - ay) / span
                            dist = float(np.hypot(foot[0] - (ox + vx * gap), foot[1] - (oy + vy * gap)))
                            if dist <= p["link_radius_frac"] * max(oh, 20.0) and dist < best_d:
                                best, best_d = oid, dist
                    if best is not None and best != tid:
                        canon[raw] = best
                        tid = best
                        n_linked += 1
                last_seen[tid] = (i, foot[0], foot[1], y2 - y1)
                trail[tid].append((i, foot[0], foot[1]))
                st = tracks.setdefault(tid, {"cls": Counter(), "t_first": t, "t_last": t, "n": 0})
                st["t_last"] = t
                if not pred:
                    st["cls"][c] += 1
                    st["n"] += 1

                lane = lane_of(foot, lanes, margin_px)
                if lane is not None:
                    if hist[tid] and (t - hist[tid][-1][0]) > p["max_gap_s"]:
                        hist[tid].clear()
                        reset_viol(tid)
                    hist[tid].append((t, foot[0], foot[1], y2 - y1))
                    hh = hist[tid]
                    if len(hh) >= win:
                        arr = np.asarray(hh, np.float64)
                        k = max(1, len(arr) // 3)
                        dt = float(arr[-k:, 0].mean() - arr[:k, 0].mean())
                        d = arr[-k:, 1:3].mean(0) - arr[:k, 1:3].mean(0)
                        disp = float(np.linalg.norm(d))
                        speed_px_s = disp / max(dt, 1e-6)
                        dmin = max(4.0, p["min_disp_frac"] * float(np.median(arr[:, 3])))
                        if not pred and disp >= dmin:
                            cosv = float(d @ lane["dir"] / disp)
                            if cosv < p["cos_thresh"]:
                                viol[tid] += 1
                                first_viol.setdefault(tid, t)
                                cos_acc[tid].append(cosv)
                                speed_acc[tid].append(speed_px_s)
                            else:
                                reset_viol(tid)
                        elif not pred:
                            reset_viol(tid)

                        if (not pred and tid not in flagged and tid in first_viol and viol[tid] >= 3
                                and t - first_viol[tid] >= p["min_viol_s"]):
                            flagged[tid] = dict(
                                t_start=first_viol[tid],
                                t_det=t,
                                viol_dur=t - first_viol[tid],
                                lane=lane["name"],
                                mean_cos=float(np.mean(cos_acc[tid])),
                                mean_speed_px_s=float(np.mean(speed_acc[tid])) if speed_acc[tid] else speed_px_s,
                            )
                else:
                    reset_viol(tid)
                    if hist[tid] and (t - hist[tid][-1][0]) > p["window_s"]:
                        hist[tid].clear()

                if render_needed:
                    bad = tid in flagged
                    col = (0, 0, 255) if bad else ((0, 200, 255) if pred else (0, 255, 0))
                    cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), col, 3 if bad else (1 if pred else 2))
                    label = f"{tid} {model.names[c]}" + ("~" if pred else "") + (" WRONG-WAY" if bad else "")
                    cv2.putText(draw, label, (int(x1), max(15, int(y1) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    pts = np.array([[int(x), int(y)] for _, x, y, _ in hist[tid]], np.int32)
                    if len(pts) > 1:
                        cv2.polylines(draw, [pts], False, col, 2)

        if save_video and writer is not None:
            writer.write(draw)
        if show and draw is not None:
            cv2.imshow("06_detect_wrong_way", draw)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    events = []
    with open(out_dir / f"{stem}_events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "track_id", "cls", "t_start", "t_det", "lane", "mean_cos", "viol_dur_s", "speed_px_s"])
        for tid, e in sorted(flagged.items(), key=lambda kv: kv[1]["t_det"]):
            cname = model.names[tracks[tid]["cls"].most_common(1)[0][0]] if tracks[tid]["cls"] else "unknown"
            row = [stem, tid, cname, round(e["t_start"], 3), round(e["t_det"], 3), e["lane"],
                   round(e["mean_cos"], 3), round(e["viol_dur"], 3), round(e["mean_speed_px_s"], 1)]
            w.writerow(row)
            events.append(dict(zip(["video", "track_id", "cls", "t_start", "t_det", "lane", "mean_cos", "viol_dur_s", "speed_px_s"], row)))

    with open(out_dir / f"{stem}_tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "track_id", "cls", "t_first", "t_last", "n_obs", "flagged"])
        for tid, st in sorted(tracks.items()):
            cname = model.names[st["cls"].most_common(1)[0][0]] if st["cls"] else "unknown"
            w.writerow([stem, tid, cname,
                        round(st["t_first"], 3), round(st["t_last"], 3), st["n"], int(tid in flagged)])

    elapsed = time.time() - t0
    meta = dict(video=stem, fps=fps, stride=stride, eff_fps=eff_fps, imgsz=imgsz, conf=conf, params=p,
                tracker=str(tracker), tracking=tracking,
                tracker_stats=dict(tracker_obj.stats) if tracker_obj else {},
                duration_s=(n_total / fps) if n_total > 0 else last_t,
                frames_processed=n_proc, runtime_s=round(elapsed, 2),
                proc_fps=round(n_proc / elapsed, 2) if elapsed else None,
                n_events=len(events), n_linked=n_linked, n_person_boxes=n_person_boxes,
                include_person=include_person, detect_person=detect_person,
                filtered_boxes=dict(filtered),
                vehicle_classes=[model.names[i] for i in cls_ids],
                person_classes=[model.names[i] for i in sorted(p_ids)])
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return events, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="file video hoặc thư mục chứa video")
    ap.add_argument("--config", required=True, help="file json hoặc thư mục configs/")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-show", action="store_true", help="tắt cửa sổ xem trực tiếp (dùng trên server/Colab)")
    ap.add_argument("--no-video", action="store_true", help="không lưu file _out.mp4 để chạy nhanh hơn")
    ap.add_argument("--include-person", action="store_true",
                    help="theo dõi cả lớp person như phương tiện (khi xe máy hay bị nhận nhầm là người)")
    ap.add_argument("--no-person", action="store_true",
                    help="bỏ qua hoàn toàn lớp người (mặc định: có phát hiện và vẽ tách biệt với phương tiện)")
    ap.add_argument("--drop-classes", nargs="*", default=[])
    ap.add_argument("--extra-classes", nargs="*", default=[])
    ap.add_argument("--cos-thresh", type=float)
    ap.add_argument("--min-viol-s", type=float)
    ap.add_argument("--window-s", type=float)
    ap.add_argument("--min-disp-frac", type=float)
    ap.add_argument("--max-box-frac", type=float)
    ap.add_argument("--lane-margin-frac", type=float)
    ap.add_argument("--link-gap-s", type=float)
    ap.add_argument("--link-radius-frac", type=float)
    ap.add_argument("--tracking", choices=["lane", "ultra"], default="lane")
    ap.add_argument("--lost-s", type=float)
    ap.add_argument("--gate-frac", type=float)
    ap.add_argument("--new-conf", type=float)
    ap.add_argument("--static-lost-s", type=float)
    ap.add_argument("--dup-iomin", type=float)
    ap.add_argument("--coast-s", type=float)
    ap.add_argument("--coast-static-s", type=float)
    ap.add_argument("--no-appearance", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    drop = {n.lower() for n in a.drop_classes}
    shown = [model.names[i] for i in vehicle_ids(model.names, a.include_person)
             if model.names[i].lower() not in drop]
    for extra in a.extra_classes:
        if extra.lower() not in {s.lower() for s in shown}:
            shown.append(extra)
    print("Lớp phương tiện (đa loại hình) được theo dõi:", shown)
    if not a.no_person:
        p_shown = [model.names[i] for i in person_ids(model.names)]
        print("Lớp người được phân tách (vẽ minh hoạ, không xét ngược chiều):", p_shown)

    src = Path(a.video)
    vids = [src] if src.is_file() else [p for p in sorted(src.iterdir()) if p.suffix.lower() in VIDEO_EXT]
    over = {k: v for k, v in dict(cos_thresh=a.cos_thresh, min_viol_s=a.min_viol_s, window_s=a.window_s,
                                  min_disp_frac=a.min_disp_frac, max_box_frac=a.max_box_frac,
                                  lane_margin_frac=a.lane_margin_frac, link_gap_s=a.link_gap_s,
                                  link_radius_frac=a.link_radius_frac, lost_s=a.lost_s,
                                  gate_frac=a.gate_frac, new_conf=a.new_conf,
                                  static_lost_s=a.static_lost_s, dup_iomin=a.dup_iomin,
                                  coast_s=a.coast_s, coast_static_s=a.coast_static_s,
                                  appearance=False if a.no_appearance else None).items() if v is not None}
    for v in vids:
        ev, meta = run_video(v, pick_cfg(a.config, v), model, a.out, imgsz=a.imgsz, conf=a.conf, stride=a.stride,
                             save_video=not a.no_video, params=over, tracker=a.tracker,
                             include_person=a.include_person, detect_person=not a.no_person, device=a.device,
                             show=not a.no_show, tracking=a.tracking,
                             drop_classes=a.drop_classes, extra_classes=a.extra_classes)
        print(f"{v.name}: {len(ev)} sự kiện ngược chiều, {meta['proc_fps']} khung/s xử lý, "
              f"người phát hiện: {meta['n_person_boxes']} box, box xe bị lọc: {meta['filtered_boxes']}, "
              f"ID nối lại: {meta['n_linked']}")
        for e in ev:
            print(f"   t={e['t_det']:.1f}s id={e['track_id']} {e['cls']} làn={e['lane']} "
                  f"cos={e['mean_cos']} speed={e.get('speed_px_s', '')}px/s")


if __name__ == "__main__":
    main()
