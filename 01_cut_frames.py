"""Bước 1: Cắt khung hình (.jpg) từ video để làm dữ liệu huấn luyện (dùng tiếp với 02_auto_label.py).

Hỗ trợ cả 2 chế độ trong một lệnh duy nhất:
  1. Cắt đều cả thư mục video (hoặc 1 video) theo số ảnh/giây (--fps) hoặc bước nhảy khung (--step):
       python 01_cut_frames.py --video Video --out data/frames --fps 2
       python 01_cut_frames.py --video Video --out data/frames --step 15

  2. Cắt chọn lọc đúng các khoảng thời gian (--range) có xe ngược chiều hoặc tình huống khó
     (tránh rò rỉ khung liền kề giữa train / val / test):
       python 01_cut_frames.py --video Video/Wrong_way_1.mp4 --out data/frames --range 5:25 40:65 --step 5

Cấu trúc thư mục xuất ra:
    data/frames/<ten_video>/<ten_video>_000000.jpg
"""
import argparse
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def imwrite_unicode(path, img, quality=95):
    """Ghi ảnh an toàn kể cả khi đường dẫn có dấu tiếng Việt trên Windows."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def parse_ranges(range_args):
    if not range_args:
        return None
    spans = []
    for item in range_args:
        if ":" not in item:
            raise SystemExit(f"Định dạng --range không hợp lệ '{item}'. Đúng dạng: bat_dau:ket_thuc (vd 5:25)")
        s_str, e_str = item.split(":", 1)
        s = float(s_str) if s_str.strip() else 0.0
        e = float(e_str) if e_str.strip() else float("inf")
        if e <= s:
            raise SystemExit(f"Khoảng thời gian rỗng hoặc ngược: {item}")
        spans.append((s, e))
    return sorted(spans)


def in_ranges(t_sec, spans):
    return True if spans is None else any(s <= t_sec <= e for s, e in spans)


def cut_video(video_path, out_dir, step=None, target_fps=2.0, spans=None, max_frames=0, quality=95):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[lỗi] Không mở được video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if step is None or step < 1:
        step = max(1, round(fps / max(target_fps, 0.1)))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    start_idx = 0
    max_t = float("inf")
    if spans:
        start_idx = max(0, int(spans[0][0] * fps))
        max_t = spans[-1][1]
        if start_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    idx = start_idx
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t_sec = idx / fps
        if t_sec > max_t:
            break
        if in_ranges(t_sec, spans) and ((idx - start_idx) % step == 0):
            out_file = out_dir / f"{stem}_{idx:06d}.jpg"
            if imwrite_unicode(out_file, frame, quality=quality):
                saved += 1
                if max_frames > 0 and saved >= max_frames:
                    break
        idx += 1

    cap.release()
    dur_s = total / fps if total > 0 else idx / fps
    extra = f", khoảng={spans}" if spans else ""
    print(f"{video_path.name} ({dur_s:.1f}s, {fps:.1f} fps): bước={step} khung{extra} -> lưu {saved} ảnh vào {out_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="file video hoặc thư mục chứa các video")
    ap.add_argument("--out", default="data/frames", help="thư mục gốc lưu ảnh (mặc định: data/frames)")
    ap.add_argument("--fps", type=float, default=2.0, help="số ảnh cắt mỗi giây (mặc định: 2 ảnh/giây)")
    ap.add_argument("--step", type=int, default=None, help="cắt 1 ảnh sau mỗi N khung (ưu tiên hơn --fps)")
    ap.add_argument("--range", nargs="*", default=None,
                    help="chỉ cắt trong các khoảng giây chỉ định, vd: --range 5:25 40:65")
    ap.add_argument("--max-frames", type=int, default=0, help="giới hạn số ảnh tối đa mỗi video (0 = không giới hạn)")
    ap.add_argument("--quality", type=int, default=95, help="chất lượng JPEG (1..100)")
    a = ap.parse_args()

    src = Path(a.video)
    if not src.exists():
        raise SystemExit(f"Không tìm thấy đường dẫn: {src}")
    if a.range and not src.is_file():
        raise SystemExit("--range chỉ dùng khi --video là 1 file video cụ thể.")

    spans = parse_ranges(a.range)
    vids = [src] if src.is_file() else [p for p in sorted(src.iterdir()) if p.suffix.lower() in VIDEO_EXT]
    if not vids:
        raise SystemExit(f"Không tìm thấy video hợp lệ trong: {src}")

    out_root = Path(a.out)
    total_saved = 0
    for v in vids:
        sub_dir = out_root if (src.is_file() and out_root.name == v.stem) else (out_root / v.stem)
        total_saved += cut_video(v, sub_dir, step=a.step, target_fps=a.fps,
                                 spans=spans, max_frames=a.max_frames, quality=a.quality)
    print(f"Hoàn tất: tổng cộng {total_saved} khung hình từ {len(vids)} video.")


if __name__ == "__main__":
    main()
