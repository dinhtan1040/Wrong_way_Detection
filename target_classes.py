"""Danh mục đối tượng mục tiêu & ánh xạ tên lớp (không phụ thuộc ID cố định).

Chỉ quan tâm 2 nhóm đối tượng:
  1. Phương tiện (đa loại hình): bicycle, car, van, truck, bus,
     motorcycle / motorbike / motor, tricycle, awning-tricycle.
  2. Người: person, pedestrian, people.
Mọi lớp nền khác (đèn giao thông, biển báo, động vật, đồ vật...) đều bị loại bỏ.

Chuẩn hóa nhóm chung (EVAL_GROUP_OF) để đối chiếu công bằng giữa các model khác bộ lớp
(ví dụ: VisDrone 10 lớp -> model Finetuned gộp lớp hoặc COCO):
    VisDrone: 0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck,
              6 tricycle, 7 awning-tricycle, 8 bus, 9 motor
    COCO    : 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
"""

VEHICLE_NAMES = {
    "bicycle", "car", "van", "truck", "bus",
    "motorcycle", "motorbike", "motor",
    "tricycle", "awning-tricycle",
}
PERSON_NAMES = {"person", "pedestrian", "people"}
TARGET_NAMES = VEHICLE_NAMES | PERSON_NAMES

# Nhóm chuẩn hóa để so sánh detector (07_eval_detector.py) giữa VisDrone và các bản Finetuned
EVAL_GROUP_OF = {
    "motor": "motor",
    "motorcycle": "motor",
    "motorbike": "motor",
    "car": "car",
    "van": "car",
    "bus": "bus",
    "truck": "truck",
    "bicycle": "bicycle",
    "tricycle": "tricycle",
    "awning-tricycle": "tricycle",
    "person": "person",
    "pedestrian": "person",
    "people": "person",
}
EVAL_VEHICLE_GROUPS = ["motor", "car", "bus", "truck", "bicycle", "tricycle"]
EVAL_PERSON_GROUPS = ["person"]
EVAL_ALL_GROUPS = EVAL_VEHICLE_GROUPS + EVAL_PERSON_GROUPS


def _items(names):
    return names.items() if isinstance(names, dict) else enumerate(names)


def resolve_ids(names, wanted):
    """Lấy danh sách ID các lớp có tên (chữ thường) nằm trong `wanted`."""
    return sorted(int(i) for i, n in _items(names) if str(n).lower() in wanted)


def vehicle_ids(names, include_person=False):
    """Lấy ID các lớp phương tiện (đa loại hình); thêm lớp người nếu include_person=True."""
    wanted = set(VEHICLE_NAMES) | (PERSON_NAMES if include_person else set())
    ids = resolve_ids(names, wanted)
    if not ids:
        raise SystemExit("Model không có lớp phương tiện nào (tên lớp: %s)" % list(dict(_items(names)).values())[:12])
    return ids


def person_ids(names):
    """Lấy ID các lớp người (person, pedestrian, people) trong model."""
    return resolve_ids(names, PERSON_NAMES)


def eval_group_of(name):
    """Quy đổi tên lớp bất kỳ về nhóm chuẩn hóa chung, hoặc None nếu là lớp nền."""
    return EVAL_GROUP_OF.get(str(name).lower())
