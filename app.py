"""
SAM Annotator v3 — Multi-Proyecto, Multi-Usuario
Roles: superadmin / local_admin / annotator
"""

import os, sys, json, base64, uuid, shutil, hashlib, time, threading, subprocess
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
import numpy as np
import cv2
from PIL import Image
import io
import torch

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "oculus-sam-secret-2026")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

DATA_DIR = Path("collab_data")
USERS_FILE     = DATA_DIR / "users.json"
PROJECTS_FILE  = DATA_DIR / "projects.json"
ACTIVITY_FILE  = DATA_DIR / "activity.json"

DATA_DIR.mkdir(exist_ok=True)

def _load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def _save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _load_users():    return _load_json(USERS_FILE, {})
def _save_users(u):   _save_json(USERS_FILE, u)
def _load_projects(): return _load_json(PROJECTS_FILE, {"projects": []})
def _save_projects(p):_save_json(PROJECTS_FILE, p)
def _load_activity(): return _load_json(ACTIVITY_FILE, {})
def _save_activity(a):_save_json(ACTIVITY_FILE, a)

def _hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def _current_user():
    return session.get("username")

def _get_role():
    return session.get("role", "annotator")

def _is_superadmin():
    return _get_role() == "superadmin"

def _is_any_admin():
    return _get_role() in ("superadmin", "local_admin")

def _is_local_admin_of(project_id):
    if _is_superadmin():
        return True
    if _get_role() == "local_admin":
        users = _load_users()
        u = users.get(_current_user(), {})
        return project_id in u.get("admin_projects", [])
    return False

def _require_login(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _current_user():
            if request.is_json:
                return jsonify({"error": "No autenticado"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return wrapper

def _require_superadmin(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _is_superadmin():
            if request.is_json:
                return jsonify({"error": "Solo el superadmin puede hacer esto"}), 403
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return wrapper

def _require_any_admin(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _is_any_admin():
            if request.is_json:
                return jsonify({"error": "Se requieren permisos de admin"}), 403
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return wrapper

def _init_default_admin():
    users = _load_users()
    if "admin" not in users:
        users["admin"] = {
            "password": _hash_pw("admin123"),
            "role": "superadmin",
            "created_at": datetime.now().isoformat(),
            "display_name": "Super Admin",
            "admin_projects": [],
        }
        _save_users(users)
        print("[INFO] Usuario superadmin creado — contraseña: admin123")

_init_default_admin()

SAM_STATE = {
    "predictor": None,
    "sam_version": "none",
    "model_name": "none",
}

USER_STATE = {}

def get_user_state(username):
    if username not in USER_STATE:
        USER_STATE[username] = {
            "image_np": None,
            "image_path": None,
            "image_set": False,
            "current_index": 0,
            "active_project_id": None,
        }
    return USER_STATE[username]

PROJECT_STATE = {}

def get_project_state(project_id):
    if project_id not in PROJECT_STATE:
        PROJECT_STATE[project_id] = {
            "image_list": [],
            "annotations": {},
            "classes": ["objeto"],
            "class_colors": {"objeto": "#3B82F6"},
            "active_class": "objeto",
            "aug_progress": {"running": False, "current": 0, "total": 0, "label": ""},
            "train_progress": {"running": False, "epoch": 0, "total_epochs": 0,
                               "loss": 0, "map50": 0, "map": 0, "log": [], "status": "idle"},
        }
    return PROJECT_STATE[project_id]

COLOR_PALETTE = [
    "#3B82F6","#EF4444","#10B981","#F59E0B","#8B5CF6",
    "#EC4899","#06B6D4","#84CC16","#F97316","#6366F1",
    "#14B8A6","#F43F5E","#A855F7","#EAB308","#22D3EE",
]

def _get_project(project_id):
    data = _load_projects()
    for p in data["projects"]:
        if p["id"] == project_id:
            return p
    return None

def _save_project(project):
    data = _load_projects()
    for i, p in enumerate(data["projects"]):
        if p["id"] == project["id"]:
            data["projects"][i] = project
            _save_projects(data)
            return
    data["projects"].append(project)
    _save_projects(data)

def _get_user_projects(username):
    users = _load_users()
    u = users.get(username, {})
    role = u.get("role", "annotator")
    data = _load_projects()
    if role == "superadmin":
        return data["projects"]
    if role == "local_admin":
        admin_pids = set(u.get("admin_projects", []))
        return [p for p in data["projects"] if p["id"] in admin_pids]
    result = []
    for p in data["projects"]:
        assignments = p.get("assignments", {})
        if username in assignments and assignments[username]:
            result.append(p)
    return result

def _get_user_images(username, project_id):
    project = _get_project(project_id)
    if not project:
        return []
    ps = get_project_state(project_id)
    assignments = project.get("assignments", {})
    if username not in assignments:
        return []
    indices = assignments[username]
    return [ps["image_list"][i] for i in indices if i < len(ps["image_list"])]

HARDWARE_INFO = {}

def detect_hardware():
    global HARDWARE_INFO
    info = {"gpu": None, "vram_gb": 0, "ram_gb": 0, "cpu_cores": 0, "cuda": False}
    try:
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(vram / 1024**3, 1)
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        info["cpu_cores"] = psutil.cpu_count(logical=False) or 1
    except Exception:
        info["cpu_cores"] = os.cpu_count() or 1
    HARDWARE_INFO = info
    return info

def get_training_recommendations(model_id, hardware=None):
    if hardware is None:
        hardware = HARDWARE_INFO or detect_hardware()
    vram = hardware.get("vram_gb", 0)
    cuda = hardware.get("cuda", False)
    table = {
        "yolov8n-seg": [(4,640,2),(8,640,4),(16,640,4),(32,640,8)],
        "yolov8s-seg": [(4,640,2),(4,640,2),(8,640,4),(16,640,4)],
        "yolov8m-seg": [(2,640,2),(4,640,2),(8,640,4),(8,640,4)],
        "yolov8l-seg": [(2,512,2),(2,640,2),(4,640,4),(8,640,4)],
        "yolov8x-seg": [(1,512,1),(2,512,2),(4,640,2),(8,640,4)],
        "yolo11n-seg": [(4,640,2),(8,640,4),(16,640,4),(32,640,8)],
        "yolo11s-seg": [(4,640,2),(4,640,2),(8,640,4),(16,640,4)],
        "yolo11m-seg": [(2,640,2),(4,640,2),(8,640,4),(8,640,4)],
        "yolo11l-seg": [(2,512,2),(2,640,2),(4,640,4),(8,640,4)],
        "yolo11x-seg": [(1,512,1),(2,512,2),(4,640,2),(8,640,4)],
    }
    vi = 0 if vram < 4 else 1 if vram < 8 else 2 if vram < 12 else 3
    if not cuda:
        vi = 0
    defaults = table.get(model_id, [(4,640,2),(8,640,4),(16,640,4),(32,640,4)])
    batch, imgsz, workers = defaults[vi]
    if os.name == "nt":
        workers = min(workers, 2)
    return {"batch": batch, "imgsz": imgsz, "workers": workers, "epochs": 100,
            "device": "0" if cuda else "cpu", "lr0": 0.01, "patience": 50}

detect_hardware()

AVAILABLE_MODELS = [
    {"id":"sam2_small","label":"SAM 2.1 Small (recomendado)","version":"sam2","checkpoint":"sam2.1_hiera_small.pt","config":"sam2.1_hiera_s.yaml"},
    {"id":"sam2_base","label":"SAM 2.1 Base+","version":"sam2","checkpoint":"sam2.1_hiera_base_plus.pt","config":"sam2.1_hiera_b+.yaml"},
    {"id":"sam2_large","label":"SAM 2.1 Large","version":"sam2","checkpoint":"sam2.1_hiera_large.pt","config":"sam2.1_hiera_l.yaml"},
    {"id":"sam1_vit_b","label":"SAM v1 ViT-B","version":"sam1","checkpoint":"sam_vit_b_01ec64.pth","config":"vit_b"},
    {"id":"sam1_vit_l","label":"SAM v1 ViT-L","version":"sam1","checkpoint":"sam_vit_l_0b3195.pth","config":"vit_l"},
    {"id":"sam1_vit_h","label":"SAM v1 ViT-H","version":"sam1","checkpoint":"sam_vit_h_4b8939.pth","config":"vit_h"},
    {"id":"demo","label":"Demo (sin GPU)","version":"demo","checkpoint":"","config":""},
]

YOLO_MODELS = [
    {"id":"yolov8n","label":"YOLOv8n — Detection (nano, rápido)","task":"det","family":"v8"},
    {"id":"yolov8s","label":"YOLOv8s — Detection (small)","task":"det","family":"v8"},
    {"id":"yolov8m","label":"YOLOv8m — Detection (medium)","task":"det","family":"v8"},
    {"id":"yolov8l","label":"YOLOv8l — Detection (large)","task":"det","family":"v8"},
    {"id":"yolov8x","label":"YOLOv8x — Detection (extra large)","task":"det","family":"v8"},
    {"id":"yolov8n-seg","label":"YOLOv8n-seg — Segmentation (nano, rápido)","task":"seg","family":"v8"},
    {"id":"yolov8s-seg","label":"YOLOv8s-seg — Segmentation (small)","task":"seg","family":"v8"},
    {"id":"yolov8m-seg","label":"YOLOv8m-seg — Segmentation (medium)","task":"seg","family":"v8"},
    {"id":"yolov8l-seg","label":"YOLOv8l-seg — Segmentation (large)","task":"seg","family":"v8"},
    {"id":"yolov8x-seg","label":"YOLOv8x-seg — Segmentation (extra large)","task":"seg","family":"v8"},
    {"id":"yolo11n","label":"YOLO11n — Detection (nano)","task":"det","family":"v11"},
    {"id":"yolo11s","label":"YOLO11s — Detection (small)","task":"det","family":"v11"},
    {"id":"yolo11m","label":"YOLO11m — Detection (medium)","task":"det","family":"v11"},
    {"id":"yolo11l","label":"YOLO11l — Detection (large)","task":"det","family":"v11"},
    {"id":"yolo11x","label":"YOLO11x — Detection (extra large)","task":"det","family":"v11"},
    {"id":"yolo11n-seg","label":"YOLO11n-seg — Segmentation (nano)","task":"seg","family":"v11"},
    {"id":"yolo11s-seg","label":"YOLO11s-seg — Segmentation (small)","task":"seg","family":"v11"},
    {"id":"yolo11m-seg","label":"YOLO11m-seg — Segmentation (medium)","task":"seg","family":"v11"},
    {"id":"yolo11l-seg","label":"YOLO11l-seg — Segmentation (large)","task":"seg","family":"v11"},
    {"id":"yolo11x-seg","label":"YOLO11x-seg — Segmentation (extra large)","task":"seg","family":"v11"},
    {"id":"yolo26n","label":"YOLO26n — Detection (nano, más reciente)","task":"det","family":"v26"},
    {"id":"yolo26s","label":"YOLO26s — Detection (small)","task":"det","family":"v26"},
    {"id":"yolo26m","label":"YOLO26m — Detection (medium)","task":"det","family":"v26"},
    {"id":"yolo26l","label":"YOLO26l — Detection (large)","task":"det","family":"v26"},
    {"id":"yolo26x","label":"YOLO26x — Detection (extra large)","task":"det","family":"v26"},
    {"id":"yolo26n-seg","label":"YOLO26n-seg — Segmentation (nano, más reciente)","task":"seg","family":"v26"},
    {"id":"yolo26s-seg","label":"YOLO26s-seg — Segmentation (small)","task":"seg","family":"v26"},
    {"id":"yolo26m-seg","label":"YOLO26m-seg — Segmentation (medium)","task":"seg","family":"v26"},
    {"id":"yolo26l-seg","label":"YOLO26l-seg — Segmentation (large)","task":"seg","family":"v26"},
    {"id":"yolo26x-seg","label":"YOLO26x-seg — Segmentation (extra large)","task":"seg","family":"v26"},
]

def _get_project_root(project_path):
    return Path(project_path).parent

def _get_split_dir(project_path, model_task):
    task_folder = "yolo_seg" if model_task == "seg" else "yolo_det"
    return _get_project_root(project_path) / "DataSet_Split" / task_folder

def _get_models_dir(project_path, model_task):
    task_folder = "yolo_seg" if model_task == "seg" else "yolo_det"
    return _get_project_root(project_path) / "Modelos_Entrenados" / task_folder

def get_available_sam_models():
    ckpt_dir = Path("checkpoints")
    result = []
    for m in AVAILABLE_MODELS:
        available = m["id"] == "demo" or (bool(m["checkpoint"]) and (ckpt_dir / m["checkpoint"]).exists())
        result.append({**m, "available": available})
    return result

def load_sam_model(model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path("checkpoints")
    model_info = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    if not model_info or model_id == "demo":
        SAM_STATE["predictor"] = None
        SAM_STATE["sam_version"] = "demo"
        SAM_STATE["model_name"] = "Demo"
        return True, "demo"
    ckpt_path = ckpt_dir / model_info["checkpoint"]
    if not ckpt_path.exists():
        return False, f"Checkpoint no encontrado: {model_info['checkpoint']}"
    try:
        if model_info["version"] == "sam2":
            import sam2 as _sam2_pkg
            cfg_base = os.path.join(os.path.dirname(_sam2_pkg.__file__), "configs")
            cfg_path = os.path.join(cfg_base, "sam2.1", model_info["config"])
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(cfg_base, "sam2", model_info["config"])
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(cfg_path, str(ckpt_path), device=device)
            SAM_STATE["predictor"] = SAM2ImagePredictor(model)
        else:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_info["config"]](checkpoint=str(ckpt_path))
            sam.to(device=device)
            SAM_STATE["predictor"] = SamPredictor(sam)
        SAM_STATE["sam_version"] = model_info["version"]
        SAM_STATE["model_name"] = model_info["label"]
        return True, model_info["label"]
    except Exception as e:
        return False, str(e)

def _embed_image(img_np):
    pred = SAM_STATE["predictor"]
    if pred is None:
        return
    try:
        if SAM_STATE["sam_version"] == "sam2":
            with torch.inference_mode():
                pred.set_image(img_np)
        else:
            pred.set_image(img_np)
    except Exception as e:
        print(f"[WARN] embed error: {e}")

def mask_to_polygon(mask):
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2).tolist()

def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0,0,0,0]
    rmin, rmax = np.where(rows)[0][[0,-1]]
    cmin, cmax = np.where(cols)[0][[0,-1]]
    return [int(cmin), int(rmin), int(cmax-cmin), int(rmax-rmin)]

def demo_mask(img_np, x, y):
    h, w = img_np.shape[:2]
    mask = np.zeros((h,w), dtype=bool)
    r = min(80, h//6, w//6)
    Y, X = np.ogrid[:h, :w]
    mask[(X-x)**2+(Y-y)**2<=r**2] = True
    return mask

def polygon_to_mask(polygon, h, w):
    mask = np.zeros((h,w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def _broadcast_project_progress(project_id):
    try:
        project = _get_project(project_id)
        if not project:
            return
        ps = get_project_state(project_id)
        users = _load_users()
        activity = _load_activity()
        assignments = project.get("assignments", {})
        stats = {}
        for uname in assignments:
            user_imgs = _get_user_images(uname, project_id)
            annotated = sum(1 for p in user_imgs if ps["annotations"].get(p))
            stats[uname] = {
                "total": len(user_imgs),
                "annotated": annotated,
                "display_name": users.get(uname, {}).get("display_name", uname),
                "last_active": activity.get(f"{uname}:{project_id}", {}).get("last_active", ""),
                "current_image": activity.get(f"{uname}:{project_id}", {}).get("current_image", ""),
            }
        socketio.emit("progress_update", {"project_id": project_id, "stats": stats},
                      to=f"progress_{project_id}")
    except Exception as e:
        print(f"[WARN] broadcast error: {e}")

def _update_activity(username, project_id, image_path=""):
    activity = _load_activity()
    key = f"{username}:{project_id}"
    activity[key] = {
        "last_active": datetime.now().isoformat(),
        "current_image": Path(image_path).name if image_path else "",
    }
    _save_activity(activity)
    _broadcast_project_progress(project_id)

def _persist_project(project_id):
    ps = get_project_state(project_id)
    if not ps["image_list"]:
        return
    by_folder = {}
    for img_path, anns in ps["annotations"].items():
        folder = str(Path(img_path).parent)
        by_folder.setdefault(folder, {})[img_path] = anns
    for folder, anns in by_folder.items():
        ann_file = Path(folder) / "annotations.json"
        ann_file.write_text(json.dumps({
            "annotations": anns,
            "classes": ps["classes"],
            "class_colors": ps["class_colors"],
        }, indent=2, ensure_ascii=False))

@app.route("/login", methods=["GET"])
def login_page():
    if _current_user():
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.json
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")
    users = _load_users()
    if username not in users:
        return jsonify({"ok": False, "error": "Usuario no encontrado"}), 401
    if users[username]["password"] != _hash_pw(password):
        return jsonify({"ok": False, "error": "Contraseña incorrecta"}), 401
    session["username"] = username
    session["role"] = users[username].get("role", "annotator")
    session["display_name"] = users[username].get("display_name", username)
    return jsonify({"ok": True, "username": username, "role": session["role"],
                    "display_name": session["display_name"]})

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/auth/me")
def api_me():
    if not _current_user():
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "username": _current_user(),
                    "role": session.get("role"), "display_name": session.get("display_name"),
                    "is_superadmin": _is_superadmin(), "is_admin": _is_any_admin()})

@app.route("/api/auth/change_password", methods=["POST"])
@_require_login
def api_change_password():
    data = request.json
    old_pw = data.get("old_password", "")
    new_pw = data.get("new_password", "")
    username = _current_user()
    users = _load_users()
    if users[username]["password"] != _hash_pw(old_pw):
        return jsonify({"ok": False, "error": "Contraseña actual incorrecta"}), 400
    if len(new_pw) < 6:
        return jsonify({"ok": False, "error": "Mínimo 6 caracteres"}), 400
    users[username]["password"] = _hash_pw(new_pw)
    _save_users(users)
    return jsonify({"ok": True})

@app.route("/admin")
@_require_login
@_require_any_admin
def admin_page():
    return render_template("admin.html")

@app.route("/api/admin/users", methods=["GET"])
@_require_login
@_require_any_admin
def api_admin_users():
    users = _load_users()
    result = []
    for uname, udata in users.items():
        result.append({"username": uname, "display_name": udata.get("display_name", uname),
                       "role": udata.get("role", "annotator"), "created_at": udata.get("created_at", ""),
                       "admin_projects": udata.get("admin_projects", [])})
    return jsonify({"users": result})

@app.route("/api/admin/create_user", methods=["POST"])
@_require_login
@_require_any_admin
def api_admin_create_user():
    data = request.json
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")
    display_name = data.get("display_name", username)
    role = data.get("role", "annotator")
    if role == "superadmin" and not _is_superadmin():
        return jsonify({"ok": False, "error": "Solo el superadmin puede crear otros superadmins"}), 403
    if not username or not password:
        return jsonify({"ok": False, "error": "Usuario y contraseña requeridos"}), 400
    if len(password) < 6:
        return jsonify({"ok": False, "error": "Contraseña mínimo 6 caracteres"}), 400
    users = _load_users()
    if username in users:
        return jsonify({"ok": False, "error": "El usuario ya existe"}), 400
    users[username] = {"password": _hash_pw(password), "role": role,
                       "created_at": datetime.now().isoformat(),
                       "display_name": display_name, "admin_projects": []}
    _save_users(users)
    return jsonify({"ok": True, "username": username})

@app.route("/api/admin/delete_user", methods=["POST"])
@_require_login
@_require_any_admin
def api_admin_delete_user():
    username = request.json.get("username")
    if username == "admin":
        return jsonify({"ok": False, "error": "No puedes eliminar al admin principal"}), 400
    users = _load_users()
    users.pop(username, None)
    _save_users(users)
    data = _load_projects()
    for p in data["projects"]:
        p.get("assignments", {}).pop(username, None)
    _save_projects(data)
    return jsonify({"ok": True})

@app.route("/api/admin/reset_password", methods=["POST"])
@_require_login
@_require_any_admin
def api_admin_reset_password():
    data = request.json
    username = data.get("username")
    new_pw = data.get("new_password", "")
    if len(new_pw) < 6:
        return jsonify({"ok": False, "error": "Mínimo 6 caracteres"}), 400
    users = _load_users()
    if username not in users:
        return jsonify({"ok": False, "error": "Usuario no encontrado"}), 404
    users[username]["password"] = _hash_pw(new_pw)
    _save_users(users)
    return jsonify({"ok": True})

@app.route("/api/admin/set_local_admin", methods=["POST"])
@_require_login
@_require_superadmin
def api_set_local_admin():
    data = request.json
    username = data.get("username")
    project_ids = data.get("project_ids", [])
    users = _load_users()
    if username not in users:
        return jsonify({"ok": False, "error": "Usuario no encontrado"}), 404
    if project_ids:
        users[username]["role"] = "local_admin"
        users[username]["admin_projects"] = project_ids
    else:
        users[username]["role"] = "annotator"
        users[username]["admin_projects"] = []
    _save_users(users)
    return jsonify({"ok": True})

@app.route("/api/projects", methods=["GET"])
@_require_login
def api_list_projects():
    username = _current_user()
    projects = _get_user_projects(username)
    result = []
    for p in projects:
        ps = get_project_state(p["id"])
        total = len(ps["image_list"])
        annotated = sum(1 for path in ps["image_list"] if ps["annotations"].get(path))
        result.append({"id": p["id"], "name": p["name"], "path": p.get("path", ""),
                       "classes": p.get("classes", []), "created_at": p.get("created_at", ""),
                       "total_images": total, "annotated_images": annotated,
                       "is_admin": _is_local_admin_of(p["id"])})
    return jsonify({"projects": result})

def _detect_classes_from_folder(folder, fallback_classes):
    detected = {}
    try:
        for ann_file in Path(folder).rglob("annotations.json"):
            try:
                saved = json.loads(ann_file.read_text(encoding="utf-8"))
                saved_classes = saved.get("classes", [])
                saved_colors = saved.get("class_colors", {})
                for cls in saved_classes:
                    if cls not in detected:
                        detected[cls] = saved_colors.get(cls, COLOR_PALETTE[len(detected) % len(COLOR_PALETTE)])
            except Exception:
                continue
    except Exception:
        pass
    if detected:
        classes = list(detected.keys())
        print(f"[INFO] Clases detectadas: {classes}")
        return classes, detected
    classes = fallback_classes if fallback_classes else ["objeto"]
    colors = {c: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, c in enumerate(classes)}
    return classes, colors

@app.route("/api/admin/create_project", methods=["POST"])
@_require_login
@_require_superadmin
def api_create_project():
    data = request.json
    name = data.get("name", "").strip()
    path = data.get("path", "").strip()
    classes = data.get("classes", ["objeto"])
    if not name:
        return jsonify({"ok": False, "error": "El nombre es requerido"}), 400
    if path and not os.path.isdir(path):
        return jsonify({"ok": False, "error": f"Carpeta no encontrada: {path}"}), 400
    project_id = str(uuid.uuid4())[:8]
    project = {"id": project_id, "name": name, "path": path, "classes": classes,
               "class_colors": {c: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, c in enumerate(classes)},
               "created_at": datetime.now().isoformat(), "assignments": {}}
    if path:
        detected_classes, detected_colors = _detect_classes_from_folder(path, classes)
        project["classes"] = detected_classes
        project["class_colors"] = detected_colors
        _save_project(project)
        _load_project_images(project_id, path, detected_classes, detected_colors)
    else:
        _save_project(project)
    return jsonify({"ok": True, "project_id": project_id, "name": name, "classes": project["classes"]})

@app.route("/api/admin/update_project", methods=["POST"])
@_require_login
def api_update_project():
    data = request.json
    project_id = data.get("project_id")
    if not _is_local_admin_of(project_id):
        return jsonify({"ok": False, "error": "Sin permisos"}), 403
    project = _get_project(project_id)
    if not project:
        return jsonify({"ok": False, "error": "Proyecto no encontrado"}), 404
    if "name" in data:
        project["name"] = data["name"]
    if "path" in data and data["path"] != project.get("path"):
        new_path = data["path"]
        if not os.path.isdir(new_path):
            return jsonify({"ok": False, "error": f"Carpeta no encontrada: {new_path}"}), 400
        project["path"] = new_path
        _load_project_images(project_id, new_path, project["classes"], project["class_colors"])
    _save_project(project)
    return jsonify({"ok": True})

@app.route("/api/admin/delete_project", methods=["POST"])
@_require_login
@_require_superadmin
def api_delete_project():
    project_id = request.json.get("project_id")
    data = _load_projects()
    data["projects"] = [p for p in data["projects"] if p["id"] != project_id]
    _save_projects(data)
    PROJECT_STATE.pop(project_id, None)
    return jsonify({"ok": True})

@app.route("/api/admin/project_classes", methods=["POST"])
@_require_login
def api_update_project_classes():
    data = request.json
    project_id = data.get("project_id")
    if not _is_local_admin_of(project_id):
        return jsonify({"ok": False, "error": "Sin permisos"}), 403
    action = data.get("action")
    project = _get_project(project_id)
    ps = get_project_state(project_id)
    if action == "add":
        name = data.get("name", "").strip()
        if name and name not in project["classes"]:
            project["classes"].append(name)
            idx = len(project["classes"]) - 1
            project["class_colors"][name] = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            ps["classes"] = project["classes"]
            ps["class_colors"] = project["class_colors"]
    elif action == "delete":
        name = data.get("name")
        if name in project["classes"] and len(project["classes"]) > 1:
            project["classes"].remove(name)
            project["class_colors"].pop(name, None)
            ps["classes"] = project["classes"]
            ps["class_colors"] = project["class_colors"]
            if ps["active_class"] == name:
                ps["active_class"] = project["classes"][0]
    elif action == "rename":
        old = data.get("old_name")
        new = data.get("new_name", "").strip()
        if old in project["classes"] and new and new not in project["classes"]:
            idx = project["classes"].index(old)
            project["classes"][idx] = new
            project["class_colors"][new] = project["class_colors"].pop(old, "#3B82F6")
            ps["classes"] = project["classes"]
            ps["class_colors"] = project["class_colors"]
            if ps.get("active_class") == old:
                ps["active_class"] = new
            for img_path, anns in ps["annotations"].items():
                for ann in anns:
                    if ann.get("label") == old:
                        ann["label"] = new
                        ann["color"] = project["class_colors"][new]
    _save_project(project)
    _persist_project(project_id)
    return jsonify({"ok": True, "classes": project["classes"], "class_colors": project["class_colors"]})

def _load_project_images(project_id, folder, classes, class_colors):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    images = sorted([str(p) for p in Path(folder).rglob("*")
                     if p.suffix.lower() in exts and p.is_file()])
    ps = get_project_state(project_id)
    ps["image_list"] = images
    ps["classes"] = list(classes)
    ps["class_colors"] = dict(class_colors)
    ps["active_class"] = classes[0] if classes else "objeto"
    ps["annotations"] = {}
    seen = set()
    for ann_file in Path(folder).rglob("annotations.json"):
        if str(ann_file) in seen:
            continue
        seen.add(str(ann_file))
        try:
            saved = json.loads(ann_file.read_text(encoding="utf-8"))
            for k, v in saved.get("annotations", {}).items():
                ps["annotations"][k] = v
            for cls in saved.get("classes", []):
                if cls not in ps["classes"]:
                    ps["classes"].append(cls)
                    idx = len(ps["classes"]) - 1
                    ps["class_colors"][cls] = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        except Exception as e:
            print(f"[WARN] Error leyendo {ann_file}: {e}")
    return len(images)

@app.route("/api/admin/open_project", methods=["POST"])
@_require_login
def api_open_project():
    data = request.json
    project_id = data.get("project_id")
    if not _is_local_admin_of(project_id):
        return jsonify({"ok": False, "error": "Sin permisos"}), 403
    project = _get_project(project_id)
    if not project:
        return jsonify({"ok": False, "error": "Proyecto no encontrado"}), 404
    path = project.get("path", "")
    if not path or not os.path.isdir(path):
        return jsonify({"ok": False, "error": "Ruta del proyecto no válida."}), 400
    count = _load_project_images(project_id, path, project["classes"], project["class_colors"])
    ps = get_project_state(project_id)
    annotated = sum(1 for p in ps["image_list"] if ps["annotations"].get(p))
    return jsonify({"ok": True, "total": count, "annotated": annotated})

@app.route("/api/admin/assign", methods=["POST"])
@_require_login
def api_admin_assign():
    data = request.json
    project_id = data.get("project_id")
    if not _is_local_admin_of(project_id):
        return jsonify({"ok": False, "error": "Sin permisos"}), 403
    username = data.get("username")
    start = int(data.get("start", 0))
    end = int(data.get("end", 0))
    ps = get_project_state(project_id)
    total = len(ps["image_list"])
    if not total:
        return jsonify({"ok": False, "error": "El proyecto no tiene imágenes cargadas"}), 400
    start = max(0, min(start, total-1))
    end = max(start, min(end, total-1))
    project = _get_project(project_id)
    if "assignments" not in project:
        project["assignments"] = {}
    project["assignments"][username] = list(range(start, end+1))
    _save_project(project)
    _broadcast_project_progress(project_id)
    return jsonify({"ok": True, "assigned": end-start+1})

@app.route("/api/admin/assign_auto", methods=["POST"])
@_require_login
def api_admin_assign_auto():
    data = request.json
    project_id = data.get("project_id")
    if not _is_local_admin_of(project_id):
        return jsonify({"ok": False, "error": "Sin permisos"}), 403
    ps = get_project_state(project_id)
    if not ps["image_list"]:
        return jsonify({"ok": False, "error": "Proyecto sin imágenes cargadas"}), 400
    users = _load_users()
    project = _get_project(project_id)
    annotators = [u for u, d in users.items()
                  if d.get("role") == "annotator" or
                  (d.get("role") == "local_admin" and project_id in d.get("admin_projects", []))]
    if not annotators:
        return jsonify({"ok": False, "error": "No hay anotadores disponibles"}), 400
    total = len(ps["image_list"])
    chunk = total // len(annotators)
    if "assignments" not in project:
        project["assignments"] = {}
    for i, uname in enumerate(annotators):
        s = i * chunk
        e = s + chunk - 1 if i < len(annotators)-1 else total-1
        project["assignments"][uname] = list(range(s, e+1))
    _save_project(project)
    _broadcast_project_progress(project_id)
    return jsonify({"ok": True, "annotators": len(annotators), "total": total})

@app.route("/api/admin/reassign", methods=["POST"])
@_require_login
def api_admin_reassign():
    data = request.json
    project_id = data.get("project_id")
    if not _is_local_admin_of(project_id):
        return jsonify({"ok": False, "error": "Sin permisos"}), 403
    from_user = data.get("from_user")
    to_user = data.get("to_user")
    project = _get_project(project_id)
    ps = get_project_state(project_id)
    assignments = project.get("assignments", {})
    from_indices = assignments.get(from_user, [])
    pending = [i for i in from_indices
               if i < len(ps["image_list"]) and not ps["annotations"].get(ps["image_list"][i])]
    if not pending:
        return jsonify({"ok": False, "error": "No hay imágenes pendientes para reasignar"}), 400
    assignments[from_user] = [i for i in from_indices if i not in pending]
    assignments[to_user] = list(set(assignments.get(to_user, []) + pending))
    project["assignments"] = assignments
    _save_project(project)
    _broadcast_project_progress(project_id)
    return jsonify({"ok": True, "moved": len(pending)})

@app.route("/api/admin/progress")
@_require_login
def api_admin_progress():
    project_id = request.args.get("project_id")
    if not project_id or not _is_local_admin_of(project_id):
        return jsonify({"error": "Sin permisos"}), 403
    project = _get_project(project_id)
    ps = get_project_state(project_id)
    users = _load_users()
    activity = _load_activity()
    result = []
    for uname, indices in project.get("assignments", {}).items():
        user_imgs = [ps["image_list"][i] for i in indices if i < len(ps["image_list"])]
        annotated = sum(1 for p in user_imgs if ps["annotations"].get(p))
        key = f"{uname}:{project_id}"
        result.append({"username": uname, "display_name": users.get(uname, {}).get("display_name", uname),
                       "total": len(user_imgs), "annotated": annotated,
                       "pending": len(user_imgs) - annotated,
                       "pct": round(100*annotated/len(user_imgs), 1) if user_imgs else 0,
                       "last_active": activity.get(key, {}).get("last_active", ""),
                       "current_image": activity.get(key, {}).get("current_image", "")})
    total = len(ps["image_list"])
    annotated = sum(1 for p in ps["image_list"] if ps["annotations"].get(p))
    return jsonify({"users": result, "global_total": total, "global_annotated": annotated,
                    "global_pct": round(100*annotated/total, 1) if total else 0})

@app.route("/api/hardware")
@_require_login
def api_hardware():
    hw = HARDWARE_INFO or detect_hardware()
    return jsonify(hw)

@app.route("/api/training/models")
@_require_login
def api_training_models():
    return jsonify({"models": YOLO_MODELS})

@app.route("/api/training/recommend", methods=["POST"])
@_require_login
def api_training_recommend():
    data = request.json
    model_id = data.get("model_id", "yolov8s-seg")
    project_id = data.get("project_id", "")
    rec = get_training_recommendations(model_id)
    hw = HARDWARE_INFO or {}
    vram = hw.get("vram_gb", 4)
    speed_factor = max(0.5, 4 / vram) if vram > 0 else 2.0
    task_factor = 1.2 if model_id.endswith("-seg") else 1.0
    mins_per_epoch = 0.3 * speed_factor * task_factor
    total_mins = mins_per_epoch * rec["epochs"]
    rec["estimated_minutes"] = round(total_mins)
    rec["estimated_display"] = (f"~{total_mins:.0f} min" if total_mins < 60 else f"~{total_mins/60:.1f}h")
    model_info = next((m for m in YOLO_MODELS if m["id"] == model_id), {})
    task = model_info.get("task", "seg")
    rec["task"] = task
    if project_id:
        project = _get_project(project_id)
        if project and project.get("path"):
            split_dir = str(_get_split_dir(project["path"], task))
            rec["suggested_split_dir"] = split_dir
            rec["suggested_yaml"] = str(Path(split_dir) / "data.yaml")
            rec["suggested_models_dir"] = str(_get_models_dir(project["path"], task))
    return jsonify({"ok": True, "params": rec, "hardware": hw})

@app.route("/api/training/export_dataset", methods=["POST"])
@_require_login
def api_training_export_dataset():
    data = request.json
    project_id = data.get("project_id")
    if not _is_local_admin_of(project_id):
        return jsonify({"ok": False, "error": "Sin permisos"}), 403
    ps = get_project_state(project_id)
    project = _get_project(project_id)
    split_train = float(data.get("split_train", 0.7))
    split_val   = float(data.get("split_val",   0.2))
    model_id_for_task = data.get("model_id", "")
    model_info_t = next((m for m in YOLO_MODELS if m["id"] == model_id_for_task), {})
    task = model_info_t.get("task", "seg")
    if project.get("path"):
        auto_out = str(_get_split_dir(project["path"], task))
    else:
        auto_out = str(Path(".") / "DataSet_Split" / ("yolo_seg" if task == "seg" else "yolo_det"))
    out_dir = data.get("output_dir", "").strip() or auto_out
    annotated_imgs = [p for p in ps["image_list"] if ps["annotations"].get(p)]
    if not annotated_imgs:
        return jsonify({"ok": False, "error": "No hay imágenes anotadas en este proyecto"}), 400
    import random
    random.shuffle(annotated_imgs)
    n = len(annotated_imgs)
    n_train = int(n * split_train)
    n_val   = int(n * split_val)
    splits = {"train": annotated_imgs[:n_train],
              "val": annotated_imgs[n_train:n_train+n_val],
              "test": annotated_imgs[n_train+n_val:]}
    cls_list = ps["classes"]
    out = Path(out_dir)
    for split_name, paths in splits.items():
        (out / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out / split_name / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in paths:
            shutil.copy(img_path, out / split_name / "images" / Path(img_path).name)
            anns = ps["annotations"].get(img_path, [])
            img_obj = Image.open(img_path)
            w, h = img_obj.size
            lines = []
            for ann in anns:
                cid = cls_list.index(ann["label"]) if ann["label"] in cls_list else 0
                poly = ann.get("polygon", [])
                if len(poly) < 3:
                    continue
                if task == "seg":
                    coords = " ".join(f"{p[0]/w:.6f} {p[1]/h:.6f}" for p in poly)
                    lines.append(f"{cid} {coords}")
                else:
                    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                    bx = min(xs)/w; by = min(ys)/h
                    bw = (max(xs)-min(xs))/w; bh = (max(ys)-min(ys))/h
                    lines.append(f"{cid} {bx+bw/2:.6f} {by+bh/2:.6f} {bw:.6f} {bh:.6f}")
            stem = Path(img_path).stem
            (out / split_name / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    yaml_content = (f"path: {out_dir}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                    f"nc: {len(cls_list)}\nnames: {cls_list}\n")
    (out / "data.yaml").write_text(yaml_content)
    return jsonify({"ok": True, "output_dir": out_dir, "task": task,
                    "train": len(splits["train"]), "val": len(splits["val"]),
                    "test": len(splits["test"]), "yaml": str(out / "data.yaml")})

@app.route("/api/training/start", methods=["POST"])
@_require_login
def api_training_start():
    data = request.json
    project_id = data.get("project_id")
    if not _is_local_admin_of(project_id):
        return jsonify({"ok": False, "error": "Sin permisos"}), 403
    ps = get_project_state(project_id)
    if ps["train_progress"]["running"]:
        return jsonify({"ok": False, "error": "Ya hay un entrenamiento en curso"}), 400
    model_id  = data.get("model_id", "yolov8s-seg")
    epochs    = int(data.get("epochs", 100))
    batch     = int(data.get("batch", 8))
    imgsz     = int(data.get("imgsz", 640))
    workers   = int(data.get("workers", 2))
    lr0       = float(data.get("lr0", 0.01))
    patience  = int(data.get("patience", 50))
    data_yaml = data.get("data_yaml", "")
    project   = _get_project(project_id)
    model_info_train = next((m for m in YOLO_MODELS if m["id"] == model_id), {})
    task_train = model_info_train.get("task", "seg")
    if not data_yaml:
        if project.get("path"):
            data_yaml = str(_get_split_dir(project["path"], task_train) / "data.yaml")
        else:
            data_yaml = "data.yaml"
    if not Path(data_yaml).exists():
        return jsonify({"ok": False, "error": f"data.yaml no encontrado: {data_yaml}. Exporta el dataset primero."}), 400
    run_name = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if project.get("path"):
        out_dir = str(_get_models_dir(project["path"], task_train) / run_name)
    else:
        out_dir = str(Path(".") / "Modelos_Entrenados" / run_name)
    def run_training():
        tp = ps["train_progress"]
        tp.update({"running": True, "epoch": 0, "total_epochs": epochs,
                   "loss": 0, "map50": 0, "map": 0, "log": [], "status": "starting"})
        try:
            from ultralytics import YOLO
            model = YOLO(f"{model_id}.pt")
            def on_train_epoch_end(trainer):
                metrics = trainer.metrics
                tp["epoch"] = trainer.epoch + 1
                tp["total_epochs"] = epochs
                tp["loss"] = round(float(trainer.loss.item()) if hasattr(trainer.loss, 'item') else 0, 4)
                tp["map50"] = round(float(metrics.get("metrics/mAP50(M)", 0)), 4)
                tp["map"]   = round(float(metrics.get("metrics/mAP50-95(M)", 0)), 4)
                tp["status"] = f"Época {tp['epoch']}/{epochs}"
                tp["log"].append({"epoch": tp["epoch"], "loss": tp["loss"],
                                  "map50": tp["map50"], "map": tp["map"]})
                if len(tp["log"]) > 300:
                    tp["log"] = tp["log"][-300:]
                socketio.emit("train_progress", {"project_id": project_id, "progress": dict(tp)},
                              to=f"train_{project_id}")
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            device = "0" if torch.cuda.is_available() else "cpu"
            model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=imgsz,
                        workers=workers, lr0=lr0, patience=patience,
                        project=out_dir, name="train", device=device, verbose=False)
            tp["status"] = "completed"
            tp["running"] = False
            best = Path(out_dir) / "train" / "weights" / "best.pt"
            tp["best_model"] = str(best) if best.exists() else ""
            socketio.emit("train_progress", {"project_id": project_id, "progress": dict(tp)},
                          to=f"train_{project_id}")
        except Exception as e:
            import traceback; traceback.print_exc()
            ps["train_progress"]["status"] = f"error: {str(e)}"
            ps["train_progress"]["running"] = False
            socketio.emit("train_progress", {"project_id": project_id,
                          "progress": ps["train_progress"]}, to=f"train_{project_id}")
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Entrenamiento iniciado en segundo plano"})

@app.route("/api/training/stop", methods=["POST"])
@_require_login
def api_training_stop():
    project_id = request.json.get("project_id")
    if not _is_local_admin_of(project_id):
        return jsonify({"ok": False, "error": "Sin permisos"}), 403
    ps = get_project_state(project_id)
    ps["train_progress"]["running"] = False
    ps["train_progress"]["status"] = "stopped"
    return jsonify({"ok": True})

@app.route("/api/training/progress")
@_require_login
def api_training_progress():
    project_id = request.args.get("project_id")
    ps = get_project_state(project_id)
    return jsonify(ps["train_progress"])

NGROK_STATE = {"process": None, "url": None, "running": False, "log": [], "error": None}
NGROK_CONFIG_FILE = DATA_DIR / "ngrok_config.json"

def _load_ngrok_config():
    return _load_json(NGROK_CONFIG_FILE, {"authtoken": "", "ngrok_path": ""})

def _save_ngrok_config(cfg):
    _save_json(NGROK_CONFIG_FILE, cfg)

def _find_ngrok():
    cfg = _load_ngrok_config()
    if cfg.get("ngrok_path") and os.path.isfile(cfg["ngrok_path"]):
        return cfg["ngrok_path"]
    candidates = [r"D:\ngrok.exe", r"C:\ngrok\ngrok.exe",
                  r"C:\Users\Projeto_Oculus\ngrok.exe",
                  "/usr/local/bin/ngrok", "/usr/bin/ngrok"]
    for c in candidates:
        if os.path.isfile(c):
            return c
    try:
        r = subprocess.run(["where" if os.name == "nt" else "which", "ngrok"],
                           capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None

def _monitor_ngrok(proc):
    import time
    NGROK_STATE["log"] = ["▶ Iniciando ngrok..."]
    NGROK_STATE["url"] = None
    time.sleep(2)
    for attempt in range(10):
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=2) as resp:
                tunnels = json.loads(resp.read())
                for t in tunnels.get("tunnels", []):
                    if t.get("proto") == "https":
                        NGROK_STATE["url"] = t["public_url"]
                        NGROK_STATE["log"].append(f"✓ URL pública: {t['public_url']}")
                        socketio.emit("ngrok_update", {"running": True, "url": NGROK_STATE["url"],
                                                       "log": NGROK_STATE["log"]})
                        break
            if NGROK_STATE["url"]:
                break
        except Exception:
            time.sleep(1)
    try:
        for line in proc.stdout:
            line = line.strip()
            if line:
                NGROK_STATE["log"].append(line)
                if len(NGROK_STATE["log"]) > 200:
                    NGROK_STATE["log"] = NGROK_STATE["log"][-200:]
                socketio.emit("ngrok_update", {"running": NGROK_STATE["running"],
                              "url": NGROK_STATE["url"], "log": NGROK_STATE["log"][-50:]})
    except Exception:
        pass
    NGROK_STATE["running"] = False
    NGROK_STATE["process"] = None
    socketio.emit("ngrok_update", {"running": False, "url": NGROK_STATE["url"], "log": NGROK_STATE["log"]})

@app.route("/api/ngrok/config", methods=["GET"])
@_require_login
@_require_any_admin
def api_ngrok_config_get():
    cfg = _load_ngrok_config()
    ngrok_path = _find_ngrok()
    return jsonify({"authtoken": cfg.get("authtoken", ""),
                    "ngrok_path": cfg.get("ngrok_path", "") or ngrok_path or "",
                    "ngrok_found": bool(ngrok_path), "running": NGROK_STATE["running"],
                    "url": NGROK_STATE["url"], "log": NGROK_STATE["log"][-50:]})

@app.route("/api/ngrok/config", methods=["POST"])
@_require_login
@_require_any_admin
def api_ngrok_config_save():
    data = request.json
    cfg = _load_ngrok_config()
    if "authtoken" in data:
        cfg["authtoken"] = data["authtoken"].strip()
    if "ngrok_path" in data:
        cfg["ngrok_path"] = data["ngrok_path"].strip()
    _save_ngrok_config(cfg)
    if cfg.get("authtoken"):
        ngrok_path = cfg.get("ngrok_path") or _find_ngrok()
        if ngrok_path:
            try:
                subprocess.run([ngrok_path, "config", "add-authtoken", cfg["authtoken"]],
                               capture_output=True, timeout=10)
                return jsonify({"ok": True, "message": "Token guardado y configurado en ngrok"})
            except Exception as e:
                return jsonify({"ok": True, "message": f"Token guardado ({e})"})
    return jsonify({"ok": True, "message": "Configuración guardada"})

@app.route("/api/ngrok/start", methods=["POST"])
@_require_login
@_require_any_admin
def api_ngrok_start():
    if NGROK_STATE["running"]:
        return jsonify({"ok": False, "error": "Ngrok ya está corriendo"}), 400
    cfg = _load_ngrok_config()
    ngrok_path = cfg.get("ngrok_path") or _find_ngrok()
    if not ngrok_path:
        return jsonify({"ok": False, "error": "No se encontró ngrok. Especifica la ruta."}), 400
    port = int(request.json.get("port", 5000))
    try:
        proc = subprocess.Popen([ngrok_path, "http", str(port)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        NGROK_STATE.update({"process": proc, "running": True, "url": None,
                            "log": [f"▶ Iniciando ngrok en puerto {port}..."], "error": None})
        t = threading.Thread(target=_monitor_ngrok, args=(proc,), daemon=True)
        t.start()
        return jsonify({"ok": True})
    except Exception as e:
        NGROK_STATE["running"] = False
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/ngrok/stop", methods=["POST"])
@_require_login
@_require_any_admin
def api_ngrok_stop():
    proc = NGROK_STATE.get("process")
    if proc:
        try:
            proc.terminate(); proc.wait(timeout=5)
        except Exception:
            try: proc.kill()
            except Exception: pass
    NGROK_STATE.update({"running": False, "process": None, "url": None})
    NGROK_STATE["log"].append("⏹ Ngrok detenido")
    socketio.emit("ngrok_update", {"running": False, "url": None, "log": NGROK_STATE["log"][-50:]})
    return jsonify({"ok": True})

@app.route("/api/ngrok/status")
@_require_login
@_require_any_admin
def api_ngrok_status():
    return jsonify({"running": NGROK_STATE["running"], "url": NGROK_STATE["url"],
                    "log": NGROK_STATE["log"][-50:]})

@socketio.on("join_project")
def on_join_project(data):
    project_id = data.get("project_id")
    if project_id:
        join_room(f"progress_{project_id}")
        join_room(f"train_{project_id}")
        _broadcast_project_progress(project_id)

@app.route("/")
def index():
    if not _current_user():
        return redirect(url_for("login_page"))
    return render_template("index.html")

@app.route("/api/pick_folder", methods=["POST"])
@_require_login
def api_pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", 1)
        folder = filedialog.askdirectory(title="Selecciona la carpeta")
        root.destroy()
        if folder:
            return jsonify({"ok": True, "folder": folder})
        return jsonify({"ok": False, "folder": ""})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/models")
@_require_login
def api_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return jsonify({"models": get_available_sam_models(), "current": SAM_STATE["model_name"],
                    "device": device, "sam_version": SAM_STATE["sam_version"]})

@app.route("/api/load_model", methods=["POST"])
@_require_login
@_require_any_admin
def api_load_model():
    model_id = request.json.get("model_id", "sam1_vit_b")
    ok, msg = load_sam_model(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return jsonify({"ok": ok, "message": msg, "device": device})

@app.route("/api/status")
@_require_login
def api_status():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ps = get_project_state(project_id) if project_id else {}
    if project_id and _is_local_admin_of(project_id):
        user_imgs = ps.get("image_list", [])
    elif project_id:
        user_imgs = _get_user_images(username, project_id)
    else:
        user_imgs = []
    return jsonify({"sam_loaded": SAM_STATE["predictor"] is not None or SAM_STATE["sam_version"] == "demo",
                    "sam_version": SAM_STATE["sam_version"], "model_name": SAM_STATE["model_name"],
                    "device": device, "image_loaded": ustate["image_set"],
                    "active_project_id": project_id, "total_images": len(user_imgs),
                    "current_index": ustate["current_index"],
                    "classes": ps.get("classes", ["objeto"]) if ps else ["objeto"],
                    "class_colors": ps.get("class_colors", {}) if ps else {},
                    "active_class": ps.get("active_class", "objeto") if ps else "objeto",
                    "username": username, "role": session.get("role"),
                    "is_superadmin": _is_superadmin(), "is_admin": _is_any_admin()})

@app.route("/api/set_active_project", methods=["POST"])
@_require_login
def api_set_active_project():
    project_id = request.json.get("project_id")
    username = _current_user()
    user_projects = _get_user_projects(username)
    if not any(p["id"] == project_id for p in user_projects):
        return jsonify({"ok": False, "error": "Sin acceso a este proyecto"}), 403
    ustate = get_user_state(username)
    ustate["active_project_id"] = project_id
    ustate["current_index"] = 0
    ustate["image_set"] = False
    project = _get_project(project_id)
    ps = get_project_state(project_id)
    if not ps["image_list"] and project and project.get("path"):
        _load_project_images(project_id, project["path"], project["classes"], project["class_colors"])
    return jsonify({"ok": True, "project_id": project_id})

@app.route("/api/load_image", methods=["POST"])
@_require_login
def api_load_image():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    if not project_id:
        return jsonify({"error": "No hay proyecto activo. Selecciona un proyecto primero."}), 400
    ps = get_project_state(project_id)
    data = request.json
    idx = data.get("index")
    if _is_local_admin_of(project_id):
        user_imgs = ps["image_list"]
    else:
        user_imgs = _get_user_images(username, project_id)
    if not user_imgs:
        return jsonify({"error": "No tienes imágenes asignadas en este proyecto."}), 400
    if idx is None:
        idx = ustate.get("current_index", 0)
    idx = max(0, min(idx, len(user_imgs)-1))
    ustate["current_index"] = idx
    path = user_imgs[idx]
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    ustate["image_np"] = img_np
    ustate["image_path"] = path
    ustate["image_set"] = True
    _embed_image(img_np)
    _update_activity(username, project_id, path)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    annotations = ps["annotations"].get(path, [])
    return jsonify({"ok": True, "image_b64": b64, "width": img_np.shape[1], "height": img_np.shape[0],
                    "filename": os.path.basename(path), "index": idx, "total": len(user_imgs),
                    "annotations": annotations, "classes": ps["classes"],
                    "class_colors": ps["class_colors"], "active_class": ps["active_class"]})

@app.route("/api/my_images")
@_require_login
def api_my_images():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    if not project_id:
        return jsonify({"images": [], "total": 0})
    ps = get_project_state(project_id)
    user_imgs = ps["image_list"] if _is_local_admin_of(project_id) else _get_user_images(username, project_id)
    result = [{"index": i, "filename": Path(p).name, "subfolder": Path(p).parent.name,
               "annotated": len(ps["annotations"].get(p, []))} for i, p in enumerate(user_imgs)]
    return jsonify({"images": result, "total": len(result)})

@app.route("/api/resume_index")
@_require_login
def api_resume_index():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    if not project_id:
        return jsonify({"index": 0})
    ps = get_project_state(project_id)
    user_imgs = ps["image_list"] if _is_local_admin_of(project_id) else _get_user_images(username, project_id)
    for i, path in enumerate(user_imgs):
        if not ps["annotations"].get(path):
            return jsonify({"index": i})
    return jsonify({"index": max(0, len(user_imgs)-1)})

@app.route("/api/predict", methods=["POST"])
@_require_login
def api_predict():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    ps = get_project_state(project_id) if project_id else {}
    data = request.json
    x, y = int(data.get("x", 0)), int(data.get("y", 0))
    label = data.get("label", ps.get("active_class", "objeto") if ps else "objeto")
    negative_points = data.get("negative_points", [])
    if not ustate["image_set"]:
        return jsonify({"error": "No hay imagen cargada"}), 400
    img_np = ustate["image_np"]
    h, w = img_np.shape[:2]
    x, y = max(0, min(x, w-1)), max(0, min(y, h-1))
    try:
        if SAM_STATE["predictor"] is None or SAM_STATE["sam_version"] == "demo":
            mask = demo_mask(img_np, x, y)
        else:
            point_coords = [[x, y]] + negative_points
            point_labels = [1] + [0]*len(negative_points)
            if SAM_STATE["sam_version"] == "sam2":
                with torch.inference_mode():
                    masks, scores, _ = SAM_STATE["predictor"].predict(
                        point_coords=np.array(point_coords), point_labels=np.array(point_labels),
                        multimask_output=True)
            else:
                masks, scores, _ = SAM_STATE["predictor"].predict(
                    point_coords=np.array(point_coords), point_labels=np.array(point_labels),
                    multimask_output=True)
            mask = masks[int(np.argmax(scores))]
        polygon = mask_to_polygon(mask)
        bbox = mask_to_bbox(mask)
        color_hex = ps.get("class_colors", {}).get(label, "#3B82F6") if ps else "#3B82F6"
        r_, g_, b_ = int(color_hex[1:3],16), int(color_hex[3:5],16), int(color_hex[5:7],16)
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        mask_bool = mask.astype(bool)
        overlay[mask_bool, 0]=r_; overlay[mask_bool, 1]=g_
        overlay[mask_bool, 2]=b_; overlay[mask_bool, 3]=150
        pil_ov = Image.fromarray(overlay, mode="RGBA")
        buf = io.BytesIO(); pil_ov.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode()
        ann = {"id": str(uuid.uuid4())[:8], "label": label, "polygon": polygon,
               "bbox": bbox, "color": color_hex, "click": [x, y]}
        return jsonify({"ok": True, "annotation": ann, "mask_b64": mask_b64})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/save_annotation", methods=["POST"])
@_require_login
def api_save_annotation():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    ps = get_project_state(project_id) if project_id else None
    ann = request.json.get("annotation")
    path = ustate["image_path"]
    if not path or not ps:
        return jsonify({"error": "Sin imagen o proyecto activo"}), 400
    if path not in ps["annotations"]:
        ps["annotations"][path] = []
    ps["annotations"][path] = [a for a in ps["annotations"][path] if a["id"] != ann["id"]]
    ps["annotations"][path].append(ann)
    _persist_project(project_id)
    _broadcast_project_progress(project_id)
    return jsonify({"ok": True})

@app.route("/api/save_polygon", methods=["POST"])
@_require_login
def api_save_polygon():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    ps = get_project_state(project_id) if project_id else None
    data = request.json
    polygon = data.get("polygon", [])
    label = data.get("label", ps["active_class"] if ps else "objeto")
    ann_id = data.get("id", str(uuid.uuid4())[:8])
    path = ustate["image_path"]
    if not path or not ps or len(polygon) < 3:
        return jsonify({"error": "Polígono inválido"}), 400
    img_np = ustate["image_np"]
    h, w = img_np.shape[:2]
    mask = polygon_to_mask(polygon, h, w)
    bbox = mask_to_bbox(mask)
    color_hex = ps["class_colors"].get(label, "#3B82F6")
    ann = {"id": ann_id, "label": label, "polygon": polygon, "bbox": bbox, "color": color_hex, "click": None}
    if path not in ps["annotations"]:
        ps["annotations"][path] = []
    ps["annotations"][path] = [a for a in ps["annotations"][path] if a["id"] != ann_id]
    ps["annotations"][path].append(ann)
    _persist_project(project_id)
    _broadcast_project_progress(project_id)
    return jsonify({"ok": True, "annotation": ann})

@app.route("/api/update_polygon", methods=["POST"])
@_require_login
def api_update_polygon():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    ps = get_project_state(project_id) if project_id else None
    data = request.json
    ann_id = data.get("id")
    polygon = data.get("polygon", [])
    path = ustate["image_path"]
    if not ps or path not in ps["annotations"]:
        return jsonify({"error": "No annotations"}), 400
    img_np = ustate["image_np"]
    h, w = img_np.shape[:2]
    mask = polygon_to_mask(polygon, h, w)
    bbox = mask_to_bbox(mask)
    for ann in ps["annotations"][path]:
        if ann["id"] == ann_id:
            ann["polygon"] = polygon; ann["bbox"] = bbox; break
    _persist_project(project_id)
    return jsonify({"ok": True})

@app.route("/api/delete_annotation", methods=["POST"])
@_require_login
def api_delete_annotation():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    ps = get_project_state(project_id) if project_id else None
    ann_id = request.json.get("id")
    path = ustate["image_path"]
    if ps and path in ps["annotations"]:
        ps["annotations"][path] = [a for a in ps["annotations"][path] if a["id"] != ann_id]
        _persist_project(project_id)
        _broadcast_project_progress(project_id)
    return jsonify({"ok": True})

@app.route("/api/update_class", methods=["POST"])
@_require_login
def api_update_class():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    ps = get_project_state(project_id) if project_id else None
    data = request.json
    action = data.get("action")
    if not ps:
        return jsonify({"error": "Sin proyecto activo"}), 400
    if action == "add":
        name = data.get("name", "").strip()
        if name and name not in ps["classes"]:
            ps["classes"].append(name)
            idx = len(ps["classes"])-1
            ps["class_colors"][name] = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        ps["active_class"] = name
    elif action == "set_active":
        name = data.get("name")
        if name in ps["classes"]:
            ps["active_class"] = name
    elif action == "delete":
        name = data.get("name")
        if name in ps["classes"] and len(ps["classes"]) > 1:
            ps["classes"].remove(name)
            ps["class_colors"].pop(name, None)
            ps["active_class"] = ps["classes"][0]
    if project_id:
        project = _get_project(project_id)
        if project:
            project["classes"] = ps["classes"]
            project["class_colors"] = ps["class_colors"]
            _save_project(project)
    _persist_project(project_id)
    return jsonify({"ok": True, "classes": ps["classes"],
                    "class_colors": ps["class_colors"], "active_class": ps["active_class"]})

@app.route("/api/annotations_summary")
@_require_login
def api_annotations_summary():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    if not project_id:
        return jsonify({"per_image": {}, "total": 0, "annotated": 0, "subfolder_stats": {}})
    ps = get_project_state(project_id)
    user_imgs = ps["image_list"] if _is_local_admin_of(project_id) else _get_user_images(username, project_id)
    per_image = {p: len(ps["annotations"].get(p, [])) for p in user_imgs}
    annotated = sum(1 for p in user_imgs if ps["annotations"].get(p))
    subfolder_stats = {}
    for img_path in user_imgs:
        name = Path(img_path).parent.name
        subfolder_stats.setdefault(name, {"total": 0, "annotated": 0})
        subfolder_stats[name]["total"] += 1
        if ps["annotations"].get(img_path):
            subfolder_stats[name]["annotated"] += 1
    return jsonify({"per_image": per_image, "total": len(user_imgs),
                    "annotated": annotated, "subfolder_stats": subfolder_stats})

@app.route("/api/aug_progress")
@_require_login
def api_aug_progress():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    if not project_id:
        return jsonify({})
    ps = get_project_state(project_id)
    return jsonify(ps.get("aug_progress", {}))

@app.route("/api/export", methods=["POST"])
@_require_login
@_require_any_admin
def api_export():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    if not project_id or not _is_local_admin_of(project_id):
        return jsonify({"error": "Sin permisos"}), 403
    data = request.json
    fmt = data.get("format", "yolo_seg")
    project = _get_project(project_id)
    out_dir = data.get("output_dir", "").strip() or str(Path(project.get("path", ".")) / "export")
    ps = get_project_state(project_id)
    try:
        fns = {"yolo_seg": export_yolo_seg, "yolo_det": export_yolo_det,
               "coco": export_coco, "pascal_voc": export_pascal_voc,
               "labelme": export_labelme, "createml": export_createml,
               "tfrecord_csv": export_tfrecord_csv}
        if fmt not in fns:
            return jsonify({"error": f"Formato desconocido: {fmt}"}), 400
        result = fns[fmt](out_dir, ps)
        return jsonify({"ok": True, "output_dir": out_dir, "details": result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def export_yolo_seg(out_dir, ps):
    cls = ps["classes"]
    img_dir = Path(out_dir) / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir = Path(out_dir) / "labels"; lbl_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path, anns in ps["annotations"].items():
        if not anns: continue
        img = Image.open(img_path); w, h = img.size
        stem = Path(img_path).stem
        shutil.copy(img_path, img_dir / Path(img_path).name)
        lines = []
        for ann in anns:
            cid = cls.index(ann["label"]) if ann["label"] in cls else 0
            poly = ann.get("polygon", [])
            if len(poly) < 3: continue
            coords = " ".join(f"{p[0]/w:.6f} {p[1]/h:.6f}" for p in poly)
            lines.append(f"{cid} {coords}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        count += 1
    (Path(out_dir) / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: images\nval: images\nnc: {len(cls)}\nnames: {cls}\n")
    return {"images": count, "format": "YOLO Segmentation"}

def export_yolo_det(out_dir, ps):
    cls = ps["classes"]
    img_dir = Path(out_dir) / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir = Path(out_dir) / "labels"; lbl_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path, anns in ps["annotations"].items():
        if not anns: continue
        img = Image.open(img_path); w, h = img.size
        stem = Path(img_path).stem
        shutil.copy(img_path, img_dir / Path(img_path).name)
        lines = []
        for ann in anns:
            cid = cls.index(ann["label"]) if ann["label"] in cls else 0
            bx, by, bw, bh = ann.get("bbox", [0,0,1,1])
            lines.append(f"{cid} {(bx+bw/2)/w:.6f} {(by+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        count += 1
    (Path(out_dir) / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: images\nval: images\nnc: {len(cls)}\nnames: {cls}\n")
    return {"images": count, "format": "YOLO Detection"}

def export_coco(out_dir, ps):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cls = ps["classes"]
    categories = [{"id": i+1, "name": c, "supercategory": "none"} for i, c in enumerate(cls)]
    images_list, anns_list, aid = [], [], 1
    for iid, (img_path, anns) in enumerate(ps["annotations"].items(), start=1):
        if not anns: continue
        img = Image.open(img_path); w, h = img.size
        images_list.append({"id": iid, "file_name": os.path.basename(img_path), "width": w, "height": h})
        for ann in anns:
            cid = cls.index(ann["label"])+1 if ann["label"] in cls else 1
            poly = ann.get("polygon", [])
            seg = [coord for p in poly for coord in p]
            bbox = ann.get("bbox", [0,0,0,0])
            anns_list.append({"id": aid, "image_id": iid, "category_id": cid,
                              "segmentation": [seg] if seg else [], "area": bbox[2]*bbox[3],
                              "bbox": bbox, "iscrowd": 0})
            aid += 1
    out = {"info": {"description": "SAM Annotator v3"},
           "categories": categories, "images": images_list, "annotations": anns_list}
    (Path(out_dir) / "annotations.json").write_text(json.dumps(out, indent=2))
    return {"annotations": aid-1, "format": "COCO JSON"}

def export_pascal_voc(out_dir, ps):
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path, anns in ps["annotations"].items():
        if not anns: continue
        img = Image.open(img_path); w, h = img.size
        root = Element("annotation")
        SubElement(root, "filename").text = os.path.basename(img_path)
        sz = SubElement(root, "size")
        SubElement(sz, "width").text = str(w); SubElement(sz, "height").text = str(h)
        SubElement(sz, "depth").text = "3"
        for ann in anns:
            obj = SubElement(root, "object")
            SubElement(obj, "name").text = ann["label"]
            SubElement(obj, "pose").text = "Unspecified"
            SubElement(obj, "truncated").text = "0"; SubElement(obj, "difficult").text = "0"
            bx, by, bw, bh = ann.get("bbox", [0,0,w,h])
            bb = SubElement(obj, "bndbox")
            SubElement(bb, "xmin").text = str(bx); SubElement(bb, "ymin").text = str(by)
            SubElement(bb, "xmax").text = str(bx+bw); SubElement(bb, "ymax").text = str(by+bh)
        xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
        (Path(out_dir) / f"{Path(img_path).stem}.xml").write_text(xml_str)
        count += 1
    return {"files": count, "format": "Pascal VOC"}

def export_labelme(out_dir, ps):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path, anns in ps["annotations"].items():
        if not anns: continue
        img = Image.open(img_path); w, h = img.size
        shapes = [{"label": ann["label"], "points": [[p[0],p[1]] for p in ann.get("polygon",[])],
                   "group_id": None, "shape_type": "polygon", "flags": {}} for ann in anns]
        data = {"version": "5.0.1", "flags": {}, "shapes": shapes,
                "imagePath": os.path.basename(img_path), "imageData": None,
                "imageHeight": h, "imageWidth": w}
        (Path(out_dir) / f"{Path(img_path).stem}.json").write_text(json.dumps(data, indent=2))
        count += 1
    return {"files": count, "format": "LabelMe JSON"}

def export_createml(out_dir, ps):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result = []
    for img_path, anns in ps["annotations"].items():
        if not anns: continue
        annotations = []
        for ann in anns:
            bx, by, bw, bh = ann.get("bbox", [0,0,0,0])
            annotations.append({"label": ann["label"],
                                "coordinates": {"x": bx+bw//2, "y": by+bh//2, "width": bw, "height": bh}})
        result.append({"image": os.path.basename(img_path), "annotations": annotations})
    (Path(out_dir) / "annotations.json").write_text(json.dumps(result, indent=2))
    return {"images": len(result), "format": "CreateML (Apple)"}

def export_tfrecord_csv(out_dir, ps):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = ["filename,width,height,class,xmin,ymin,xmax,ymax"]
    for img_path, anns in ps["annotations"].items():
        if not anns: continue
        img = Image.open(img_path); w, h = img.size
        fname = os.path.basename(img_path)
        for ann in anns:
            bx, by, bw, bh = ann.get("bbox", [0,0,0,0])
            rows.append(f"{fname},{w},{h},{ann['label']},{bx},{by},{bx+bw},{by+bh}")
    (Path(out_dir) / "labels.csv").write_text("\n".join(rows))
    return {"rows": len(rows)-1, "format": "TFRecord CSV"}

def apply_augmentations(img, params, preview=False):
    import random
    from PIL import ImageEnhance, ImageFilter, ImageOps
    result = img.copy()
    if params.get('flip_h') and (preview or random.random() < 0.5):
        result = ImageOps.mirror(result)
    if params.get('flip_v') and (preview or random.random() < 0.5):
        result = ImageOps.flip(result)
    rot = float(params.get('rotation', 0))
    if rot > 0:
        angle = rot if preview else random.uniform(-rot, rot)
        result = result.rotate(angle, expand=False, fillcolor=(128,128,128))
    br = float(params.get('brightness', 0))
    if br > 0:
        f = 1+br if preview else 1+random.uniform(-br, br)
        result = ImageEnhance.Brightness(result).enhance(max(0.1, f))
    co = float(params.get('contrast', 0))
    if co > 0:
        f = 1+co if preview else 1+random.uniform(-co, co)
        result = ImageEnhance.Contrast(result).enhance(max(0.1, f))
    sa = float(params.get('saturation', 0))
    if sa > 0:
        f = 1+sa if preview else 1+random.uniform(-sa, sa)
        result = ImageEnhance.Color(result).enhance(max(0, f))
    hue = float(params.get('hue', 0))
    if hue > 0:
        arr = np.array(result.convert('HSV'))
        shift = int(hue*255/180) if preview else int(random.uniform(-hue,hue)*255/180)
        arr[:,:,0] = (arr[:,:,0].astype(int)+shift) % 256
        result = Image.fromarray(arr, 'HSV').convert('RGB')
    bl = float(params.get('blur', 0))
    if bl > 0:
        r = bl if preview else random.uniform(0, bl)
        if r > 0.1:
            result = result.filter(ImageFilter.GaussianBlur(radius=r))
    noise = float(params.get('noise', 0))
    if noise > 0:
        arr = np.array(result)
        prob = noise * 0.05
        rng = np.random.default_rng()
        arr[rng.random(arr.shape[:2]) < prob/2] = 255
        arr[rng.random(arr.shape[:2]) < prob/2] = 0
        result = Image.fromarray(arr)
    crop = float(params.get('crop', 0))
    if crop > 0:
        w, h = result.size
        p = crop if preview else random.uniform(0, crop)
        mx, my = int(w*p/2), int(h*p/2)
        if mx > 0 and my > 0:
            result = result.crop((mx, my, w-mx, h-my)).resize((w,h), Image.LANCZOS)
    if params.get('grayscale') and (preview or random.random() < 0.3):
        result = ImageOps.grayscale(result).convert('RGB')
    if params.get('cutout') and (preview or random.random() < 0.5):
        arr = np.array(result)
        h2, w2 = arr.shape[:2]
        cx = random.randint(w2//4, 3*w2//4); cy = random.randint(h2//4, 3*h2//4)
        bw = random.randint(w2//8, w2//3); bh = random.randint(h2//8, h2//3)
        x1, y1 = max(0, cx-bw//2), max(0, cy-bh//2)
        x2, y2 = min(w2, cx+bw//2), min(h2, cy+bh//2)
        arr[y1:y2, x1:x2] = 128
        result = Image.fromarray(arr)
    jpeg_q = int(params.get('jpeg', 100))
    if jpeg_q < 100:
        buf = io.BytesIO()
        result.save(buf, format='JPEG', quality=jpeg_q)
        buf.seek(0)
        result = Image.open(buf).copy()
    return result

@app.route('/api/preview_augmentation', methods=['POST'])
@_require_login
def api_preview_augmentation():
    username = _current_user()
    ustate = get_user_state(username)
    if not ustate['image_set']:
        return jsonify({'error': 'No hay imagen cargada'}), 400
    params = request.json.get('params', {})
    try:
        img = Image.fromarray(ustate['image_np'])
        aug = apply_augmentations(img, params, preview=True)
        buf = io.BytesIO()
        aug.save(buf, format='JPEG', quality=85)
        return jsonify({'ok': True, 'image_b64': base64.b64encode(buf.getvalue()).decode()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_augmented', methods=['POST'])
@_require_login
@_require_any_admin
def api_export_augmented():
    username = _current_user()
    ustate = get_user_state(username)
    project_id = ustate.get("active_project_id")
    if not project_id or not _is_local_admin_of(project_id):
        return jsonify({'error': 'Sin permisos'}), 403
    ps = get_project_state(project_id)
    project = _get_project(project_id)
    if not ps['image_list']:
        return jsonify({'error': 'Proyecto sin imágenes'}), 400
    data = request.json
    params = data.get('params', {})
    copies = max(1, int(data.get('copies', 2)))
    out_dir = data.get('output_dir', '').strip()
    mode = data.get('mode', 'dataset')
    if not out_dir:
        out_dir = str(Path(project.get('path', '.')) / ('augmented' if mode == 'dataset' else 'Filtros_Transformaciones'))
    processed, errors = 0, []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ann_dict = {}
    total_to_process = len(ps['image_list'])
    ps['aug_progress'] = {"running": True, "current": 0, "total": total_to_process, "label": "Iniciando..."}
    for img_path in ps['image_list']:
        try:
            orig = Image.open(img_path).convert('RGB')
            w, h = orig.size
            stem = Path(img_path).stem
            ext = Path(img_path).suffix
            anns = ps['annotations'].get(img_path, [])
            for i in range(copies):
                aug_img, aug_anns = apply_augmentations_with_annotations(orig, anns, params, w, h)
                new_name = f"{stem}_aug{i+1}{ext}"
                aug_img.save(str(Path(out_dir) / new_name))
                if aug_anns:
                    ann_dict[str(Path(out_dir) / new_name)] = aug_anns
            processed += 1
            ps['aug_progress'] = {"running": True, "current": processed, "total": total_to_process,
                                  "label": f"Procesando: {Path(img_path).name}"}
        except Exception as e:
            errors.append(f'{Path(img_path).name}: {e}')
    ps['aug_progress'] = {"running": False, "current": processed, "total": total_to_process, "label": "Completado"}
    ann_file = Path(out_dir) / 'annotations.json'
    ann_file.write_text(json.dumps({'annotations': ann_dict, 'classes': ps['classes'],
                                    'class_colors': ps['class_colors']}, indent=2))
    return jsonify({'ok': True, 'processed': processed, 'output_dir': out_dir,
                    'total_images': processed * copies, 'errors': errors[:5]})

def apply_augmentations_with_annotations(img, anns, params, w, h, deterministic=False):
    import random, math
    from PIL import ImageEnhance, ImageFilter, ImageOps as _io
    do_flip_h = params.get('flip_h') and (deterministic or random.random() < 0.5)
    do_flip_v = params.get('flip_v') and (deterministic or random.random() < 0.5)
    rot_max = float(params.get('rotation', 0))
    angle = rot_max if deterministic else (random.uniform(-rot_max, rot_max) if rot_max > 0 else 0)
    crop_max = float(params.get('crop', 0))
    crop_p = crop_max if deterministic else (random.uniform(0, crop_max) if crop_max > 0 else 0)
    br = float(params.get('brightness', 0))
    br_f = 1+br if deterministic else (1+random.uniform(-br, br) if br > 0 else 1)
    co = float(params.get('contrast', 0))
    co_f = 1+co if deterministic else (1+random.uniform(-co, co) if co > 0 else 1)
    sa = float(params.get('saturation', 0))
    sa_f = 1+sa if deterministic else (1+random.uniform(-sa, sa) if sa > 0 else 1)
    hue = float(params.get('hue', 0))
    hue_shift = int(hue*255/180) if deterministic else (int(random.uniform(-hue,hue)*255/180) if hue > 0 else 0)
    bl = float(params.get('blur', 0))
    blur_r = bl if deterministic else (random.uniform(0, bl) if bl > 0 else 0)
    do_gray = params.get('grayscale') and (deterministic or random.random() < 0.3)
    do_cutout = params.get('cutout') and (deterministic or random.random() < 0.5)
    jpeg_q = int(params.get('jpeg', 100))
    result = img.copy()
    if do_flip_h: result = _io.mirror(result)
    if do_flip_v: result = _io.flip(result)
    if angle != 0: result = result.rotate(angle, expand=False, fillcolor=(128,128,128))
    if crop_p > 0:
        mx, my = int(w*crop_p/2), int(h*crop_p/2)
        if mx > 0 and my > 0:
            result = result.crop((mx, my, w-mx, h-my)).resize((w, h), Image.LANCZOS)
    if br_f != 1: result = ImageEnhance.Brightness(result).enhance(max(0.1, br_f))
    if co_f != 1: result = ImageEnhance.Contrast(result).enhance(max(0.1, co_f))
    if sa_f != 1: result = ImageEnhance.Color(result).enhance(max(0, sa_f))
    if hue_shift != 0:
        arr = np.array(result.convert('HSV'))
        arr[:,:,0] = (arr[:,:,0].astype(int)+hue_shift) % 256
        result = Image.fromarray(arr, 'HSV').convert('RGB')
    if blur_r > 0.1: result = result.filter(ImageFilter.GaussianBlur(radius=blur_r))
    if params.get('noise', 0) > 0:
        arr = np.array(result)
        prob = float(params['noise']) * 0.05
        rng = np.random.default_rng()
        arr[rng.random(arr.shape[:2]) < prob/2] = 255
        arr[rng.random(arr.shape[:2]) < prob/2] = 0
        result = Image.fromarray(arr)
    if do_gray: result = _io.grayscale(result).convert('RGB')
    if do_cutout:
        arr = np.array(result)
        cx = random.randint(w//4, 3*w//4); cy_c = random.randint(h//4, 3*h//4)
        bw2 = random.randint(w//8, w//3); bh2 = random.randint(h//8, h//3)
        x1, y1 = max(0, cx-bw2//2), max(0, cy_c-bh2//2)
        x2, y2 = min(w, cx+bw2//2), min(h, cy_c+bh2//2)
        arr[y1:y2, x1:x2] = 128
        result = Image.fromarray(arr)
    if jpeg_q < 100:
        buf = io.BytesIO()
        result.save(buf, format='JPEG', quality=jpeg_q)
        buf.seek(0)
        result = Image.open(buf).copy()
    aug_anns = []
    for ann in anns:
        new_ann = dict(ann)
        poly = ann.get('polygon', [])
        if not poly:
            aug_anns.append(new_ann); continue
        pts = [[float(p[0]), float(p[1])] for p in poly]
        if do_flip_h: pts = [[w-p[0], p[1]] for p in pts]
        if do_flip_v: pts = [[p[0], h-p[1]] for p in pts]
        if angle != 0:
            angle_rad = math.radians(-angle)
            cx_r, cy_r = w/2, h/2
            new_pts = []
            for p in pts:
                dx, dy = p[0]-cx_r, p[1]-cy_r
                rx = dx*math.cos(angle_rad)-dy*math.sin(angle_rad)+cx_r
                ry = dx*math.sin(angle_rad)+dy*math.cos(angle_rad)+cy_r
                new_pts.append([max(0,min(w,rx)), max(0,min(h,ry))])
            pts = new_pts
        if crop_p > 0:
            mx, my = w*crop_p/2, h*crop_p/2
            new_w, new_h = w-2*mx, h-2*my
            if new_w > 0 and new_h > 0:
                pts = [[(p[0]-mx)*w/new_w, (p[1]-my)*h/new_h] for p in pts]
                pts = [[max(0,min(w,x)), max(0,min(h,y))] for x, y in pts]
        new_poly = [[int(x), int(y)] for x, y in pts]
        new_ann['polygon'] = new_poly
        xs = [p[0] for p in new_poly]; ys = [p[1] for p in new_poly]
        if xs and ys:
            new_ann['bbox'] = [int(min(xs)), int(min(ys)), int(max(xs)-min(xs)), int(max(ys)-min(ys))]
        aug_anns.append(new_ann)
    return result, aug_anns

# ─── Auto-patch de templates al arrancar ──────────────────────────────────────
# Aplica correcciones a admin.html automáticamente si aún no están presentes.
# No requiere ningún paso manual: se ejecuta cada vez que corres python app.py.

def _autopatch_admin():
    import shutil as _shutil
    from pathlib import Path as _Path

    admin_path = _Path("templates") / "admin.html"
    if not admin_path.exists():
        print("[PATCH] templates/admin.html no encontrado, omitiendo.")
        return

    content = admin_path.read_text(encoding="utf-8")
    MARKER = "// [PATCH-v3.1-APPLIED]"
    if MARKER in content:
        print("[PATCH] admin.html ya esta actualizado.")
        return

    _shutil.copy(str(admin_path), str(admin_path) + ".bak")
    ok_list, fail_list = [], []

    def ap(old, new, name):
        nonlocal content
        if old in content:
            content = content.replace(old, new, 1)
            ok_list.append(name)
        else:
            fail_list.append(name)

    ap(
        "  [trainSel, progSel].forEach(sel => {\n"
        "    sel.innerHTML = '<option value=\"\">— Selecciona proyecto —</option>' +\n"
        "      projects.map(p => `<option value=\"${p.id}\">${p.name}</option>`).join(\"\");\n"
        "  });\n"
        "}",
        "  const prevTrain = trainSel.value;\n"
        "  const prevProg = progSel.value;\n"
        "  [trainSel, progSel].forEach(sel => {\n"
        "    sel.innerHTML = '<option value=\"\">— Selecciona proyecto —</option>' +\n"
        "      projects.map(p => `<option value=\"${p.id}\">${p.name}</option>`).join(\"\");\n"
        "  });\n"
        "  if (prevTrain) trainSel.value = prevTrain;\n"
        "  if (prevProg) progSel.value = prevProg;\n"
        "}",
        "FIX-1a"
    )

    ap(
        "let selectedProjectId = null;\nlet meRole = \"\";\nlet projects = [];\nlet trainChart = null;\nlet socket = null;",
        "let selectedProjectId = null;\nlet meRole = \"\";\nlet projects = [];\nlet trainChart = null;\nlet socket = null;\nlet _trainParamsLocked = false;\nlet _lastTrainProject = null;\nlet _lastTrainModel = null;",
        "FIX-1b"
    )

    ap(
        "async function onTrainProjectChange() {\n"
        "  const project_id = document.getElementById(\"train-project-sel\").value;\n"
        "  if (!project_id) return;\n"
        "  loadRecommendations();\n"
        "  if (socket && project_id) socket.emit(\"join_project\", {project_id});\n"
        "  const tp = await (await fetch(`/api/training/progress?project_id=${project_id}`)).json();\n"
        "  updateTrainUI(tp);\n"
        "}",
        "async function onTrainProjectChange() {\n"
        "  const project_id = document.getElementById(\"train-project-sel\").value;\n"
        "  if (!project_id) return;\n"
        "  if (project_id !== _lastTrainProject) {\n"
        "    _trainParamsLocked = false;\n"
        "    _lastTrainProject = project_id;\n"
        "    _lastTrainModel = null;\n"
        "  }\n"
        "  loadRecommendations();\n"
        "  if (socket && project_id) socket.emit(\"join_project\", {project_id});\n"
        "  const tp = await (await fetch(`/api/training/progress?project_id=${project_id}`)).json();\n"
        "  updateTrainUI(tp);\n"
        "}",
        "FIX-1c"
    )

    ap(
        "  const d = await post(\"/api/training/recommend\", {model_id, project_id});\n"
        "  if (!d.ok) return;\n"
        "  const p = d.params;\n"
        "  document.getElementById(\"param-epochs\").value = p.epochs;\n"
        "  document.getElementById(\"param-batch\").value = p.batch;\n"
        "  document.getElementById(\"param-imgsz\").value = p.imgsz;\n"
        "  document.getElementById(\"param-workers\").value = p.workers;\n"
        "  document.getElementById(\"param-lr\").value = p.lr0;\n"
        "  document.getElementById(\"param-patience\").value = p.patience;\n"
        "  // Rutas automaticas sugeridas\n"
        "  if (p.suggested_yaml) {\n"
        "    document.getElementById(\"param-yaml\").value = p.suggested_yaml;\n"
        "  }",
        "  const d = await post(\"/api/training/recommend\", {model_id, project_id});\n"
        "  if (!d.ok) return;\n"
        "  const p = d.params;\n"
        "  const modelChanged = model_id !== _lastTrainModel;\n"
        "  if (!_trainParamsLocked || modelChanged) {\n"
        "    document.getElementById(\"param-epochs\").value = p.epochs;\n"
        "    document.getElementById(\"param-batch\").value = p.batch;\n"
        "    document.getElementById(\"param-imgsz\").value = p.imgsz;\n"
        "    document.getElementById(\"param-workers\").value = p.workers;\n"
        "    document.getElementById(\"param-lr\").value = p.lr0;\n"
        "    document.getElementById(\"param-patience\").value = p.patience;\n"
        "    _lastTrainModel = model_id;\n"
        "    _trainParamsLocked = false;\n"
        "  }\n"
        "  if (p.suggested_yaml && !document.getElementById(\"param-yaml\").value) {\n"
        "    document.getElementById(\"param-yaml\").value = p.suggested_yaml;\n"
        "  }",
        "FIX-1d"
    )

    ap(
        "setInterval(() => { loadProjects(); }, 30000);\ninit();",
        "setInterval(() => { loadProjects(); }, 30000);\ninit();\n\n"
        "[\"param-epochs\",\"param-batch\",\"param-imgsz\",\"param-workers\",\"param-lr\",\"param-patience\"].forEach(id => {\n"
        "  const el = document.getElementById(id);\n"
        "  if (el) el.addEventListener(\"input\", () => { _trainParamsLocked = true; });\n"
        "});",
        "FIX-1e"
    )

    ap(
        "  <div class=\"nav-links\">\n"
        "    <a href=\"/\">\U0001f3e0 Anotador</a>\n"
        "    <a href=\"/admin\" class=\"active\">\u2699 Admin</a>\n"
        "    <span id=\"hdr-role\" style=\"font-size:11px;color:var(--muted)\"></span>\n"
        "    <a href=\"#\" onclick=\"doLogout()\" style=\"color:var(--red)\">Salir</a>\n"
        "  </div>",
        "  <div class=\"nav-links\">\n"
        "    <a href=\"/\">\U0001f3e0 Anotador</a>\n"
        "    <a href=\"/admin\" class=\"active\">\u2699 Admin</a>\n"
        "    <span id=\"hdr-role\" style=\"font-size:11px;color:var(--muted)\"></span>\n"
        "    <div style=\"display:flex;gap:2px;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:2px\">\n"
        "      <button onclick=\"setLang('es')\" id=\"lang-es\" style=\"padding:3px 8px;border:none;border-radius:4px;font-size:11px;font-weight:600;cursor:pointer;background:transparent;color:var(--muted);transition:.15s\">ES</button>\n"
        "      <button onclick=\"setLang('pt')\" id=\"lang-pt\" style=\"padding:3px 8px;border:none;border-radius:4px;font-size:11px;font-weight:600;cursor:pointer;background:transparent;color:var(--muted);transition:.15s\">PT</button>\n"
        "      <button onclick=\"setLang('en')\" id=\"lang-en\" style=\"padding:3px 8px;border:none;border-radius:4px;font-size:11px;font-weight:600;cursor:pointer;background:transparent;color:var(--muted);transition:.15s\">EN</button>\n"
        "    </div>\n"
        "    <a href=\"#\" onclick=\"doLogout()\" style=\"color:var(--red)\">Salir</a>\n"
        "  </div>",
        "FIX-2a"
    )

    ap(
        "function applyLang() {",
        "function setLang(lang) {\n"
        "  if (!LANGS[lang]) return;\n"
        "  currentLang = lang;\n"
        "  localStorage.setItem(\"sam_lang\", lang);\n"
        "  applyLang();\n"
        "  updateLangButtons();\n"
        "}\n\n"
        "function updateLangButtons() {\n"
        "  [\"es\",\"pt\",\"en\"].forEach(l => {\n"
        "    const btn = document.getElementById(\"lang-\" + l);\n"
        "    if (!btn) return;\n"
        "    if (l === currentLang) { btn.style.background = \"var(--accent)\"; btn.style.color = \"#fff\"; }\n"
        "    else { btn.style.background = \"transparent\"; btn.style.color = \"var(--muted)\"; }\n"
        "  });\n"
        "}\n\n"
        "function applyLang() {",
        "FIX-2b"
    )

    ap(
        "  applyLang();\n  await loadHardware();",
        "  applyLang();\n  updateLangButtons();\n  await loadHardware();",
        "FIX-2c"
    )

    ap(
        "// Sync lang from annotator\nconst savedLang = localStorage.getItem(\"sam_lang\");\nif (savedLang && LANGS[savedLang]) { currentLang = savedLang; }",
        "// Sync lang desde index.html via localStorage\nconst savedLang = localStorage.getItem(\"sam_lang\");\nif (savedLang && LANGS[savedLang]) { currentLang = savedLang; }\n\n"
        "window.addEventListener(\"storage\", (e) => {\n"
        "  if (e.key === \"sam_lang\" && e.newValue && LANGS[e.newValue]) {\n"
        "    currentLang = e.newValue;\n"
        "    applyLang();\n"
        "    updateLangButtons();\n"
        "  }\n"
        "});",
        "FIX-2d"
    )

    content = content.replace("</body>", f"<!-- {MARKER} -->\n</body>", 1)
    admin_path.write_text(content, encoding="utf-8")

    print("[PATCH] admin.html actualizado:")
    for name in ok_list:   print(f"  OK  {name}")
    for name in fail_list: print(f"  !!  {name} — no encontrado (posiblemente ya aplicado)")
    print("[PATCH] Backup en templates/admin.html.bak")

_autopatch_admin()
# ─── Fin auto-patch ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    hw = HARDWARE_INFO
    print("\n" + "="*60)
    print("  SAM Annotator v3 — Multi-Proyecto")
    print("="*60)
    print(f"  CUDA: {hw.get('cuda', False)}")
    if hw.get('cuda'):
        print(f"  GPU:  {hw.get('gpu')} ({hw.get('vram_gb')}GB VRAM)")
    print(f"  RAM:  {hw.get('ram_gb', '?')}GB  |  CPU: {hw.get('cpu_cores','?')} núcleos")
    print("\n  Local:  http://localhost:5000")
    print("  Red:    http://<IP-del-PC>:5000")
    print("\n  SuperAdmin: usuario=admin  contraseña=admin123")
    print("="*60+"\n")
    socketio.run(app, debug=False, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)