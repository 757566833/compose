import sys
import os
import math
import uuid
import asyncio
import aiofiles
import base64
import httpx
import uvicorn
import bpy
import mathutils
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from contextlib import asynccontextmanager

# ==========================================
# 0. 环境初始化
# ==========================================

def silence_blender():
    logfile = os.devnull
    out = open(logfile, 'w')
    os.dup2(out.fileno(), sys.stdout.fileno())
    os.dup2(out.fileno(), sys.stderr.fileno())

silence_blender()

RENDER_LOCK = asyncio.Lock()
TEMP_DIR = Path("/tmp/omniframe")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_gpu_acceleration()
    yield

app = FastAPI(title="OmniFrame RTX-Vision Server", lifespan=lifespan)

# ==========================================
# 1. 核心渲染引擎配置 (针对 RTX 4060)
# ==========================================

def init_gpu_acceleration():
    """针对 RTX 系列显卡强制开启 OptiX 加速"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    prefs = bpy.context.preferences.addons['cycles'].preferences
    # RTX 4060 优先使用 OPTIX，其次 CUDA
    for device_type in ['OPTIX', 'CUDA']:
        try:
            prefs.compute_device_type = device_type
            prefs.get_devices()
            break
        except:
            continue

    has_gpu = False
    for device in prefs.devices:
        if device.type in ['OPTIX', 'CUDA']:
            device.use = True
            has_gpu = True
    
    scene.cycles.device = 'GPU' if has_gpu else 'CPU'
    scene.cycles.samples = 64  # 4060 下 64 采样极快且清晰
    scene.cycles.use_denoising = True
    print(f"--- [INFO] Rendering Device: {scene.cycles.device} ({prefs.compute_device_type}) ---")

# ==========================================
# 2. 修复 Context 错误的逻辑函数
# ==========================================

def cleanup_blender_data():
    """安全清理所有数据块"""
    # 显式使用 bpy.data 而非 context
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    for block in [bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.materials, bpy.data.images]:
        for item in block:
            if item.users == 0:
                try: block.remove(item)
                except: pass
    
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def setup_render_scene(model_path: str):
    """导入并居中模型，修复 Context 引用报错"""
    cleanup_blender_data()
    
    ext = Path(model_path).suffix.lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=model_path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=model_path)
    elif ext == '.stl':
        bpy.ops.wm.stl_import(filepath=model_path)

    # 获取当前场景所有 Mesh
    obs = [o for o in bpy.data.objects if o.type == 'MESH']
    if not obs: return None

    # 计算全局包围盒
    all_coords = []
    for o in obs:
        all_coords.extend([o.matrix_world @ v.co for v in o.data.vertices])
    
    if not all_coords: return None
    
    min_c = mathutils.Vector((min(v.x for v in all_coords), min(v.y for v in all_coords), min(v.z for v in all_coords)))
    max_c = mathutils.Vector((max(v.x for v in all_coords), max(v.y for v in all_coords), max(v.z for v in all_coords)))
    center = (min_c + max_c) / 2
    max_dim = (max_c - min_c).length

    # 修复点：直接操作对象 location 而不依赖 context.object
    for o in obs:
        if not o.parent: # 仅移动根级物体
            o.location -= center
            
    return max_dim

def perform_render(max_dim: float, resolution: int):
    """执行渲染流程"""
    scene = bpy.context.scene
    
    # 设置灯光
    world = scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (1, 1, 1, 1)
    bg.inputs['Strength'].default_value = 1.0

    # 显式添加灯光，避免使用 bpy.context.object
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, 5, 10)
    light_data.energy = 2.0

    # 设置相机
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = max_dim * 1.1
    dist = max_dim * 2

    views = {
        "front":  ((0, -dist, 0), (1.5708, 0, 0)),
        "back":   ((0, dist, 0),  (1.5708, 0, 3.14159)),
        "left":   ((-dist, 0, 0), (1.5708, 0, -1.5708)),
        "right":  ((dist, 0, 0),  (1.5708, 0, 1.5708)),
        "top":    ((0, 0, dist),  (0, 0, 0)),
        "bottom": ((0, 0, -dist), (3.14159, 0, 0))
    }

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    
    results = {}
    render_path = TEMP_DIR / f"out_{uuid.uuid4()}.png"

    for name, (pos, rot) in views.items():
        cam_obj.location = pos
        cam_obj.rotation_euler = rot
        scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)
        
        with open(render_path, "rb") as f:
            results[name] = base64.b64encode(f.read()).decode('utf-8')
    
    if render_path.exists(): render_path.unlink()
    return results

# ==========================================
# 3. FastAPI 路由与多线程包装
# ==========================================

def sync_render_worker(file_path: Path, res: int):
    """同步工作线程，由 run_in_executor 调用"""
    try:
        max_dim = setup_render_scene(str(file_path))
        if max_dim is None: return None
        return perform_render(max_dim, res)
    finally:
        cleanup_blender_data()
        if file_path.exists(): file_path.unlink()

@app.post("/six/view/file")
async def render_upload(file: UploadFile = File(...), res: int = 512):
    save_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(save_path, 'wb') as f:
        await f.write(await file.read())

    async with RENDER_LOCK:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, sync_render_worker, save_path, res)
        if not data: raise HTTPException(400, "Model processing failed")
        return {"status": "success", "data": data}

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": bpy.context.scene.cycles.device}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)