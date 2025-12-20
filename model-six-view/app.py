import sys
import os
from pathlib import Path

# 彻底屏蔽掉底层标准输出
def silence_blender():
    logfile = os.devnull
    out = open(logfile, 'w')
    # 将标准输出和标准错误重定向到 /dev/null
    os.dup2(out.fileno(), sys.stdout.fileno())
    os.dup2(out.fileno(), sys.stderr.fileno())

# 在脚本最开始调用
silence_blender()


# ==========================================
# 0. 环境路径自适应
# ==========================================

# 在 Dockerfile 中我们会设置这个环境变量
IS_DOCKER = os.getenv("RUNNING_IN_DOCKER") == "true"

if not IS_DOCKER:
    # 仅在本地开发环境（Mac）下添加个人库路径
    user_site_packages = os.path.expanduser("~/.local/lib/python3.11/site-packages")
    if os.path.exists(user_site_packages) and user_site_packages not in sys.path:
        sys.path.append(user_site_packages)
        print(f"--- [ENV] Local Dev Mode: Added {user_site_packages} ---")
else:
    print("--- [ENV] Docker Mode: Using container site-packages ---")

import bpy
import math
import mathutils  # Blender 专用的数学库
import os
import base64
import sys
import uuid
import asyncio
import aiofiles
import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from contextlib import asynccontextmanager
import httpx
import bpy
from pydantic import BaseModel, HttpUrl

# ==========================================
# 1. 全局配置与状态
# ==========================================

RENDER_LOCK = asyncio.Lock()
TEMP_DIR = Path("/tmp/omniframe")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理：启动时初始化 GPU"""
    init_gpu_acceleration()
    yield
    # 可以在这里添加清理逻辑

app = FastAPI(title="OmniFrame AI-Vision Server", lifespan=lifespan)

# ==========================================
# 2. 增强型渲染引擎
# ==========================================

def init_gpu_acceleration():
    """初始化 GPU，优先使用 GPU，失败则回退 CPU"""
    print("\n--- [INIT] Configuring Rendering Engine ---")
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    try:
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        # 根据 Mac 平台选择 METAL
        device_type = "METAL" if sys.platform == "darwin" else "CUDA"
        
        cycles_prefs.compute_device_type = device_type
        cycles_prefs.get_devices()
        
        has_gpu = False
        for device in cycles_prefs.devices:
            if device.type in [device_type, 'CPU']:
                device.use = True
                if device.type == device_type: has_gpu = True
        
        scene.cycles.device = 'GPU' if has_gpu else 'CPU'
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
        print(f"--- [SUCCESS] Engine: Cycles | Device: {scene.cycles.device} ---")
    except Exception as e:
        print(f"--- [WARNING] GPU Init Failed: {e}. Falling back to CPU ---")
        scene.cycles.device = 'CPU'

def cleanup_blender_data():
    # 停止任何正在进行的渲染流程（可选）
    
    # 移除对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # 清理数据块
    for block in [bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.materials]:
        for item in block:
            block.remove(item)

    # 专门处理图像，避开正在使用的系统级图像
    for img in bpy.data.images:
        if img.users == 0 or not img.name.startswith("Render Result"):
            try:
                bpy.data.images.remove(img)
            except:
                pass
    
    # 彻底递归清理孤立块
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def setup_render_scene(model_path: str):
    """模型导入与构图优化"""
    cleanup_blender_data()
    
    ext = Path(model_path).suffix.lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=model_path)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=model_path)
    elif ext == '.stl':
        bpy.ops.wm.stl_import(filepath=model_path)

    obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not obs: return None

    # 统一缩放与归一化位置
    for o in obs: o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # 计算整体包围盒
    all_coords = [v.co @ o.matrix_world for o in obs for v in o.data.vertices]
    if not all_coords: return None
    
    min_coord = [min(c[i] for c in all_coords) for i in range(3)]
    max_coord = [max(c[i] for c in all_coords) for i in range(3)]
    center = [(min_coord[i] + max_coord[i]) / 2 for i in range(3)]
    
    # 修正点：使用 mathutils.Vector
    for o in obs:
        o.location -= mathutils.Vector(center)
    
    max_dim = max([(max_coord[i] - min_coord[i]) for i in range(3)])
    return max_dim

def perform_render(max_dim: float, resolution: int):
    """执行六视图渲染并返回 Base64"""
    setup_ai_lighting()
    
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = max_dim * 1.1 
    dist = max_dim * 2

    views = {
        "front":  ((0, -dist, 0), (math.radians(90), 0, 0)),
        "back":   ((0, dist, 0),  (math.radians(90), 0, math.radians(180))),
        "left":   ((-dist, 0, 0), (math.radians(90), 0, math.radians(-90))),
        "right":  ((dist, 0, 0),  (math.radians(90), 0, math.radians(90))),
        "top":    ((0, 0, dist),  (0, 0, 0)),
        "bottom": ((0, 0, -dist), (math.radians(180), 0, 0))
    }

    results = {}
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    
    render_file = TEMP_DIR / f"buffer_{uuid.uuid4()}.png"

    for name, (pos, rot) in views.items():
        cam.location, cam.rotation_euler = pos, rot
        scene.render.filepath = str(render_file)
        bpy.ops.render.render(write_still=True)
        
        with open(render_file, "rb") as f:
            results[name] = base64.b64encode(f.read()).decode('utf-8')
    
    if render_file.exists(): render_file.unlink()
    return results

def setup_ai_lighting():
    """光照优化逻辑"""
    scene = bpy.context.scene
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (1, 1, 1, 1) # 纯白背景
    bg.inputs['Strength'].default_value = 1.5

    directions = [
        (10, 10, 10), (-10, -10, 10), (10, -10, 10), (-10, 10, 10)
    ]
    for pos in directions:
        bpy.ops.object.light_add(type='SUN', location=pos)
        bpy.context.object.data.energy = 1.0

async def _process_render(file_path: Path, res: int, task_id: str):
    """
    统一的内部处理逻辑：
    1. 执行渲染 -> 2. 清理环境 -> 3. 返回结果
    """
    try:
        # 这里的操作是阻塞型的，但在 RENDER_LOCK 保护下是安全的
        max_dim = setup_render_scene(str(file_path))
        if max_dim is None:
            raise ValueError("Model is empty or invalid (no meshes found)")
            
        images_dict = perform_render(max_dim, res)
        
        return {
            "status": "success",
            "task_id": task_id,
            "data": images_dict
        }
    finally:
        # 无论成功失败，都在这里统一删除临时文件并清理 Blender 内存
        if file_path.exists():
            file_path.unlink()
        cleanup_blender_data()
# ==========================================
# FastAPI 路由
# ==========================================

# 1. 直接上传文件
@app.post("/six/view/file")
async def render_upload(
    file: UploadFile = File(...),
    res: int = Query(512, ge=128, le=2048)
):
    ext = Path(file.filename).suffix.lower()
    task_id = str(uuid.uuid4())
    save_path = TEMP_DIR / f"{task_id}{ext}"

    async with RENDER_LOCK:
        async with aiofiles.open(save_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        return await _process_render(save_path, res, task_id)

# --- 定义 URL 请求模型 ---
class SixViewRequest(BaseModel):
    url: HttpUrl
    res: int = 512
# 2. 通过 URL 渲染
@app.post("/six/view/url")
async def render_url(request_data: SixViewRequest):
    url_str = str(request_data.url)
    task_id = str(uuid.uuid4())
    
    async with RENDER_LOCK:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url_str)
                response.raise_for_status()
                
                # 智能识别后缀：优先看 URL，如果没有则看 Content-Type
                ext = Path(url_str).suffix.lower()
                if ext not in ['.glb', '.gltf', '.obj', '.stl']:
                    content_type = response.headers.get("content-type", "")
                    mime_map = {
                        "model/gltf-binary": ".glb",
                        "model/gltf+json": ".gltf",
                        "application/octet-stream": ".obj" # 很多时候 OBJ 被识别为二进制流
                    }
                    ext = mime_map.get(content_type, ".glb")

                save_path = TEMP_DIR / f"{task_id}{ext}"
                
                async with aiofiles.open(save_path, 'wb') as out_file:
                    await out_file.write(response.content)
            
            return await _process_render(save_path, request_data.res, task_id)
            
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch model from URL: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu": bpy.context.scene.cycles.device}

# ==========================================
# 4. 服务器启动入口
# ==========================================

if __name__ == "__main__":
    # 启动服务器，由于是后台模式运行在 Blender 中，这里会挂起进程等待请求
    uvicorn.run(app, host="0.0.0.0", port=8000)