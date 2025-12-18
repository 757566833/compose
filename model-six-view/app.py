import bpy
import math
import os
import base64
import sys
import shutil
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query

# ==========================================
# 1. 核心渲染逻辑 (OmniFrame Engine)
# ==========================================

def get_gpu_info():
    """检测 GPU 状态用于健康检查"""
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        device_type = "METAL" if sys.platform == "darwin" else "CUDA"
        prefs.get_devices()
        active_devices = [d.name for d in prefs.devices if d.use]
        return {
            "status": "Healthy" if active_devices else "Warning: No GPU found",
            "type": device_type,
            "devices": active_devices
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}
# 在 init_gpu_acceleration() 中添加更健壮的错误处理
def init_gpu_acceleration():
    """【全局初始化】配置 GPU 加速与 Cycles 渲染器"""
    print("\n--- 正在初始化全局渲染引擎 (GPU 加速) ---")
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # AI 识别建议使用实色背景，避免透明通道处理异常
    scene.render.film_transparent = False 
    
    try:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons['cycles'].preferences
        device_type = "METAL" if sys.platform == "darwin" else "CUDA"
        
        cycles_prefs.compute_device_type = device_type
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            if device.type in [device_type, 'CPU']:
                device.use = True
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 128  # AI 识别不需要极高的采样，128 足够清晰
        print(f"--- {device_type} 加速初始化成功 ---")
    except Exception as e:
        print(f"--- GPU 初始化失败: {e}, 使用CPU渲染 ---")
        scene.cycles.device = 'CPU'

def clear_scene():
    """清空场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for m in bpy.data.meshes: bpy.data.meshes.remove(m)
    for c in bpy.data.cameras: bpy.data.cameras.remove(c)
    for l in bpy.data.lights: bpy.data.lights.remove(l)

def setup_ai_lighting():
    """
    专为 AI 视觉任务优化的光照：
    1. 高环境亮度：消除所有死角
    2. 全方位太阳光：减少阴影硬度，模拟无影灯效果
    """
    scene = bpy.context.scene
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes['Background']
    bg_node.inputs['Color'].default_value = (1, 1, 1, 1) # 纯白世界背景
    bg_node.inputs['Strength'].default_value = 1.0       # 提升基础亮度

    # 1. 顶光 (Top)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    bpy.context.object.data.energy = 2.0
    bpy.context.object.data.angle = 0.5 # 阴影边缘软化

    # 2. 正前方光 (Front)
    bpy.ops.object.light_add(type='SUN', location=(0, -10, 5))
    bpy.context.object.data.energy = 1.5

    # 3. 正后方光 (Back)
    bpy.ops.object.light_add(type='SUN', location=(0, 10, 5))
    bpy.context.object.data.energy = 1.5

    # 4. 底光 (Bottom) - 确保 AI 能看清底部特征
    bpy.ops.object.light_add(type='SUN', location=(0, 0, -10))
    bpy.context.object.data.energy = 1.5

def render_to_base64(model_path, resolution):
    """渲染六视图核心流程"""
    clear_scene()
    
    # 导入模型
    ext = os.path.splitext(model_path)[1].lower()
    try:
        if ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=model_path)
        elif ext == '.obj':
            bpy.ops.wm.obj_import(filepath=model_path)
        elif ext == '.stl':
            bpy.ops.wm.stl_import(filepath=model_path)
    except Exception as e:
        print(f"导入失败: {e}")
        return {}

    # 设置光照
    setup_ai_lighting()
    
    obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not obs: return {}

    # 自动居中与缩放
    for o in obs: o.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    max_dim = max([max(o.dimensions) for o in obs])

    # 相机配置 (正交相机，防止形变)
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = max_dim * 1.05  # 紧凑构图，最大化利用像素
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
    # 分辨率设为 768，更适合现代 VLM 模型
    scene.render.resolution_x, scene.render.resolution_y = resolution, resolution
    
    tmp_path = os.path.join("/tmp", f"render_{uuid.uuid4()}.png")

    for name, (pos, rot) in views.items():
        cam.location, cam.rotation_euler = pos, rot
        scene.render.filepath = tmp_path
        bpy.ops.render.render(write_still=True)
        with open(tmp_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode('utf-8')
            results[name] = b64_str
    
    if os.path.exists(tmp_path): os.remove(tmp_path)
    return results

# ==========================================
# 2. HTTP 服务接口 (FastAPI)
# ==========================================

app = FastAPI(title="OmniFrame AI-Vision Server")

# 启动时执行一次 GPU 初始化
init_gpu_acceleration()

@app.get("/health")
async def health_check():
    return {
        "status": "alive",
        "service": "OmniFrame",
        "blender": bpy.app.version_string,
        "gpu_info": get_gpu_info()
    }

@app.post("/render")
async def handle_render(
    file: UploadFile = File(...),
    res: int = Query(512, description="分辨率大小，默认为 512") # 在 URL 中接收 ?res=512
):
    # 格式白名单
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.glb', '.gltf', '.obj', '.stl']:
        raise HTTPException(status_code=400, detail="Unsupported model format")

    task_id = str(uuid.uuid4())
    save_path = f"/tmp/{task_id}_{file.filename}"
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 执行六视图渲染
        images_dict = render_to_base64(save_path, res)
        
        return {
            "status": "success",
            "task_id": task_id,
            "data": images_dict,  # 包含 front, back, left, right, top, bottom
            "resolution": f"{res}x{res}",
            "format":"image/png"
        }
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(save_path): os.remove(save_path)

if __name__ == "__main__":
    print("\n[OmniFrame] 专为 AI 识别设计的渲染服务已就绪...")
    uvicorn.run(app, host="0.0.0.0", port=8000)