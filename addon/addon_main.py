bl_info = {
    "name": "xrhumanlab_blender_renderer",
    "blender": (4, 0, 0),
    "category": "Render",
    "description": "rendering obj add-on",
    "author": "SHKim",
    "version": (1, 0, 0),
    "location": "View3D > Sidebar > xrhuman_renderer",
    "warning": "",
    "doc_url": "",
    "tracker_url": ""
}


import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, Panel
from mathutils import Vector, Color, Euler
import math
import numpy as np
import random
import os

cc4_addon_light1 = {'name': 'Light1', 'type': 'SPOT', 'energy': 125.0, 'color': Color((1.0, 1.0, 1.0)), 'size': 2.0950000286102295, 'location': Vector((0.2256695032119751, -1.7913637161254883, 0.5483174324035645)), 'rotation_euler': Euler((1.2529067993164062, 0.3425169289112091, -0.012558246962726116), 'XYZ'), 'radius': 0.22499999403953552}
cc4_addon_light2 = {'name': 'Light2', 'type': 'SPOT', 'energy': 125.0, 'color': Color((1.0, 1.0, 1.0)), 'size': 2.0950000286102295, 'location': Vector((0.6399987936019897, -1.3198518753051758, -0.8029392957687378)), 'rotation_euler': Euler((1.8845493793487549, 0.50091552734375, 0.6768553256988525), 'XYZ'), 'radius': 1.0}
cc4_addon_light3 = {'name': 'Light3', 'type': 'SPOT', 'energy': 261.1000061035156, 'color': Color((1.1554443836212158, 0.7907306551933289, 0.7156423926353455)), 'size': 1.4479999542236328, 'location': Vector((0.16602079570293427, 1.6661170721054077, 0.4980148673057556)), 'rotation_euler': Euler((-1.3045594692230225, 0.11467886716127396, 0.03684665635228157), 'XYZ'), 'radius': 1.0}


####util####
def empty_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def empty_light():
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' or "Empty" in obj.name:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.delete()
        

def delete_all_cameras():
    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

def create_light_with_info(light_info):
    light_data = bpy.data.lights.new(name=light_info['name'], type=light_info['type'])
    light_data.energy = light_info['energy']
    light_data.color = light_info['color']
    light_data.spot_size = light_info['size']
    
    if light_info['type'] in ['SPOT', 'SUN']:
        light_data.shadow_soft_size = light_info['radius']
    
    light_object = bpy.data.objects.new(name=light_info['name'], object_data=light_data)
    light_object.location = light_info['location']
    light_object.rotation_euler = light_info['rotation_euler']
    
    bpy.context.collection.objects.link(light_object)

def get_scene_center_and_scale():
    # 숨겨지지 않은 MESH만 사용
    for obj in bpy.context.scene.objects:
        print(obj.type, obj.name, obj.hide_viewport, obj.hide_render)
    mesh_objects = [obj for obj in bpy.context.scene.objects if (obj.type == 'MESH' and not obj.hide_viewport and not obj.hide_render)]
    all_corners = []

    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            all_corners.append(world_corner)

    # 모든 꼭짓점을 포함하는 바운딩 박스를 계산
    min_corner = Vector((min([corner.x for corner in all_corners]), 
                        min([corner.y for corner in all_corners]), 
                        min([corner.z for corner in all_corners])))

    max_corner = Vector((max([corner.x for corner in all_corners]), 
                        max([corner.y for corner in all_corners]), 
                        max([corner.z for corner in all_corners])))

    # 씬의 중심점과 크기 계산
    scene_center = (min_corner + max_corner) / 2
    scene_size = max_corner - min_corner
    return scene_center, scene_size

# USD로부터 변환된 Scene 데이터를 다루기 위해 parent 가 아니면 옮기지 않는 방식으로 변경
def move_and_scale_scene():
    bpy.ops.object.select_all(action='DESELECT')
    scene_center, scene_size = get_scene_center_and_scale()    
    for obj in bpy.context.scene.objects:
        if obj.parent is None:
            obj.location -= scene_center

            # Origin이 객체의 중심에 있지 않으면 scaling이 Origin중심으로 이루어지므로 Origin을 객체의 중심으로 옮김
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

            obj.scale /= max(scene_size)

# 사용안함. move_and_scale_scene 으로 교체
def select_and_move_mesh():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        
        if obj.type == 'MESH':
            if obj.hide_viewport and not obj.hide_render: # 숨겨진 객체는 무시하고 가장 처음 탐색된 객체 사용
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                break
        
    obj = bpy.context.active_object
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_center = sum(bbox_corners, Vector((0.0, 0.0, 0.0))) / 8
    
    # armature를 가진 fbx mesh의 경우 일반적으로 mesh가 armature에 종속되므로 armature의 위치를 옮김. 예외가 있을 경우 처리 필요
    armature = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break  

    if armature is not None:
        armature.location -= bbox_center
    else:
        obj.location -= bbox_center
    
def get_camera_parameters_extrinsic(camera_obj):
    # ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    bcam = camera_obj
    R_bcam2cv = np.array([[1,  0,  0], [0, -1,  0], [0,  0, -1]])

    location = np.array([bcam.matrix_world.decompose()[0]]).T
    R_world2bcam = np.array(bcam.matrix_world.decompose()[1].to_matrix().transposed())
    T_world2bcam = np.matmul(R_world2bcam.dot(-1), location)
    print(location, T_world2bcam)
    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)

    extr = np.concatenate((R_world2cv, T_world2cv), axis=1)
    extr = np.concatenate((extr, np.array([[0,0,0,1]])), axis=0)
    return extr

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL': return sensor_y
    return sensor_x

def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y: return 'HORIZONTAL'
        else: return 'VERTICAL'
    return sensor_fit

def get_camera_parameters_intrinsic(camera_obj, scene):
    """ Get intrinsic camera parameters: focal length and principal point. """
    # ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera/120063#120063
    focal_length = camera_obj.data.lens # [mm]
    res_x, res_y = bpy.context.scene.render.resolution_x,  bpy.context.scene.render.resolution_y
    
    cam_data = camera_obj.data
    sensor_size_in_mm = get_sensor_size(cam_data.sensor_fit, cam_data.sensor_width, cam_data.sensor_height)
    sensor_fit = get_sensor_fit(cam_data.sensor_fit,scene.render.pixel_aspect_x * res_x,scene.render.pixel_aspect_y * res_y)
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL': view_fac_in_px = res_x
    else: view_fac_in_px = pixel_aspect_ratio * res_y
    #pixel_size_mm_per_px = (sensor_size_in_mm / focal_length) / view_fac_in_px
    pixel_size_mm_per_px = (sensor_size_in_mm / focal_length) / view_fac_in_px
    f_x = 1.0 / pixel_size_mm_per_px
    f_y = (1.0 / pixel_size_mm_per_px) / pixel_aspect_ratio
    c_x = (res_x - 1) / 2.0 - cam_data.shift_x * view_fac_in_px
    c_y = (res_y - 1) / 2.0 + (cam_data.shift_y * view_fac_in_px) / pixel_aspect_ratio
    print(f_x, f_y)
    
    intrinsic_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    return intrinsic_matrix


# 속도와 안정성 위해 포인트 샘플링은 기본적으로 5천개만 수행
def save_pointcloud(path, max_points=5000):
    with open(path + "/test.obj", 'w') as file:
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH" and not obj.hide_viewport and not obj.hide_render:

                #bpy.context.view_layer.objects.active = obj
                #bpy.ops.object.duplicate(linked=False)
                #dup_obj = bpy.context.active_object
                
                
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                #bpy.ops.export_scene.obj(filepath="cache_obj", use_selection=True, use_mesh_modifiers=True)
                bpy.ops.wm.obj_export(filepath=path+"/cache_obj.obj", export_selected_objects=True, apply_modifiers=True)
                bpy.ops.object.select_all(action='DESELECT')
                
                #bpy.ops.import_scene.obj(filepath="cache_obj")
                bpy.ops.wm.obj_import(filepath=path+"/cache_obj.obj")
                if len(bpy.context.selected_objects):
                    imported_obj = bpy.context.selected_objects[0]
                else:
                    continue

                mesh = imported_obj.to_mesh()
                
                ################ texture에서 칼라를 가져오는건 아직 미구현. ################
                color_layer = mesh.vertex_colors.active.data if mesh.vertex_colors else None

                vert_indices = list(range(len(mesh.vertices)))
                random.shuffle(vert_indices)
                sampled_indices = vert_indices[:min(max_points, len(vert_indices))] # 하나의 mesh 당 최대 max_points 만큼만 샘플링

                for i in sampled_indices:
                    vert = mesh.vertices[i] 
                    v = vert.co # 버텍스 위치
                    c = [1.0, 1.0, 1.0] # 버텍스 색상 (기본값: 흰색)
                    if color_layer:
                        # 버텍스 색상 데이터 수집
                        loop_index = mesh.loops[i].index
                        c = color_layer[loop_index].color
                    
                    # up axis가 Blender에서 강제로 바뀌는 문제 때문에 아래와 같이 저장!
                    # Blender는 자신의 월드 좌표계에서는 기본적으로 Z UP으로 물체를 배치하고 랜더링함.
                    # 그러나 OBJ 및 FBX를 다룰 때 데이터가 Y_UP임을 가정하고, 읽어온 후 강제로 Z_UP으로 변환함. 
                    # 이 때 실제 OBJ의 vertex의 좌표를 Z_UP으로 변환시키지는않고 Y_UP형태로 유지함.
                    # 그래서 vertex를 직접 읽어서 저장하는 경우 Blender의 월드좌표계와 일치하지 않는 문제가 발생하여 아래와 같이 변경함
                    #file.write(f"v {v.x} {v.y} {v.z} {c[0]} {c[1]} {c[2]}\n")
                    file.write(f"v {v.x} {-v.z} {v.y} {c[0]} {c[1]} {c[2]}\n")

                bpy.data.objects.remove(imported_obj, do_unlink=True)
                #os.remove(temp_file_path)
    file.close()


def save_camera(path, cameras):
    cam_path = path + "/camera/"
    os.makedirs(cam_path, exist_ok=True)
    
    cam_number = 0
    for cam in cameras:
        cam_number += 1
        extrinsic = get_camera_parameters_extrinsic(cam)
        intrinsic = get_camera_parameters_intrinsic(cam, bpy.context.scene)
        print(extrinsic)

        np.save(f"{cam_path}{cam_number:03d}_intrinsic.npy", intrinsic)
        np.save(f"{cam_path}{cam_number:03d}_extrinsic.npy", extrinsic)
    

def setup_camera(azimuth_deg, elevation_deg, radius):

    # 라디안으로 변환
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    # 카메라 위치 계산
    x = radius * math.sin(elevation_rad) * math.cos(azimuth_rad)
    y = radius * math.sin(elevation_rad) * math.sin(azimuth_rad)
    z = - radius * math.cos(elevation_rad) # 직관성을 위해 negative 값 사용

    if bpy.context.scene.world_axis_type == "Y_UP":
        y_before, z_before = y, z
        y, z = z_before, -y_before
    
        bpy.ops.object.camera_add(location=(x, y, z), rotation=(0,0,0))
        camera = bpy.context.active_object

        # Track to 안쓰고 Euler 회전각 직접 계산해서 사용
        direction = Vector((0, 0, 0)) - Vector((x, y, z))
        rot_y = math.atan2(-direction.x, -direction.z)
        horizontal_dist = (direction.x ** 2 + direction.z ** 2) ** 0.5
        rot_x = math.atan2(-direction.y, horizontal_dist)
        
        camera.rotation_euler[1] = rot_y
        camera.rotation_euler[0] = -rot_x

    else:
        bpy.ops.object.camera_add(location=(x, y, z), rotation=(0,0,0))
        camera = bpy.context.active_object
        direction = Vector((0, 0, 0)) - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        

    # Track to constraint 에 문제가 있어서 폐기. Y_UP으로 세팅할경우 카메라가 원하는데로 제어가 안됨
    '''
    # 카메라가 원점을 바라보게 설정
    origin_point = bpy.data.objects.get("origin_point")
    bpy.ops.object.constraint_add(type='TRACK_TO')
    camera.constraints['Track To'].target = origin_point  # 원점을 가정
    camera.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    camera.constraints['Track To'].up_axis = 'UP_Y'
    '''

    
def set_cam():
    azimuth_min, azimuth_max, horizontal_split = bpy.context.scene.azimuth_min, bpy.context.scene.azimuth_max, bpy.context.scene.horizontal_split
    elevation_min, elevation_max, vertical_split = bpy.context.scene.elevation_min, bpy.context.scene.elevation_max, bpy.context.scene.vertical_split
    radius = bpy.context.scene.radius
    
    # 직관성을 위해 90도를 더함 (일반적으로 xy평면상에서 elevation이 움직이는 것이 직관적. 더하지 않을 경우 z축을 기준으로 카메라가 배치되어 직관성이 떨어짐)
    elevation_max += 90
    elevation_min += 90

    azimuth_min, azimuth_max = azimuth_min % 360, azimuth_max % 360
    elevation_min, elevation_max = elevation_min % 360, elevation_max % 360

    
    # 둘이 같으면 그냥 360도 회전으로 설정
    if azimuth_min == azimuth_max : azimuth_max = azimuth_max + 360
    if elevation_min == elevation_max : elevation_max = elevation_max + 360

    # split 이 1이나 그 이하의 값으로 설정되어있으면 강제로 1로 설정. 그 외에는 -1을 해주어 범위의 직관성 향상
    #horizontal_interval = horizontal_split - 1 if  horizontal_split > 1 else 1 # 사용해보니 horizontal 방향은 직관성이 오히려 떨어짐.
    vertical_interval = vertical_split - 1 if  vertical_split > 1 else 1

    azimuth_interval = int( (azimuth_max - azimuth_min) / (horizontal_split) )
    elevation_interval = int( (elevation_max - elevation_min) / (vertical_interval) )

    azimuth_list = [azimuth_min + azimuth_interval * i for i in range(horizontal_split)]
    elevation_list = [elevation_min + elevation_interval * i for i in range(vertical_split)]

    #bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    #origin_point = bpy.context.object
    #origin_point.name = "origin_point"

    delete_all_cameras()
    for j in range(vertical_split): # 직관성을 위해 vertical 먼저
        for i in range(horizontal_split):    
            setup_camera(azimuth_list[i], elevation_list[j], radius)

def render_with_multiple_cameras(output_path, frame=0):
    scene = bpy.context.scene
    original_camera = scene.camera

    cameras = [obj for obj in scene.objects if obj.type == 'CAMERA']
    if bpy.context.scene.save_point3d == "ON" : save_pointcloud(f"{bpy.context.scene.save_path}")
    if bpy.context.scene.save_bone3d == "ON" : a=1 #미구현
    if bpy.context.scene.render_skinning_weight == "ON" : a=1 #미구현
    
    save_camera(bpy.context.scene.save_path, cameras)

    # 폴더 만들기
    if bpy.context.scene.render_rgb == "ON" : os.makedirs(f"{bpy.context.scene.save_path}/image/", exist_ok=True) 
    if bpy.context.scene.render_normal == "ON" : os.makedirs(f"{bpy.context.scene.save_path}/normal/", exist_ok=True)
    if bpy.context.scene.render_depth == "ON" : os.makedirs(f"{bpy.context.scene.save_path}/depth/", exist_ok=True)
    if bpy.context.scene.render_mask == "ON" : os.makedirs(f"{bpy.context.scene.save_path}/mask/", exist_ok=True)
    
    
    bpy.context.scene.use_nodes = True # RGB 말고 다른 출력 지원을 위해 Node ON
    tree = bpy.context.scene.node_tree
    view_layer = bpy.context.view_layer
    for node in tree.nodes: # 아래의 노드들은 use_nodes = True 로 설정시 자동으로 생성되는 노드들임. 없을경우... 예외처리 필요
        if node.type == "R_LAYERS" : rl_node = node
        elif node.type == "COMPOSITE" : rgb_node = node

    
    # NODE 설정
    if bpy.context.scene.render_normal == "ON":
        view_layer.use_pass_normal = True


    if bpy.context.scene.render_depth == "ON":
        view_layer.use_pass_normal = True
        #rl_node = tree.nodes.new(type="CompositorNodeRLayers")
        map_node = tree.nodes.new(type="CompositorNodeMapRange")
        viewer_depth = tree.nodes.new(type="CompositorNodeViewer")
        viewer_depth.name = 'Depth Viewer'

        # depth 값은 [0, 1] 로 normalize 됨
        map_node.inputs['From Min'].default_value = 0
        map_node.inputs['From Max'].default_value = bpy.context.scene.radius * 2 # Radius의 두배 값으로 depth 범위 설정 
        map_node.use_clamp = True

        #tree.links.new(rl_node.outputs['Depth'], map_node.inputs['Value'])
        #tree.links.new(map_node.outputs['Value'], viewer_depth.inputs['Image'])

        output_file_node = tree.nodes.new('CompositorNodeOutputFile')
        output_file_node.base_path = f"{bpy.context.scene.save_path}/depth/"
        #scene.view_layers["View Layer"].use_path_z = True

        
        

        tree.links.new(rl_node.outputs['Depth'], map_node.inputs['Value'])
        tree.links.new(map_node.outputs['Value'], output_file_node.inputs['Image'])


    cam_number = 0
    for cam in cameras:
        cam_number += 1
        scene.camera = cam
        print(f"Rendering with camera: {cam.name}")


        output_file_node.file_slots.new("Depth")
        output_file_node.file_slots[-1].path = "_frame_{frame:04d}_cam_{cam_number:03d}"

        if bpy.context.scene.render_rgb == "ON" : 
            scene.render.filepath = f"{bpy.context.scene.save_path}/image/_frame_{frame:04d}_cam_{cam_number:03d}"
            bpy.ops.render.render(write_still=True)
        

        if bpy.context.scene.render_normal == "ON":
            view_layer.use_pass_normal = True



        if bpy.context.scene.render_depth == "ON":
            view_layer.use_pass_normal = True
            ##depth_image = bpy.data.images['Depth Viewer']
            #depth_image.filepath_raw = f"{bpy.context.scene.save_path}/depth/_frame_{frame:04d}_cam_{cam_number:03d}"
            #depth_image.file_format = 'PNG'
            #depth_image.save()

        if bpy.context.scene.render_mask == "ON":
            # GPU 가속기능?? 필요시 추가 고려
            bpy.context.scene.render.engine = 'CYCLES' # CYCLES는 포인트 샘플링을 하기 때문에 매우 느림. RGB 랜더링때는 끄기
            view_layer.use_pass_z = True
            
            view_layer.use_pass_object_index = True



            bpy.context.scene.render.engine = 'BLENDER_EEVEE' # CYCLES는 포인트 샘플링을 하기 때문에 매우 느림. RGB 랜더링때는 끄기

    
    scene.camera = original_camera

def find_scene_frame():

    max_frame, min_frame = 0, 99999
    for obj in bpy.context.scene.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe_point in fcurve.keyframe_points:
                    max_frame = max(max_frame, keyframe_point.co.x)
                    min_frame = min(min_frame, keyframe_point.co.x)
    return int(max_frame), int(min_frame)

def auto_frame_update(self, context):
    if self.auto_frame_mode == "ON":
        max_frame, min_frame = find_scene_frame()
        bpy.context.scene.frame_max = max_frame
        bpy.context.scene.frame_min = min_frame
        bpy.context.scene.frame_auto_search = True # Toggle auto search
        print("AUTO ON, Find first and last frame", min_frame, max_frame, bpy.context.scene.frame_auto_search)
    else:
        bpy.context.scene.frame_auto_search = False
        print("AUTO OFF")

def light_update(self, context):
    if self.light_mode == "ON":
        empty_light()
        create_light_with_info(cc4_addon_light1)
        create_light_with_info(cc4_addon_light2)
        create_light_with_info(cc4_addon_light3)
        print("Use Fixed Light")
    else:
        empty_light()
        # 구현해야함
        print("Use Random Light")

def rotate_object_90_deg_x(obj, degree):
    obj.rotation_euler[0] += math.radians(degree)
    # 위치를 변경하기 전에 현재 위치를 저장
    original_location = obj.location.copy()
    # 위치 변경: Z 위치를 Y로, Y 위치는 음수 Z로 변경
    if degree < 0:
        obj.location[1] = original_location[2]
        obj.location[2] = -original_location[1]

    else :
        obj.location[1] = -original_location[2]
        obj.location[2] = original_location[1]
    


def ZUP_YUP_Swap(self, context):
    if self.axis_mode == "Y_UP" and bpy.context.scene.world_axis_type == "Z_UP":
        bpy.context.scene.world_axis_type = "Y_UP"
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.parent is None:

                if obj.type != "CAMERA":
                    rotate_object_90_deg_x(obj, -90)

                if obj.type != "ARMATURE":
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

        set_cam() 
                
    elif self.axis_mode == "Z_UP" and bpy.context.scene.world_axis_type == "Y_UP":    
        bpy.context.scene.world_axis_type = "Z_UP"
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.parent is None:

                if obj.type != "CAMERA":
                    rotate_object_90_deg_x(obj, 90)
                if obj.type != "ARMATURE":
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

        set_cam()

def RGB_Mode_Toggle(self, context):
    bpy.context.scene.render_rgb = "ON" if self.rgb_render == "ON" else "OFF"

def DEPTH_Mode_Toggle(self, context):
    bpy.context.scene.render_depth = "ON" if self.depth_map_render == "ON" else "OFF"

def NORMAL_Mode_Toggle(self, context):
    bpy.context.scene.render_normal = "ON" if self.normal_map_render == "ON" else "OFF"

def MASK_Mode_Toggle(self, context):
    bpy.context.scene.render_mask = "ON" if self.render_mask == "ON" else "OFF"

def POINTCLOUD_Mode_Toggle(self, context):
    bpy.context.scene.save_point3d = "ON" if self.pointcloud_export == "ON" else "OFF"

def BONE_Mode_Toggle(self, context):
    bpy.context.scene.save_bone3d = "ON" if self.bone_export == "ON" else "OFF"

def SKINNINGWEIGHT_Mode_Toggle(self, context):
    bpy.context.scene.render_skinning_weight = "ON" if self.render_skinning_weight == "ON" else "OFF"

####util####
class CustomProperty(bpy.types.PropertyGroup):    
    auto_frame_mode: bpy.props.EnumProperty(items=[
            ("OFF", "No Auto", "No automatic animation frame setting"),
            ("ON", "Auto", "Automatically set animation frame"),],
        default="OFF", update = auto_frame_update)
    
    light_mode: bpy.props.EnumProperty(items=[
            ("ON", "FIX", "Use CC4 addon's light setting"),
            ("OFF", "RANDOM", "USE Random light setting"),],
        default="OFF", update = light_update)
    
    depth_map_render: bpy.props.EnumProperty(items=[
            ("ON", "ON", "Render depth map"),
            ("OFF", "OFF", "Don't render depth map"),],
        default="OFF", update = DEPTH_Mode_Toggle)
    
    normal_map_render: bpy.props.EnumProperty(items=[
            ("ON", "ON", "Render normal map"),
            ("OFF", "OFF", "Don't render normal map"),],
        default="OFF", update = NORMAL_Mode_Toggle)
    
    rgb_render: bpy.props.EnumProperty(items=[
            ("ON", "ON", "Render rgb image"),
            ("OFF", "OFF", "Don't render rgb image"),],
        default="ON", update = RGB_Mode_Toggle)
    
    pointcloud_export: bpy.props.EnumProperty(items=[
            ("ON", "ON", "Save point cloud with color as obj"),
            ("OFF", "OFF", "Don't save point cloud"),],
        default="OFF", update = POINTCLOUD_Mode_Toggle)
    
    bone_export: bpy.props.EnumProperty(items=[
            ("ON", "ON", "Save 3D bone as obj"),
            ("OFF", "OFF", "Don't save bone"),],
        default="OFF", update = BONE_Mode_Toggle)
    
    render_skinning_weight: bpy.props.EnumProperty(items=[
            ("ON", "ON", "Render skinning weight map"),
            ("OFF", "OFF", "Don't render skinning weight map"),],
        default="OFF", update = SKINNINGWEIGHT_Mode_Toggle)
    
    render_mask: bpy.props.EnumProperty(items=[
            ("ON", "ON", "Render mask map"),
            ("OFF", "OFF", "Don't render mask map"),],
        default="OFF", update = MASK_Mode_Toggle)
    
    axis_mode: bpy.props.EnumProperty(items=[
            ("Y_UP", "Y_UP", "Use Y-UP Axis"),
            ("Z_UP", "Z_UP", "Use Z-UP Axis"),],
        default="Z_UP", update = ZUP_YUP_Swap)

    
class SelectSaveDirectoryOperator(Operator, ImportHelper):
    bl_idname = "select_save_directory.character"
    bl_label = "Select Save Directory"
    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    
    def execute(self, context):
        context.scene.save_path = self.directory
        self.report({'INFO'}, "Selected Directory: " + context.scene.save_path)
        return {'FINISHED'}

class SelectDataDirectoryOperator(Operator, ImportHelper):
    bl_idname = "select_data_directory.character"
    bl_label = "Select Data Directory"
    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    
    def execute(self, context):
        context.scene.data_path = self.directory
        self.report({'INFO'}, "Selected Directory: " + context.scene.data_path)
        return {'FINISHED'}

class Render_Start(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "render_start.character"
    bl_label = "Render_Start_Class"
    
    
    def execute(self, context):
        set_cam()
        render_with_multiple_cameras(bpy.context.scene.save_path)
        # Rendering 
        return {'FINISHED'}
    
class Render_Start_With_Animation(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "render_start_with_animation.character"
    bl_label = "Render_Start_Class"
    
    
    def execute(self, context):
        set_cam()
        for frame in range(bpy.context.scene.frame_min, bpy.context.scene.frame_max + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.scene.frame_cur = frame
            
            render_with_multiple_cameras(bpy.context.scene.save_path, frame)
            #bpy.context.scene.render.filepath = f"{bpy.context.scene.save_path}frame_{frame:04d}"
            #bpy.ops.render.render(write_still=True)
    


class OBJ_importer(bpy.types.Operator, ImportHelper):
    bl_idname = "import_test_obj.character"
    bl_label = "Import Character(OBJ)"
    
    # ImportHelper mixin 클래스 사용
    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={'HIDDEN'}, maxlen=255,)

    def execute(self, context):
        print("Selected file:", self.filepath)
        empty_scene()
        bpy.ops.import_scene.obj(filepath=self.filepath)
        move_and_scale_scene()
        return {'FINISHED'}


class FBX_importer(bpy.types.Operator, ImportHelper):
    bl_idname = "import_test.character"
    bl_label = "Import Character(FBX)"
    
    filename_ext = ".fbx"
    filter_glob: StringProperty(default="*.fbx", options={'HIDDEN'}, maxlen=255,)

    def execute(self, context):
        print("Selected file:", self.filepath)
        empty_scene()
        bpy.ops.import_scene.fbx(filepath=self.filepath)
        move_and_scale_scene()
        return {'FINISHED'}

class CC_importer(bpy.types.Operator, ImportHelper):
    bl_idname = "import_test_cc.character"
    bl_label = "Import Character CC Addon"
    
    filename_ext = ".fbx"
    filter_glob: StringProperty(default="*.fbx",options={'HIDDEN'}, maxlen=255,)

    def execute(self, context):
        print("Selected file:", self.filepath)
        empty_scene()
        bpy.ops.cc3.importer(param="IMPORT", filepath=self.filepath)
        move_and_scale_scene()
        return {'FINISHED'}
    
class Set_camera(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "set_camera.character"
    bl_label = "Set camera on world"
    
    def execute(self, context):
        set_cam()
        return {'FINISHED'}
    

class Search_Last_Frame(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "search_last_frame.character"
    bl_label = "Search last frame of scene animation"

    def execute(self, context):
        max_frame, min_frame = find_scene_frame()
        
        bpy.context.scene.frame_max = max_frame
        bpy.context.scene.frame_min = min_frame

        bpy.context.scene.frame_auto_search = True if bpy.context.scene.frame_auto_search is False else False # Toggle auto search
        print("Find first and last frame", min_frame, max_frame, bpy.context.scene.frame_auto_search)
        return {'FINISHED'}


class MainPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "xrhuman_renderer"
    bl_idname = "xrhuman_renderer_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'xrhuman_renderer'


    # ICON은 여기서 조회가능 https://docs.blender.org/api/current/bpy_types_enum_items/icon_items.html
    def draw(self, context):
        layout = self.layout
        props = bpy.context.scene.CustomProperty
        
        row = layout.row()
        #row.label(text="Import FBX character")
        layout.operator(FBX_importer.bl_idname)

        
        row = layout.row()
        #row.label(text="Import CC4 character")
        layout.operator(CC_importer.bl_idname)

        
        row = layout.row()
        #row.label(text="Import OBJ character")
        layout.operator(OBJ_importer.bl_idname)

        #row = layout.row()
        #row.label(text="Camera Setting")
        
        
        box = layout.box()
        box.label(text="Data Path", icon='IMPORT')
        col = box.column(align=True)
        
        row = col.row(align=True)
        row.prop(context.scene, "data_path")
        row = col.row(align=True)
        row.operator(SelectDataDirectoryOperator.bl_idname)
        
        row = col.row(align=True)
        row.prop(context.scene, "save_path")
        row = col.row(align=True)
        row.operator(SelectSaveDirectoryOperator.bl_idname)
        


        
        # Camera Box
        box = layout.box()
        box.label(text="Camera Settings", icon='CAMERA_DATA')

        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(props, "axis_mode", expand=True)

        col = box.column(align=True)
        col.label(text="Azimuth: (min/max/split)")
        row = col.row(align=True)
        row.prop(context.scene, "azimuth_min")
        row.prop(context.scene, "azimuth_max")
        row.prop(context.scene, "horizontal_split")
        
        col = box.column(align=True)
        col.label(text="Elevation: (min/max/split)")
        row = col.row(align=True)
        row.prop(context.scene, "elevation_min")
        row.prop(context.scene, "elevation_max")
        row.prop(context.scene, "vertical_split")

        col = box.column(align=True)
        col.label(text="Radius:")
        row = col.row(align=True)
        row.prop(context.scene, "radius")

        col = box.column(align=True)
        row = col.row(align=True)
        row.operator(Set_camera.bl_idname)


        # Light Box
        box = layout.box()
        box.label(text="Light Settings", icon='LIGHT')
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(props, "light_mode", expand=True)





        # Animation Box
        box = layout.box()
        box.label(text="Animation Settings", icon='ARMATURE_DATA')

        col = box.column(align=True)
        col.label(text="Frame: (min/max)")
        row = col.row(align=True)
        row.prop(context.scene, "frame_min")
        row.prop(context.scene, "frame_max")
        #row = col.row(align=True)
        #row.operator(Search_Last_Frame.bl_idname, text="Set Frame Auto")

        row = col.row(align=True)
        row.prop(props, "auto_frame_mode", expand=True)
        


        # Output Data Box
        box = layout.box()
        box.label(text="Output Data ", icon='IMAGE_DATA')

        col = box.column(align=True)
        row = col.row(align=True)
        row.label(text="RGB")
        row.prop(props, "rgb_render", expand=True)

        col = box.column(align=True)
        row = col.row(align=True)
        row.label(text="Depth")
        row.prop(props, "depth_map_render", expand=True)

        col = box.column(align=True)
        row = col.row(align=True)
        row.label(text="Normal")
        row.prop(props, "normal_map_render", expand=True)

        col = box.column(align=True)
        row = col.row(align=True)
        row.label(text="Point3D")
        row.prop(props, "pointcloud_export", expand=True)

        col = box.column(align=True)
        row = col.row(align=True)
        row.label(text="Bone")
        row.prop(props, "bone_export", expand=True)

        col = box.column(align=True)
        row = col.row(align=True)
        row.label(text="Skinning")
        row.prop(props, "render_skinning_weight", expand=True)

        col = box.column(align=True)
        row = col.row(align=True)
        row.label(text="Mask")
        row.prop(props, "render_mask", expand=True)


        # Start Button
        row = layout.row()
        row.scale_y = 2
        row.operator(Render_Start.bl_idname,  text="Render")
        row.operator(Render_Start_With_Animation.bl_idname,  text="Render Anim")
        row = layout.row()




def register():
    # ADDON에서 사용할 변수들! 사용자가 UI를 통해 손쉽게 조작할 수 있음.  
    # bpy.types.Scene.azimuth_min 와 같이 선언하고, 함수에서는 bpy.context.scene.azimuth_min 를 통해 코드내에서 접근할 수 있음
    
    bpy.types.Scene.data_path = bpy.props.StringProperty(name="Data Path", default="", description="Path of data folder")
    bpy.types.Scene.save_path = bpy.props.StringProperty(name="Save Path", default="", description="Path of save folder")

    bpy.types.Scene.world_axis_type = bpy.props.StringProperty(name="Axis", default="Z_UP", description="World Axis type")

    # Camera Extrinsic Hyperparameter
    bpy.types.Scene.azimuth_min = bpy.props.IntProperty(name="Min", default=0, description="Minimum azimuth angle")
    bpy.types.Scene.azimuth_max = bpy.props.IntProperty(name="Max", default=360, description="Maximum azimuth angle")
    bpy.types.Scene.elevation_min = bpy.props.IntProperty(name="Min", default=-20, description="Minimum elevation angle")
    bpy.types.Scene.elevation_max = bpy.props.IntProperty(name="Max", default=30, description="Maximum elevation angle")
    bpy.types.Scene.radius = bpy.props.FloatProperty(name="Radius", default=3, description="Radius")
    bpy.types.Scene.horizontal_split = bpy.props.IntProperty(name="Horizontal Split", default=4, description="horizontal_split")
    bpy.types.Scene.vertical_split = bpy.props.IntProperty(name="Vertical Split", default=1, description="vertical_split")

    # Animation Hyperparameter
    bpy.types.Scene.frame_cur = bpy.props.IntProperty(name="Current Frame", default=0, description="Current Frame of Animation")
    bpy.types.Scene.frame_max = bpy.props.IntProperty(name="Max", default=1, description="Last Frame of Animation")
    bpy.types.Scene.frame_min = bpy.props.IntProperty(name="Min", default=0, description="First Frame of Animation")
    bpy.types.Scene.frame_auto_search = bpy.props.BoolProperty(name="Auto Search Frame", default=False, description="Automatically search frame toggle")


    bpy.types.Scene.render_rgb = bpy.props.StringProperty(name="Rgb MODE", default="ON", description="Rgb MODE")
    bpy.types.Scene.render_depth = bpy.props.StringProperty(name="Depth MODE", default="OFF", description="Depth MODE")
    bpy.types.Scene.render_normal = bpy.props.StringProperty(name="Normal MODE", default="OFF", description="Normal MODE")
    bpy.types.Scene.save_point3d = bpy.props.StringProperty(name="Point MODE", default="OFF", description="Point MODE")
    bpy.types.Scene.save_bone3d = bpy.props.StringProperty(name="Bone MODE", default="OFF", description="Bone MODE")
    bpy.types.Scene.render_skinning_weight = bpy.props.StringProperty(name="Skinning Mode", default="OFF", description="Skinning Mode")
    bpy.types.Scene.render_mask = bpy.props.StringProperty(name="Mask MODE", default="OFF", description="Mask Mode")


    # Class
    bpy.utils.register_class(Render_Start)
    bpy.utils.register_class(Render_Start_With_Animation)
    bpy.utils.register_class(OBJ_importer)
    bpy.utils.register_class(FBX_importer)
    bpy.utils.register_class(CC_importer)
    bpy.utils.register_class(MainPanel)
    bpy.utils.register_class(Set_camera)

    bpy.utils.register_class(Search_Last_Frame)
    bpy.utils.register_class(SelectSaveDirectoryOperator)
    bpy.utils.register_class(SelectDataDirectoryOperator)
    
    # Property 의 경우 Toggle 가능한 버튼을 생성하기 위해 사용
    bpy.utils.register_class(CustomProperty)
    bpy.types.Scene.CustomProperty = bpy.props.PointerProperty(type=CustomProperty)

def unregister():
    # Class
    bpy.utils.unregister_class(Render_Start)
    bpy.utils.unregister_class(Render_Start_With_Animation)
    bpy.utils.unregister_class(OBJ_importer)
    bpy.utils.unregister_class(FBX_importer)
    bpy.utils.unregister_class(CC_importer)
    bpy.utils.unregister_class(MainPanel)
    bpy.utils.unregister_class(Set_camera)

    bpy.utils.unregister_class(Search_Last_Frame)
    bpy.utils.unregister_class(SelectSaveDirectoryOperator)
    bpy.utils.unregister_class(SelectDataDirectoryOperator)

    bpy.utils.unregister_class(CustomProperty)
    del(bpy.types.Scene.CustomProperty) # Property는 del을 통해 삭제

if __name__ == "__main__":
    register()