# Copyright (C) 2021 Victor Soupday
# This file is part of CC/iC Blender Tools <https://github.com/soupday/cc_blender_tools>
#
# CC/iC Blender Tools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CC/iC Blender Tools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CC/iC Blender Tools.  If not, see <https://www.gnu.org/licenses/>.

import os
import shutil
import bpy

from . import (imageutils, jsonutils, materials, modifiers, nodeutils, physics,
               scene, channel_mixer, shaders, basic, properties, utils, vars)

debug_counter = 0


def delete_import(chr_cache):
    props = bpy.context.scene.CC3ImportProps

    for obj_cache in chr_cache.object_cache:
        obj = obj_cache.object
        if props.paint_object == obj:
            props.paint_object = None
            props.paint_material = None
            props.paint_image = None
        utils.try_remove(obj, True)

    all_materials_cache = chr_cache.get_all_materials_cache()
    for mat_cache in all_materials_cache:
        mat = mat_cache.material
        utils.try_remove(mat, True)

    chr_cache.import_file = ""
    chr_cache.import_type = ""
    chr_cache.import_name = ""
    chr_cache.import_dir = ""
    chr_cache.import_main_tex_dir = ""
    chr_cache.import_space_in_name = False
    chr_cache.import_embedded = False
    chr_cache.import_has_key = False
    chr_cache.import_key_file = ""

    chr_cache.tongue_material_cache.clear()
    chr_cache.teeth_material_cache.clear()
    chr_cache.head_material_cache.clear()
    chr_cache.skin_material_cache.clear()
    chr_cache.tearline_material_cache.clear()
    chr_cache.eye_occlusion_material_cache.clear()
    chr_cache.eye_material_cache.clear()
    chr_cache.hair_material_cache.clear()
    chr_cache.pbr_material_cache.clear()
    chr_cache.object_cache.clear()

    utils.remove_from_collection(props.import_cache, chr_cache)

    utils.clean_collection(bpy.data.images)
    utils.clean_collection(bpy.data.materials)
    utils.clean_collection(bpy.data.textures)
    utils.clean_collection(bpy.data.meshes)
    utils.clean_collection(bpy.data.armatures)
    utils.clean_collection(bpy.data.node_groups)
    # as some node_groups are nested...
    utils.clean_collection(bpy.data.node_groups)


def process_material(chr_cache, obj, mat, object_json):
    props = bpy.context.scene.CC3ImportProps
    mat_cache = chr_cache.get_material_cache(mat)
    mat_json = jsonutils.get_material_json(object_json, mat)

    if not mat_cache: return

    # don't process user added materials
    if mat_cache.user_added: return

    if not mat.use_nodes:
        mat.use_nodes = True

    if chr_cache.setup_mode == "ADVANCED":

        if mat_cache.is_cornea() or mat_cache.is_eye():
            shaders.connect_eye_shader(obj, mat, object_json, mat_json)

        elif mat_cache.is_tearline():
            shaders.connect_tearline_shader(obj, mat, mat_json)

        elif mat_cache.is_eye_occlusion():
            shaders.connect_eye_occlusion_shader(obj, mat, mat_json)

        elif mat_cache.is_skin() or mat_cache.is_nails():
            shaders.connect_skin_shader(obj, mat, mat_json)

        elif mat_cache.is_teeth():
            shaders.connect_teeth_shader(obj, mat, mat_json)

        elif mat_cache.is_tongue():
            shaders.connect_tongue_shader(obj, mat, mat_json)

        elif mat_cache.is_hair():
            shaders.connect_hair_shader(obj, mat, mat_json)

        elif mat_cache.is_sss():
            shaders.connect_sss_shader(obj, mat, mat_json)

        else:
            shaders.connect_pbr_shader(obj, mat, mat_json)

    else:

        nodeutils.clear_cursor()
        nodeutils.reset_cursor()

        if mat_cache.is_eye_occlusion():
            basic.connect_eye_occlusion_material(obj, mat, mat_json)

        elif mat_cache.is_tearline():
            basic.connect_tearline_material(obj, mat, mat_json)

        elif mat_cache.is_cornea():
            basic.connect_basic_eye_material(obj, mat, mat_json)

        else:
            basic.connect_basic_material(obj, mat, mat_json)

        nodeutils.move_new_nodes(-600, 0)

    # apply cached alpha settings
    #if character_cache.generation == "ActorCore":
    #    materials.set_material_alpha(mat, "CLIP")
    if mat_cache is not None:
        if mat_cache.alpha_mode != "NONE":
            materials.apply_alpha_override(obj, mat, mat_cache.alpha_mode)
        if mat_cache.culling_sides > 0:
            materials.apply_backface_culling(obj, mat, mat_cache.culling_sides)

    # apply any channel mixers
    if mat_cache is not None:
        if mat_cache.mixer_settings:
                mixer_settings = mat_cache.mixer_settings
                if mixer_settings.rgb_image or mixer_settings.id_image:
                    channel_mixer.rebuild_mixers(chr_cache, mat, mixer_settings)


def process_object(chr_cache, obj, objects_processed, character_json):
    props = bpy.context.scene.CC3ImportProps
    prefs = bpy.context.preferences.addons[__name__.partition(".")[0]].preferences

    if obj is None or obj in objects_processed:
        return

    objects_processed.append(obj)

    object_json = jsonutils.get_object_json(character_json, obj)
    physics_json = None

    utils.log_info("")
    utils.log_info("Processing Object: " + obj.name + ", Type: " + obj.type)
    utils.log_indent()

    obj_cache = chr_cache.get_object_cache(obj)

    if obj.type == "MESH":

        # remove any modifiers for refractive eyes
        modifiers.remove_eye_modifiers(obj)

        # process any materials found in the mesh object
        for mat in obj.data.materials:
            if mat and mat not in objects_processed:
                utils.log_info("")
                utils.log_info("Processing Material: " + mat.name)
                utils.log_indent()
                process_material(chr_cache, obj, mat, object_json)
                utils.log_recess()
                objects_processed.append(mat)

        # setup special modifiers for displacement, UV warp, etc...
        if chr_cache.setup_mode == "ADVANCED":
            if obj_cache.is_eye():
                modifiers.add_eye_modifiers(obj)
            elif obj_cache.is_eye_occlusion():
                modifiers.add_eye_occlusion_modifiers(obj)
            elif obj_cache.is_tearline():
                modifiers.add_tearline_modifiers(obj)

    elif obj.type == "ARMATURE":

        # set the frame range of the scene to the active action on the armature
        if prefs.physics == "ENABLED" and props.physics_mode == "ON":
            scene.fetch_anim_range(bpy.context)

    utils.log_recess()


def cache_object_materials(character_cache, obj, character_json, processed):
    props = bpy.context.scene.CC3ImportProps

    if obj is None or obj in processed:
        return

    obj_json = jsonutils.get_object_json(character_json, obj)
    obj_cache = character_cache.add_object_cache(obj)

    if obj.type == "MESH":

        utils.log_info(f"Caching Object: {obj.name}")
        utils.log_indent()

        for mat in obj.data.materials:

            if mat and mat.node_tree is not None and not mat in processed:

                object_type, material_type = materials.detect_materials(character_cache, obj, mat, obj_json)
                obj_cache.object_type = object_type
                mat_cache = character_cache.add_material_cache(mat, material_type)
                mat_cache.dir = imageutils.get_material_tex_dir(character_cache, obj, mat)
                utils.log_indent()
                materials.detect_embedded_textures(character_cache, obj, obj_cache, mat, mat_cache)
                materials.detect_mixer_masks(character_cache, obj, obj_cache, mat, mat_cache)
                utils.log_recess()
                processed.append(mat)

            elif mat in processed:

                object_type, material_type = materials.detect_materials(character_cache, obj, mat, obj_json)
                obj_cache.object_type = object_type

        utils.log_recess()

    processed.append(obj)


def apply_edit_shapekeys(obj):
    """For objects with shapekeys, set the active visible and edit mode shapekey to the basis.
    """
    # shapekeys data path:
    #   bpy.context.active_object.data.shape_keys.key_blocks['Basis']
    if obj.type == "MESH":
        shape_keys = obj.data.shape_keys
        if shape_keys is not None:
            blocks = shape_keys.key_blocks
            if blocks is not None:
                # if the object has shape keys
                if len(blocks) > 0:
                    try:
                        # set the active shapekey to the basis and apply shape keys in edit mode.
                        obj.active_shape_key_index = 0
                        obj.show_only_shape_key = False
                        obj.use_shape_key_edit_mode = True
                    except Exception as e:
                        utils.log_error("Unable to set shape key edit mode!", e)


def init_shape_key_range(obj):
    #bpy.context.active_object.data.shape_keys.key_blocks['Basis']
    if obj.type == "MESH":
        shape_keys: bpy.types.Key = obj.data.shape_keys
        if shape_keys is not None:
            blocks = shape_keys.key_blocks
            if blocks is not None:
                if len(blocks) > 0:
                    for block in blocks:
                        # expand the range of the shape key slider to include negative values...
                        block.slider_min = -1.0

            # re-set a value in the shapekey action keyframes to force
            # the shapekey action to update to the new ranges:
            try:
                action = utils.safe_get_action(shape_keys)
                if action:
                    co = action.fcurves[0].keyframe_points[0].co
                    action.fcurves[0].keyframe_points[0].co = co
            except:
                pass


def detect_generation(chr_cache, json_data):

    if json_data:
        json_generation = jsonutils.get_character_generation_json(json_data, chr_cache.import_name, chr_cache.character_id)
        if json_generation is not None and json_generation != "":
            try:
                return vars.CHARACTER_GENERATION[json_generation]
            except:
                pass

    generation = "UNKNOWN"

    arm = chr_cache.get_armature()

    if arm:
        chr_cache.parent_object = arm
        if utils.find_pose_bone_in_armature(arm, "RootNode_0_", "RL_BoneRoot"):
            generation = "ActorCore"
        elif utils.find_pose_bone_in_armature(arm, "CC_Base_L_Pinky3", "L_Pinky3"):
            generation = "G3"
        elif utils.find_pose_bone_in_armature(arm, "pinky_03_l"):
            generation = "GameBase"
        elif utils.find_pose_bone_in_armature(arm, "CC_Base_L_Finger42", "L_Finger42"):
            generation = "G1"
        utils.log_info(f"Generation could be: {generation} detected from pose bones.")

    if generation == "UNKNOWN":
        for obj_cache in chr_cache.object_cache:
            obj = obj_cache.object
            if obj.type == "MESH":
                name = obj.name.lower()
                if "cc_game_body" in name or "cc_game_tongue" in name:
                    generation = "GameBase"
                elif "cc_base_body" in name:
                    if utils.object_has_material(obj, "ga_skin_body"):
                        generation = "GameBase"
                    elif utils.object_has_material(obj, "std_skin_body"):
                        generation = "G3"
                    elif utils.object_has_material(obj, "skin_body"):
                        generation = "G1"
        if generation != "UNKNOWN":
            utils.log_info(f"Generation could be: {generation} detected from materials.")

    if generation == "UNKNOWN" or generation == "G3":

        for obj_cache in chr_cache.object_cache:
            obj = obj_cache.object
            if obj.type == "MESH" and obj.name == "CC_Base_Body":

                # try vertex count
                if len(obj.data.vertices) == 14164:
                    utils.log_info("Generation: G3Plus detected by vertex count.")
                    return "G3Plus"
                elif len(obj.data.vertices) == 13286:
                    utils.log_info("Generation: G3 detected by vertex count.")
                    return "G3"

                #try UV map test
                if materials.test_for_material_uv_coords(obj, 0, [[0.5, 0.763], [0.7973, 0.6147], [0.1771, 0.0843], [0.912, 0.0691]]):
                    utils.log_info("Generation: G3Plus detected by UV test.")
                    return "G3Plus"
                elif materials.test_for_material_uv_coords(obj, 0, [[0.5, 0.034365], [0.957562, 0.393431], [0.5, 0.931725], [0.275117, 0.961283]]):
                    utils.log_info("Generation: G3 detected by UV test.")
                    return "G3"

    return generation


def is_iclone_temp_motion(name : str):
    u_idx = utils.safe_index_of(name, '_', 0)
    if u_idx == -1:
        return False
    if not name[:u_idx].isdigit():
       return False
    search = "TempMotion"
    if utils.partial_match(name, "TempMotion", u_idx + 1):
        return True
    else:
        return False


def remap_action_names(actions, objects, name):
    key_map = {}
    num_keys = 0

    armature_actions = []
    shapekey_actions = []

    for obj in objects:
        if obj.type == "MESH":
            new_obj_name = utils.get_action_shape_key_object_name(obj.name)
            if obj.data.shape_keys:
                key_map[new_obj_name] = obj.data.shape_keys.name
                utils.log_info(f"ShapeKey: {obj.data.shape_keys.name} belongs to: {new_obj_name}")
                num_keys += 1

    for action in actions:
        action_key_name = action.name.split("|")[0]
        new_action_name = action.name.split("|")[-1]
        if is_iclone_temp_motion(new_action_name):
            new_action_name = "iCTM"
        elif new_action_name == "AvatarCurrentMotion":
            new_action_name = "CCPose"
        if action.name.startswith("Armature"):
            new_name = f"{name}|A|{new_action_name}"
            utils.log_info(f"Renaming action: {action.name} to {new_name}")
            action.name = new_name
            armature_actions.append(action)
        else:
            for new_obj_name in key_map:
                key_name = key_map[new_obj_name]
                if action_key_name == key_name:
                    new_name = f"{name}|K|{new_obj_name}|{new_action_name}"
                    utils.log_info(f"Renaming action: {action.name} to {new_name}")
                    action.name = new_name
                    shapekey_actions.append(action)

    return armature_actions, shapekey_actions


def detect_character(file_path, type, objects, actions, json_data, warn):
    props = bpy.context.scene.CC3ImportProps
    prefs = bpy.context.preferences.addons[__name__.partition(".")[0]].preferences

    utils.log_info("")
    utils.log_info("Detecting Characters:")
    utils.log_info("---------------------")

    dir, name = os.path.split(file_path)
    type = name[-3:].lower()
    name = name[:-4]

    chr_json = jsonutils.get_character_json(json_data, name, name)
    chr_cache = props.import_cache.add()
    chr_cache.import_file = file_path
    chr_cache.import_type = type
    chr_cache.import_name = name
    chr_cache.import_dir = dir
    chr_cache.import_space_in_name = " " in name
    chr_cache.character_index = 0
    chr_cache.character_name = name
    chr_cache.character_id = name
    processed = []

    if type == "fbx":

        # key file
        chr_cache.import_key_file = os.path.join(chr_cache.import_dir, chr_cache.import_name + ".fbxkey")
        chr_cache.import_has_key = os.path.exists(chr_cache.import_key_file)

        # determine the main texture dir
        chr_cache.import_main_tex_dir = os.path.join(dir, name + ".fbm")
        if os.path.exists(chr_cache.import_main_tex_dir):
            chr_cache.import_embedded = False
        else:
            chr_cache.import_main_tex_dir = ""
            chr_cache.import_embedded = True

        # process the armature(s)
        arm_count = 0
        for arm in objects:
            if arm.type == "ARMATURE":
                arm_count += 1
                arm.name = name
                if arm.data:
                    arm.data.name = name
                # in case of duplicate names: character_name contains the name currently in Blender.
                #                             import_name contains the original name.
                chr_cache.character_name = arm.name
                # add armature to object_cache
                chr_cache.add_object_cache(arm)

        if arm_count > 1:
            warn.append("Multiple characters detected in Fbx.")
            warn.append("Character exports from iClone to Blender do not fully support multiple characters.")
            warn.append("Characters should be exported individually for best results.")

        # add child objects to object_cache
        for obj in objects:
            if obj.type == "MESH":
                chr_cache.add_object_cache(obj)

        # remame actions
        utils.log_info("Renaming actions:")
        utils.log_indent()
        remap_action_names(actions, objects, chr_cache.character_name)
        utils.log_recess()

        # determine character generation
        chr_cache.generation = detect_generation(chr_cache, json_data)
        utils.log_info("Generation: " + chr_cache.character_name + " (" + chr_cache.generation + ")")

        # cache materials
        for obj_cache in chr_cache.object_cache:
            obj = obj_cache.object
            if obj.type == "MESH":
                cache_object_materials(chr_cache, obj, chr_json, processed)

        properties.init_character_property_defaults(chr_cache, chr_json)

    elif type == "obj":

        # key file
        chr_cache.import_key_file = os.path.join(chr_cache.import_dir, chr_cache.import_name + ".ObjKey")
        chr_cache.import_has_key = os.path.exists(chr_cache.import_key_file)

        # determine the main texture dir
        chr_cache.import_main_tex_dir = os.path.join(dir, name)
        chr_cache.import_embedded = False
        if not os.path.exists(chr_cache.import_main_tex_dir):
            chr_cache.import_main_tex_dir = ""

        for obj in objects:
            if obj.type == "MESH":
                chr_cache.add_object_cache(obj)

        for obj_cache in chr_cache.object_cache:
            # scale obj import by 1/100
            obj = obj_cache.object
            obj.scale = (0.01, 0.01, 0.01)
            if not chr_cache.import_has_key: # objkey import is a single mesh with no materials
                cache_object_materials(chr_cache, obj, json_data, processed)

        properties.init_character_property_defaults(chr_cache, chr_json)

    # set preserve volume on armature modifiers
    for obj in objects:
        if obj.type == "MESH":
            arm_mod = modifiers.get_object_modifier(obj, "ARMATURE")
            if arm_mod:
                arm_mod.use_deform_preserve_volume = False

    # material setup mode
    if chr_cache.import_has_key:
        chr_cache.setup_mode = prefs.morph_mode
    else:
        chr_cache.setup_mode = prefs.quality_mode

    # character render target
    chr_cache.render_target = prefs.render_target

    utils.log_info("")
    return chr_cache


# Import operator
#

class CC3Import(bpy.types.Operator):
    """Import CC3 Character and build materials"""
    bl_idname = "cc3.importer"
    bl_label = "Import"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(
        name="Filepath",
        description="Filepath of the fbx or obj to import.",
        subtype="FILE_PATH"
        )

    filter_glob: bpy.props.StringProperty(
        default="*.fbx;*.obj;*.glb;*.gltf;*.vrm",
        options={"HIDDEN"},
        )

    param: bpy.props.StringProperty(
            name = "param",
            default = "",
            options={"HIDDEN"}
        )

    use_anim: bpy.props.BoolProperty(name = "Import Animation", description = "Import animation with character.\nWarning: long animations take a very long time to import in Blender 2.83", default = True)

    running = False
    imported = False
    built = False
    lighting = False
    timer = None
    clock = 0
    invoked = False
    imported_character = None
    imported_materials = []
    imported_images = []
    is_rl_character = False


    def import_character(self, warn):
        props = bpy.context.scene.CC3ImportProps

        utils.start_timer()

        utils.log_info("")
        utils.log_info("Importing Character Model:")
        utils.log_info("--------------------------")

        self.detect_import_mode()

        import_anim = self.use_anim

        dir, name = os.path.split(self.filepath)
        type = name[-3:].lower()
        imported = None
        actions = None

        json_data = jsonutils.read_json(self.filepath)

        if type == "fbx":

            # invoke the fbx importer
            utils.tag_objects()
            utils.tag_images()
            utils.tag_actions()
            bpy.ops.import_scene.fbx(filepath=self.filepath, directory=dir, use_anim=import_anim)
            imported = utils.untagged_objects()
            actions = utils.untagged_actions()
            self.imported_images = utils.untagged_images()

            # detect characters and objects
            self.imported_character = detect_character(self.filepath, type, imported, actions, json_data, warn)

            utils.log_timer("Done .Fbx Import.")

        elif type == "obj":

            # invoke the obj importer
            utils.tag_objects()
            utils.tag_images()
            if self.param == "IMPORT_MORPH":
                bpy.ops.import_scene.obj(filepath = self.filepath, split_mode = "OFF",
                        use_split_objects = False, use_split_groups = False,
                        use_groups_as_vgroups = True)
            else:
                bpy.ops.import_scene.obj(filepath = self.filepath, split_mode = "ON",
                        use_split_objects = True, use_split_groups = True,
                        use_groups_as_vgroups = False)

            imported = utils.untagged_objects()
            self.imported_images = utils.untagged_images()

            # detect characters and objects
            self.imported_character = detect_character(self.filepath, type, imported, actions, json_data, warn)

            #if self.param == "IMPORT_MORPH":
            #    if self.imported_character.import_main_tex_dir != "":
            #        reconstruct_obj_materials(obj)
            #        pass

            utils.log_timer("Done .Obj Import.")

        elif type == "gltf" or type == "glb":

            # invoke the GLTF importer
            utils.tag_objects()
            utils.tag_images()
            bpy.ops.import_scene.gltf(filepath = self.filepath)
            imported = utils.untagged_objects()
            self.imported_images = utils.untagged_images()

            utils.log_timer("Done .GLTF Import.")

        elif type == "vrm":

            # copy .vrm to .glb
            glb_path = os.path.join(dir, name + "_temp.glb")
            shutil.copyfile(self.filepath, glb_path)
            self.filepath = glb_path

            # invoke the GLTF importer
            utils.tag_objects()
            utils.tag_images()
            bpy.ops.import_scene.gltf(filepath = self.filepath)
            imported = utils.untagged_objects()
            self.imported_images = utils.untagged_images()

            # find the armature and rotate it 180 degrees in Z
            arm : bpy.types.Object = utils.get_armature_in_objects(imported)
            if arm:
                arm.rotation_mode = "XYZ"
                arm.rotation_euler = (0, 0, 3.1415926535897)
                utils.set_only_active_object(arm)
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
                utils.try_select_objects(imported)

            os.remove(glb_path)

            utils.log_timer("Done .vrm Import.")


    def build_materials(self, context):
        objects_processed = []
        props: properties.CC3ImportProps = bpy.context.scene.CC3ImportProps
        prefs = bpy.context.preferences.addons[__name__.partition(".")[0]].preferences

        utils.start_timer()

        utils.log_info("")
        utils.log_info("Building Character Materials:")
        utils.log_info("-----------------------------")

        nodeutils.check_node_groups()

        chr_cache: properties.CC3CharacterCache = None
        if self.imported_character:
            chr_cache = self.imported_character
            json_data = jsonutils.read_json(self.filepath)
        else:
            chr_cache = props.get_context_character_cache(context)
            if chr_cache:
                self.imported_character = chr_cache
                json_data = jsonutils.read_json(chr_cache.import_file)
                # when rebuilding, use the currently selected render target
                chr_cache.render_target = prefs.render_target

        if chr_cache:

            chr_json = jsonutils.get_character_json(json_data, chr_cache.import_name, chr_cache.character_id)

            if self.param == "BUILD":
                chr_cache.check_material_types(chr_json)

            if props.build_mode == "IMPORTED":
                for cache in chr_cache.object_cache:
                    if cache.object:
                        process_object(chr_cache, cache.object, objects_processed, chr_json)

            # only processes the selected objects that are listed in the import_cache (character)
            elif props.build_mode == "SELECTED":
                for cache in chr_cache.object_cache:
                    if cache.object and cache.object in bpy.context.selected_objects:
                        process_object(chr_cache, cache.object, objects_processed, chr_json)

            # setup default physics
            if prefs.physics == "ENABLED" and props.physics_mode == "ON":
                utils.log_info("")
                physics.add_all_physics(chr_cache)

            # enable SSR
            if prefs.refractive_eyes == "SSR":
                bpy.context.scene.eevee.use_ssr = True
                bpy.context.scene.eevee.use_ssr_refraction = True

        utils.log_timer("Done Build.", "s")


    def detect_import_mode(self):
        # detect if we are importing a character for morph/accessory editing (i.e. has a key file)
        dir, name = os.path.split(self.filepath)
        type = name[-3:].lower()
        name = name[:-4]

        textures_path = os.path.join(dir, "textures", name)
        json_path = os.path.join(dir, name + ".json")

        self.is_rl_character = False

        if type == "obj":
            obj_key_path = os.path.join(dir, name + ".ObjKey")
            if os.path.exists(obj_key_path):
                self.param = "IMPORT_MORPH"
                utils.log_info("Importing as character morph with ObjKey.")
                self.is_rl_character = True
                return

        elif type == "fbx":
            obj_key_path = os.path.join(dir, name + ".fbxkey")
            if os.path.exists(obj_key_path):
                self.param = "IMPORT_MORPH"
                utils.log_info("Importing as editable character with fbxkey.")
                self.is_rl_character = True
                return

        if os.path.exists(textures_path):
            self.is_rl_character = True

        if os.path.exists(json_path):
            self.is_rl_character = True

        if self.is_rl_character:
            utils.log_info("Importing for rendering without key file.")
        else:
            utils.log_info("Importing generic character.")

        self.param = "IMPORT_QUALITY"


    def run_import(self, context):

        warn = []
        self.import_character(warn)

        if len(warn) > 0:
            utils.message_box_multi("Import Warning!", "ERROR", warn)

        self.imported = True


    def run_build(self, context):

        if self.is_rl_character:
            self.build_materials(context)

        self.built = True


    def run_finish(self, context):
        props = bpy.context.scene.CC3ImportProps
        prefs = bpy.context.preferences.addons[__name__.partition(".")[0]].preferences

        chr_cache = self.imported_character

        if chr_cache:

            # for any objects with shape keys expand the slider range to -1.0 <> 1.0
            # Character Creator and iClone both use negative ranges extensively.
            for obj_cache in chr_cache.object_cache:
                init_shape_key_range(obj_cache.object)

            if self.param == "IMPORT_MORPH" or self.param == "IMPORT_ACCESSORY":
                # for any objects with shape keys select basis and enable show in edit mode
                for obj_cache in chr_cache.object_cache:
                    apply_edit_shapekeys(obj_cache.object)

            if self.param == "IMPORT_MORPH" or self.param == "IMPORT_ACCESSORY":
                if prefs.lighting == "ENABLED" and props.lighting_mode == "ON":
                    if chr_cache.import_type == "fbx":
                        scene.setup_scene_default(prefs.pipeline_lighting)
                    else:
                        scene.setup_scene_default(prefs.morph_lighting)

            # use portrait lighting for quality mode
            if self.param == "IMPORT_QUALITY":
                if prefs.lighting == "ENABLED" and props.lighting_mode == "ON":
                    scene.setup_scene_default(prefs.quality_lighting)

            if prefs.refractive_eyes == "SSR":
                bpy.context.scene.eevee.use_ssr = True
                bpy.context.scene.eevee.use_ssr_refraction = True

            # set a minimum of 50 max transparency bounces:
            if bpy.context.scene.cycles.transparent_max_bounces < 50:
                bpy.context.scene.cycles.transparent_max_bounces = 50

            scene.zoom_to_character(chr_cache)
            scene.active_select_body(chr_cache)

            # clean up unused images from the import
            if len(self.imported_images) > 0:
                utils.log_info("Cleaning up unused images:")
                img: bpy.types.Image = None
                for img in self.imported_images:
                    num_users = img.users
                    if (img.use_fake_user and img.users == 1) or img.users == 0:
                        utils.log_info("Removing Image: " + img.name)
                        bpy.data.images.remove(img)
            utils.clean_collection(bpy.data.images)

        self.imported_character = None
        self.imported_materials = []
        self.imported_images = []
        self.lighting = True


    def modal(self, context, event):

        # 60 second timeout
        if event.type == 'TIMER':
            self.clock = self.clock + 1
            if self.clock > 600:
                self.cancel(context)
                self.report({'INFO'}, "Import operator timed out!")
                return {'CANCELLED'}

        if event.type == 'TIMER' and not self.running:
            if not self.imported:
                self.running = True
                self.run_import(context)
                if not self.is_rl_character:
                    self.cancel(context)
                    self.report({'INFO'}, "None Standard Character Done!")
                    return {'FINISHED'}
                self.clock = 0
                self.running = False
            elif not self.built:
                self.running = True
                self.run_build(context)
                self.clock = 0
                self.running = False
            elif not self.lighting:
                self.running = True
                self.run_finish(context)
                self.clock = 0
                self.running = False

            if self.imported and self.built and self.lighting:
                self.cancel(context)
                self.report({'INFO'}, "All Done!")
                return {'FINISHED'}

        return {'PASS_THROUGH'}


    def cancel(self, context):
        if self.timer is not None:
            context.window_manager.event_timer_remove(self.timer)
            self.timer = None


    def execute(self, context):
        props = bpy.context.scene.CC3ImportProps
        prefs = bpy.context.preferences.addons[__name__.partition(".")[0]].preferences
        self.imported_character = None
        self.imported_materials = []
        self.imported_images = []

        # import character
        if "IMPORT" in self.param:
            if self.filepath != "" and os.path.exists(self.filepath):
                if self.invoked and self.timer is None:
                    self.imported = False
                    self.built = False
                    self.lighting = False
                    self.running = False
                    self.clock = 0
                    self.report({'INFO'}, "Importing Character, please wait for import to finish and materials to build...")
                    bpy.context.window_manager.modal_handler_add(self)
                    self.timer = context.window_manager.event_timer_add(1.0, window = bpy.context.window)
                    return {'PASS_THROUGH'}
                elif not self.invoked:
                    self.run_import(context)
                    if self.is_rl_character:
                        self.run_build(context)
                        self.run_finish(context)
                    return {'FINISHED'}
            else:
                utils.log_error("No valid filepath to import!")

        # build materials
        elif self.param == "BUILD":
            self.build_materials(context)

        # rebuild the node groups for advanced materials
        elif self.param == "REBUILD_NODE_GROUPS":
            nodeutils.rebuild_node_groups()
            utils.clean_collection(bpy.data.images)

        elif self.param == "DELETE_CHARACTER":
            chr_cache = props.get_context_character_cache(context)
            if chr_cache:
                delete_import(chr_cache)

        return {"FINISHED"}


    def invoke(self, context, event):
        if "IMPORT" in self.param:
            context.window_manager.fileselect_add(self)
            self.invoked = True
            return {"RUNNING_MODAL"}

        return self.execute(context)


    @classmethod
    def description(cls, context, properties):
        if "IMPORT" in properties.param:
            return "Import a new .fbx or .obj character exported by Character Creator 3.\n" \
                   "Notes for exporting from CC3:\n" \
                   " - For round trip-editing (exporting character back to CC3), export as FBX: 'Mesh Only' or 'Mesh and Motion' with Calibration, from CC3, as this guarantees generation of the .fbxkey file needed to re-import the character back to CC3.\n" \
                   " - For creating morph sliders, export as OBJ: Nude Character in Bind Pose from CC3, as this is the only way to generate the .ObjKey file for morph slider creation in CC3.\n" \
                   " - FBX export with motion in 'Current Pose' or 'Custom Motion' does not export an .fbxkey and cannot be exported back to CC3.\n" \
                   " - OBJ export 'Character with Current Pose' does not create an .objkey and cannot be exported back to CC3.\n" \
                   " - OBJ export 'Nude Character in Bind Pose' .obj does not export any materials"
        elif properties.param == "IMPORT_ACCESSORY":
            return "Import .fbx or .obj character from CC3 for accessory creation. This will import current pose or animation.\n" \
                   "Notes for exporting from CC3:\n" \
                   "1. OBJ or FBX exports in 'Current Pose' are good for accessory creation as they import back into CC3 in exactly the right place"
        elif properties.param == "BUILD":
            return "Rebuild materials for the current imported character with the current build settings"
        elif properties.param == "DELETE_CHARACTER":
            return "Removes the character and any associated objects, meshes, materials, nodes, images, armature actions and shapekeys. Basically deletes everything not nailed down.\n**Do not press this if there is anything you want to keep!**"
        elif properties.param == "REBUILD_NODE_GROUPS":
            return "Rebuilds the shader node groups for the advanced and eye materials."
        return ""


class CC3ImportAnimations(bpy.types.Operator):
    """Import CC3 animations"""
    bl_idname = "cc3.anim_importer"
    bl_label = "Import Animations"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(
        name="Filepath",
        description="Filepath of the fbx to import.",
        subtype="FILE_PATH"
    )

    directory: bpy.props.StringProperty(subtype='DIR_PATH')

    files: bpy.props.CollectionProperty(
            type=bpy.types.OperatorFileListElement,
            options={'HIDDEN', 'SKIP_SAVE'}
    )

    filter_glob: bpy.props.StringProperty(
        default="*.fbx",
        options={"HIDDEN"}
    )

    remove_meshes: bpy.props.BoolProperty(
        default = True,
        description="Remove all imported mesh objects.",
        name="Remove Meshes",
    )

    remove_materials_images: bpy.props.BoolProperty(
        default = True,
        description="Remove all imported materials and image textures.",
        name="Remove Materials & Images",
    )

    remove_shape_keys: bpy.props.BoolProperty(
        default = False,
        description="Remove Shapekey actions along with their meshes.",
        name="Remove Shapekey Actions",
    )

    param: bpy.props.StringProperty(
            name = "param",
            default = "",
            options={"HIDDEN"}
    )

    def import_animation_fbx(self, dir, file):
        path = os.path.join(dir, file)
        name = file[:-4]

        utils.log_info(f"Importing Fbx file: {path}")

        # invoke the fbx importer
        utils.tag_objects()
        utils.tag_images()
        utils.tag_actions()
        utils.tag_materials()
        bpy.ops.import_scene.fbx(filepath=path, directory=dir, use_anim=True)
        objects = utils.untagged_objects()
        actions = utils.untagged_actions()
        images = utils.untagged_images()
        materials = utils.untagged_materials()

        #if "Imported Animations" not in bpy.data.collections:
        #    collection = bpy.data.collections.new("Imported Animations")
        #    bpy.context.scene.collection.children.link(collection)
        #else:
        #    collection = bpy.data.collections["Imported Animations"]

        utils.log_info("Renaming actions:")
        utils.log_indent()
        armature_actions, shapekey_actions = remap_action_names(actions, objects, name)
        utils.log_recess()

        utils.log_info("Cleaning up:")
        utils.log_indent()

        for obj in objects:
            if obj.type == "ARMATURE":
                obj.name = name
                if obj.data:
                    obj.data.name = name

        if self.remove_meshes:
            # only interested in actions, delete the rest
            for obj in objects:
                if obj.type != "ARMATURE":
                    utils.log_info(f"Removing Object: {obj.name}")
                    utils.delete_mesh_object(obj)
            # and optionally remove the shape keys
            if self.remove_shape_keys:
                for action in shapekey_actions:
                    utils.log_info(f"Removing Shapekey Action: {action.name}")
                    bpy.data.actions.remove(action)

        if self.remove_materials_images:
            for img in images:
                utils.log_info(f"Removing Image: {img.name}")
                bpy.data.images.remove(img)

            for mat in materials:
                utils.log_info(f"Removing Material: {mat.name}")
                bpy.data.materials.remove(mat)

        utils.log_recess()

        return

    def execute(self, context):
        props = bpy.context.scene.CC3ImportProps
        prefs = bpy.context.preferences.addons[__name__.partition(".")[0]].preferences

        utils.start_timer()

        utils.log_info("")
        utils.log_info("Importing FBX Animations:")
        utils.log_info("-------------------------")

        for fbx_file in self.files:
            self.import_animation_fbx(self.directory, fbx_file.name)

        utils.log_timer("Done Build.", "s")

        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    @classmethod
    def description(cls, context, properties):
        return ""


