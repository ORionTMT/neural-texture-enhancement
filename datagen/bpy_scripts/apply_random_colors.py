import bpy
import random

def create_unlit_material(name, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    for node in nodes:
        nodes.remove(node)
    
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (200, 0)
    
    emission_node = nodes.new(type="ShaderNodeEmission")
    emission_node.location = (0, 0)
    emission_node.inputs[0].default_value = color
    
    links.new(emission_node.outputs[0], output_node.inputs[0])
    
    return material

def replace_scene_materials(scene):
    for obj in scene.objects:
        if obj.type in {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}:
            for slot in obj.material_slots:
                random_color = [random.random(), random.random(), random.random(), 1.0]
                material_name = f"Unlit_{obj.name}_{random.randint(0, 1000)}"
                unlit_material = create_unlit_material(material_name, random_color)
                slot.material = unlit_material

replace_scene_materials(bpy.context.scene)
