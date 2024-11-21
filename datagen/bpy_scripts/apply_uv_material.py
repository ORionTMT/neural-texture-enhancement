import bpy

# Replace all materials used in this scene with the material named "UV"
material = bpy.data.materials.get('UV')

for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        for i, slot in enumerate(obj.material_slots):
            if "nodraw" not in slot.material.name.lower():  # No-draw brushes should not be included
                obj.material_slots[i].material = material

