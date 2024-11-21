import bpy
import json

camera = bpy.data.objects.get("Camera")

camera_data = {}

for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    
    position = camera.location
    rotation = camera.rotation_euler
    
    camera_data[frame] = {
        "position": {"x": position.x, "y": position.y, "z": position.z},
        "rotation": {"x": rotation.x, "y": rotation.y, "z": rotation.z},
    }

with open("campath.json", "w") as f:
    json.dump(camera_data, f, indent=4)
