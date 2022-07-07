import argparse
import logging
import os
import statistics
import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator
from kubric.safeimport.bpy import bpy
import numpy as np
from scipy.spatial import transform

logging.basicConfig(level="INFO")

parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=6)

parser.add_argument("--rotate_pole", type=int, default=0)
parser.add_argument("--angular_vel", type=float, default=20)

parser.add_argument("--trans_camera", type=int, default=0)
parser.add_argument("--trans_vel", type=float, default=0.5)

parser.add_argument("--use_motion_blur", type=int, default=0)                  # default 0.1 | Max 1.0, 
parser.add_argument("--rolling_shutter_type", type=str, default='NONE')             # ['TOP', 'NONE'] 
parser.add_argument("--rolling_shutter", type=float, default=0.025)                 # default 0.0025 | Max 1.0, 
parser.add_argument("--motion_blur_shutter", type=float, default=0.5)               # default 0.5 | Max 1.0,
parser.add_argument("--motion_blur_position", type=str, default='CENTER')               # default 0.5 | Max 1.0,

parser.add_argument("--out_dir", type=str, default='./output')                      # dir

FLAGS = parser.parse_args()
print('FLAGS {}'.format(FLAGS))

out_dir = FLAGS.out_dir
os.makedirs(out_dir, exist_ok=True)
logging.info('makedir {}'.format(out_dir))

# ? --- create scene and attach a renderer and simulator
scene = kb.Scene(resolution=(640, 480))
scene.frame_end = FLAGS.num_frames      # < numbers of frames to render
scene.frame_rate = 24                   # < rendering framerate
scene.step_rate = 240                   # < simulation framerate

renderer = KubricBlender(scene) # Assign the renderer to the scene
simulator = KubricSimulator(scene) # Assign the Simulator to the scene

# ? --- populate the scene with a Floor, lights, a camera and a long thin rectangle with the angular velocity
# Create a floor
floor = kb.Cube(
    name="floor",
    scale=(10, 10, 0.1),
    position=(0, 0, -0.1),
    static=True,
    background=True,
    segmentation_id=1,
    material=kb.PrincipledBSDFMaterial(color=kb.get_color("black"))
)

# Create a directional light
sun = kb.DirectionalLight(
    name="sun",
    position=(0, 0, 6),
    look_at=(0, 0, 0),
    intensity=1.5
    )

# Create a camera
scene.camera = kb.PerspectiveCamera(
    name="camera",
    position=(0, 0, 5),
    look_at=(0, 0, 0),
    # focal_length=35.,
    # sensor_width=32
)

print('intrinsics: ', scene.camera.intrinsics)

# ? Create a rectangle with the angular velocity
color = kb.random_hue_color() # Get a random color
material = kb.PrincipledBSDFMaterial(color=color) # Create a material with the random color

POSITION = (0, 0, 0) # Position of the rectangle
VELOCITY = (0,0,FLAGS.angular_vel) if FLAGS.rotate_pole else (0, 0, 0)
SCALE = (0.1, 5, 0.1) # Scale of the rectangle
print('POSITION {}, VELOCITY {}, SCALE {}'.format(POSITION, VELOCITY, SCALE))

obj = kb.Cube(
    name="rectangle",
    scale=SCALE,
    velocity=(0,0,0),
    angular_velocity = VELOCITY,
    position=POSITION,
    mass=0.2,
    restitution=1,
    material=material,
    friction=1,
    segmentation_id=2,
    static=True
)

# I like being lazy, thats why I made an array and used a for loop to add the objects to the scene
objects = [floor, sun, obj]
for obj in objects:
    scene += obj

# ! Rolling Shutter
bpy.context.scene.render.use_motion_blur = FLAGS.use_motion_blur                                            # Turn on motion blur inside the Renderer
logging.info("Motion blur is turned on in the Scene Renderer, use_motion_blur: {}".format(FLAGS.use_motion_blur)) #loggy loggy log
bpy.context.scene.cycles.motion_blur_position=FLAGS.motion_blur_position
logging.info("Motion blur position is turned on the {} of the frame".format(FLAGS.motion_blur_position)) 
bpy.context.scene.cycles.rolling_shutter_type = FLAGS.rolling_shutter_type                                  # Now when we set this it will apply as the setting is on
logging.info(f"Rolling Shutter type is {bpy.context.scene.cycles.rolling_shutter_type}")                    #loggy loggy log
bpy.context.scene.cycles.rolling_shutter_duration = FLAGS.rolling_shutter                                   # Set the duration of the motion blur
logging.info(f"Rolling Shutter duration is {bpy.context.scene.cycles.rolling_shutter_duration}")            #loggy loggy log
bpy.context.scene.render.motion_blur_shutter = FLAGS.motion_blur_shutter                                    # Set the shutter speed of the motion blur
logging.info(f"Motion Blur Shutter speed is {bpy.context.scene.render.motion_blur_shutter}")                #loggy loggy log

# ! trans camera
cam_params = []
if FLAGS.trans_camera:
  original_camera_position = scene.camera.position
  num_phi_values_per_theta = 1
  theta_change = FLAGS.trans_vel / \
      ((scene.frame_end - scene.frame_start) / num_phi_values_per_theta)

  for frame in range(scene.frame_start, scene.frame_end + 1):
    x, y, z = scene.camera.position
    x += FLAGS.trans_vel
    # y += FLAGS.trans_vel(1)
    # z += FLAGS.trans_vel(2)
    scene.camera.position = (x, y, z)

    # scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

    cam_param = np.zeros([1, 8])
    quat = scene.camera.quaternion
    rot = transform.Rotation.from_quat(quat)
    inv_quat = rot.inv().as_quat()

    cam_param[0, 0] = scene.camera.focal_length
    cam_param[0, 1] = x
    cam_param[0, 2] = y
    cam_param[0, 3] = quat[3]
    cam_param[0, 4:7] = quat[:3]
    cam_param[0, 7] = z
    cam_params.append(cam_param)


# --- executes the simulation (and store keyframes)
simulator.run()

# --- renders the output
kb.as_path("output").mkdir(exist_ok=True)
renderer.save_state(f"{out_dir}/camera_shutter_example.blend")
frames_dict = renderer.render()

frames_dict.pop('segmentation')
frames_dict.pop('object_coordinates')
frames_dict.pop('forward_flow')
frames_dict.pop('backward_flow')
frames_dict.pop('normal')
frames_dict.pop('depth')

# --- renders the output
kb.write_image_dict(frames_dict, out_dir)