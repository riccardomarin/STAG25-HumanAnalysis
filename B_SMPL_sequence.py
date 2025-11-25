# Room Model:
# https://sketchfab.com/3d-models/the-king-s-hall-d18155613363445b9b68c0c67196d98d
# 
#
# smpl_uv can be obtained from the SMPL website. Here it is slightly modified to accomodate
# our visualization tools
#
# texture is from meshcapade wiki: https://meshcapade.wiki/SMPL#sample-objs-with-textures
#

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer 
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes
from aitviewer.utils.so3 import aa2rot_numpy

C.smplx_models = './dataset/body_models/'
import numpy as np 
import smplx
import torch 
import open3d as o3d
import numpy as np 
import pickle as pkl
import os 
import tqdm 

import trimesh 
from PIL import Image

# Utility Functions
o3d_float      = o3d.utility.Vector3dVector
o3d_integer    = o3d.utility.Vector3iVector
visualizer     = o3d.visualization.draw   #in case of issues, use o3d.visualization.draw_geometries
TriMesh        = o3d.geometry.TriangleMesh
PointCloud     = o3d.geometry.PointCloud
o3d_read       = o3d.io.read_triangle_mesh
o3d_write      = o3d.io.write_triangle_mesh

# Initializing SMPL Layer
smpl_layer = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')
smpl_layer()['vertices'].shape

# Loading the poses
dict = np.load('dataset/motions/poses.npz')

# Look into the MoCap dictionary
for k in dict.keys():
    print(k)

# Check how long is the sequence 
print(len(dict['poses']))

# Select an initial and final frame and a step. Save a mesh every 200 frames
start_frame = 180  
end_frame   = 5000
step = 20

# Iterate over the poses and save the meshes. 
# Set the poses, betas, and translation accordingly
poses = torch.zeros((1, 23 * 3), dtype=torch.float32)
betas = torch.zeros((1, 10), dtype=torch.float32)
trans = torch.zeros((1, 3), dtype=torch.float32) 

seq_vertices = []
for i in tqdm.tqdm(np.arange(start_frame, end_frame, step)):
    
    # Setting the SMPL parameters for the current frame
    trasl           = torch.tensor(dict['trans'][i]     ,dtype=torch.float32).unsqueeze(0)
    body_pose       = torch.tensor(dict['poses'][i][3:24*3] ,dtype=torch.float32).unsqueeze(0)
    global_orient   = torch.tensor(dict['poses'][i][0:3] ,dtype=torch.float32).unsqueeze(0)
    betas           = torch.tensor(dict['betas'][0:10]  ,dtype=torch.float32).unsqueeze(0)
    
    # Forward pass 
    output = smpl_layer(  betas = betas,
                          body_pose = body_pose,
                          global_orient = global_orient,
                          transl = trasl)

    # Saving the frame 
    v = output['vertices'].detach().cpu().numpy().squeeze()
    f = smpl_layer.faces
    frame_pose = TriMesh(o3d_float(v),o3d_integer(f))
    seq_vertices.append(v)

    # Optinal: Saving frames locally
    # frame_pose = TriMesh(o3d_float(v),o3d_integer(f))
    # o3d_write('dataset/output/task_2/out_' + str(i) + '.ply',frame_pose)

# Save the motion sequence
numpy_seq_vertices = np.asarray(seq_vertices)

# Loading the UV coordinates for SMPL
# Note: the original smpl_uv has per_wedge_textures_coordinates.
# however, this is not supported by  aitviewer. So I had to convert them to
# per_vertex_textures_coordinates, which can create some artifacts.

smpl_texture = trimesh.load('./dataset/body_models/smpl_uv2.obj', process=True)

# Rotations to make SMPL oriented up-right
r2 = aa2rot_numpy(np.repeat(np.array([[-np.pi/2, 0, 0]]),len(numpy_seq_vertices),0)) 
r3 = aa2rot_numpy(np.repeat(np.array([[0, 0, np.pi]]),len(numpy_seq_vertices),0)) 
rots = r2 @ r3

texture_image = "./dataset/body_models/f_02_alb.002.png"

# Create the node with the vertices and colors we- computed.
cubesphere = Meshes(
    numpy_seq_vertices,
    f,
    name="SMPL",
    position=[0, 0, 0],
    scale=1,
    flat_shading=True,
    rotation = rots,
    uv_coords= smpl_texture.visual.uv,
    path_to_texture= texture_image
)

# Loading a room model
r = trimesh.load('./dataset/model.obj')
texture_image = "./dataset/tex_u1_v1.jpg"
room_viewer = Meshes(
    np.asarray(r.vertices),
    np.asarray(r.faces),
    position=[1.72,2,-9],
    uv_coords= r.visual.uv,
    path_to_texture= texture_image
)

# Create a viewer.
v = Viewer()

# Set the camera position.
v.scene.camera.position = (7.32, 0.8, -2.9)

# Add the animated mesh to the scene.
v.scene.add(cubesphere)
v.scene.add(room_viewer)

v.run()

