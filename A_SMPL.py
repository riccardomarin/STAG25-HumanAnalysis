import smplx
import torch 
import open3d as o3d
import numpy as np 
import pickle as pkl
import os 

# Utility Functions
o3d_float      = o3d.utility.Vector3dVector
o3d_integer    = o3d.utility.Vector3iVector
visualizer     = o3d.visualization.draw   #in case of issues, use o3d.visualization.draw_geometries
TriMesh        = o3d.geometry.TriangleMesh
PointCloud     = o3d.geometry.PointCloud
o3d_read       = o3d.io.read_triangle_mesh
o3d_write      = o3d.io.write_triangle_mesh
o3d_write_pc   = o3d.io.write_point_cloud

#### TASK 0 -- Looking into SMPL.pkl
smpl_pickle = pkl.load(open('./dataset/body_models/smpl/SMPL_NEUTRAL.pkl', 'rb'), encoding='latin1')
print(smpl_pickle.keys())

print("Joint Regressor:" + str(smpl_pickle['J_regressor'].shape))
print("Kinematic tree:" + str(smpl_pickle['kintree_table'].shape))
print("Kinematic tree - fathers:\n" + str(smpl_pickle['kintree_table'][0]))
print("Kinematic tree - Joints IDs:\n" + str(smpl_pickle['kintree_table'][1]))
print("Joints:" + str(smpl_pickle['J'].shape))
print("Skinning Weights:" + str(smpl_pickle['weights'].shape))

print("PCA - Shape:" + str(smpl_pickle['shapedirs'].shape))
print("PCA - Pose:"  + str(smpl_pickle['posedirs'].shape))

print("Neutral Pose Mesh Vertices:" + str(smpl_pickle['v_template'].shape))
print("Faces:" + str(smpl_pickle['f'].shape))

#### TASK 1 -- Visualizing T-Pose SMPL

# Initialize SMPL model
smpl_layer = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Foward-pass
output = smpl_layer.forward()

# Selecting the vertices
v = output['vertices'].detach().cpu().numpy().squeeze()

# Visualization
f = smpl_layer.faces
t_pose = TriMesh(o3d_float(v),o3d_integer(f))
t_pose.compute_vertex_normals()

visualizer([t_pose])

# Save a model
os.makedirs('./dataset/output/task_1', exist_ok=True)
o3d_write('./dataset/output/out_tpose.ply',t_pose)

## Task 1.5 -- Inspecting the output
for k in output.keys():
    if output[k] is not None:
        print(k.ljust(13),":", output[k].shape)
    else:
        print(k.ljust(13),":", None)


####  TASK 2 -- Visualizing SMPL Skeleton
# Initialize SMPL model
smpl_layer = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Foward-pass
output = smpl_layer()

# Selecting the joints
v = output['joints'].detach().cpu().numpy().squeeze()[0:24]
key_pc = PointCloud(o3d_float(v))

# smpl_layer.parents contains the kinematic chain
bones = np.vstack((np.arange(len(smpl_layer.parents)), smpl_layer.parents.numpy())).T[1:]
print(bones)

# We define the bones as lines between the joints
colors = [[1, 0, 0] for i in range(len(bones))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(o3d_float(v)),
    lines=o3d.utility.Vector2iVector(bones),
)

# We enumerate the joints
node_labels = []
for i in np.arange(len(v)):
    hello_open3d_mesh = o3d.t.geometry.TriangleMesh.create_text(str(i), depth=2).to_legacy()
    hello_open3d_mesh.paint_uniform_color([1, 0.706, 0])
    hello_open3d_mesh.transform([[0.005, 0, 0, v[i,0]], [0, 0.005, 0,  v[i,1]], [0, 0, 0.005,  v[i,2]],
                             [0, 0, 0, 1]])
    node_labels.append(hello_open3d_mesh)

node_labels.append(key_pc)
node_labels.append(line_set)
visualizer(node_labels)

####  TASK 3 -- Visualizing SMPL in a different pose and identity
# Set random values for translation, body pose, global orientation, and betas
rand_trasl             = torch.rand((1, 3), dtype=torch.float32)
rand_body_pose         = torch.rand((1, 23 * 3), dtype=torch.float32)/3
rand_global_orient     = torch.rand((1, 3), dtype=torch.float32)
rand_betas             = torch.rand((1, 10), dtype=torch.float32)*5

output = smpl_layer(betas = rand_betas,
                    global_orient = rand_global_orient,
                    body_pose = rand_body_pose,
                    transl = rand_trasl)

v = output['vertices'].detach().cpu().numpy().squeeze()
f = smpl_layer.faces
rand_pose = TriMesh(o3d_float(v),o3d_integer(f))
rand_pose.compute_vertex_normals()

visualizer([rand_pose])

####  TASK 4 -- Apply a MoCap Sequence

# Initializing SMPL Layer
smpl_layer = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Loading the poses
dict = np.load('dataset/motions/poses.npz')

# Looking into the MoCap dictionary
for k in dict.keys():
    print(k)

# Checking how long is the sequence 
print(len(dict['poses']))

# Selecting an initial and final frame and a step. Save a mesh every 200 frames
start_frame = 500 
end_frame   = 1000 
step = 20

# Iterating over the poses and save the meshes. 
# Set the poses, betas, and translation accordingly
poses = torch.zeros((1, 23 * 3), dtype=torch.float32)
betas = torch.zeros((1, 10), dtype=torch.float32)
trans = torch.zeros((1, 3), dtype=torch.float32) 

for i in np.arange(start_frame, end_frame, step):
    
    # Setting the SMPL parameters for the current frame
    trasl           = torch.tensor(dict['trans'][i]     ,dtype=torch.float32).unsqueeze(0)
    body_pose       = torch.tensor(dict['poses'][i][3:24*3] ,dtype=torch.float32).unsqueeze(0)
    global_orient   = torch.tensor(dict['poses'][i][0:3] ,dtype=torch.float32).unsqueeze(0)
    betas           = torch.tensor(dict['betas'][0:10]  ,dtype=torch.float32).unsqueeze(0)
    
    # Forwarding pass 
    output = smpl_layer(  betas = betas,
                          body_pose = body_pose,
                          global_orient = global_orient,
                          transl = trasl)

    # Saving the frame 
    v = output['vertices'].detach().cpu().numpy().squeeze()
    f = smpl_layer.faces
    frame_pose = TriMesh(o3d_float(v),o3d_integer(f))

    os.makedirs('dataset/output/task_2', exist_ok=True)
    o3d_write('dataset/output/task_2/out_' + str(i) + '.ply',frame_pose)



########################################################################3
##### ASSIGNEMENTS
#### TASK A1 -- Move the left elbow of 90 degrees 
# Initializing SMPL model
smpl_layer = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Initializing T-pose values
trasl             = ... #
body_pose         = ... #
global_orient     = ... #
betas             = ... #

# Identifying the correct left elbow joint and move it
# such that it bends 90 degrees in front of the chest

body_pose[ ... ] = ...

output = smpl_layer(betas         = betas,
                    global_orient = global_orient,
                    body_pose     = body_pose,
                    transl        = trasl)

# Writing the Output
v = ...
f = ...

t_pose = TriMesh(o3d_float(v),o3d_integer(f))
o3d_write('output/task_1/out.ply',t_pose)

# QUESTIONS
# 1) How did you find the correct \theta to update? Can you explain why you set the values in that way?
# 2) There is more than one set for \theta parameters that can produce the same pose. Can you find another one?
# 3) Let's say you also want to add a tilt of the left elbow vertically of 90 degrees. How would you modify the parameters?

##### TASK A2 -- Apply the joint regressor to SMPL in T-pose
# Initializing SMPL layer to the T-pose 
smpl_layer = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')
trasl             = ...
body_pose         = ...
global_orient     = ...
betas             = ...

# Forwarding passo3d_write_pc   = o3d.io.write_pointcloud
output = ...

# Retrieving the 24 joints
output_joints = ...

# Applying the joint regressor to the vertices
regressed_joints = ...

# Quantifying the difference between output_joints and regressed_joints. You can compute the RMSE
difference = ...

# Saving the output
A_output = PointCloud(o3d_float(output_joints.numpy().squeeze()))
o3d_write_pc('output/task_3/A_output.ply',A_output)
A_regressed = PointCloud(o3d_float(regressed_joints.numpy().squeeze()))
o3d_write_pc('output/task_3/A_regressed.ply',A_regressed)

#### TASK A3 -- Apply the joint regressor to SMPL in a different pose
# Loading the MoCap poses you downloaded 
dict = ...

# Choosing a frame (let's use 5190 for replicability)
frame = 5190

trasl             = ...
body_pose         = ...
global_orient     = ...
betas             = ...

# Forwarding pass
output    = ...

# Retrieving the 24 joints
output_joints = ...

# Applying the joint regressor to the vertices
regressed_joints = ...

# Quantifying the difference between output_joints and regressed_joints. You can compute the RMSE
difference = ...


#Saving the output
B_output = PointCloud(o3d_float(output_joints.numpy().squeeze()))
o3d_write_pc('output/task_3/B_output.ply',B_output)
B_regressed = PointCloud(o3d_float(regressed_joints.numpy().squeeze()))
o3d_write_pc('output/task_3/B_regressed.ply',B_regressed)

# QUESTIONS
# 1) How do the "difference" terms of Task 3.A and 3.B change? Do you have an intuition on why?
# 2) Do you think it is possible to obtain a join regressor that does not have a similar phenomenon?
#    If no, why? If yes, do you have an idea of how it could be realized?