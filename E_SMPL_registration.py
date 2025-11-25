## For this assignement you can reuse the usal environment.
## The only new libraries to install should be: tqdm, robust_laplacian

import smplx
import torch 
import open3d as o3d
import numpy as np 
from pytorch3d.loss import (
    chamfer_distance
)
import pickle as pkl
from os import path as osp
import robust_laplacian 
from matplotlib import pyplot as plt
from pytorch3d.ops import knn_points
from scipy.sparse.linalg import eigs,eigsh
import tqdm 

# Utility Functions
o3d_float   = o3d.utility.Vector3dVector
o3d_integer = o3d.utility.Vector3iVector
visualizer  = o3d.visualization.draw
TriMesh     = o3d.geometry.TriangleMesh
PointCloud  = o3d.geometry.PointCloud
o3d_read    = o3d.io.read_triangle_mesh
o3d_write   = o3d.io.write_triangle_mesh

# Setting the device
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
device = torch.device("cpu")
print("WARNING: CPU only, this will be slow!")


target_shape_path = './dataset/registration/target.ply'

######################################################################
##### TASK 1 - v2v registration
##### The target shape has the same vertices of SMPL. This means that the two are in
##### correspondence by their triangulation\parametrization. Tough, we do not know which
##### SMPL parameters generated the target. We can recover them by optimization.

# Creating SMPL layer
smpl_layer_opt = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Loading the target shape
target = o3d_read(target_shape_path)

target.compute_vertex_normals()
visualizer([target])

if np.asarray(target.vertices).shape[0] != 6890:
    pass
else:
    # Parsing the target vertices into pytorch
    target_v = torch.tensor(np.asarray(target.vertices), dtype=torch.float32)

    # Initializing optimizer
    optim = torch.optim.Adam(smpl_layer_opt.parameters(), lr=0.1)

    # Setting the number of iterations
    n_iters = 1000
    pbar = tqdm.tqdm(range(0, n_iters))

    # Optimization cycle
    for i in pbar:
        # Getting the SMPL vertices
        predictions = smpl_layer_opt()['vertices']

        # Computing the v2v loss
        loss = torch.sum((predictions - target_v)**2)

        # Backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        # Logging
        pbar.set_description("Loss: %f" % loss)

    # Storing the result
    v = predictions.detach().cpu().numpy().squeeze()
    f = smpl_layer_opt.faces
    registration = TriMesh(o3d_float(v),o3d_integer(f))
    registration.compute_vertex_normals()
    registration.vertex_colors = o3d_float(v)

    o3d_write('./dataset/output/registration/out_gt.ply', registration)

    # Visualizing the result 
    visualizer([registration, target])

## QUESTIONS
# 1) Do you think we reach a global minimum, or is it possible to obtain even a better alignement?
# 2) Do the intermidiate shapes all represent realistic humans?
######################################################################

##### TASK 2 - Chamfer registration
##### Of course, we generally cannot assume to have a 1:1 correspondence between the shapes
##### hence, a v2v loss cannot be applied. Instead, we can rely on Chamfer distance, as seen
##### during our lectures.

# Creating a SMPL layer
smpl_layer_opt = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Loading the target shape
target = o3d_read(target_shape_path)
target.compute_vertex_normals()

# Parsing the target vertices into pytorch
target_v = torch.tensor(np.asarray(target.vertices), dtype=torch.float32)

# Setting the number of iterations
n_iters = 5000
pbar = tqdm.tqdm(range(0, n_iters))

# Initializing optimizer
optim = torch.optim.Adam(smpl_layer_opt.parameters(), lr=0.001)

for i in pbar:
  # Getting the SMPL vertices
  predictions = smpl_layer_opt()['vertices']

  # Define chamfer distance as a loss. 
  # NOTE: Since chamfer can be quite computational expensive, you can subsample the shape
  # by considering one point every 10
  loss, _ = chamfer_distance(predictions[:,::5,:], target_v[::5,:].unsqueeze(0))

  loss.backward()
  optim.step()
  optim.zero_grad()

  # Logging
  pbar.set_description("Loss: %f" % loss)

# Storing the result
v = predictions.detach().cpu().numpy().squeeze()
f = smpl_layer_opt.faces
registration = TriMesh(o3d_float(v),o3d_integer(f))
registration.compute_vertex_normals()
registration.vertex_colors = o3d_float(v)
o3d_write('./dataset/output/registration/out_chamfer.ply', registration)

# Compute an error w.r.t. the GT (v2v distance)
if np.asarray(target.vertices).shape[0] != 6890:
    pass
else:
    GT_error = torch.sum((predictions - target_v)**2)

# Visualizing the result 
visualizer([registration, target])

## QUESTIONS
# 1) Compared to the v2v fitting, we are using more iterations and a lower learning rate. Why?
# 2) Do you think that using some of the regularizations that we have seen during the lecture (e.g., edge loss, laplacian loss, normal loss,...) can help? 
#    If yes, why and which? If no, why?

######################################################################

##### TASK 3 - Chamfer registration + Pose Prior
##### To regularize the optimization, we can rely on a learned prior on the SMPL parameters. 
##### Specifically, someone has fitted a Gaussian model to the distribution of the SMPL poses (\theta parameters)
##### and stored the mean and the inverse of the covariance matrix into './dataset/priors/body_prior.pkl'.
##### During the optimization, we can use the (squared) Mahalanobis distance to assess how plausible is a certain pose.
##### https://en.wikipedia.org/wiki/Mahalanobis_distance
##### 

# Create SMPL layer
smpl_layer_opt = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Loading the target shape
target = o3d_read(target_shape_path)
target.compute_vertex_normals()

# Parsing the target vertices into pytorch
target_v = torch.tensor(np.asarray(target.vertices), dtype=torch.float32)

# Loading the pose prior
body_prior = pkl.load(open(osp.join('dataset/prior', 'body_prior.pkl'), 'rb'), encoding='latin')

# Loading the mean of a gaussian distribution
gmm_mean = torch.from_numpy(body_prior['mean']).float().unsqueeze(0).to(device)

# Loading the inverse of covariance matrix (Σ⁻¹). REMARK: it is stored as Cholesky decomposition
gmm_precision_ch = torch.from_numpy(body_prior['precision']).float().to(device)
gmm_precision = gmm_precision_ch @ gmm_precision_ch.T

# Setting the number of iterations
n_iters = 5000
pbar = tqdm.tqdm(range(0, n_iters))

# Initializing optimizer
optim = torch.optim.Adam(smpl_layer_opt.parameters(), lr=0.001)

# Defyining the squared mahalanobis distance
def mahalanobis(u, mu, cov):
    delta = u - mu
    m = torch.matmul(delta.squeeze(), torch.matmul(cov, delta.squeeze()))
    return m

# Running optimization with annealing on prior regularization
# epochs from 0 to 1500:    0.01
# epochs from 1500 to 3500: 0.001
# epochs from 3500 to 4500: 0.0001
# After 4500:               0.0

for i in pbar:
  # Getting the SMPL vertices
  predictions  = smpl_layer_opt()['vertices']

  # Getting the SMPL pose
  pose         = smpl_layer_opt.body_pose 

  # Computing the prior loss
  loss_prior   = mahalanobis(pose[:, :63],gmm_mean, gmm_precision) 

  # Using chamfer distance as a loss. 
  # NOTE: Since chamfer can be quite computational expensive, you can subsample the shape
  # by considering one point every 10
  loss_data, _ = chamfer_distance(predictions[:,::5,:], target_v[::5,:].unsqueeze(0))

  # Annealing
  if i < 1500:
    wh = 0.01
  if i > 1500:
    wh = 0.001
  if i > 3500:
    wh = 0.0001
  if i > 4500:
    wh = 0

  # Loss composition
  loss = loss_data + loss_prior * wh 

  # Backward pass
  loss.backward()
  optim.step()
  optim.zero_grad()

  # Logging
  pbar.set_description(f"Cham_Loss: {loss_data:.4f} / prior_Loss: {loss_prior:.4f}")

# Storing the result
v = predictions.detach().cpu().numpy().squeeze()
f = smpl_layer_opt.faces
registration = TriMesh(o3d_float(v),o3d_integer(f))
registration.compute_vertex_normals()
registration.vertex_colors = o3d_float(v)
o3d_write('./dataset/output/registration/out_chamfer+pose.ply', registration)

# Compute an error w.r.t. the GT (v2v distance)
if np.asarray(target.vertices).shape[0] != 6890:
    pass
else:
    GT_error = torch.sum((predictions - target_v)**2)

# Visualizing the result
visualizer([registration, target])

## QUESTIONS
# 1) Run it again but keep the regularization weight constant across the optimization. What happen?
# 2) How the GT_error and the Chamfer loss changed from Task 2?
# 3) What do you think is the problem that limit this optimization?


######################################################################
##### TASK 4 - Functional Maps guidance
##### We want to introduce another term that might help our optimization: a point-to-point Functional Maps correspondence.
##### First, we need to obtain a Functional correspondence between these two shapes.
##### In the previous lectures/assignement, you have experimented with the conversions between a point-to-point correspondence
##### and the functional map C. We need two ingredients:
##### 1) The functional bases for registration and target shapes. We will use the eigenfunctions of the LBO
##### 2) A correspondence between registration and target. We will use the nearest-neighbor pairing after the last registration
##### NOTE: This step assumes you already run the TASK 3, as we are proceeding from that.
#####        My solution is based on robust_laplacian library, but if you prefer other implementation everything should work really similar

# Getting the vertices and faces
v_reg = np.asarray(registration.vertices)
f_reg = np.asarray(registration.triangles)
v_tar = np.asarray(target.vertices)
f_tar = np.asarray(target.triangles)

# Getting the laplacians
reg_laplacian, reg_mass = robust_laplacian.mesh_laplacian(v_reg,f_reg)
tar_laplacian, tar_mass = robust_laplacian.mesh_laplacian(v_tar,f_tar)

# Getting the eigenfunctions. Let's use just the first 10.
k = 15
evals_reg, evecs_reg     = eigsh(reg_laplacian, k, reg_mass, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
evals_target, evecs_target = eigsh(tar_laplacian, k, tar_mass, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)

# Finding a nearest-neighbor matching between the two shapes
queries = knn_points(torch.tensor(v_tar,dtype=torch.float32).unsqueeze(0),torch.tensor(v_reg,dtype=torch.float32).unsqueeze(0))
Pi_nn = np.squeeze(queries[1])

# Converting Pi into the matrix C using Pi_nn as correspondence
C = evecs_reg.T @ reg_mass @ evecs_target[Pi_nn]

# Converting the C into a correspondence Pi
evecs_reg_C = torch.tensor((C.T @evecs_reg.T).T, dtype=torch.float32)
queries = knn_points(evecs_reg_C.unsqueeze(0), torch.tensor(evecs_target,dtype=torch.float32).unsqueeze(0))
Pi = np.squeeze(queries[1])

# Visualizing the correspondence
colors_target = (v_tar - np.min(v_tar,0))/(np.max(v_tar,0) - np.min(v_tar,0))

# Storing the matrix C
plt.imshow(C)
plt.show()
plt.savefig('./dataset/output/registration/C.png')

# Visualizing the result

## QUESTIONS
# 1) Visualize Pi and Pi_nn correspondences. How does Pi_nn and Pi change? Do you have an intuition about why?
# 2) Visualize the first three non-constant eigenfunction: a) of the registration; b) of the registration, but after applying the C to them; c) of the target shape.
#    What do you notice?
# 3) Run it again, but with C of different sizes (e.g., 5x5 or 100x100). How does it affect the Pi correspondence? Do you have an intuition about why?

######################################################################
##### TASK 5 - Optimizing using Chamfer Distance, Pose Prior, Functional Maps
##### Now we are all set to run the final optimization. We will use three losses:
##### A) Chamfer distance
##### B) Pose Prior
##### C) v2v using the Functional Maps correspondence Pi
##### NOTE: This step starts from the output of TASK 3 and TASK 4. 

# Initializing optimizer
optim = torch.optim.Adam(smpl_layer_opt.parameters(), lr=0.005)

# Setting the number of iterations
n_iters = 5000
pbar = tqdm.tqdm(range(0, n_iters))

# Running optimization with annealing on prior regularization and FMAP loss. Use the following weights:
# epochs from 0 to 2500:    0.001 and 0.1
# epochs from 2500 to 4000: 0.0001 and 0.01
# After 4000:               0.0 and 0.0

for i in pbar:
  # Getting the SMPL vertices
  predictions  = smpl_layer_opt()['vertices']

  # Getting the SMPL pose
  pose         = smpl_layer_opt.body_pose 

  # Computing the prior loss
  loss_prior   = mahalanobis(pose[:, :63],gmm_mean, gmm_precision) 

  # Computing the chamfer loss
  loss_data, _ = chamfer_distance(predictions[:,::5,:], target_v[::5,:].unsqueeze(0))

  # Computing the FMAP loss
  loss_fmap = torch.sqrt(torch.sum((predictions - target_v[Pi,:])**2))

  # Annealing
  if i < 2500:
    wh = 0.0001
    wfmap = 0.1
  elif i<4000:
    wh = 0.00001
    wfmap = 0.01
  else:
    wh = 0
    wfmap = 0
  
  # Obtaining the full loss
  loss = loss_data + loss_prior * wh  + loss_fmap*wfmap

  loss.backward()
  optim.step()
  optim.zero_grad()

  # Logging
  pbar.set_description(f"Cham_Loss: {loss_data:.4f} / Prior_Loss: {loss_prior:.4f} / FMAP_Loss: {loss_prior:.4f}")

v = predictions.detach().cpu().numpy().squeeze()
f = smpl_layer_opt.faces

# Compute an error w.r.t. the GT (v2v distance)
if np.asarray(target.vertices).shape[0] != 6890:
    pass
else:
    GT_error = torch.sum((predictions - target_v)**2)


# Storing the result
registration = TriMesh(o3d_float(v),o3d_integer(f))
registration.compute_vertex_normals()
registration.vertex_colors = o3d_float(v)

visualizer([registration, target])
o3d_write('./dataset/output/registration/out_final.ply', registration)

## QUESTIONS
# 1) How the final registration looks like? What do you think is the key change introduced by FMAP correspondence?
# 2) Do you see any significant error still present? If yes, do you see any possible solution?
# 3) Run the last three tasks again, but for TASK 4 try to use a bigger C, e.g., using 100 functions. Does it work better or worse? Do you have an intuition about why?
# 4) Run the TASK 5 but re-initialize SMPL in t-pose. Does the optimization still work?