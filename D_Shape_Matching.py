import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import scipy.io as sio
import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
import torch  
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import open3d as o3d
from torch.autograd import Variable
import tqdm 
import scipy as sp
import smplx
import robust_laplacian
import matplotlib.pyplot as plt

# Utility Functions
o3d_float      = o3d.utility.Vector3dVector
o3d_integer    = o3d.utility.Vector3iVector
visualizer     = o3d.visualization.draw   #in case of issues, use o3d.visualization.draw_geometries
TriMesh        = o3d.geometry.TriangleMesh
PointCloud     = o3d.geometry.PointCloud
o3d_read       = o3d.io.read_triangle_mesh
o3d_write      = o3d.io.write_triangle_mesh

# Initialize SMPL model
smpl_layer = smplx.create('./dataset/body_models/', model_type="smpl", gender='neutral')

# Foward-pass
output = smpl_layer.forward()
V_SMPL = output['vertices'].detach().cpu().numpy().squeeze()
F_SMPL = smpl_layer.faces

mesh1 = o3d.geometry.TriangleMesh(o3d_float(V_SMPL),o3d_integer(F_SMPL))

# mesh1 = o3d_read('./dataset/spectral/tr_reg_090.off')
v_src, f_src = np.asarray(mesh1.vertices), np.asarray(mesh1.triangles)

mesh2 = o3d_read('./dataset/spectral/tr_reg_043.off')
mesh2.vertices = o3d_float(np.asarray(mesh2.vertices))
v_tar, f_tar = np.asarray(mesh2.vertices), np.asarray(mesh2.triangles)

funz_ = (v_tar - np.min(v_tar,0))/np.tile((np.max(v_tar,0)-np.min(v_tar,0)),(np.size(v_tar,0),1));
colors = np.cos(funz_)
funz_tar = (colors-np.min(colors))/(np.max(colors) - np.min(colors));

mesh1.vertex_colors = o3d_float(funz_tar)
mesh2.vertex_colors = o3d_float(funz_tar)
visualizer([mesh1, mesh2])

dtype = 'float32'
k = 100

L_src, A_src = robust_laplacian.mesh_laplacian(v_src, f_src)

try:
    evals_src, evecs_src = eigsh(L_src, k, A_src, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
except:
    evals_src, evecs_src = eigsh(L_src- 1e-8* scipy.sparse.identity(v_src.shape[0]), k,
                         A_src, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)

L_tar, A_tar = robust_laplacian.mesh_laplacian(v_tar, f_tar)

try:
    evals_tar, evecs_tar = eigsh(L_tar, k, A_tar, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
except:
    evals_tar, evecs_tar = eigsh(L_tar- 1e-8* scipy.sparse.identity(v_tar.shape[0]), k,
                         A_tar, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)

# Point-to-point correspondence computation
treesearch = sp.spatial.cKDTree(v_src)
p2p = treesearch.query(v_tar, k=1)[1]

# To see the quality of the matching we plot a function on one shape and we transfer it to the other
funz_ = (v_src - np.min(v_src,0))/np.tile((np.max(v_src,0)-np.min(v_src,0)),(np.size(v_src,0),1));
colors = np.cos(funz_);
funz_src = (colors-np.min(colors))/(np.max(colors) - np.min(colors));

mesh1.vertex_colors = o3d_float(funz_src)
mesh2.vertex_colors = o3d_float(funz_src[p2p,:])
visualizer([mesh1, mesh2])

# Computing (euclidean) error evaluation
err = np.sum(np.square(v_tar - v_tar[p2p,:]))
print("Matching error - Euclidean: " + str (err))

#############
# Computing descriptors
n_land =  v_tar.shape[0] // 100
n_evals = 10

# Landmarks, as step functions randomly sampled
step = np.int32(np.ceil(v_src.shape[0] / n_land))
a = np.arange(0,v_src.shape[0],step)
landmarks = np.zeros((v_src.shape[0], a.size))
landmarks[a,np.arange(a.size)] = 1

# Optimization Process
desc_src = landmarks
desc_tar = landmarks

# Descriptor normalization
no = np.sqrt(np.diag(np.matmul(A_src.__matmul__(desc_src).T, desc_src)))
no_s = np.tile(no.T,(v_src.shape[0],1))
no_t = np.tile(no.T,(v_tar.shape[0],1))
fct_src = np.divide(desc_src,no_s)
fct_tar = np.divide(desc_tar,no_t)

# Coefficents of the obtained descriptors
Fct_src = np.matmul(A_src.T.__matmul__(evecs_src[:, 0:n_evals]).T, fct_src)
Fct_tar = np.matmul(A_tar.T.__matmul__(evecs_tar[:, 0:n_evals]).T, fct_tar)

# The relation between the two constant functions can be computed in a closed form
constFct = np.zeros((n_evals,1))
constFct[0, 0] = np.sign(evecs_src[0, 0] * evecs_tar[0, 0]) * np.sqrt(np.sum(A_tar)/np.sum(A_src))

# Energy weights
w1 = 1e-1 # Descriptors preservation
w2 = 1e-8 # Commutativity with Laplacian

# Define objects
fs = torch.Tensor(Fct_src)
ft = torch.Tensor(Fct_tar)
evals = torch.diag(torch.Tensor(np.reshape(np.float32(evals_src[0:n_evals]), (n_evals,))))
evalt = torch.diag(torch.Tensor(np.reshape(np.float32(evals_tar[0:n_evals]), (n_evals,))))

C_ini = np.zeros((n_evals,n_evals))
C_ini[0,0]=constFct[0,0]
C = Variable(torch.Tensor(C_ini), requires_grad=True)

optimizer = torch.optim.Adam([C], lr=5e-2)

for it in tqdm.tqdm(range(1500)):   
    optimizer.zero_grad()

    # Loss computation
    loss1 = w1 * torch.sum(((torch.matmul(C, fs) - ft) ** 2)) / 2 # Descriptor preservation
    loss2 = w2 * torch.sum((torch.matmul(C, evals) - torch.matmul(evalt,C))**2) # Commute with Laplacian
    loss = torch.sum(loss1  + loss2)

    # Gradient descent
    loss.backward()
    optimizer.step()

# Showing C matrix
C_np = C.detach().numpy()
plt.imshow(C_np)
plt.colorbar()
plt.show()

# Point-to-point correspondence computation
treesearch = sp.spatial.cKDTree(np.matmul(evecs_src[:,0:n_evals], C_np.T))
p2p = treesearch.query(evecs_tar[:,0:n_evals], k=1)[1]

# Correspondence visualization
funz_ = (v_src - np.min(v_src,0))/np.tile((np.max(v_src,0)-np.min(v_src,0)),(np.size(v_src,0),1));
colors = np.cos(funz_);
funz_src = (colors-np.min(colors))/(np.max(colors) - np.min(colors));

mesh1.vertex_colors = o3d_float(funz_src)
mesh2.vertex_colors = o3d_float(funz_src[p2p,:])
visualizer([mesh1, mesh2])

# Computing (euclidean) error evaluation
err = np.sum(np.square(v_tar - v_tar[p2p,:]))
print("Matching error - Euclidean: " + str (err))


