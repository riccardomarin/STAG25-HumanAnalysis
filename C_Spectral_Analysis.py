
import scipy
import scipy.io as sio
from scipy.sparse.linalg import eigsh

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import robust_laplacian

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap

# Utility Functions
o3d_float      = o3d.utility.Vector3dVector
o3d_integer    = o3d.utility.Vector3iVector
visualizer     = o3d.visualization.draw   #in case of issues, use o3d.visualization.draw_geometries
TriMesh        = o3d.geometry.TriangleMesh
PointCloud     = o3d.geometry.PointCloud
o3d_read       = o3d.io.read_triangle_mesh
o3d_write      = o3d.io.write_triangle_mesh

# Colormap for visualizaiton
colors = ['blue', 'white', 'red']
cmap = LinearSegmentedColormap.from_list('blue_white_red', colors)
def values_to_rgb(values, vmin=None, vmax=None):
    values = np.array(values)
    
    # Imposta limiti di normalizzazione
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    # Normalizza i valori tra 0 e 1
    norm = (values - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)

    # Applica la colormap
    rgba_colors = cmap(norm)
    
    # Estrai solo l'RGB (ignora Alpha)
    rgb_colors = rgba_colors[:, :3]
    
    return rgb_colors

# In this case the mesh is saved in a matlab-like file
shape_path1 = './dataset/spectral/pose.mat'
dtype = 'float32'
x = sio.loadmat(shape_path1)

# Converting the dictionary in convenient variables
vertices = x['M']['VERT'][0,0].astype(dtype)
triv = x['M']['TRIV'][0,0].astype('long')-1

# Number of eigenvalues we would compute
k = 200

# Robust LBO
L, A = robust_laplacian.mesh_laplacian(vertices, triv)

try:
    evals_head, evecs = eigsh(L, k, A, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
except:
    evals_head, evecs = eigsh(L- 1e-8* scipy.sparse.identity(vertices.shape[0]), 200,
                         A, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)

# Visualizing the eingenvectors
meshes = []
for j in np.arange(5):
    rgb = values_to_rgb(evecs[:,j], vmin=np.min(evecs[:,1]), vmax=np.max(evecs[:,j]))
    mesh = TriMesh(o3d_float(vertices), o3d_integer(triv))
    mesh.vertex_colors = o3d_float(rgb)
    mesh.compute_vertex_normals()
    meshes.append(mesh)

visualizer(meshes)

# Visualizing the spectra
plt.plot(evals_head, 'r', linewidth=4)    #Dense method
plt.show()


#### Let's do a low pass filtering of the coordinates
k = 150
evecs_trim = evecs[:,0:k]

# Remember: the inner product is defined with the Areas!
v = evecs[:,0:k] @ evecs_trim.T @ (A * vertices)

mesh = TriMesh(o3d_float(vertices), o3d_integer(triv))
mesh.compute_vertex_normals()
mesh_low = TriMesh(o3d_float(v), o3d_integer(triv)).translate((1.0,0,0))
mesh_low.compute_vertex_normals()

visualizer([mesh,mesh_low])

####### Let's do that on a human shape

mesh = o3d_read('./dataset/spectral/tr_reg_090.off')

# Loading an humanoid shape
vertices = np.asarray(mesh.vertices)
triv = np.asarray(mesh.triangles)

k = 150

# LBO
L, A = robust_laplacian.mesh_laplacian(vertices, triv)

try:
    evals, evecs = eigsh(L, k, A, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
except:
    evals, evecs = eigsh(L- 1e-8* scipy.sparse.identity(vertices.shape[0]), 200,
                         A, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    
# Low
evecs_trim = evecs[:,0:k]
v = evecs[:,0:k] @ evecs_trim.T @ (A * vertices)

mesh = TriMesh(o3d_float(vertices), o3d_integer(triv))
mesh.compute_vertex_normals()
mesh_low = TriMesh(o3d_float(v), o3d_integer(triv)).translate((1.0,0,0))
mesh_low.compute_vertex_normals()

visualizer([mesh, mesh_low])

####### Spectral Clustering (Segmentation)
# As a first application, we can cluster the surface using the spectral embedding of the shapes.

# The steps are:

# - Choosing the number of clusters
# - Running KMeans on the eigenvectors
# - Visualizing the clusters
# Number of cluster

n_c = 6

mesh1 = o3d_read('./dataset/spectral/tr_reg_090.off')
vertices1 = np.asarray(mesh1.vertices)
triv1 = np.asarray(mesh1.triangles)
L1, A1 = robust_laplacian.mesh_laplacian(vertices1, triv1)

try:
    evals1, evecs1 = eigsh(L1, k, A1, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
except:
    evals1, evecs1 = eigsh(L1- 1e-8* scipy.sparse.identity(vertices.shape[0]), 200,
                         A1, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    
mesh2 = o3d_read('./dataset/spectral/tr_reg_043.off')
vertices2 = np.asarray(mesh2.vertices)
triv2 = np.asarray(mesh2.triangles)
L2, A2 = robust_laplacian.mesh_laplacian(vertices2, triv2)

try:
    evals2, evecs2 = eigsh(L2, k, A2, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
except:
    evals2, evecs2 = eigsh(L2- 1e-8* scipy.sparse.identity(vertices.shape[0]), 200,
                         A2, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    
# Number of eigenvalues we would compute
k = 50

#KMeans
kmeans1 = KMeans(n_clusters=n_c, random_state=1).fit(evecs1[:,1:n_c])
kmeans2 = KMeans(n_clusters=n_c, random_state=1).fit(evecs2[:,1:n_c])

def discrete_to_rgb_jet(labels):
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Colormap 'jet' con n_labels suddivisioni
    cmap = get_cmap('jet', n_labels)

    # Mappatura etichetta -> colore RGB
    label_to_color = {
        label: cmap(i)[:3]  # solo RGB, ignoriamo alpha
        for i, label in enumerate(unique_labels)
    }

    # Applica la mappatura a tutte le etichette
    rgb_colors = np.array([label_to_color[label] for label in labels])

    return rgb_colors

rgb1 = discrete_to_rgb_jet(kmeans1.labels_)
rgb2 = discrete_to_rgb_jet(kmeans2.labels_)

mesh1.vertex_colors = o3d_float(rgb1)
mesh1.vertices = o3d_float(vertices1+[1,0,0]) # Shift mesh1 for better visualization
mesh2.vertex_colors = o3d_float(rgb2)

visualizer([mesh1, mesh2])

###### Shapes Classification
# Eigenvalues can be also used as shape descriptors for classification tasks.
# Here we visualize the spectra of different shapes.
# We load three different shapes: two humans in different poses and an head.
# You can observe that similar shapes have similar spectra.

plt.plot(evals1, 'b', linewidth=4)
plt.plot(evals2,'r',linewidth=4)
plt.plot(evals_head[0:len(evals1)],'g', linewidth=4,marker='o', markersize=4)
plt.show()

##### ASSIGNEMENTS
# 1) Generate a bunch of SMPL shapes with different identities (betas) and poses. How the eigenvalues evolve?
# 2) 