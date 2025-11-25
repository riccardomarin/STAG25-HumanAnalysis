# Virtual Humans Under a Shape Analysis Spotlight
Repository for the STAG25 Lecture: Virtual Humans Under a Shape Analysis Spotlight, November 2025

### 💻 Code & Environment Setup

Setup: To run the scripts, you need to set up a local environment (a GPU is **not required**).

```
conda create -n stag25 python=3.9.23
conda activate stag25

conda install -c conda-forge libstdcxx-ng
pip install smplx[all] open3d plyfile moderngl-window==2.4.6 pyglet aitviewer scikit-learn pandas robust-laplacian

pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17
pip install git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47
```
### 🔑 Required Accounts & Data
You need to create accounts on the following platforms to download the necessary models and data:
- SMPLfy: https://smplify.is.tue.mpg.de/
- AMASS: https://amass.is.tue.mpg.de/
 
**Data Fetching:**
After creating your accounts, run the scripts in the **`scripts`** folder to fetch the data needed for the course scripts. Both Linux Bash and Windows PowerShell versions are available.
In case of trouble, just perform the instructions manually.

----

## 📚 References & Further Reading

This tutorial takes inspiration from a number of sources that are useful for diving deeper into the topics:

* **SMPL Made Simple Tutorial:** [smpl-made-simple.is.tue.mpg.de](https://smpl-made-simple.is.tue.mpg.de/)
* **"Virtual Humans" Lecture (University of Tuebingen):** [YouTube Playlist](https://www.youtube.com/watch?v=DFHuV7nOgsI&list=PL05umP7R6ij13it8Rptqo7lycHozvzCJn)
* **Meshcapade Wiki:** [meshcapade.wiki](https://meshcapade.wiki/)
* **FAUST Dataset:** [faust-leaderboard.is.tuebingen.mpg.de](https://faust-leaderboard.is.tuebingen.mpg.de/)
