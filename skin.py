
# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Various dimensionality reductions running on the Iris dataset
import pylab as pl
import numpy as np

skin = np.loadtxt('Skin_NonSkin.txt')
index1=np.where(skin[:,3]==1)
index2=np.where(skin[:,3]==2)
skin1=skin[index1,:][0]
skin2=skin[index2,:][0]

skin1d=skin1[:,0:3]
ave1d=np.mean(skin1d,axis=0)
X=skin1d-ave1d
A,sv1d,ev1d=np.linalg.svd(X,full_matrices=False)

skind=skin[:,0:3]
aved=np.mean(skind,axis=0)
Xt=skind-aved
At,svd,evd=np.linalg.svd(X,full_matrices=False)


"""
target= skin[:,3]
skin=skin[:,0:3]
ave=np.mean(skin,axis=0)
X=skin-ave
A,sval,evec=np.linalg.svd(X,full_matrices=False)
skin_p2=np.dot(evec,skin.T)






#--------------------3d plotting------------------------
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =skin_p2[0]
y =skin_p2[1]
z =skin_p2[2]

ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('B')
ax.set_ylabel('G')
ax.set_zlabel('R')
ax.view_init(elev=10., azim=25)
plt.show()
ax.view_init(elev=10., azim=115)
plt.show()
#--------------------3d plotting--------------------
"""