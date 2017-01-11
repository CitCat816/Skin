
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The iris classification example

import numpy as np
skin1 = np.loadtxt('Skin_NonSkin_mod1.txt')
skin2 = np.loadtxt('Skin_NonSkin_mod2.txt')
order=list(range(np.shape(skin1)[0]))
np.random.shuffle(order)
skin1=skin1[order,:]
order =list(range(np.shape(skin2)[0]))
np.random.shuffle(order)
skin2=skin2[order,:]
skin1=np.vstack((skin1,skin2[0:50859]))

order =list(range(np.shape(skin1)[0]))
np.random.shuffle(order)
skin1=skin1[order,:]
skin2=skin2[50859:]

target=skin1[:,3:5]
skin1=skin1[:,0:3]
skin1=skin1
target2=skin2[:,3:5]
skin2=skin2[:,0:3]

train = skin1[::2]
traint = target[::2]
valid =np.vstack((skin1[1::4],skin2[::2]))
validt = np.vstack((target[1::4],target2[::2]))
test = np.vstack((skin1[3::4],skin2[1::2]))
testt = np.vstack((target[3::4],target2[1::2]))




print(sum(traint))
# Train the network'
import mlp
net = mlp.mlp(train,traint,10,outtype='softmax')
net.earlystopping(train,traint,valid,validt,0.3)
net.confmat(test,testt)