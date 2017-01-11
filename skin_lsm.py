# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:04:40 2015

@author: 이수민
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
feature_dict = {i:label for i,label in zip(
            range(3),
              ('R',
              'G',
              'B',
               ))}


import pandas as pd

df = pd.io.parsers.read_csv(
    filepath_or_buffer='http://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt',
    header=None,
    sep='	',
    )
    
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all" , inplace=True) 

df.tail()

X = df[[0,1,2]].values
y = df['class label'].values

import numpy as np

np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1,3):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))

S_W = np.zeros((3,3))
for cl,mv in zip(range(1,3), mean_vectors):
    class_sc_mat = np.zeros((3,3))                  
    for row in X[y == cl]:
        row, mv = row.reshape(3,1), mv.reshape(3,1)
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             
print('within-class Scatter Matrix:\n', S_W)

overall_mean = np.mean(X, axis=0)

S_B = np.zeros((3,3))
for i,mean_vec in enumerate(mean_vectors):
    n = X[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(3,1)
    overall_mean = overall_mean.reshape(3,1) 
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between-class Scatter Matrix:\n', S_B)
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigv = eig_vecs[:,i].reshape(3,1)
    np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                         eig_vals[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)
print('ok')


eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]


eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)



print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

W = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
print('Matrix W:\n', W.real)

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

lable_dict = {1:'nonskin',2:'skin'}

X_lda = X.dot(W)
assert X_lda.shape == (245057, 2), ""
from matplotlib import pyplot as plt
def plot_step_lda():
    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,3),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_lda[:,0][y == label],
                y=X_lda[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=lable_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

   
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()

