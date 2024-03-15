#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist 
import sys
sys.path.append(r'C:\Users\aggar\New folder')

import wkpp as wkpp 
import numpy as np

import random
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import cv2
imag = cv2.imread('fruits.jpg')
imag.shape
m = imag.shape[0]
n = imag.shape[1]
weight = np.ones((m,n))*1/(m*n)
k=17


# In[3]:


def d(x,B):
    min_dist = 1e+20
    B_close = -1
    for i in range(len(B)):
        dist = dist = np.linalg.norm(x-B[i])
        if(dist<min_dist):
            min_dist = dist
            B_close = i
    return dist,B_close


# In[4]:


def D2(data, k):
    flattened_data = data.reshape(-1, data.shape[-1])
    centroids = []
    centroids.append(flattened_data[np.random.randint(flattened_data.shape[0])])
    for i in range(1, k):
        distances = cdist(flattened_data, np.array(centroids))
        min_distances = np.min(distances, axis=1)
        probabilities = min_distances ** 2 / np.sum(min_distances ** 2)
        new_centroid_index = np.random.choice(flattened_data.shape[0], p=probabilities)
        centroids.append(flattened_data[new_centroid_index])

    return np.array(centroids)
centroids = D2(imag, k)


# In[5]:


def Sampling(data, k, centers, Sample_size):
    m = data.shape[0]
    n = data.shape[1]
    alpha = 16 * (np.log(k) + 2)
    cluster = np.zeros((m, n))
    d_sum = 0
    d_f = np.zeros((m, n, 2))
    B_i = np.zeros((len(centers), 2))

    for i in range(m):
        for j in range(n):
            d_f[i][j] = d(data[i][j], centers)
            cluster[i][j] = d_f[i][j][1]
            d_sum += d_f[i][j][0]
            B_i[int(d_f[i][j][1])][1] += 1
            B_i[int(d_f[i][j][1])][0] += d_f[i][j][0]
            
    c_phi = d_sum / (m * n)
    S = np.zeros((m, n))
    pr = np.zeros((m, n))
    sum_S = 0

    for i in range(m):
        for j in range(n):
            S[i][j] = alpha * d_f[i][j][0] / c_phi + 2 * alpha * B_i[int(cluster[i][j])][0] / (
                    B_i[int(cluster[i][j])][1] * c_phi) + 4 * m * n / B_i[int(cluster[i][j])][1]
            sum_S += S[i][j]
    
    for i in range(m):
        for j in range(n):
            pr[i][j] = S[i][j] / sum_S

    index_set = np.ones((m * n, 2))
    for i in range(m * n):
        index_set[i, 0] = i
        index_set[i, 1] = pr[i // n][i % n]
        
    C_index = np.random.choice(index_set[:, 0], p=index_set[:, 1], size=int(Sample_size) + 1)
    coreset = np.zeros((int(Sample_size) + 1, 3))
    weight = np.zeros((int(Sample_size) + 1))

    for i in range(int(Sample_size + 1)):
        coreset[i] = np.array(data[int(C_index[i] // n)][int(C_index[i] % n)])
        weight[i] = 1 / (Sample_size * pr[int(C_index[i] // n)][int(C_index[i] % n)] + 1e-8)
        
    return coreset, weight


# In[6]:


coreset, weight = Sampling(imag,k,centroids,100)


# In[7]:


from PIL import Image
imag = np.copy(imag)
kmeans = KMeans(n_clusters=k,  init='k-means++', max_iter=10).fit(coreset,sample_weight=weight)
centers = kmeans.cluster_centers_
centers = np.array(centers)
for i in range(m):
    for j in range(n):
        d_f = d(imag[i][j],centers)
        imag[i][j] = centers[int(d_f[1])]
        

data12 =  Image.fromarray((imag * 255).astype(np.uint8))
data12.save("Segmented_Orignal.png")


# In[8]:


imag.shape


# In[9]:


centers.shape


# In[10]:


coreset.shape


# In[11]:


for i in range (coreset.shape[0]):
    d_f=d(coreset[i],centers)
    coreset[i]=centers[int(d_f[1])]
    
data12 =  Image.fromarray((imag * 255).astype(np.uint8))
data12.save("Segmented_Coreset.png")    
    


# In[ ]:




