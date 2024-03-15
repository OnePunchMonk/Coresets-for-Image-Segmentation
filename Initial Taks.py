#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""*****************************************************************************************
IIIT Delhi License
Copyright (c) 2023 Supratim Shit
*****************************************************************************************"""

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


# # Real data input
# dataset = fetch_kddcup99()								# Fetch kddcup99 
# data = dataset.data										# Load data
# data = np.delete(data,[0,1,2,3],1) 						# Preprocess
# data = data.astype(float)								# Preprocess
# data = StandardScaler().fit_transform(data)				# Preprocess



# In[3]:


import cv2
imag = cv2.imread('fruits.jpg')
imag.shape
m = imag.shape[0]
n = imag.shape[1]
weight = np.ones((m,n))*1/(m*n)
data=imag


# In[4]:


n = np.size(data,0)										# Number of points in the dataset
d = np.size(data,1)										# Number of dimension/features in the dataset.
k = 17													# Number of clusters (say k = 17)
Sample_size = 100										# Desired coreset size (say m = 100)




# In[ ]:





# In[5]:


def d(x,B):
    min_dist = 1e+20
    B_close = -1
    for i in range(len(B)):
        dist = dist = np.linalg.norm(x-B[i])
        if(dist<min_dist):
            min_dist = dist
            B_close = i
    return dist,B_close


# In[6]:


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



# In[7]:


centers = D2(data,k)									# Call D2-Sampling (D2())


# In[ ]:





# In[8]:


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



# In[9]:


coreset, weight = Sampling(data,k,centers,Sample_size)	# Call coreset construction algorithm (Sampling())



# In[10]:


import numpy as np

# Assuming your tuple is stored in a variable named 'my_tuple'
my_array = np.array(data)

# Print the dimensions of the NumPy array
print("Dimensions of the array:", my_array.shape)
# print(my_array[0].shape)
# print(my_array[1].shape)
# print(my_array[2].shape)

print(my_array[0])


# In[11]:


# import cv2
# grayscale_image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
# edges_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=k)
# edges_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=k)
# edges = np.sqrt(edges_x**2 + edges_y**2)

#     # Reshape and concatenate features
# reshaped_grayscale = grayscale_image.reshape(-1, 1)
# reshaped_edges = edges.reshape(-1, 1)
# features = np.concatenate((reshaped_grayscale, reshaped_edges), axis=1)


# In[12]:


#     # Reshape data to 2D if necessary (assuming pixel-wise features)
# if len(data.shape) == 3:
#     data = data.reshape(-1, 3)  # Reshape to (height * width, channels)
if len(data.shape) == 3:
        data = data.reshape(-1, 3)
    # Apply PCA for dimensionality reduction (adjust n_components as needed)
# pca = PCA(n_components=10)  # Increased to preserve more information

#     # **Optional: Feature Scaling (consider if features have different scales)**
#     # from sklearn.preprocessing import StandardScaler
#     # scaler = StandardScaler()
#     # scaled_data = scaler.fit_transform(data)
#     # reduced_features = pca.fit_transform(scaled_data)

# reduced_features = pca.fit_transform(data)


# In[13]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(data)


# In[14]:


# import numpy as np

# # Assuming your data is stored in a list called 'data'
# for i in range(len(data)):
#   # Access the element at index 'i'
#   element = data[i]

#   # Reshape the element to a 3D NumPy array for convenient manipulation
#   reshaped_data = np.reshape(element, (900,3))

#   # Calculate the average along the last dimension (axis=2) to get the average for each pixel channel
#   averaged_data = np.mean(reshaped_data, axis=0)

#   # Update the element in the list with the averaged data
#   data[i] = averaged_data


# In[ ]:





# In[15]:


#---Running KMean Clustering---#
fkmeans = KMeans(n_clusters=k,init='k-means++')
fkmeans.fit_predict(data)



# In[16]:


reduced_features.shape


# In[17]:


data.shape


# In[18]:


# Coreset_centers = Coreset_centers.reshape(1, -1)
# Coreset_centers.shape


# In[ ]:





# In[19]:


#----Practical Coresets performance----# 	
Coreset_centers, _ = wkpp.kmeans_plusplus_w(coreset, k, w=weight, n_local_trials=100)						# Run weighted kMeans++ on coreset points
wt_kmeansclus = KMeans(n_clusters=k, init=Coreset_centers, max_iter=10).fit(coreset,sample_weight = weight)	# Run weighted KMeans on the coreset, using the inital centers from the above line.
Coreset_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
coreset_cost = np.sum(np.min(cdist(data,Coreset_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_practicalCoreset = abs(coreset_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from practical coreset, here fkmeans.inertia_ is the optimal cost on the complete data.



# In[20]:


#-----Uniform Sampling based Coreset-----#
tmp = np.random.choice(range(n),size=Sample_size,replace=False)		
sample = data[tmp][:]																						# Uniform sampling
sweight = n*np.ones(Sample_size)/Sample_size 																# Maintain appropriate weight
sweight = sweight/np.sum(sweight)																			# Normalize weight to define a distribution



# In[21]:


#-----Uniform Samling based Coreset performance-----# 	
wt_kmeansclus = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(sample,sample_weight = sweight)		# Run KMeans on the random coreset
Uniform_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
uniform_cost = np.sum(np.min(cdist(data,Uniform_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_unifromCoreset = abs(uniform_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from random coreset, here fkmeans.inertia_ is the optimal cost on the full data.
	



# In[22]:


print("Relative error from Practical Coreset is",reative_error_practicalCoreset)
print("Relative error from Uniformly random Coreset is",reative_error_unifromCoreset)


# In[23]:


# imag = np.copy(imag)
centers = fkmeans.cluster_centers_
centers = np.array(centers)


# In[24]:


# print(d_f)


# In[25]:


# for i in range(m):
#     for j in range(n):
#         for k in range
#         d_f = d(imag[i][j],centers)
#         imag[i][j] = centers[int(d_f[1])]
        



# In[26]:


# from PIL import Image
# data12 =  Image.fromarray((imag * 255).astype(np.uint8))
# data12.save("segmentedF.png")


# In[ ]:





# In[27]:


Coreset_centers.shape


# In[28]:


coreset.shape


# In[29]:


imag.shape


# In[30]:


centers.shape


# In[31]:


# for i in range(coreset.shape[0]):
#     for j in range(coreset.shape[1]):
#         d_f = d(coreset[i][j],centers)
#         nearest_cluster_index = d_f[1] 
#         coreset[i][j] = centers[nearest_cluster_index]
        


# In[ ]:





# In[ ]:




