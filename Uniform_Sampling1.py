from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import wkpp as wkpp 
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
dataset = fetch_kddcup99()								# Fetch kddcup99 
data = dataset.data										# Load data
data = np.delete(data,[0,1,2,3],1) 						# Preprocess
data = data.astype(float)								# Preprocess
data = StandardScaler().fit_transform(data)				# Preprocess

n = np.size(data,0)										# Number of points in the dataset
d = np.size(data,1)										# Number of dimension/features in the dataset.
k = 17													# Number of clusters (say k = 17)
Sample_size = 100										# Desired coreset size (say m = 100)

def dist_to_B(x, B, return_closest_index=False):
    min_dist = np.inf
    closest_index = -1
    for i, b in enumerate(B):
        dist = np.linalg.norm(x - b)
        if dist < min_dist:
            min_dist = dist
            closest_index = i
    if return_closest_index:
        return min_dist, closest_index
    return min_dist

def D2(data, k):
    # Implementation of Algorithm1
    centers = []
    centers.append(data[np.random.choice(len(data))])
    
    for _ in range(k - 1):
        p = np.zeros(len(data))
        for i, x in enumerate(data):
            p[i] = dist_to_B(x, centers) ** 2
        p = p / sum(p)
        centers.append(data[np.random.choice(len(data), p=p)])
    
    return centers

def Sampling(data, k, centers, Sample_size):
    # Implementation of Algorithm2
    
    alpha = 16 * (np.log2(k) + 2)
    
    B_i_totals = [0] * len(centers)
    B_i = [np.empty_like(data) for _ in range(len(centers))]
    for x in data:
        _, closest_index = dist_to_B(x, centers, return_closest_index=True)
        B_i[closest_index][B_i_totals[closest_index]] = x
        B_i_totals[closest_index] += 1        
        
    c_phi = sum([dist_to_B(x, centers) ** 2 for x in data]) / len(data)

    p = np.zeros(len(data))
    
    sum_dist = {i: 0 for i in range(len(centers))}
    for i, x in enumerate(data):
        dist, closest_index = dist_to_B(x, centers, return_closest_index=True)
        sum_dist[closest_index] += dist ** 2
    
    for i, x in enumerate(data):
        p[i] = 2 * alpha * dist_to_B(x, centers) ** 2 / c_phi
        
        _, closest_index = dist_to_B(x, centers, return_closest_index=True)
        p[i] += 4 * alpha * sum_dist[closest_index] / (B_i_totals[closest_index] * c_phi)

        p[i] += 4 * len(data) / B_i_totals[closest_index]
    p = p / sum(p)

    chosen_indices = np.random.choice(len(data), size=Sample_size, p=p)
    weights = [1 / (Sample_size * p[i]) for i in chosen_indices]
    coresets=[data[i] for i in chosen_indices]
    return coresets, weights

fkmeans = KMeans(n_clusters=k,init='k-means++')
fkmeans.fit_predict(data)

tmp = np.random.choice(range(n),size=Sample_size,replace=False)		
sample = data[tmp][:]																						# Uniform sampling
sweight = n*np.ones(Sample_size)/Sample_size 																# Maintain appropriate weight
sweight = sweight/np.sum(sweight)																			# Normalize weight to define a distribution

#-----Uniform Samling based Coreset performance-----# 	
wt_kmeansclus = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(sample,sample_weight = sweight)		# Run KMeans on the random coreset
Uniform_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
uniform_cost = np.sum(np.min(cdist(data,Uniform_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_unifromCoreset = abs(uniform_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from random coreset, here fkmeans.inertia_ is the optimal cost on the full data.
	

# print("Relative error from Practical Coreset is",reative_error_practicalCoreset)
print("Relative error from Uniformly random Coreset is",reative_error_unifromCoreset) 

##Relative error from Uniformly random Coreset is 1.5379973941340326