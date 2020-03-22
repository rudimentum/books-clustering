from sklearn.cluster import KMeans
import numpy as np
#2d model
import matplotlib.pyplot as plt
#3d model
from mpl_toolkits.mplot3d import Axes3D

N_CLUSTERS = 5

# words count, reviews count, year
data = np.genfromtxt(fname='data.csv',delimiter=';',dtype=float, usecols=(1,2,3))
#titles of books
labels = np.genfromtxt(fname='data.csv',delimiter=';',dtype=str, usecols=0)

kmeans = KMeans(init='k-means++', n_clusters=N_CLUSTERS, n_init=10)
kmeans.fit(data)
pred_classes = kmeans.predict(data)

centroids = kmeans.cluster_centers_
print(centroids)

for cluster in range(N_CLUSTERS):
    print('cluster: ', cluster)
    print(labels[np.where(pred_classes == cluster)])
  
#2d model
plt.figure(figsize=(6,4))
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.scatter(data[:,0], data[:,1], c= kmeans.labels_.astype(float))
plt.title('K-Means Clustering 2D')
plt.show()

#3d model
fig = plt.figure() 
ax = Axes3D(fig)

ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=50)
ax.scatter(data[:,0], data[:,1], data[:,2], c= kmeans.labels_.astype(float))
ax.set_title('K-Means Clustering 3D')
plt.show()