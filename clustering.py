import matplotlib

matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\aeadod\\Desktop\\project\\exercise\\e11\\data\\Mall_Customers.csv',sep=',')
training_set=df[['Annual Income (k$)','Spending Score (1-100)']].values
#print(training_set)

# train your model here as kmeans_algo
kmeans_algo=KMeans(n_clusters = 5).fit(training_set)
# assign variable centroids
centroids=kmeans_algo.cluster_centers_
# assign variable labels
labels=kmeans_algo.labels_


step_size = 0.02
x_min, x_max = training_set[:, 0].min() - 1, training_set[:, 0].max() + 1
y_min, y_max = training_set[:, 1].min() - 1, training_set[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
all_points_in_min_max_range = np.c_[xx.ravel(), yy.ravel()]
prediction_result = kmeans_algo.predict(all_points_in_min_max_range)

plt.figure(1, figsize=(15, 7))
plt.clf()
prediction_result = prediction_result.reshape(xx.shape)
plt.imshow(prediction_result, extent=(x_min, x_max, y_min, y_max), cmap=plt.cm.Pastel2, origin='lower')

plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, c=labels, s=200)
plt.scatter(x=centroids[:, 0], y=centroids[:, 1], s=300, c='red', alpha=0.5)
plt.ylabel('Spending Score (1-100)'), plt.xlabel('Annual Income (k$)')
plt.show()
print(labels)