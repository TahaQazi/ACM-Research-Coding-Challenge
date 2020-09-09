# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:54:04 2020

@author: Pc
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
kmeans = KMeans(n_clusters=5, max_iter=10)
df = pd.read_csv('ClusterPlot.csv', delimiter=',', nrows=150)
y = df['V2'];
x = df['V1'];
plt.xlabel('V1'); plt.ylabel('V2')
plt.scatter(x,y)
X_std = StandardScaler().fit_transform(df)
kmeans.fit(X_std)
y_kmeans = kmeans.predict(X_std)
plt.scatter(x,y, c=y_kmeans,s = 75, cmap ='Dark2')