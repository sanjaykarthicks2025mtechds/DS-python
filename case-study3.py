

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


x,y_true=make_blobs(
    n_samples=500,
    centers=3,
    cluster_std=0.7,
    random_state=42,
)

n_outliers=20
outliers=np.random.uniform(low=-8,high=8,size=(n_outliers,2))
x_full=np.vstack([x,outliers])

iso=IsolationForest(
    contamination=n_outliers/x_full.shape[0],
    random_state=42,
)

iso_labels=iso.fit_predict(x_full)
lof=LocalOutlierFactor(
    n_neighbors=20,
    contamination=n_outliers/x_full.shape[0],
)
lof_labels=lof.fit_predict(x_full)

def plot_anomalies(x,labels,title):
    normal=labels==1
    anom=labels==-1
    plt.figure()
    plt.scatter(x[normal,0],x[normal,1],s=20,label="Normal")
    plt.scatter(x[anom,0],x[anom,1],s=40,label="Anomaly")
    plt.title(title)
    plt.legend()
    plt.show()

plot_anomalies(x_full,iso_labels,"Isolation Forest Anomaly Detection")
plot_anomalies(x_full,lof_labels,"Local Outlier Factor Anomaly Detection")

inertias=[]
k_range=range(1,11)
for k in k_range:
    km=KMeans(n_clusters=k,random_state=42,n_init="auto")
    km.fit(x)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(list(k_range),inertias,marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method for K-Means")
plt.show()

k_opt=3
kmeans=KMeans(n_clusters=k_opt,random_state=42,n_init="auto")
labels=kmeans.fit_predict(x)

plt.figure()
plt.scatter(x[:,0],x[:,1],c=labels,s=20,cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=150,marker="*",edgecolor="k")

plt.title(f"K-Means clustering (k={k_opt})")
plt.show()
