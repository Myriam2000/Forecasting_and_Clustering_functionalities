import numpy as np
import pandas as pd

import sklearn

from kneed import KneeLocator

from numpy import unique
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score, adjusted_rand_score
from tabulate import tabulate


def param_Kmeans(Cluster_scaled): # setting the number of clusters for Kmeans
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 100, "random_state": 42}

    # SSD = Sum of Square distance of samples to thier closest center
    ssd = [] 
    for num_clusters in list(range(1,10)):
        model_clus = KMeans(n_clusters = num_clusters, **kmeans_kwargs)
        model_clus.fit(Cluster_scaled)
        ssd.append(model_clus.inertia_)

    # knee detection
    kl = KneeLocator(range(1, 10), ssd, curve="convex", direction="decreasing")
    num_cluster = kl.elbow
    return num_cluster


def kmeans_clustering(Cluster, Cluster_scaled):
    # parameter setting
    num_cluster = param_Kmeans(Cluster_scaled)

    # k-means clustering
    model_k = KMeans(init="k-means++", n_clusters= num_cluster, max_iter=50)
    model_k.fit(Cluster_scaled)

    # assign clusters in a new column of the dataset 
    output = Cluster.copy()
    output['Category_Kmeans'] = model_k.predict(Cluster_scaled)

    return output


def param_DBSCAN(Cluster): # setting epsilon
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(Cluster.values)
    distances, indices = nbrs.kneighbors(Cluster.values)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    epsilon = int(distances[int(0.90*len(distances))]) # epsilon is chosen in a way 90% of the pairs' distances are bellow epsilon
    return epsilon


def DBSCAN_clustering(Cluster, Cluster_scaled):
    # parameter setting
    epsilon = param_DBSCAN(Cluster)

    # define the model
    model = DBSCAN(eps=epsilon, min_samples=5)
        
    # assign a cluster to each example
    output = Cluster.copy()
    output['Category_DBSCAN'] = model.fit_predict(Cluster_scaled)

    return output
    

def Agglomerative_clustering(Cluster, Cluster_scaled):
    # parameter setting
    num_cluster = param_Kmeans(Cluster_scaled)

    # define the model
    agglomerative = AgglomerativeClustering(n_clusters=num_cluster)

    # assign clusters in a new column of the dataset 
    output = Cluster.copy()
    output['Category_agglomerative'] = agglomerative.fit_predict(Cluster_scaled)

    return output


def clus_output(output, method, points, input_name): # save new dataset with new clusters column
    output.to_csv('../output/{}_clus_output_{}_{}.csv'.format(input_name, method.lower(), points))
    print("the csv file was successfully saved.")


def evaluate_clustering(data, labels, method): # metric used is silhouette_score
    if method=='dbscan':
        if len(unique(labels))!=1:
            sc = silhouette_score(data, labels, metric='euclidean')
            print ("Silhouette Score :", sc)
        else :
            print("error: only one cluster. The score can't be calculated")
    else:
        sc = silhouette_score(data, labels)
        print ("Silhouette Score :", sc)


def stats_insight(output, X): #table output describing the clusters
    # count
    number = output.groupby(output.columns[-1]).count()[X[0]]

    # mean
    mean = output.groupby(output.columns[-1]).mean()

    # table construction
    h = list(unique(output.iloc[ :, -1:]))
    h = np.concatenate((['Clusters'],[' '], h))
    table = []
    table.append(np.concatenate((['count'],[' '], number)))
    table.append(np.concatenate((['mean'],[X[0]], mean[X[0]])))
    for i in range (1,len(X)):
        table.append(np.concatenate(([' '],[X[i]], mean[X[i]])))
    return tabulate(table, headers=h, tablefmt="github")


def use_clustering_method(Cluster, points, X, func, input_name, method='KMEANS'): # main function for clustering
    # normalization
    if func!='count':
        Cluster_scaled =  pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(Cluster))
    else:
        Cluster_scaled = Cluster

    if method == 'KMEANS':
        output = kmeans_clustering(Cluster, Cluster_scaled)
    
    elif method == 'DBSCAN':
        output = DBSCAN_clustering(Cluster, Cluster_scaled)
    
    elif method == 'AGGLOMERATIVE':
        output = Agglomerative_clustering(Cluster, Cluster_scaled)

    # output insight
    file = open("../output/stats_insight_{}_{}.txt".format(method, points), "w+")
    file.write("Statistics on {}'s clustering with {} method :\n".format(points, method))
    file.write(stats_insight(output, X))
    file.close()

    clus_output(output, method, points, input_name)
    return output
