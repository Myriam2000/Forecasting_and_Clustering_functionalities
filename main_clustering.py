import sys 
import data_preprocessing as dp
import clustering as clus
import vizualisation as v


def main_clustering():
    # define the parameters 
    data_path = sys.argv[1]
    points = sys.argv[2] 
    X = sys.argv[3].split(' ')
    func = sys.argv[4]
    method = sys.argv[5].upper() # kmeans, Dbscan, agglomerative

    #find input file name
    i = data_path.rfind('/')
    input_name = data_path[i+1:]

    # data processing
    data = dp.upload_data_clus(data_path, points, X)
    Cluster = dp.prep_data_clus(data, points, X, func)
    # execute clustering
    output = clus.use_clustering_method(Cluster, points, X, func, input_name, method)

    # vizualize
    if len(X)==2:
        v.plot_clus_2D(X, points, output, method)




if __name__ == "__main__":
    main_clustering()