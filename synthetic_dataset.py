import numpy as np
import matplotlib.pyplot as plt

class SyntheticClassificationDataset2d:
    '''
    Class for generating synthetic 2d data
    :n_samples: - number of samples separate for clusters, e.g. [10, 20, 25]
    :cluster_center: - cluster centers of shape n_clusters x 2, e.g. [[1, 1], [5, 5], [9, 8]]
    :cluster_std: - cluster std per each dim of shape n_clusters x 2, e.g.  [[1.0, 1.0], [5.0, 5.5], [9.0, 1.0]]
    :rotation: - rotation of the cluster in degrees anti-clockwise, e.g. [45, 60, 90]
    :random_state: - seed, e.g. 42
    '''
    def __init__(self, n_samples, cluster_center, cluster_std, rotation, random_state=None):
        if type(n_samples)==int:
            self.n_samples = np.array([n_samples])
            self.cluster_center = cluster_center
            self.cluster_std = cluster_std
            self.rotation = np.array([rotation])
            self.random_state = random_state
        else:
            self.n_samples = n_samples
            self.cluster_center = cluster_center
            self.cluster_std = cluster_std
            self.rotation = rotation
            self.random_state = random_state
        assert len(self.n_samples) == len(self.cluster_center) == len(self.cluster_std) == len(self.rotation), f"Different number of clusters in parameters given."
        self.X, self.y = self._generate_dataset()
        self.clustering_labels = None
   
    def _generate_dataset(self):
        if self.random_state is not None: np.random.seed(self.random_state)
        points = np.empty((0,2), float)
        labels = np.empty(0, int)
        for index, n_cluster in np.ndenumerate(self.n_samples):
            points_cluster = np.empty((0,2), float)
            labels_cluster = np.empty(0, int)
            for _ in range(n_cluster):
                x1 = np.random.normal(loc=0.0, scale=self.cluster_std[index[0]][0])
                x2 = np.random.normal(loc=0.0, scale=self.cluster_std[index[0]][1])
                points_cluster = np.append(points_cluster, np.array([[x1, x2]]), axis=0)
                labels_cluster = np.append(labels_cluster, np.array([index[0]]), axis=0)
            #Rotate entire cluster
            theta = np.radians(self.rotation[index[0]])
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            points_cluster = np.matmul(R, points_cluster.T).T
            #Shift entire cluster to a given center
            points_cluster[:, 0] += self.cluster_center[index[0]][0]
            points_cluster[:, 1] += self.cluster_center[index[0]][1]
            points = np.append(points, points_cluster, axis=0)
            labels = np.append(labels, labels_cluster, axis=0)
        return(points, labels)

    def plot(self, ground_truth=True):
        '''Draw the dataset with matplotlib'''
        plt.subplot()
        if ground_truth:
            plt.scatter(self.X[:,0],
                        self.X[:,1],
                        c=self.y,
                        vmin=min(self.y),
                        vmax=max(self.y))
        else:
            plt.scatter(self.X[:,0],
                        self.X[:,1],
                        c=self.clustering_labels,
                        vmin=min(self.clustering_labels),
                        vmax=max(self.clustering_labels))
        plt.axis('scaled')
        plt.show()

    def read(self):
        '''Initiate object with external data'''
        pass

    def write(self):
        '''Save generated data locally'''
        pass

    def save_plot(self):
        '''Save image of the plotted data locally'''
        pass
