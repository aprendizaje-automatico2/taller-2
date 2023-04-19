# A medoid can be defined as a point in the cluster, 
# whose dissimilarities with all the other points in the cluster are minimum. 
# The dissimilarity of the medoid(Ci) and object(Pi) is calculated by using E = |Pi – Ci|  -->   Distancia de manhatan?

import numpy as np
import matplotlib.pyplot as plt

##  implementacion propia
import numpy as np

import numpy as np

class KMedoidsr:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    def fit(self, X):
        # Inicialización de los centroides como puntos aleatorios del conjunto de datos
        self.medoids = X[np.random.choice(len(X), self.n_clusters, replace=False)]
        
        # Asignación inicial de los puntos a los clusters
        self.labels = np.argmin(np.linalg.norm(X[:, np.newaxis, :] - self.medoids, axis=2), axis=1)
        
        # Bucle principal
        for _ in range(self.max_iter):
            # Cálculo de la distancia entre los puntos y los medoids de su cluster
            distances = np.zeros((len(X), self.n_clusters))
            for i, x in enumerate(X):
                for j, medoid in enumerate(self.medoids):
                    distances[i, j] = np.sum(np.abs(x - medoid))
            
            # Creación de una copia de las asignaciones actuales
            old_labels = np.copy(self.labels)
            
            # Actualización de los clusters
            for j in range(self.n_clusters):
                mask = self.labels == j
                if np.sum(mask) > 0:
                    cluster_distances = distances[mask, :]
                    medoid_index = np.argmin(np.sum(cluster_distances, axis=0))
                    self.medoids[j] = X[mask][medoid_index]
            
            # Asignación de los puntos a los clusters
            self.labels = np.argmin(distances, axis=1)
            
            # Comprobación de convergencia
            if np.all(old_labels == self.labels):
                break
        
        return self
    
    def predict(self, X):
        distances = np.zeros((len(X), self.n_clusters))
        for i, x in enumerate(X):
            for j, medoid in enumerate(self.medoids):
                distances[i, j] = np.sum(np.abs(x - medoid))
        return np.argmin(distances, axis=1)


# Generación de datos aleatorios
np.random.seed(42)
X = np.random.rand(100, 2)

# Creación de una instancia de la clase KMedoids
kmedoids = KMedoidsr(n_clusters=3, max_iter=100)

# Ejecución del algoritmo
kmedoids.fit(X)

# Obtención de las etiquetas de los clusters
labels = kmedoids.predict(X)

# Representación gráfica de los clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(kmedoids.medoids[:, 0], kmedoids.medoids[:, 1], marker='*', s=200, c='r')
plt.show()