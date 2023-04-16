import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class KMedoids:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
    
    def fit(self, X):
        # Inicializar los medoids aleatoriamente
        m = X.shape[0]
        medoids_idx = np.random.choice(m, self.k, replace=False)
        self.medoids = X[medoids_idx]

        # Iterar hasta que la asignación de clústeres converja
        for i in range(self.max_iterations):
            # Calcular la distancia entre cada punto y cada medoid
            distances = pairwise_distances(X, self.medoids)

            # Encontrar el medoid más cercano para cada punto
            self.labels = np.argmin(distances, axis=1)

            # Actualizar los medoids con los puntos más cercanos
            for j in range(self.k):
                mask = self.labels == j
                cluster_points = X[mask]
                cluster_distances = pairwise_distances(cluster_points)
                total_distance = np.sum(cluster_distances, axis=1)
                new_medoid_idx = np.argmin(total_distance)
                self.medoids[j] = cluster_points[new_medoid_idx]

        # Calcular la distancia total de la asignación final de clústeres
        distances = pairwise_distances(X, self.medoids)
        self.total_distance = np.sum(np.min(distances, axis=1))

        return self

    def predict(self, X):
        distances = pairwise_distances(X, self.medoids)
        labels = np.argmin(distances, axis=1)
        return labels


X = np.asarray([[1, 2], [1, 4], [1, 0],
                [4, 2], [4, 4], [4, 0]])
kmedoids = KMedoids(k=2, max_iterations=200)
kmedoids.fit(X)
labels = kmedoids.predict(X)
medoids = kmedoids.medoids
total_distance = kmedoids.total_distance
print(f'Etiquetas: {labels}')
print(f'\nMEDOIDS: {medoids}')
print(f'DISTANCIA TOTAL: {total_distance}')
