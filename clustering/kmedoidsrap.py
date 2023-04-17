# A medoid can be defined as a point in the cluster, 
# whose dissimilarities with all the other points in the cluster are minimum. 
# The dissimilarity of the medoid(Ci) and object(Pi) is calculated by using E = |Pi – Ci|  -->   Distancia de manhatan?


import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")


class KMedoidsClass:
    def __init__(self,data,k,iters):
        self.data= data
        self.k = k
        self.iters = iters
        self.medoids = np.array([data[i] for i in range(self.k)])
        self.colors = np.array(np.random.randint(0, 255, size =(self.k, 4)))/255
        self.colors[:,3]=1

    def manhattan(self,p1, p2):
        return np.abs((p1[0]-p2[0])) + np.abs((p1[1]-p2[1]))

    def get_costs(self, medoids, data):
        tmp_clusters = {i:[] for i in range(len(medoids))}
        cst = 0
        for d in data:
            dst = np.array([self.manhattan(d, md) for md in medoids])
            c = dst.argmin()
            tmp_clusters[c].append(d)
            cst+=dst.min()

        tmp_clusters = {k:np.array(v) for k,v in tmp_clusters.items()}
        return tmp_clusters, cst

    def fit(self):

        samples,_ = self.data.shape

        self.clusters, cost = self.get_costs(data=self.data, medoids=self.medoids)
        count = 0

        colors =  np.array(np.random.randint(0, 255, size =(self.k, 4)))/255
        colors[:,3]=1

        plt.title(f"Step : 0")
        [plt.scatter(self.clusters[t][:, 0], self.clusters[t][:, 1], marker="*", s=100,
                                        color = colors[t]) for t in range(self.k)]
        plt.scatter(self.medoids[:, 0], self.medoids[:, 1], s=200, color=colors)
        plt.show()

        while True:
            swap = False
            for i in range(samples):
                if not i in self.medoids:
                    for j in range(self.k):
                        tmp_meds = self.medoids.copy()
                        tmp_meds[j] = i
                        clusters_, cost_ = self.get_costs(data=self.data, medoids=tmp_meds)

                        if cost_<cost:
                            self.medoids = tmp_meds
                            cost = cost_
                            swap = True
                            self.clusters = clusters_
                            # print(f"Medoids Changed to: {self.medoids}.")
                            # plt.title(f"Step : {count+1}")  
                            # [plt.scatter(self.clusters[t][:, 0], self.clusters[t][:, 1], marker="*", s=100,
                            #             color = colors[t]) for t in range(self.k)]
                            # plt.scatter(self.medoids[:, 0], self.medoids[:, 1], s=200, color=colors)
                            # plt.show()
            count+=1

            if count>=self.iters:
                print("End of the iterations.")
                break
            if not swap:
                print("No changes.")
                break
        return self

#dt = np.random.randint(0,100, (100,2))
dt = np.asarray([[1, 2], [1, 4], [1, 0],
                [4, 2], [4, 4], [4, 0]])
kmedoid = KMedoidsClass(dt,2,5)
kmedoid.fit()
#print(f"Medoids Changed to: {self.medoids}.")
plt.title(f"Final step :")  
[plt.scatter(kmedoid.clusters[t][:, 0], kmedoid.clusters[t][:, 1], marker="*", s=100,
            color = kmedoid.colors[t]) for t in range(kmedoid.k)]
plt.scatter(kmedoid.medoids[:, 0], kmedoid.medoids[:, 1], s=200, color=kmedoid.colors)
plt.show()