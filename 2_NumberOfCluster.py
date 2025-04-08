import matplotlib.pyplot as plt
from sklearn.cluster import KMeans #algorithm clusters data by trying to separate samples in n groups of equal variance



x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))

#Elbow method
wcss_list = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    wcss_list.append(kmeans.inertia_)


plt.plot(range(1,11), wcss_list, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show() 