import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Загрузка данных
iris_dataset = load_iris()
features = iris_dataset.data

# Список значений инерции
inertia_values = []

for cluster_count in range(1, 11):
    kmeans_model = KMeans(n_clusters=cluster_count, random_state=42)
    kmeans_model.fit(features)
    inertia_values.append(kmeans_model.inertia_)

# Вычисление первой производной инерции
inertia_differences = np.diff(inertia_values)

# Построение графика первой производной
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), -inertia_differences, marker='o', color='orange', linestyle='--')
plt.xlabel('Количество кластеров', fontsize=12)
plt.ylabel('Изменение инерции (производная)', fontsize=12)
plt.title('Определение оптимального числа кластеров через анализ производной', fontsize=14)
plt.grid(alpha=0.5)
plt.show()
