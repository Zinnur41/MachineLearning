import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

# Загрузка набора данных
iris_dataset = load_iris()
data_points = iris_dataset.data  # Массив признаков


def custom_kmeans(data, num_clusters, max_iterations=100):
    """
    Реализация алгоритма k-means.

    :param data: Массив данных.
    :param num_clusters: Количество кластеров.
    :param max_iterations: Максимальное количество итераций.
    :return: Центроиды и метки кластеров.
    """
    # Инициализация центроид случайным образом
    initial_indices = np.random.choice(len(data), num_clusters, replace=False)
    cluster_centers = data[initial_indices]

    for iteration in range(max_iterations):
        # Вычисляем расстояния от каждой точки до центроидов
        distances = np.linalg.norm(data[:, np.newaxis] - cluster_centers, axis=2)

        # Присваиваем метки на основе минимального расстояния
        cluster_labels = np.argmin(distances, axis=1)

        # Сохраняем предыдущие центроиды для проверки сходимости
        previous_centers = cluster_centers.copy()

        # Пересчитываем центроиды как средние значения кластеров
        for cluster_idx in range(num_clusters):
            points_in_cluster = data[cluster_labels == cluster_idx]
            if len(points_in_cluster) > 0:  # Проверяем, что кластер не пустой
                cluster_centers[cluster_idx] = np.mean(points_in_cluster, axis=0)

        # Если центроиды не изменились, алгоритм сходит
        if np.allclose(previous_centers, cluster_centers):
            break

    return cluster_centers, cluster_labels


# Оптимальное количество кластеров
optimal_clusters = 3

# Запуск k-means
final_centers, final_labels = custom_kmeans(data_points, optimal_clusters)

# Визуализация начальных данных
plt.figure(figsize=(8, 5))
plt.scatter(data_points[:, 0], data_points[:, 1], c='gray', label='Точки данных')
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='red', marker='x', s=100, label='Центроиды (конечные)')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Итоговая кластеризация')
plt.legend()
plt.show()

# Визуализация шагов алгоритма
for step in range(10):  # Максимум 10 шагов
    step_centers, step_labels = custom_kmeans(data_points, optimal_clusters)

    plt.figure(figsize=(8, 5))
    colors = ['blue', 'green', 'orange']  # Цвета для кластеров
    for cluster_idx in range(optimal_clusters):
        cluster_points = data_points[step_labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[cluster_idx],
                    label=f'Кластер {cluster_idx + 1}')

    # Отображение центроидов
    plt.scatter(step_centers[:, 0], step_centers[:, 1], c='red', marker='x', s=100, label='Центроиды')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(f'Шаг {step + 1}')
    plt.legend()
    plt.show()
