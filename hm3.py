import requests
import numpy as np
import itertools
from random import randint, shuffle, random, sample

# Ваш API ключ
YANDEX_API_KEY = "c51f053b-4792-49bd-a342-8ad8a698c44a"

# Города для анализа
cities = ["Москва", "Санкт-Петербург", "Казань", "Нижний Новгород", "Екатеринбург", "Самара"]

def get_city_coordinates(cities):
    """Получить координаты городов через API."""
    coordinates = []
    for city in cities:
        url = f"https://geocode-maps.yandex.ru/1.x/"
        params = {
            "apikey": YANDEX_API_KEY,
            "geocode": city,
            "format": "json"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            geo_object = response.json()['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']
            coords = geo_object['Point']['pos'].split()
            coordinates.append((float(coords[1]), float(coords[0])))
        else:
            print(f"Ошибка при получении данных для {city}: {response.status_code}")
    return coordinates

def get_distances_matrix(coords):
    """Получить матрицу расстояний между точками."""
    n = len(coords)
    distances = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        url = "https://api.routing.yandex.net/v2/route"
        params = {
            "apikey": YANDEX_API_KEY,
            "waypoints": f"{coords[i][0]},{coords[i][1]}|{coords[j][0]},{coords[j][1]}",
            "mode": "driving"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            distance = data["routes"][0]["legs"][0]["distance"]["value"]
            distances[i][j] = distances[j][i] = distance
        else:
            print(f"Ошибка при расчете расстояния: {response.status_code}")
    return distances

# Генетический алгоритм
class GeneticAlgorithmTSP:
    def __init__(self, distance_matrix, population_size=100, generations=500, mutation_rate=0.05):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_cities = len(distance_matrix)
        self.population = self.initialize_population()

    def initialize_population(self):
        """Инициализация популяции."""
        population = []
        for _ in range(self.population_size):
            path = list(range(self.num_cities))
            shuffle(path)
            population.append(path)
        return population

    def fitness(self, path):
        """Оценка пути."""
        return sum(self.distance_matrix[path[i - 1]][path[i]] for i in range(len(path)))

    def select(self):
        """Рулетка для выбора родителя."""
        fitness_scores = [1 / self.fitness(ind) for ind in self.population]
        total = sum(fitness_scores)
        probs = [score / total for score in fitness_scores]
        return sample(self.population, weights=probs, k=2)

    def crossover(self, parent1, parent2):
        """Одноточечный кроссовер."""
        point = randint(1, self.num_cities - 2)
        child = parent1[:point] + [gene for gene in parent2 if gene not in parent1[:point]]
        return child

    def mutate(self, child):
        """Мутация (обмен двух генов)."""
        if random() < self.mutation_rate:
            a, b = randint(0, self.num_cities - 1), randint(0, self.num_cities - 1)
            child[a], child[b] = child[b], child[a]

    def evolve(self):
        """Эволюция одного поколения."""
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = self.select()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def solve(self):
        """Основной цикл алгоритма."""
        best_path = None
        best_distance = float("inf")
        for generation in range(self.generations):
            self.evolve()
            current_best = min(self.population, key=self.fitness)
            current_distance = self.fitness(current_best)
            if current_distance < best_distance:
                best_path, best_distance = current_best, current_distance
            print(f"Поколение {generation}, Лучшая длина пути: {best_distance}")
        return best_path, best_distance

# Основной код
coords = get_city_coordinates(cities)
distances = get_distances_matrix(coords)
tsp_solver = GeneticAlgorithmTSP(distances)
best_path, best_distance = tsp_solver.solve()

print("Лучший маршрут:", [cities[i] for i in best_path])
print("Длина маршрута:", best_distance)
