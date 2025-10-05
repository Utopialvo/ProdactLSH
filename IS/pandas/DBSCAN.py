import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_score
import time
from collections import defaultdict
from typing import Union, Optional, Tuple, List, Dict
import math

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

class TorchKDTree:
    """
    Реализация KD-Tree на чистом PyTorch для ускорения поиска соседей.
    
    Attributes:
        leaf_size (int): Максимальное количество точек в листе
        tree (dict): Структура дерева
        data (torch.Tensor): Исходные данные
    """
    
    def __init__(self, leaf_size: int = 30):
        """
        Инициализация KD-Tree.
        
        Args:
            leaf_size: Максимальное количество точек в листе
        """
        self.leaf_size = leaf_size
        self.tree = None
        self.data = None
        
    def fit(self, X: torch.Tensor) -> 'TorchKDTree':
        """
        Построение KD-Tree из данных.
        
        Args:
            X: Тензор данных [n_samples, n_features]
            
        Returns:
            self: Возвращает экземпляр класса
        """
        self.data = X
        self.tree = self._build_tree(torch.arange(len(X)))
        return self
    
    def _build_tree(self, indices: torch.Tensor, depth: int = 0) -> Dict:
        """
        Рекурсивное построение KD-Tree.
        
        Args:
            indices: Индексы точек для текущего узла
            depth: Глубина текущего узла
            
        Returns:
            Словарь представляющий узел дерева
        """
        n_points = len(indices)
        
        # Если точек меньше leaf_size, создаем лист
        if n_points <= self.leaf_size:
            return {
                'type': 'leaf',
                'indices': indices,
                'depth': depth
            }
        
        # Выбираем ось для разделения (чередуем по глубине)
        axis = depth % self.data.shape[1]
        
        # Находим медиану по выбранной оси
        points = self.data[indices]
        values = points[:, axis]
        median_idx = torch.argsort(values)[len(values) // 2]
        median_value = values[median_idx]
        
        # Разделяем точки
        left_mask = values <= median_value
        right_mask = values > median_value
        
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]
        
        # Проверяем, что разделение имеет смысл
        if len(left_indices) == 0 or len(right_indices) == 0:
            return {
                'type': 'leaf',
                'indices': indices,
                'depth': depth
            }
        
        return {
            'type': 'node',
            'axis': axis,
            'value': median_value,
            'left': self._build_tree(left_indices, depth + 1),
            'right': self._build_tree(right_indices, depth + 1),
            'depth': depth
        }
    
    def query_radius(self, point: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Поиск всех точек в заданном радиусе.
        
        Args:
            point: Точка запроса
            radius: Радиус поиска
            
        Returns:
            Индексы точек в радиусе
        """
        if self.tree is None:
            raise ValueError("Дерево не построено. Вызовите fit() сначала.")
            
        indices = []
        self._query_recursive(self.tree, point, radius, indices)
        
        if len(indices) == 0:
            return torch.tensor([], dtype=torch.long, device=point.device)
        
        return torch.cat(indices)
    
    def _query_recursive(self, node: Dict, point: torch.Tensor, radius: float, 
                        indices: List[torch.Tensor]):
        """
        Рекурсивный поиск в KD-Tree.
        
        Args:
            node: Текущий узел дерева
            point: Точка запроса
            radius: Радиус поиска
            indices: Список для сбора индексов
        """
        if node['type'] == 'leaf':
            # Проверяем все точки в листе
            leaf_points = self.data[node['indices']]
            distances = torch.norm(leaf_points - point, dim=1)
            mask = distances <= radius
            if mask.any():
                indices.append(node['indices'][mask])
        else:
            axis = node['axis']
            
            # Проверяем, нужно ли исследовать левое поддерево
            if point[axis] - radius <= node['value']:
                self._query_recursive(node['left'], point, radius, indices)
            
            # Проверяем, нужно ли исследовать правое поддерево
            if point[axis] + radius > node['value']:
                self._query_recursive(node['right'], point, radius, indices)
    
    def batch_query_radius(self, points: torch.Tensor, radius: float) -> List[torch.Tensor]:
        """
        Пакетный поиск соседей для нескольких точек.
        
        Args:
            points: Точки запроса [n_queries, n_features]
            radius: Радиус поиска
            
        Returns:
            Список тензоров с индексами соседей для каждой точки
        """
        results = []
        for i in range(len(points)):
            neighbors = self.query_radius(points[i], radius)
            results.append(neighbors)
        return results


class TorchRadiusNeighbors:
    """
    Полная замена NearestNeighbors из sklearn на чистый PyTorch.
    Вычисляет соседей в радиусе используя матричные операции PyTorch или KD-Tree.
    
    Attributes:
        radius (float): Радиус поиска соседей
        metric (str): Метрика расстояния ('euclidean', 'cosine')
        use_kdtree (bool): Использовать ли KD-Tree для ускорения
        leaf_size (int): Размер листа для KD-Tree
        X (torch.Tensor): Данные для поиска соседей
        kdtree (TorchKDTree): KD-Tree для ускорения поиска
    """
    
    def __init__(self, radius: float = 0.5, metric: str = 'euclidean', 
                 use_kdtree: bool = True, leaf_size: int = 30):
        """
        Инициализация TorchRadiusNeighbors.
        
        Args:
            radius: Радиус поиска соседей
            metric: Метрика расстояния ('euclidean', 'cosine')
            use_kdtree: Использовать ли KD-Tree для ускорения
            leaf_size: Размер листа для KD-Tree
            
        Raises:
            ValueError: Если параметры невалидны
        """
        if radius <= 0:
            raise ValueError("radius должен быть положительным")
        if metric not in ['euclidean', 'cosine']:
            raise ValueError("Метрика должна быть 'euclidean' или 'cosine'")
        if leaf_size <= 0:
            raise ValueError("leaf_size должен быть положительным")
            
        self.radius = radius
        self.metric = metric
        self.use_kdtree = use_kdtree and metric == 'euclidean'  # KD-Tree только для евклидовой метрики
        self.leaf_size = leaf_size
        self.X = None
        self.kdtree = None
        
    def fit(self, X: Union[torch.Tensor, np.ndarray]) -> 'TorchRadiusNeighbors':
        """
        Сохраняет данные для поиска соседей.
        
        Args:
            X: Данные для поиска соседей [n_samples, n_features]
            
        Returns:
            self: Возвращает экземпляр класса
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
            
        self.X = X
        
        # Строим KD-Tree если нужно
        if self.use_kdtree:
            self.kdtree = TorchKDTree(leaf_size=self.leaf_size)
            self.kdtree.fit(X)
            
        return self
    
    def radius_neighbors(self, X: Optional[Union[torch.Tensor, np.ndarray]] = None, 
                        return_distance: bool = False) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]]:
        """
        Находит соседей в радиусе для каждой точки.
        
        Args:
            X: Точки для которых ищем соседей (если None, используем fit данные)
            return_distance: Возвращать ли расстояния
            
        Returns:
            Если return_distance=True: (distances, indices)
            Иначе: indices
            
        Raises:
            ValueError: Если не вызван метод fit()
        """
        if X is None:
            X = self.X
        elif not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
            
        if self.X is None:
            raise ValueError("Не вызван метод fit()")
        
        # Используем KD-Tree если доступно
        if self.use_kdtree and self.kdtree is not None:
            return self._radius_neighbors_kdtree(X, return_distance)
        else:
            return self._radius_neighbors_brute(X, return_distance)
    
    def _radius_neighbors_kdtree(self, X: torch.Tensor, return_distance: bool) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]]:
        """
        Поиск соседей с использованием KD-Tree.
        """
        neighbors_indices = self.kdtree.batch_query_radius(X, self.radius)
        
        if return_distance:
            neighbors_distances = []
            for i, indices in enumerate(neighbors_indices):
                if len(indices) > 0:
                    distances = torch.norm(X[i] - self.X[indices], dim=1)
                    neighbors_distances.append(distances)
                else:
                    neighbors_distances.append(torch.tensor([], dtype=torch.float32, device=X.device))
            
            return neighbors_distances, neighbors_indices
        else:
            return neighbors_indices
    
    def _radius_neighbors_brute(self, X: torch.Tensor, return_distance: bool) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]]:
        """
        Поиск соседей полным перебором.
        """
        # Вычисляем матрицу расстояний
        distance_matrix = self._compute_distance_matrix(X, self.X)
        
        # Находим соседей в радиусе
        neighbors_indices = []
        neighbors_distances = [] if return_distance else None
        
        for i in range(len(X)):
            mask = distance_matrix[i] <= self.radius
            indices = torch.where(mask)[0]
            neighbors_indices.append(indices)
            
            if return_distance:
                distances = distance_matrix[i][mask]
                neighbors_distances.append(distances)
        
        # Освобождаем память
        del distance_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_distance:
            return neighbors_distances, neighbors_indices
        else:
            return neighbors_indices
    
    def _compute_distance_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет матрицу расстояний между x и y.
        
        Args:
            x: Первый тензор [n_samples_x, n_features]
            y: Второй тензор [n_samples_y, n_features]
            
        Returns:
            Матрица расстояний [n_samples_x, n_samples_y]
            
        Raises:
            ValueError: Если метрика не поддерживается
        """
        if self.metric == 'euclidean':
            return self._euclidean_distance_matrix(x, y)
        elif self.metric == 'cosine':
            return self._cosine_distance_matrix(x, y)
        else:
            raise ValueError(f"Метрика {self.metric} не поддерживается")
    
    def _euclidean_distance_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет евклидову матрицу расстояний через матричные операции.
        
        Формула: ||x - y||² = ||x||² + ||y||² - 2*x*yᵀ
        
        Args:
            x: Первый тензор [n_samples_x, n_features]
            y: Второй тензор [n_samples_y, n_features]
            
        Returns:
            Евклидова матрица расстояний [n_samples_x, n_samples_y]
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        
        distances_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        # Из-за ошибок округления могут быть маленькие отрицательные числа
        distances_sq = torch.clamp(distances_sq, min=0.0)
        return torch.sqrt(distances_sq)
    
    def _cosine_distance_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет косинусную матрицу расстояний.
        
        Args:
            x: Первый тензор [n_samples_x, n_features]
            y: Второй тензор [n_samples_y, n_features]
            
        Returns:
            Косинусная матрица расстояний [n_samples_x, n_samples_y]
        """
        # Нормализуем векторы
        x_norm = x / torch.norm(x, dim=1, keepdim=True).clamp(min=1e-10)
        y_norm = y / torch.norm(y, dim=1, keepdim=True).clamp(min=1e-10)
        
        # Косинусное сходство
        cosine_sim = torch.mm(x_norm, y_norm.t())
        # Косинусное расстояние = 1 - cosine_similarity
        return 1 - cosine_sim


class DBSCAN:
    """
    Улучшенная реализация DBSCAN для эмбеддингов на PyTorch с поддержкой:
    - Батч-обучения и дообучения
    - Предсказания через центроиды кластеров
    - Различных метрик расстояния
    - KD-Tree для ускорения поиска
    - Чистого PyTorch без зависимостей от sklearn
    
    Attributes:
        eps (float): Радиус окрестности для поиска соседей
        min_samples (int): Минимальное количество соседей для core point
        metric (str): Метрика расстояния ('euclidean', 'cosine')
        use_kdtree (bool): Использовать ли KD-Tree для ускорения
        leaf_size (int): Размер листа для KD-Tree
        labels_ (torch.Tensor): Метки кластеров после обучения
        cluster_centroids_ (dict): Центроиды кластеров
        cluster_sizes_ (dict): Размеры кластеров для взвешенного обновления
        core_samples_ (dict): Core samples для каждого кластера
        is_fitted (bool): Флаг обучения модели
        device_ (torch.device): Устройство вычислений
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean',
                 use_kdtree: bool = True, leaf_size: int = 30):
        """
        Инициализация параметров DBSCAN.
        
        Args:
            eps: Радиус окрестности для поиска соседей
            min_samples: Минимальное количество соседей для core point
            metric: Метрика расстояния ('euclidean', 'cosine')
            use_kdtree: Использовать ли KD-Tree для ускорения
            leaf_size: Размер листа для KD-Tree
            
        Raises:
            ValueError: Если параметры невалидны
        """
        if eps <= 0:
            raise ValueError("eps должен быть положительным")
        if min_samples <= 0:
            raise ValueError("min_samples должен быть положительным")
        if metric not in ['euclidean', 'cosine']:
            raise ValueError("Метрика должна быть 'euclidean' или 'cosine'")
        if leaf_size <= 0:
            raise ValueError("leaf_size должен быть положительным")
            
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.use_kdtree = use_kdtree
        self.leaf_size = leaf_size
        
        self.labels_ = None
        self.cluster_centroids_ = {}
        self.cluster_sizes_ = {}
        self.core_samples_ = defaultdict(list)
        self.is_fitted = False
        self.device_ = torch.device('cpu')
        
    def fit(self, X: Union[torch.Tensor, np.ndarray], batch_size: Optional[int] = None) -> 'DBSCAN':
        """
        Обучение DBSCAN на данных.
        
        Args:
            X: Тензор эмбеддингов [n_samples, n_features]
            batch_size: Размер батча для инкрементального обучения
            
        Returns:
            self: Возвращает экземпляр класса
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
            
        # Сохраняем устройство для консистентности
        self.device_ = X.device
        
        if batch_size and batch_size < len(X):
            return self._fit_batch(X, batch_size)
        else:
            return self._fit_single_batch(X)
    
    def fit_predict(self, X: Union[torch.Tensor, np.ndarray], 
                   batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Обучение и предсказание в одном методе.
        
        Args:
            X: Тензор эмбеддингов [n_samples, n_features]
            batch_size: Размер батча для инкрементального обучения
            
        Returns:
            Метки кластеров
        """
        self.fit(X, batch_size)
        return self.labels_
    
    def _fit_single_batch(self, X: torch.Tensor) -> 'DBSCAN':
        """
        Обычное обучение на всех данных используя TorchRadiusNeighbors.
        
        Args:
            X: Тензор данных для обучения
            
        Returns:
            self: Возвращает экземпляр класса
        """
        n_samples = X.shape[0]
        labels = -torch.ones(n_samples, dtype=torch.int32, device=self.device_)
        
        # Используем наш TorchRadiusNeighbors вместо sklearn
        neighbors_model = TorchRadiusNeighbors(
            radius=self.eps, 
            metric=self.metric,
            use_kdtree=self.use_kdtree,
            leaf_size=self.leaf_size
        )
        neighbors_model.fit(X)
        neighborhoods = neighbors_model.radius_neighbors(X, return_distance=False)
        
        cluster_label = 0
        
        # Используем tqdm для прогресс-бара
        for i in tqdm(range(n_samples), desc="DBSCAN clustering"):
            if labels[i] != -1:
                continue
                
            if torch.is_tensor(neighborhoods[i]):
                neighbors = neighborhoods[i]
            else:
                neighbors = torch.tensor(neighborhoods[i], dtype=torch.long, device=self.device_)
            
            if len(neighbors) < self.min_samples:
                labels[i] = 0  # Шум
                continue
                
            cluster_label += 1
            labels[i] = cluster_label
            self._expand_cluster(i, neighbors, labels, neighborhoods, cluster_label, X)
        
        self.labels_ = labels
        self._compute_centroids(X, labels)
        self.is_fitted = True
        return self
    
    def _fit_batch(self, X: torch.Tensor, batch_size: int) -> 'DBSCAN':
        """
        Инкрементальное обучение батчами.
        
        Args:
            X: Тензор данных для обучения
            batch_size: Размер батча
            
        Returns:
            self: Возвращает экземпляр класса
        """
        n_samples = X.shape[0]
        all_labels = -torch.ones(n_samples, dtype=torch.int32, device=self.device_)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_X = X[start_idx:end_idx]
            
            if not self.is_fitted:
                # Первый батч - обучаем с нуля
                batch_labels = -torch.ones(len(batch_X), dtype=torch.int32, device=self.device_)
                
                neighbors_model = TorchRadiusNeighbors(
                    radius=self.eps, 
                    metric=self.metric,
                    use_kdtree=self.use_kdtree,
                    leaf_size=self.leaf_size
                )
                neighbors_model.fit(batch_X)
                neighborhoods = neighbors_model.radius_neighbors(batch_X, return_distance=False)
                
                cluster_label = 0
                for i in range(len(batch_X)):
                    if batch_labels[i] != -1:
                        continue
                        
                    if torch.is_tensor(neighborhoods[i]):
                        neighbors = neighborhoods[i]
                    else:
                        neighbors = torch.tensor(neighborhoods[i], dtype=torch.long, device=self.device_)
                        
                    if len(neighbors) < self.min_samples:
                        batch_labels[i] = 0
                        continue
                        
                    cluster_label += 1
                    batch_labels[i] = cluster_label
                    self._expand_cluster_batch(i, neighbors, batch_labels, neighborhoods, cluster_label, batch_X)
                
                all_labels[start_idx:end_idx] = batch_labels
                self._compute_centroids(batch_X, batch_labels)
                self.is_fitted = True
                
            else:
                # Последующие батчи - дообучение через центроиды
                batch_labels = self._predict_batch(batch_X)
                all_labels[start_idx:end_idx] = batch_labels
                
                # Обновляем центроиды с новыми точками
                self._update_centroids(batch_X, batch_labels)
        
        self.labels_ = all_labels
        return self
    
    def partial_fit(self, X: Union[torch.Tensor, np.ndarray]) -> 'DBSCAN':
        """
        Дообучение модели на новых данных.
        
        Args:
            X: Новые эмбеддинги для дообучения
            
        Returns:
            self: Возвращает экземпляр класса
        """
        if not self.is_fitted:
            return self.fit(X)
            
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
            
        # Предсказываем кластеры для новых данных
        new_labels = self._predict_batch(X)
        
        # Обновляем центроиды
        self._update_centroids(X, new_labels)
        
        # Объединяем с предыдущими результатами
        if self.labels_ is not None:
            current_labels = self.labels_
            updated_labels = torch.cat([current_labels, new_labels])
            self.labels_ = updated_labels
        else:
            self.labels_ = new_labels
            
        return self
    
    def _expand_cluster(self, point_idx: int, neighbors: torch.Tensor, labels: torch.Tensor, 
                       neighborhoods: List[torch.Tensor], cluster_label: int, X: torch.Tensor):
        """
        Расширение кластера (стандартная версия).
        
        Args:
            point_idx: Индекс начальной точки
            neighbors: Соседи начальной точки
            labels: Тензор меток кластеров
            neighborhoods: Список соседей для всех точек
            cluster_label: Метка текущего кластера
            X: Исходные данные
        """
        cluster_points = [point_idx]
        
        i = 0
        while i < len(cluster_points):
            current_point = cluster_points[i]
            
            if torch.is_tensor(neighborhoods[current_point]):
                current_neighbors = neighborhoods[current_point]
            else:
                current_neighbors = torch.tensor(neighborhoods[current_point], dtype=torch.long, device=self.device_)
            
            if len(current_neighbors) >= self.min_samples:
                for neighbor_idx in current_neighbors:
                    neighbor_idx = neighbor_idx.item()
                    if labels[neighbor_idx] == 0:  # Шум -> border point
                        labels[neighbor_idx] = cluster_label
                    elif labels[neighbor_idx] == -1:  # Не посещен -> добавляем
                        labels[neighbor_idx] = cluster_label
                        cluster_points.append(neighbor_idx)
            i += 1
            
        # Сохраняем core samples для кластера
        core_points = [idx for idx in cluster_points 
                      if len(neighborhoods[idx]) >= self.min_samples]
        self.core_samples_[cluster_label].extend([X[idx] for idx in core_points])
    
    def _expand_cluster_batch(self, point_idx: int, neighbors: torch.Tensor, labels: torch.Tensor,
                             neighborhoods: List[torch.Tensor], cluster_label: int, X: torch.Tensor):
        """
        Расширение кластера (батч-версия).
        
        Args:
            point_idx: Индекс начальной точки
            neighbors: Соседи начальной точки
            labels: Тензор меток кластеров
            neighborhoods: Список соседей для всех точек
            cluster_label: Метка текущего кластера
            X: Исходные данные
        """
        cluster_points = [point_idx]
        
        i = 0
        while i < len(cluster_points):
            current_point = cluster_points[i]
            
            if torch.is_tensor(neighborhoods[current_point]):
                current_neighbors = neighborhoods[current_point]
            else:
                current_neighbors = torch.tensor(neighborhoods[current_point], dtype=torch.long, device=self.device_)
            
            if len(current_neighbors) >= self.min_samples:
                for neighbor_idx in current_neighbors:
                    neighbor_idx = neighbor_idx.item()
                    if labels[neighbor_idx] == 0:
                        labels[neighbor_idx] = cluster_label
                    elif labels[neighbor_idx] == -1:
                        labels[neighbor_idx] = cluster_label
                        cluster_points.append(neighbor_idx)
            i += 1
    
    def _compute_centroids(self, X: torch.Tensor, labels: torch.Tensor):
        """
        Вычисление центроидов для всех кластеров используя PyTorch.
        
        Args:
            X: Исходные данные
            labels: Метки кластеров
        """
        unique_labels = torch.unique(labels)
        self.cluster_centroids_ = {}
        self.cluster_sizes_ = {}
        
        for label in unique_labels:
            label = label.item()
            if label == 0:  # Пропускаем шум
                continue
                
            cluster_mask = (labels == label)
            cluster_points = X[cluster_mask]
            
            if len(cluster_points) > 0:
                centroid = torch.mean(cluster_points, dim=0)
                self.cluster_centroids_[label] = centroid
                self.cluster_sizes_[label] = len(cluster_points)
    
    def _update_centroids(self, X: torch.Tensor, labels: torch.Tensor):
        """
        Обновление центроидов с новыми данными используя PyTorch.
        
        Args:
            X: Новые данные
            labels: Метки кластеров для новых данных
        """
        if not self.cluster_centroids_:
            self._compute_centroids(X, labels)
            return
            
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            label = label.item()
            if label == 0:
                continue
                
            cluster_mask = (labels == label)
            cluster_points = X[cluster_mask]
            n_new_points = len(cluster_points)
            
            if n_new_points > 0:
                if label in self.cluster_centroids_:
                    # Взвешенное среднее с учетом количества точек
                    old_centroid = self.cluster_centroids_[label]
                    old_size = self.cluster_sizes_.get(label, 1)
                    total_size = old_size + n_new_points
                    
                    sum_old = old_centroid * old_size
                    sum_new = cluster_points.sum(dim=0)
                    new_centroid = (sum_old + sum_new) / total_size
                    
                    self.cluster_centroids_[label] = new_centroid
                    self.cluster_sizes_[label] = total_size
                else:
                    # Новый кластер
                    new_centroid = torch.mean(cluster_points, dim=0)
                    self.cluster_centroids_[label] = new_centroid
                    self.cluster_sizes_[label] = n_new_points
    
    def predict(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Предсказание кластеров для новых данных через центроиды.
        
        Args:
            X: Новые эмбеддинги для предсказания
            
        Returns:
            Метки кластеров
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
            
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
            
        return self._predict_batch(X)
    
    def _predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Предсказание для батча данных через центроиды используя PyTorch.
        
        Args:
            X: Данные для предсказания
            
        Returns:
            Метки кластеров
        """
        if not self.cluster_centroids_:
            return torch.zeros(len(X), dtype=torch.int32, device=X.device)
            
        # Собираем все центроиды в один тензор
        centroids_list = []
        labels_list = []
        for label, centroid in self.cluster_centroids_.items():
            centroids_list.append(centroid)
            labels_list.append(label)
        
        if not centroids_list:
            return torch.zeros(len(X), dtype=torch.int32, device=X.device)
            
        centroids_tensor = torch.stack(centroids_list).to(X.device)  # [n_centroids, n_features]
        labels_tensor = torch.tensor(labels_list, dtype=torch.int32, device=X.device)
        
        # Векторизованное вычисление расстояний
        if self.metric == 'euclidean':
            # [n_samples, n_centroids]
            distances = torch.cdist(X.unsqueeze(0), centroids_tensor.unsqueeze(0)).squeeze(0)
        elif self.metric == 'cosine':
            X_norm = X / torch.norm(X, dim=1, keepdim=True).clamp(min=1e-10)
            centroids_norm = centroids_tensor / torch.norm(centroids_tensor, dim=1, keepdim=True).clamp(min=1e-10)
            distances = 1 - torch.mm(X_norm, centroids_norm.t())
        
        # Находим ближайший центроид в пределах eps
        min_distances, min_indices = torch.min(distances, dim=1)
        predictions = labels_tensor[min_indices]
        predictions[min_distances > self.eps] = 0  # Шум
        
        return predictions
    
    def get_cluster_centroids(self) -> Tuple[Optional[torch.Tensor], List[int]]:
        """
        Возвращает центроиды кластеров в виде тензора PyTorch.
        
        Returns:
            Кортеж (центроиды, метки кластеров) или (None, []) если нет центроидов
        """
        if not self.cluster_centroids_:
            return None, []
            
        centroids_list = []
        labels_list = []
        
        for label, centroid in sorted(self.cluster_centroids_.items()):
            centroids_list.append(centroid)
            labels_list.append(label)
            
        return torch.stack(centroids_list), labels_list