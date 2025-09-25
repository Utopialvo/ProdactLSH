import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.cluster import KMeans
import heapq
from collections import defaultdict
import math
from itertools import product
from typing import List, Tuple, Dict, Union, Optional, Callable
import warnings


class LSH:
    """
    Locality-Sensitive Hashing (LSH) с поддержкой GPU и различных метрик расстояния.
    
    Математическая основа (из LSH.pdf):
    Семейство хэш-функций H называется (r, cr, p1, p2)-чувствительным для метрики d, если:
    1. Если d(p, q) ≤ r, то Pr[h(p) = h(q)] ≥ p1
    2. Если d(p, q) ≥ cr, то Pr[h(p) = h(q)] ≤ p2
    
    AND-construction: p_and = p1^k (усиливает строгость)
    OR-construction: p_or = 1 - (1 - p1)^L (увеличивает recall)
    
    Args:
        input_dim: размерность входных векторов
        num_tables: количество хэш-таблиц (L) - OR-construction
        hash_size: количество хэш-функций на таблицу (k) - AND-construction  
        bucket_width: ширина бакета (w)
        distance_type: тип расстояния ('euclidean', 'cosine', 'manhattan')
        projection_type: тип проекций ('random', 'orthogonal', 'random_rotation')
        device: 'auto', 'cuda' или 'cpu'
    """
    
    def __init__(self, input_dim: int, num_tables: int = 10, hash_size: int = 16, 
                 bucket_width: float = 1.0, distance_type: str = 'euclidean',
                 projection_type: str = 'random', device: str = 'auto'):
        
        self.input_dim = input_dim
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.bucket_width = bucket_width
        self.distance_type = distance_type
        self.projection_type = projection_type
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Инициализация проекций в зависимости от типа
        self._init_projections()
        
        # Смещения для квантования (только для евклидова и манхеттенского расстояний)
        if distance_type in ['euclidean', 'manhattan']:
            # b ~ U(0, w) для обеспечения равномерного квантования
            self.biases = torch.rand(num_tables, hash_size, device=self.device) * bucket_width
        else:
            self.biases = None
            
        self.buckets = [{} for _ in range(num_tables)]
        self.data_points = []
        
        # Для косинусного расстояния нормализуем векторы
        self._normalize_for_cosine = (distance_type == 'cosine')
        
    def _init_projections(self):
        """
        Инициализация проекционных матриц в зависимости от типа.
        
        Теоретическое обоснование (LSH.pdf, раздел 3):
        - Случайные проекции: a ~ N(0, I) для евклидова расстояния
        - Ортогональные проекции: лучше сохраняют расстояния
        - Случайные повороты: равномерное покрытие сферы
        """
        if self.projection_type == 'random':
            # Случайные проекции из нормального распределения (p-стабильное распределение)
            self.projections = torch.randn(
                self.num_tables, self.hash_size, self.input_dim, 
                device=self.device
            )
            
        elif self.projection_type == 'orthogonal':
            # Ортогональные проекции (лучше сохраняют расстояния)
            projections = []
            for _ in range(self.num_tables):
                # Генерируем случайную матрицу и ортогонализируем ее
                random_matrix = torch.randn(self.hash_size, self.input_dim, device=self.device)
                Q, _ = torch.linalg.qr(random_matrix.T)  # QR-разложение для ортогонализации
                projections.append(Q.T[:self.hash_size])  # Берем первые hash_size строк
            self.projections = torch.stack(projections)
            
        elif self.projection_type == 'random_rotation':
            # Случайные повороты (равномерно распределенные на сфере)
            projections = []
            for _ in range(self.num_tables * self.hash_size):
                # Генерируем случайный вектор и нормализуем его
                vec = torch.randn(self.input_dim, device=self.device)
                vec = F.normalize(vec, p=2, dim=0)
                projections.append(vec)
                
            self.projections = torch.stack(projections).reshape(
                self.num_tables, self.hash_size, self.input_dim
            )
        else:
            raise ValueError(f"Неизвестный тип проекции: {self.projection_type}")
            
        # Нормализация для разных типов расстояний
        if self.distance_type == 'manhattan':
            # Для манхеттенского расстояния используем распределение Коши
            cauchy = torch.tensor(
                np.random.standard_cauchy(self.projections.shape),
                device=self.device, dtype=torch.float32
            )
            self.projections = cauchy
            
        elif self.distance_type == 'cosine':
            # Для косинусного расстояния нормализуем проекции
            self.projections = F.normalize(self.projections, p=2, dim=-1)
    
    def _normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Нормализация векторов для косинусного расстояния"""
        if self._normalize_for_cosine:
            return F.normalize(vectors, p=2, dim=-1)
        return vectors
    
    def _compute_hashes(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Вычисление хэшей для векторов в зависимости от типа расстояния.
        
        Для евклидова расстояния (E2LSH):
        h(v) = floor((a·v + b) / w)
        
        Для косинусного расстояния:
        h(v) = sign(a·v)
        """
        batch_size = vectors.shape[0]
        
        # Нормализуем векторы если нужно
        vectors = self._normalize_vectors(vectors)
        
        # Проецируем вектора для всех таблиц
        projections = self.projections.unsqueeze(0)  # [1, L, k, D]
        vectors_expanded = vectors.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        
        # Вычисляем скалярные произведения
        dot_products = (projections * vectors_expanded).sum(dim=-1)  # [B, L, k]
        
        if self.distance_type == 'cosine':
            # Для косинусного расстояния используем знак проекции
            hashes = (dot_products > 0).long()
        else:
            # Для евклидова и манхеттенского используем квантование
            hashes = torch.floor((dot_products + self.biases.unsqueeze(0)) / self.bucket_width)
        
        return hashes.long()
    
    def _compute_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Вычисление расстояния между векторами в зависимости от типа метрики"""
        if self.distance_type == 'euclidean':
            return torch.norm(vec1 - vec2, dim=1)
        elif self.distance_type == 'manhattan':
            return torch.abs(vec1 - vec2).sum(dim=1)
        elif self.distance_type == 'cosine':
            return 1 - F.cosine_similarity(vec1, vec2)
        else:
            raise ValueError(f"Неизвестный тип расстояния: {self.distance_type}")
    
    def add(self, vectors: torch.Tensor, ids: Optional[List[int]] = None) -> None:
        """Добавление векторов в индекс"""
        if len(vectors.shape) == 1:
            vectors = vectors.unsqueeze(0)
            
        # Проверка размерности
        if vectors.shape[1] != self.input_dim:
            raise ValueError(f"Ожидалась размерность {self.input_dim}, получена {vectors.shape[1]}")
            
        vectors = vectors.to(self.device)
        batch_size = vectors.shape[0]
        
        if ids is None:
            start_idx = len(self.data_points)
            ids = list(range(start_idx, start_idx + batch_size))
        elif len(ids) != batch_size:
            raise ValueError("Количество ID должно совпадать с количеством векторов")
        
        # Сохраняем вектора
        self.data_points.extend(vectors.cpu())
        
        # Вычисляем хэши
        hashes = self._compute_hashes(vectors)  # [B, L, k]
        
        # Добавляем в бакеты
        for i in range(batch_size):
            vector_id = ids[i]
            for table_idx in range(self.num_tables):
                # Создаем ключ бакета
                hash_key = tuple(hashes[i, table_idx].cpu().numpy())
                
                if hash_key not in self.buckets[table_idx]:
                    self.buckets[table_idx][hash_key] = []
                
                self.buckets[table_idx][hash_key].append(vector_id)
    
    def _generate_probes(self, main_hash: tuple, num_probes: int) -> List[np.ndarray]:
        """
        Генерация соседних бакетов для multi-probe поиска.
        
        Multi-probe LSH (LSH.pdf, раздел 1.3): 
        Исследует соседние бакеты для увеличения recall без добавления хэш-таблиц.
        """
        probes = []
        hash_array = np.array(main_hash)
        
        if self.distance_type == 'cosine':
            # Для косинусного расстояния переворачиваем биты
            for num_flips in range(1, min(len(main_hash), 3)):  # Переворачиваем 1-2 бита
                for flip_positions in product([0, 1], repeat=len(main_hash)):
                    if sum(flip_positions) != num_flips:
                        continue
                    
                    new_hash = hash_array.copy()
                    for pos in np.where(flip_positions)[0]:
                        new_hash[pos] = 1 - new_hash[pos]  # Переворачиваем бит
                    
                    probes.append(new_hash)
                    
                    if len(probes) >= num_probes:
                        return probes[:num_probes]
        else:
            # Для евклидова и манхеттенского расстояний сдвигаем на ±1
            for combinations in product([-1, 0, 1], repeat=len(main_hash)):
                if all(delta == 0 for delta in combinations):
                    continue  # Пропускаем исходный бакет
                    
                new_hash = hash_array + np.array(combinations)
                probes.append(new_hash)
                
                if len(probes) >= num_probes:
                    return probes[:num_probes]
        
        return probes[:num_probes]
    
    def query(self, query_vector: torch.Tensor, k: int = 10, 
              num_probes: int = 5) -> Tuple[List[int], List[float]]:
        """
        Поиск k ближайших соседей.
        
        Вероятностные гарантии (LSH.pdf, раздел 2):
        P[success] ≥ 1 - (1 - p1^k)^L
        P[false positive] ≤ p2^k * L
        """
        if len(query_vector.shape) == 1:
            query_vector = query_vector.unsqueeze(0)
            
        if query_vector.shape[1] != self.input_dim:
            raise ValueError(f"Ожидалась размерность {self.input_dim}, получена {query_vector.shape[1]}")
            
        query_vector = query_vector.to(self.device)
        query_vector = self._normalize_vectors(query_vector)
        
        # Вычисляем хэши для запроса
        query_hashes = self._compute_hashes(query_vector)  # [1, L, k]
        
        # Собираем кандидатов из бакетов
        candidates = set()
        
        for table_idx in range(self.num_tables):
            # Основной бакет
            main_hash = tuple(query_hashes[0, table_idx].cpu().numpy())
            
            if main_hash in self.buckets[table_idx]:
                candidates.update(self.buckets[table_idx][main_hash])
            
            # Multi-probe: соседние бакеты
            if num_probes > 1:
                for probe in self._generate_probes(main_hash, num_probes-1):
                    probe_key = tuple(probe)
                    if probe_key in self.buckets[table_idx]:
                        candidates.update(self.buckets[table_idx][probe_key])
        
        # Вычисляем расстояния до кандидатов
        if not candidates:
            return [], []
            
        candidate_vectors = torch.stack([self.data_points[i] for i in candidates]).to(self.device)
        query_expanded = query_vector.expand(len(candidates), -1)
        
        # Вычисляем расстояния в зависимости от типа метрики
        distances = self._compute_distance(query_expanded, candidate_vectors)
        
        # Выбираем k ближайших
        if len(candidates) > k:
            topk_values, topk_indices = torch.topk(distances, k, largest=False)
            candidate_list = list(candidates)
            nearest_ids = [candidate_list[i] for i in topk_indices.cpu().numpy()]
            nearest_distances = topk_values.cpu().numpy()
        else:
            nearest_ids = list(candidates)
            nearest_distances = distances.cpu().numpy()
        
        return nearest_ids, nearest_distances


class LSHWithSampling(LSH):
    """
    LSH с поддержкой Importance Sampling для эффективной выборки данных.
    
    Методологическое обоснование (IS_LSH.pdf, раздел 2):
    LSH осуществляет естественную стратификацию данных. Importance Sampling позволяет
    проводить оценку математического ожидания с помощью взвешенной выборки:
    
    E_p[L] ≈ (1/n) * Σ w(x_i) * L(x_i), где w(x) = p(x)/q(x)
    
    В контексте LSH-бакетов, q(x) соответствует вероятности выборки из конкретного бакета.
    """

    def __init__(self, input_dim: int, num_tables: int = 10, hash_size: int = 16, 
                 bucket_width: float = 1.0, distance_type: str = 'euclidean',
                 projection_type: str = 'random', device: str = 'auto'):
        
        super().__init__(input_dim, num_tables, hash_size, bucket_width, 
                        distance_type, projection_type, device)
        
        # Статистика бакетов для семплинга
        self.bucket_stats = defaultdict(lambda: {'size': 0, 'ids': []})
        self.total_points = 0
        
    def add(self, vectors: torch.Tensor, ids: Optional[List[int]] = None) -> None:
        """Добавление векторов с сохранением статистики бакетов"""
        if len(vectors.shape) == 1:
            vectors = vectors.unsqueeze(0)
            
        vectors = vectors.to(self.device)
        batch_size = vectors.shape[0]
        
        if ids is None:
            start_idx = len(self.data_points)
            ids = list(range(start_idx, start_idx + batch_size))
        
        self.data_points.extend(vectors.cpu())
        self.total_points += batch_size
        
        hashes = self._compute_hashes(vectors)
        
        for i in range(batch_size):
            vector_id = ids[i]
            
            for table_idx in range(self.num_tables):
                hash_key = tuple(hashes[i, table_idx].cpu().numpy())
                
                if hash_key not in self.buckets[table_idx]:
                    self.buckets[table_idx][hash_key] = []
                
                self.buckets[table_idx][hash_key].append(vector_id)
                
                # Обновляем статистику бакетов
                bucket_id = f"table_{table_idx}_{hash_key}"
                self.bucket_stats[bucket_id]['size'] += 1
                self.bucket_stats[bucket_id]['ids'].append(vector_id)
    
    def importance_sampling(self, target_function: Callable, sample_size: int, 
                          strategy: str = 'proportional', temperature: float = 1.0, 
                          max_bucket_size: Optional[int] = None) -> Tuple[List[int], List[float]]:
        """
        Importance Sampling на основе LSH бакетов.
        
        Стратегии выборки (IS_LSH.pdf, раздел 3.2):
        - Пропорциональная: n_i ∝ |B_i| * S/N
        - Сбалансированная: n_i = min(c, |B_i|) * S/N  
        - На основе дисперсии: n_i ∝ |B_i| * σ_B_i
        
        Args:
            target_function: функция f(x) для оценки важности
            sample_size: размер выборки
            strategy: стратегия выборки ('proportional', 'balanced', 'variance_based')
            temperature: параметр для смягчения распределения
            max_bucket_size: максимальный размер выборки из одного бакета
            
        Returns:
            Кортеж (индексы выбранных точек, веса для importance sampling)
        """
        if max_bucket_size is None:
            max_bucket_size = max(1, sample_size // 10)
            
        if not self.bucket_stats:
            raise ValueError("Нет данных для семплинга. Сначала добавьте векторы.")
        
        # Вычисляем важность для каждого бакета согласно IS_LSH.pdf
        bucket_importances = {}
        bucket_sizes = {}
        
        for bucket_id, stats in self.bucket_stats.items():
            if stats['size'] == 0:
                continue
                
            bucket_size = stats['size']
            bucket_sizes[bucket_id] = bucket_size
            
            # Вычисляем важность в зависимости от стратегии
            if strategy == 'proportional':
                # n_i ∝ |B_i| * S/N
                importance = bucket_size
            elif strategy == 'balanced':
                # n_i = min(c, |B_i|) * S/N
                importance = min(bucket_size, max_bucket_size)
            elif strategy == 'variance_based':
                # n_i ∝ |B_i| * σ_B_i (дисперсия целевой функции в бакете)
                bucket_vectors = torch.stack([self.data_points[i] for i in stats['ids']])
                with torch.no_grad():
                    target_values = target_function(bucket_vectors)
                    if len(target_values) > 1:
                        importance = target_values.var().item() * bucket_size
                    else:
                        importance = bucket_size
            else:
                raise ValueError(f"Неизвестная стратегия: {strategy}")
            
            bucket_importances[bucket_id] = importance
        
        if not bucket_importances:
            return [], []
        
        # Нормализуем важности с температурой
        importances = torch.tensor(list(bucket_importances.values()))
        probs = F.softmax(importances / temperature, dim=0)
        
        # Распределяем sample_size по бакетам согласно IS_LSH.pdf
        bucket_ids = list(bucket_importances.keys())
        selected_buckets = []
        
        remaining_size = sample_size
        total_importance = sum(bucket_importances.values())
        
        for bucket_id, prob in zip(bucket_ids, probs):
            if remaining_size <= 0:
                break
                
            # Вычисляем размер выборки для бакета
            bucket_sample_size = min(
                math.ceil(prob * sample_size),
                bucket_sizes[bucket_id],
                max_bucket_size,
                remaining_size
            )
            
            if bucket_sample_size > 0:
                selected_buckets.append((bucket_id, bucket_sample_size))
                remaining_size -= bucket_sample_size
        
        # Собираем выборку с правильными весами согласно формуле 3.3 IS_LSH.pdf
        sampled_indices = []
        sampling_weights = []
        
        for bucket_id, size in selected_buckets:
            stats = self.bucket_stats[bucket_id]
            bucket_size = stats['size']
            
            # Случайная выборка из бакета
            if size >= bucket_size:
                selected_indices = list(range(bucket_size))
            else:
                selected_indices = torch.randperm(bucket_size)[:size].tolist()
            
            # КОРРЕКТНЫЙ расчет весов согласно IS_LSH.pdf, раздел 3.3
            bucket_importance = bucket_importances[bucket_id]
            bucket_prob = bucket_importance / total_importance
            
            # Вероятность выбора точки: P_select(B_i) * (n_i / |B_i|)
            point_prob = bucket_prob * (size / bucket_size)
            
            for idx in selected_indices:
                vector_id = stats['ids'][idx]
                sampled_indices.append(vector_id)
                
                # Правильная формула веса согласно IS_LSH.pdf:
                # w(x) = 1 / (P_select(B_i) * (n_i / |B_i|) * N)
                weight = 1.0 / (point_prob * self.total_points) if point_prob > 0 else 1.0
                sampling_weights.append(weight)
        
        return sampled_indices, sampling_weights
    
    def create_sampler(self, target_function: Callable, sample_size: int, 
                     strategy: str = 'proportional', **kwargs) -> WeightedRandomSampler:
        """Создание PyTorch Sampler для DataLoader"""
        indices, weights = self.importance_sampling(
            target_function, sample_size, strategy, **kwargs
        )
        
        if not indices:
            raise ValueError("Не удалось создать выборку")
            
        # Преобразуем веса в tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        
        return WeightedRandomSampler(weights_tensor, len(indices), replacement=True)


class ProductQuantizer:
    """
    Product Quantization с поддержкой GPU для эффективного сжатия и поиска векторов.
    
    Математическая основа (PQ_IS.pdf, раздел 2):
    Цель PQ - найти отображение q: ℝ^D → C, минимизирующее ошибку квантования:
    min_q Σ ||x_i - q(x_i)||^2
    
    Пространство разбивается на M подпространств, каждое квантуется независимо.
    
    Args:
        input_dim: размерность входных векторов
        num_subspaces: количество подпространств (M)
        num_centroids: количество центроидов в каждом подпространстве (K)
        device: 'auto', 'cuda' или 'cpu'
    """
    
    def __init__(self, input_dim: int, num_subspaces: int = 8, 
                 num_centroids: int = 256, device: str = 'auto'):
        
        assert input_dim % num_subspaces == 0, "input_dim должен делиться на num_subspaces"
        
        self.input_dim = input_dim
        self.num_subspaces = num_subspaces
        self.subspace_dim = input_dim // num_subspaces
        self.num_centroids = num_centroids
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.codebooks = None
        self.is_trained = False
        
    def train(self, data: torch.Tensor, max_iter: int = 100, 
              batch_size: int = 1000) -> torch.Tensor:
        """Обучение кодбуков с помощью k-means"""
        if len(data.shape) != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"Ожидалась размерность (N, {self.input_dim})")
            
        data = data.to(self.device)
        num_samples = data.shape[0]
        
        # Разделяем данные на подпространства
        data_split = data.view(num_samples, self.num_subspaces, self.subspace_dim)
        
        self.codebooks = torch.zeros(self.num_subspaces, self.num_centroids, 
                                   self.subspace_dim, device=self.device)
        
        # Обучаем k-means для каждого подпространства
        for subspace_idx in range(self.num_subspaces):
            subspace_data = data_split[:, subspace_idx, :]
            
            # Используем мини-батчи для больших datasets
            if num_samples > batch_size:
                centroids = self._minibatch_kmeans(subspace_data, max_iter, batch_size)
            else:
                centroids = self._exact_kmeans(subspace_data, max_iter)
            
            self.codebooks[subspace_idx] = centroids
            
        self.is_trained = True
        return self.codebooks
    
    def _minibatch_kmeans(self, data: torch.Tensor, max_iter: int, 
                         batch_size: int) -> torch.Tensor:
        """Приближенный k-means с мини-батчами"""
        num_samples = data.shape[0]
        
        # Инициализация центроидов
        indices = torch.randperm(num_samples)[:self.num_centroids]
        centroids = data[indices].clone()
        
        for iteration in range(max_iter):
            # Случайный мини-батч
            batch_indices = torch.randint(0, num_samples, (batch_size,))
            batch_data = data[batch_indices]
            
            # Находим ближайшие центроиды
            distances = torch.cdist(batch_data, centroids)
            closest = torch.argmin(distances, dim=1)
            
            # Обновляем центроиды
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.num_centroids, device=self.device)
            
            for i in range(batch_size):
                centroid_idx = closest[i]
                new_centroids[centroid_idx] += batch_data[i]
                counts[centroid_idx] += 1
            
            # Избегаем деления на ноль
            mask = counts > 0
            if mask.any():
                new_centroids[mask] /= counts[mask].unsqueeze(1)
            
            # Заменяем пустые центроиды случайными точками
            empty_mask = counts == 0
            if empty_mask.any():
                new_indices = torch.randint(0, num_samples, (empty_mask.sum(),))
                new_centroids[empty_mask] = data[new_indices]
            
            # Проверяем сходимость
            if torch.allclose(centroids, new_centroids, rtol=1e-6):
                break
                
            centroids = new_centroids
            
        return centroids
    
    def _exact_kmeans(self, data: torch.Tensor, max_iter: int) -> torch.Tensor:
        """Точный k-means (для CPU через sklearn)"""
        data_cpu = data.cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.num_centroids, max_iter=max_iter, n_init=1)
        kmeans.fit(data_cpu)
        
        return torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32)
    
    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        """Кодирование векторов в PQ-коды"""
        if not self.is_trained:
            raise ValueError("Сначала обучите модель с помощью train()")
            
        if len(vectors.shape) != 2 or vectors.shape[1] != self.input_dim:
            raise ValueError(f"Ожидалась размерность (N, {self.input_dim})")
            
        vectors = vectors.to(self.device)
        batch_size = vectors.shape[0]
        
        # Разделяем на подпространства
        vectors_split = vectors.view(batch_size, self.num_subspaces, self.subspace_dim)
        
        codes = torch.zeros(batch_size, self.num_subspaces, dtype=torch.long, device=self.device)
        
        for subspace_idx in range(self.num_subspaces):
            subspace_vectors = vectors_split[:, subspace_idx, :]
            subspace_codebook = self.codebooks[subspace_idx]
            
            # Вычисляем расстояния до всех центроидов
            distances = torch.cdist(subspace_vectors, subspace_codebook)
            
            # Находим ближайшие центроиды
            codes[:, subspace_idx] = torch.argmin(distances, dim=1)
        
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Декодирование PQ-кодов обратно в векторы"""
        batch_size = codes.shape[0]
        
        reconstructed = torch.zeros(batch_size, self.input_dim, device=self.device)
        
        for subspace_idx in range(self.num_subspaces):
            subspace_codes = codes[:, subspace_idx]
            subspace_vectors = self.codebooks[subspace_idx][subspace_codes]
            
            start_idx = subspace_idx * self.subspace_dim
            end_idx = start_idx + self.subspace_dim
            
            reconstructed[:, start_idx:end_idx] = subspace_vectors
        
        return reconstructed
    
    def search(self, query_vector: torch.Tensor, encoded_data: torch.Tensor, 
               k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск ближайших соседей с помощью асимметричного расстояния.
        
        Асимметричное вычисление расстояний (PQ_IS.pdf, раздел 2.5):
        d(q, y)^2 ≈ Σ ||q[m] - c_jm[m]||^2
        
        Args:
            query_vector: вектор запроса [D]
            encoded_data: закодированные данные [N, M]
            k: количество ближайших соседей
            
        Returns:
            Кортеж (индексы ближайших соседей, расстояния)
        """
        if not self.is_trained:
            raise ValueError("Сначала обучите модель с помощью train()")
            
        query_vector = query_vector.to(self.device)
        encoded_data = encoded_data.to(self.device)
        
        num_samples = encoded_data.shape[0]
        
        # Разделяем запрос на подпространства
        query_split = query_vector.view(self.num_subspaces, self.subspace_dim)
        
        # Предвычисляем расстояния для каждого подпространства
        distance_tables = torch.zeros(self.num_subspaces, self.num_centroids, device=self.device)
        
        for subspace_idx in range(self.num_subspaces):
            query_subspace = query_split[subspace_idx]
            codebook_subspace = self.codebooks[subspace_idx]
            
            # Евклидово расстояние от запроса до центроидов
            distances = torch.norm(codebook_subspace - query_subspace, dim=1)
            distance_tables[subspace_idx] = distances
        
        # Вычисляем расстояния до всех закодированных векторов
        total_distances = torch.zeros(num_samples, device=self.device)
        
        for subspace_idx in range(self.num_subspaces):
            subspace_codes = encoded_data[:, subspace_idx]
            total_distances += distance_tables[subspace_idx, subspace_codes]
        
        # Находим k ближайших
        if k < num_samples:
            distances, indices = torch.topk(total_distances, k, largest=False)
        else:
            distances = total_distances
            indices = torch.arange(num_samples)
        
        return indices.cpu().numpy(), distances.cpu().numpy()


class PQWithSampling(ProductQuantizer):
    """
    Product Quantization с поддержкой Importance Sampling.
    
    Методологическое обоснование (PQ_IS.pdf, раздел 4):
    PQ-кластеризация определяет распределение q(x) = Σ w_l * q_l(x),
    где w_l - вес кластера, q_l(x) - распределение внутри кластера.
    
    Importance Sampling позволяет оценить μ = E_p(x)[f(x)] через:
    μ = E_q(x)[f(x) * p(x)/q(x)]
    """

    def __init__(self, input_dim: int, num_subspaces: int = 8, 
                 num_centroids: int = 256, device: str = 'auto'):
        super().__init__(input_dim, num_subspaces, num_centroids, device)
        self.cluster_stats = None
        self.encoded_data = None
        self.data_points = None
        self.total_points = 0
        
    def train(self, data: torch.Tensor, max_iter: int = 100, 
              batch_size: int = 1000) -> torch.Tensor:
        """Обучение с сохранением статистики кластеров"""
        result = super().train(data, max_iter, batch_size)
        
        # Сохраняем данные для семплинга
        self.data_points = data.clone().to(self.device)
        self.encoded_data = self.encode(data)
        self.total_points = data.shape[0]
        
        # Вычисляем статистику кластеров
        self._compute_cluster_stats()
        
        return result
    
    def _compute_cluster_stats(self) -> None:
        """Вычисление статистики по кластерам PQ"""
        if self.encoded_data is None:
            return
            
        num_samples = self.encoded_data.shape[0]
        self.cluster_stats = {}
        
        # Для каждого подпространства считаем статистику
        for subspace_idx in range(self.num_subspaces):
            subspace_codes = self.encoded_data[:, subspace_idx]
            unique_codes, counts = torch.unique(subspace_codes, return_counts=True)
            
            self.cluster_stats[subspace_idx] = {
                'codes': unique_codes,
                'sizes': counts,
                'centroids': self.codebooks[subspace_idx]
            }
    
    def importance_sampling(self, target_function: Callable, sample_size: int,
                          strategy: str = 'proportional', use_residuals: bool = True,
                          temperature: float = 1.0) -> Tuple[List[int], List[float]]:
        """
        Importance Sampling на основе PQ кластеров.
        
        Алгоритм соответствует PQ_IS.pdf, Algorithm 1:
        1. Оценить μ_l, Σ_l для каждого кластера
        2. Вычислить важность I_l = Var_{x∼q_l}[f(x)]
        3. Нормировать веса P(l) = (w_l * I_l) / Σ_j w_j * I_j
        4. Выбирать кластер l ∼ P(l) и генерировать x_s ∼ q_l(x)
        5. Вычислять вес w_s = p(x_s)/q(x_s)
        """
        if self.cluster_stats is None:
            raise ValueError("Сначала обучите модель с помощью train()")
            
        # Используем первое подпространство для семплинга (можно расширить на все)
        selected_subspace = 0
        stats = self.cluster_stats[selected_subspace]
        
        # Вычисляем важности для каждого кластера
        cluster_importances = self._compute_cluster_importances(
            selected_subspace, target_function, strategy, use_residuals
        )
        
        if cluster_importances is None or len(cluster_importances) == 0:
            warnings.warn("Не удалось вычислить важности кластеров, используется равномерное распределение")
            # Равномерное распределение по кластерам
            cluster_importances = torch.ones(len(stats['codes']), device=self.device)
        
        # Заменяем NaN и отрицательные значения
        cluster_importances = torch.nan_to_num(cluster_importances, nan=1.0, posinf=1.0, neginf=0.0)
        cluster_importances = torch.clamp(cluster_importances, min=0.0)
        
        # Если все важности нулевые, используем равномерное распределение
        if cluster_importances.sum() == 0:
            cluster_importances = torch.ones_like(cluster_importances)
        
        # Нормализуем с температурой (softmax с температурой)
        probs = F.softmax(cluster_importances / temperature, dim=0)
        
        # Распределяем sample_size по кластерам
        cluster_sizes = stats['sizes'].cpu().numpy()
        cluster_sample_sizes = self._distribute_sample_size(
            probs.cpu().numpy(), cluster_sizes, sample_size
        )
        
        # Собираем выборку из кластеров с правильными весами
        sampled_indices = []
        sampling_weights = []
        
        total_importance = cluster_importances.sum().item()
        
        for cluster_idx, (code, size) in enumerate(zip(stats['codes'], cluster_sample_sizes)):
            if size == 0:
                continue
                
            # Находим индексы точек в этом кластере
            mask = self.encoded_data[:, selected_subspace] == code
            cluster_indices = torch.where(mask)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Случайная выборка из кластера
            if size >= len(cluster_indices):
                selected_indices = cluster_indices
            else:
                selected_indices = cluster_indices[torch.randperm(len(cluster_indices))[:size]]
            
            # Вычисляем веса согласно PQ_IS.pdf
            cluster_prob = probs[cluster_idx].item()
            point_prob = cluster_prob * (size / len(cluster_indices))
            
            for idx in selected_indices:
                sampled_indices.append(idx.item())
                # w(x) = 1 / (P(l) * (n_l/|B_l|) * N)
                weight = 1.0 / (point_prob * self.total_points) if point_prob > 0 else 1.0
                sampling_weights.append(weight)
        
        return sampled_indices, sampling_weights
    
    def _compute_cluster_importances(self, subspace_idx: int, target_function: Callable,
                                   strategy: str, use_residuals: bool) -> Optional[torch.Tensor]:
        """Вычисление важностей кластеров для заданной стратегии"""
        stats = self.cluster_stats[subspace_idx]
        importances = []
        
        for code in stats['codes']:
            # Находим точки в кластере
            mask = self.encoded_data[:, subspace_idx] == code
            cluster_points = self.data_points[mask]
            
            if len(cluster_points) == 0:
                importances.append(0.0)
                continue
                
            if strategy == 'proportional':
                # Важность пропорциональна размеру кластера
                importance = len(cluster_points)
            elif strategy == 'residual_variance' and use_residuals:
                # Дисперсия остатков относительно центроида (только в подпространстве)
                importance = self._compute_residual_variance(
                    cluster_points, stats['centroids'][code], subspace_idx, target_function
                )
            elif strategy == 'cluster_density':
                # Плотность кластера (можно использовать более сложные метрики)
                importance = self._compute_cluster_density(cluster_points)
            else:
                # По умолчанию используем размер кластера
                importance = len(cluster_points)
                
            importances.append(importance)
        
        return torch.tensor(importances, device=self.device)

    def _compute_residual_variance(self, cluster_points: torch.Tensor, centroid: torch.Tensor,
                                 subspace_idx: int, target_function: Callable) -> float:
        """Вычисление дисперсии остатков в кластере (только для подпространства)"""
        if len(cluster_points) < 2:
            return len(cluster_points)  # Для кластеров с 1 точкой используем размер
        
        # Вычисляем индексы подпространства
        start_idx = subspace_idx * self.subspace_dim
        end_idx = start_idx + self.subspace_dim
        
        # Берем только часть векторов, соответствующую подпространству
        cluster_points_subspace = cluster_points[:, start_idx:end_idx]
        
        # Вычисляем остатки (расстояние до центроида в подпространстве)
        residuals = torch.norm(cluster_points_subspace - centroid, dim=1)
        
        # Оцениваем важность через target_function (на полных векторах)
        with torch.no_grad():
            target_vals = target_function(cluster_points)
            
            # Проверяем, что дисперсии вычисляются корректно
            if len(target_vals) > 1:
                residual_var = residuals.var().item()
                target_var = target_vals.var().item()
                # Комбинируем дисперсию остатков и целевой функции
                importance = residual_var * target_var * len(cluster_points)
            else:
                importance = len(cluster_points)
            
        return importance
    
    def _compute_cluster_density(self, cluster_points: torch.Tensor) -> float:
        """Вычисление плотности кластера через среднее расстояние до центроида"""
        if len(cluster_points) == 0:
            return 0.0
            
        # Простая оценка плотности через обратное среднее расстояние
        centroid = cluster_points.mean(dim=0)
        avg_distance = torch.norm(cluster_points - centroid, dim=1).mean().item()
        
        # Чем меньше среднее расстояние, тем выше плотность
        density = len(cluster_points) / (1.0 + avg_distance)
        return density
    
    def _distribute_sample_size(self, probs: np.ndarray, cluster_sizes: np.ndarray,
                              total_size: int) -> np.ndarray:
        """Распределение размера выборки по кластерам"""
        # Заменяем NaN и бесконечности на 0
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Если все вероятности нулевые, распределяем равномерно
        if probs.sum() == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()
        
        sample_sizes = np.zeros_like(cluster_sizes, dtype=int)
        remaining_size = total_size
        
        # Распределяем пропорционально вероятностям с учетом ограничений
        while remaining_size > 0:
            allocations = (probs * remaining_size).astype(int)
            # Ограничиваем максимальным размером кластера
            allocations = np.minimum(allocations, cluster_sizes - sample_sizes)
            
            if allocations.sum() == 0:
                # Распределяем оставшиеся сэмплы по кластерам, которые еще могут принять точки
                available_mask = (cluster_sizes - sample_sizes) > 0
                if available_mask.any():
                    allocations = np.zeros_like(sample_sizes)
                    allocations[available_mask] = 1
                    allocations = np.minimum(allocations, cluster_sizes - sample_sizes)
                    allocations = np.minimum(allocations, remaining_size)
                else:
                    break
                    
            sample_sizes += allocations
            remaining_size -= allocations.sum()
        
        return sample_sizes
    
    def create_sampler(self, target_function: Callable, sample_size: int,
                     strategy: str = 'proportional', **kwargs) -> WeightedRandomSampler:
        """Создание PyTorch Sampler для DataLoader"""
        indices, weights = self.importance_sampling(
            target_function, sample_size, strategy, **kwargs
        )
        
        if not indices:
            raise ValueError("Не удалось создать выборку")
            
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        return WeightedRandomSampler(weights_tensor, len(indices), replacement=True)


class SamplingEvaluator:
    """Утилиты для оценки качества семплинга."""
    
    @staticmethod
    def estimate_expectation(samples: torch.Tensor, weights: List[float], 
                           target_function: Callable) -> float:
        """
        Оценка матожидания с помощью Importance Sampling.
        
        Формула из IS_LSH.pdf: E_p[f] ≈ (Σ w_i * f(x_i)) / (Σ w_i)
        """
        with torch.no_grad():
            target_values = target_function(samples)
            weighted_sum = (target_values * torch.tensor(weights, device=samples.device)).sum()
            weight_sum = torch.tensor(weights, device=samples.device).sum()
            
            if weight_sum == 0:
                return 0.0
                
            return (weighted_sum / weight_sum).item()
    
    @staticmethod
    def compare_distributions(original_data: torch.Tensor, sampled_data: torch.Tensor,
                            original_weights: Optional[List[float]] = None,
                            sampled_weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Сравнение распределений исходных данных и выборки.
        
        Метрики сравнения:
        - Разница средних: ||μ_original - μ_sampled||
        - Разница ковариационных матриц: ||Σ_original - Σ_sampled||_F
        """
        # Взвешенное или обычное среднее
        if original_weights is not None:
            original_weights_tensor = torch.tensor(original_weights, device=original_data.device)
            original_mean = (original_data * original_weights_tensor.unsqueeze(1)).sum(dim=0) / original_weights_tensor.sum()
        else:
            original_mean = original_data.mean(dim=0)
            
        if sampled_weights is not None:
            sampled_weights_tensor = torch.tensor(sampled_weights, device=sampled_data.device)
            sampled_mean = (sampled_data * sampled_weights_tensor.unsqueeze(1)).sum(dim=0) / sampled_weights_tensor.sum()
        else:
            sampled_mean = sampled_data.mean(dim=0)
        
        mean_diff = torch.norm(original_mean - sampled_mean).item()
        
        # Сравнение ковариационных матриц (только для некоррелированных весов)
        if original_weights is None and sampled_weights is None:
            original_cov = torch.cov(original_data.T)
            sampled_cov = torch.cov(sampled_data.T)
            cov_diff = torch.norm(original_cov - sampled_cov).item()
        else:
            cov_diff = float('nan')  # Для взвешенных данных ковариация сложнее
            
        return {
            'mean_difference': mean_diff,
            'covariance_difference': cov_diff
        }


def demo_all_features():
    """Демонстрация всех функций библиотеки"""
    torch.manual_seed(42)
    
    # Генерация тестовых данных
    data = torch.randn(1000, 128)  # 1000 векторов по 128 размеров
    query = torch.randn(128)
    
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ LSH С РАЗНЫМИ МЕТРИКАМИ И ПРОЕКЦИЯМИ")
    print("=" * 60)
    
    # Тестируем разные метрики и проекции
    configs = [
        ('euclidean', 'random', "Евклидово расстояние, случайные проекции"),
        ('cosine', 'orthogonal', "Косинусное расстояние, ортогональные проекции"),
        ('manhattan', 'random', "Манхеттенское расстояние, случайные проекции"),
    ]
    
    for distance_type, projection_type, description in configs:
        print(f"\n--- {description} ---")
        
        lsh = LSH(
            input_dim=128, 
            num_tables=5, 
            hash_size=8,
            distance_type=distance_type,
            projection_type=projection_type
        )
        lsh.add(data)
        
        indices, distances = lsh.query(query, k=5)
        print(f"Найдены соседи: {indices[:5]}")
        print(f"Расстояния: {distances[:5]}")
    
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ IMPORTANCE SAMPLING")
    print("=" * 60)
    
    # Определяем target function (например, норма вектора)
    def target_function(x):
        return torch.norm(x, dim=1)
    
    # Демонстрация LSH с семплингом
    print("\n--- LSH с Importance Sampling ---")
    lsh_sampler = LSHWithSampling(
        input_dim=128, 
        num_tables=5, 
        hash_size=8,
        distance_type='cosine',
        projection_type='orthogonal',
    )
    lsh_sampler.add(data)
    
    indices, weights = lsh_sampler.importance_sampling(
        target_function, 
        sample_size=100,
        strategy='variance_based'
    )
    
    print(f"Размер выборки: {len(indices)}")
    print(f"Пример весов: {weights[:5]}")
    
    # Оценка матожидания
    if len(indices) > 0:
        sampled_data = torch.stack([lsh_sampler.data_points[i] for i in indices])
        expectation = SamplingEvaluator.estimate_expectation(sampled_data, weights, target_function)
        print(f"Оценка матожидания: {expectation:.4f}")
    
    # Демонстрация PQ с семплингом
    print("\n--- PQ с Importance Sampling ---")
    pq_sampler = PQWithSampling(input_dim=128, num_subspaces=8, num_centroids=256)
    pq_sampler.train(data)
    
    codes = pq_sampler.encode(data)
    print(f"Размер закодированных данных: {codes.shape}")
    
    indices, distances = pq_sampler.search(query, codes, k=5)
    print(f"Найдены соседи: {indices}")
    print(f"Расстояния: {distances}")
    
    # Проверка реконструкции
    reconstructed = pq_sampler.decode(codes[:5])
    reconstruction_error = torch.mean((data[:5] - reconstructed.cpu())**2)
    print(f"Ошибка реконструкции: {reconstruction_error:.6f}")
    
    # Демонстрация семплинга PQ
    indices, weights = pq_sampler.importance_sampling(
        target_function,
        sample_size=100,
        strategy='residual_variance'
    )
    
    print(f"Размер выборки PQ: {len(indices)}")
    if len(indices) > 0:
        print(f"Пример весов PQ: {weights[:5]}")
    
    # Сравнение распределений
    if len(indices) > 0:
        sampled_data_pq = torch.stack([pq_sampler.data_points[i] for i in indices])
        comparison = SamplingEvaluator.compare_distributions(data, sampled_data_pq)
        print(f"Сравнение распределений: {comparison}")
    
    return lsh_sampler, pq_sampler


if __name__ == "__main__":
    # Запуск демонстрации
    lsh_sampler, pq_sampler = demo_all_features()
