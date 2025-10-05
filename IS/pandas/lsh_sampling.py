# lsh_sampling.py
"""
Модуль Locality-Sensitive Hashing (LSH) с поддержкой Importance Sampling.

Реализует:
- Базовый LSH с различными метриками расстояния и типами проекций
- LSH с поддержкой Importance Sampling для эффективной выборки данных
- Оптимизированные алгоритмы для работы с большими наборами данных

Математическая основа:
Семейство хэш-функций H называется (r, cr, p1, p2)-чувствительным для метрики d, если:
1. Если d(p, q) ≤ r, то Pr[h(p) = h(q)] ≥ p1
2. Если d(p, q) ≥ cr, то Pr[h(p) = h(q)] ≤ p2

AND-construction: p_and = p1^k (усиливает строгость)
OR-construction: p_or = 1 - (1 - p1)^L (увеличивает recall)
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from itertools import product
from typing import List, Tuple, Dict, Union, Optional, Callable
import warnings
from collections import defaultdict

from intelligent_caching import MemoryEfficientTensorStorage
from torch.utils.data import WeightedRandomSampler

class LSH:
    """
    Locality-Sensitive Hashing (LSH) с поддержкой GPU и различных метрик расстояния.
    
    Attributes:
        input_dim (int): Размерность входных векторов
        num_tables (int): Количество хэш-таблиц (L) - OR-construction
        hash_size (int): Количество хэш-функций на таблицу (k) - AND-construction
        bucket_width (float): Ширина бакета (w)
        distance_type (str): Тип расстояния ('euclidean', 'cosine', 'manhattan')
        projection_type (str): Тип проекций ('random', 'orthogonal', 'random_rotation')
        device (str): Устройство вычислений ('cuda', 'cpu', или 'auto')
        cache_enabled (bool): Включено ли кэширование на диск
        max_memory_points (int): Максимальное количество точек в оперативной памяти
        storage (MemoryEfficientTensorStorage): Хранилище для эффективного управления данными
        projections (torch.Tensor): Проекционные матрицы для хэширования
        biases (torch.Tensor): Случайные смещения для квантования
        buckets (list): Хэш-таблицы для хранения данных
        _normalize_for_cosine (bool): Флаг нормализации для косинусного расстояния
    """
    
    def __init__(self, input_dim: int, num_tables: int = 10, hash_size: int = 16, 
                 bucket_width: float = 1.0, distance_type: str = 'euclidean',
                 projection_type: str = 'random', device: str = 'auto',
                 cache_enabled: bool = True, max_memory_points: int = 10000):
        """
        Инициализация LSH индекса.
        
        Args:
            input_dim: Размерность входных векторов
            num_tables: Количество хэш-таблиц (L)
            hash_size: Количество хэш-функций на таблицу (k)
            bucket_width: Ширина бакета для квантования
            distance_type: Тип расстояния ('euclidean', 'cosine', 'manhattan')
            projection_type: Тип проекций ('random', 'orthogonal', 'random_rotation')
            device: Устройство вычислений ('auto', 'cuda', 'cpu')
            cache_enabled: Включено ли кэширование на диск
            max_memory_points: Максимальное количество точек в оперативной памяти
        """
        self.input_dim = input_dim
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.bucket_width = bucket_width
        self.distance_type = distance_type
        self.projection_type = projection_type
        self.cache_enabled = cache_enabled
        self.max_memory_points = max_memory_points
        
        # Автоматическое определение устройства вычислений
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Инициализация проекционных матриц
        self._init_projections()
        
        # Инициализация смещений для квантования (только для евклидова и манхеттенского расстояний)
        if distance_type in ['euclidean', 'manhattan']:
            self.biases = torch.rand(num_tables, hash_size, device=self.device) * bucket_width
        else:
            self.biases = None
            
        # Инициализация хэш-таблиц
        self.buckets = [{} for _ in range(num_tables)]
        
        # Инициализация системы хранения данных
        self.storage = MemoryEfficientTensorStorage(
            cache_enabled=cache_enabled,
            max_memory_points=max_memory_points
        )
        
        # Флаг нормализации для косинусного расстояния
        self._normalize_for_cosine = (distance_type == 'cosine')
        
    def _init_projections(self):
        """
        Инициализация проекционных матриц в зависимости от типа.
        
        Теоретическое обоснование:
        - Случайные проекции: a ~ N(0, I) для евклидова расстояния (p-стабильные распределения)
        - Ортогональные проекции: лучше сохраняют расстояния в пространстве
        - Случайные повороты: равномерное покрытие сферы для косинусного расстояния
        """
        if self.projection_type == 'random':
            # Случайные проекции из нормального распределения
            self.projections = torch.randn(
                self.num_tables, self.hash_size, self.input_dim, 
                device=self.device
            )
            
        elif self.projection_type == 'orthogonal':
            # Ортогональные проекции (лучше сохраняют расстояния)
            projections = []
            for _ in range(self.num_tables):
                # Генерируем случайную матрицу и ортогонализируем ее через QR-разложение
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
            
        # Специальная обработка для разных типов расстояний
        if self.distance_type == 'manhattan':
            # Для манхеттенского расстояния используем распределение Коши (1-стабильное)
            cauchy = torch.tensor(
                np.random.standard_cauchy(self.projections.shape),
                device=self.device, dtype=torch.float32
            )
            self.projections = cauchy
            
        elif self.distance_type == 'cosine':
            # Для косинусного расстояния нормализуем проекции
            self.projections = F.normalize(self.projections, p=2, dim=-1)
    
    def _normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Нормализация векторов для косинусного расстояния.
        
        Args:
            vectors: Входные векторы для нормализации
            
        Returns:
            Нормализованные векторы
        """
        if self._normalize_for_cosine:
            return F.normalize(vectors, p=2, dim=-1)
        return vectors
    
    def _get_data_point(self, idx: int) -> torch.Tensor:
        """
        Получение точки данных с автоматической загрузкой из хранилища.
        
        Args:
            idx: Индекс точки данных
            
        Returns:
            Тензор точки данных на нужном устройстве
        """
        return self.storage.get_tensor(idx, self.device)
    
    def compute_hashes(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Вычисление хэшей для векторов в зависимости от типа расстояния.
        
        Для евклидова расстояния (E2LSH):
        h(v) = floor((a·v + b) / w)
        
        Для косинусного расстояния:
        h(v) = sign(a·v)
        
        Args:
            vectors: Входные векторы для хэширования [batch_size, input_dim]
            
        Returns:
            Хэши векторов [batch_size, num_tables, hash_size]
        """
        batch_size = vectors.shape[0]
        
        # Нормализуем векторы если нужно (для косинусного расстояния)
        vectors = self._normalize_vectors(vectors)
        
        # Подготавливаем тензоры для батчевого вычисления
        # projections: [1, L, k, D], vectors_expanded: [B, 1, 1, D]
        projections = self.projections.unsqueeze(0)  
        vectors_expanded = vectors.unsqueeze(1).unsqueeze(2)  
        
        # Вычисляем скалярные произведения (проекции)
        dot_products = (projections * vectors_expanded).sum(dim=-1)  # [B, L, k]
        
        if self.distance_type == 'cosine':
            # Для косинусного расстояния используем знак проекции
            hashes = (dot_products > 0).long()
        else:
            # Для евклидова и манхеттенского используем квантование со смещениями
            hashes = torch.floor((dot_products + self.biases.unsqueeze(0)) / self.bucket_width)
        
        return hashes.long()
    
    def _compute_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """
        Вычисление расстояния между векторами в зависимости от типа метрики.
        
        Args:
            vec1: Первый вектор
            vec2: Второй вектор
            
        Returns:
            Расстояние между векторами
            
        Raises:
            ValueError: Если указан неизвестный тип расстояния
        """
        if self.distance_type == 'euclidean':
            return torch.norm(vec1 - vec2, dim=1)
        elif self.distance_type == 'manhattan':
            return torch.abs(vec1 - vec2).sum(dim=1)
        elif self.distance_type == 'cosine':
            return 1 - F.cosine_similarity(vec1, vec2)
        else:
            raise ValueError(f"Неизвестный тип расстояния: {self.distance_type}")
    
    def add(self, vectors: torch.Tensor, ids: Optional[List[int]] = None) -> None:
        """
        Добавление векторов в индекс с поддержкой эффективного хранения.
        
        Args:
            vectors: Векторы для добавления [batch_size, input_dim]
            ids: Опциональные идентификаторы векторов
            
        Raises:
            ValueError: Если размерность векторов не совпадает с ожидаемой
            ValueError: Если количество ID не совпадает с количеством векторов
        """
        # Обработка случая с одним вектором
        if len(vectors.shape) == 1:
            vectors = vectors.unsqueeze(0)
            
        # Проверка размерности входных данных
        if vectors.shape[1] != self.input_dim:
            raise ValueError(f"Ожидалась размерность {self.input_dim}, получена {vectors.shape[1]}")
            
        # Перемещаем данные на нужное устройство
        vectors = vectors.to(self.device)
        batch_size = vectors.shape[0]
        
        # Сохраняем вектора в хранилище
        tensor_list = [vectors[i] for i in range(batch_size)]
        stored_indices = self.storage.add_tensors(tensor_list)
        
        # Используем сохраненные индексы если ID не предоставлены
        if ids is None:
            ids = stored_indices
        elif len(ids) != batch_size:
            raise ValueError("Количество ID должно совпадать с количеством векторов")
        
        # Вычисляем хэши для всех векторов
        hashes = self.compute_hashes(vectors)  # [B, L, k]
        
        # Добавляем векторы в соответствующие бакеты хэш-таблиц
        for i in range(batch_size):
            vector_id = ids[i]
            for table_idx in range(self.num_tables):
                # Создаем ключ бакета из хэша
                hash_key = tuple(hashes[i, table_idx].cpu().numpy())
                
                # Создаем бакет если он не существует
                if hash_key not in self.buckets[table_idx]:
                    self.buckets[table_idx][hash_key] = []
                
                # Добавляем ID вектора в бакет
                self.buckets[table_idx][hash_key].append(vector_id)
    
    def _generate_probes(self, main_hash: tuple, num_probes: int) -> List[np.ndarray]:
        """
        Генерация соседних бакетов для multi-probe поиска.
        
        Multi-probe LSH: исследует соседние бакеты для увеличения recall 
        без добавления дополнительных хэш-таблиц.
        
        Args:
            main_hash: Основной хэш для исследования
            num_probes: Количество соседних бакетов для генерации
            
        Returns:
            Список соседних хэшей
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
        Поиск k ближайших соседей с поддержкой кэшированных данных.
        
        Вероятностные гарантии:
        P[success] ≥ 1 - (1 - p1^k)^L
        P[false positive] ≤ p2^k * L
        
        Args:
            query_vector: Вектор запроса [input_dim]
            k: Количество ближайших соседей для поиска
            num_probes: Количество соседних бакетов для исследования
            
        Returns:
            Кортеж (индексы ближайших соседей, расстояния)
            
        Raises:
            ValueError: Если размерность вектора запроса не совпадает с ожидаемой
        """
        # Обработка случая с одним вектором запроса
        if len(query_vector.shape) == 1:
            query_vector = query_vector.unsqueeze(0)
            
        # Проверка размерности вектора запроса
        if query_vector.shape[1] != self.input_dim:
            raise ValueError(f"Ожидалась размерность {self.input_dim}, получена {query_vector.shape[1]}")
            
        # Перемещаем запрос на нужное устройство и нормализуем
        query_vector = query_vector.to(self.device)
        query_vector = self._normalize_vectors(query_vector)
        
        # Вычисляем хэши для запроса
        query_hashes = self.compute_hashes(query_vector)  # [1, L, k]
        
        # Собираем кандидатов из бакетов
        candidates = set()
        
        for table_idx in range(self.num_tables):
            # Основной бакет запроса
            main_hash = tuple(query_hashes[0, table_idx].cpu().numpy())
            
            # Добавляем кандидатов из основного бакета
            if main_hash in self.buckets[table_idx]:
                candidates.update(self.buckets[table_idx][main_hash])
            
            # Multi-probe: добавляем кандидатов из соседних бакетов
            if num_probes > 1:
                for probe in self._generate_probes(main_hash, num_probes-1):
                    probe_key = tuple(probe)
                    if probe_key in self.buckets[table_idx]:
                        candidates.update(self.buckets[table_idx][probe_key])
        
        # Если кандидатов не найдено, возвращаем пустые результаты
        if not candidates:
            return [], []
            
        # Загружаем векторы кандидатов с поддержкой кэширования
        candidate_vectors = []
        for i in candidates:
            candidate_vectors.append(self._get_data_point(i))
        
        candidate_vectors = torch.stack(candidate_vectors).to(self.device)
        query_expanded = query_vector.expand(len(candidates), -1)
        
        # Вычисляем расстояния до кандидатов в зависимости от типа метрики
        distances = self._compute_distance(query_expanded, candidate_vectors)
        
        # Выбираем k ближайших соседей
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
    
    Attributes:
        bucket_stats (defaultdict): Статистика по бакетам для семплинга
        total_points (int): Общее количество точек данных
    """
    
    def __init__(self, input_dim: int, num_tables: int = 10, hash_size: int = 16, 
                 bucket_width: float = 1.0, distance_type: str = 'euclidean',
                 projection_type: str = 'random', device: str = 'auto',
                 cache_enabled: bool = True, max_memory_points: int = 10000):
        """
        Инициализация LSH с поддержкой семплинга.
        
        Args:
            input_dim: Размерность входных векторов
            num_tables: Количество хэш-таблиц
            hash_size: Количество хэш-функций на таблицу
            bucket_width: Ширина бакета для квантования
            distance_type: Тип расстояния
            projection_type: Тип проекций
            device: Устройство вычислений
            cache_enabled: Включено ли кэширование на диск
            max_memory_points: Максимальное количество точек в оперативной памяти
        """
        super().__init__(input_dim, num_tables, hash_size, bucket_width, 
                        distance_type, projection_type, device, cache_enabled, max_memory_points)
        
        # Статистика бакетов для семплинга
        self.bucket_stats = defaultdict(lambda: {'size': 0, 'ids': []})
        self.total_points = 0
        
    def add(self, vectors: torch.Tensor, ids: Optional[List[int]] = None) -> None:
        """
        Добавление векторов с сохранением статистики бакетов и поддержкой кэширования.
        
        Args:
            vectors: Векторы для добавления
            ids: Опциональные идентификаторы векторов
        """
        if len(vectors.shape) == 1:
            vectors = vectors.unsqueeze(0)
            
        vectors = vectors.to(self.device)
        batch_size = vectors.shape[0]
        
        # Сохраняем вектора в хранилище
        tensor_list = [vectors[i] for i in range(batch_size)]
        stored_indices = self.storage.add_tensors(tensor_list)
        
        if ids is None:
            ids = stored_indices
        
        # Обновляем общее количество точек
        self.total_points += batch_size
        
        # Вычисляем хэши для векторов
        hashes = self.compute_hashes(vectors)
        
        for i in range(batch_size):
            vector_id = ids[i]
            
            for table_idx in range(self.num_tables):
                hash_key = tuple(hashes[i, table_idx].cpu().numpy())
                
                # Добавляем в бакет хэш-таблицы
                if hash_key not in self.buckets[table_idx]:
                    self.buckets[table_idx][hash_key] = []
                
                self.buckets[table_idx][hash_key].append(vector_id)
                
                # Обновляем статистику бакетов для семплинга
                bucket_id = f"table_{table_idx}_{hash_key}"
                self.bucket_stats[bucket_id]['size'] += 1
                self.bucket_stats[bucket_id]['ids'].append(vector_id)
    
    def _batch_get_data_points(self, indices: List[int]) -> torch.Tensor:
        """
        Оптимизированная батчевая загрузка данных.
        
        Args:
            indices: Список индексов для загрузки
            
        Returns:
            Батч тензоров
        """
        tensors = []
        for idx in indices:
            tensors.append(self._get_data_point(idx))
        return torch.stack(tensors)
    
    def _precompute_target_function(self, target_function: Callable, 
                                  batch_size: int = 1000) -> Dict[str, float]:
        """
        Предварительное вычисление целевой функции для всех бакетов.
        
        Args:
            target_function: Функция для оценки важности
            batch_size: Размер батча для обработки
            
        Returns:
            Словарь со статистикой по бакетам
        """
        bucket_target_stats = {}
        
        for bucket_id, stats in self.bucket_stats.items():
            if stats['size'] == 0:
                continue
                
            # Батчевая обработка для больших бакетов
            if stats['size'] > batch_size:
                target_values = []
                for i in range(0, stats['size'], batch_size):
                    batch_indices = stats['ids'][i:i + batch_size]
                    batch_data = self._batch_get_data_points(batch_indices)
                    with torch.no_grad():
                        target_values.append(target_function(batch_data))
                
                target_values = torch.cat(target_values)
            else:
                # Обработка маленьких бакетов за один раз
                bucket_data = self._batch_get_data_points(stats['ids'])
                with torch.no_grad():
                    target_values = target_function(bucket_data)
            
            # Сохраняем статистику для бакета
            if len(target_values) > 1:
                variance = target_values.var().item()
                mean = target_values.mean().item()
            else:
                variance = 0.0
                mean = target_values.item() if len(target_values) == 1 else 0.0
                
            bucket_target_stats[bucket_id] = {
                'variance': variance,
                'mean': mean,
                'size': stats['size']
            }
        
        return bucket_target_stats
    
    def importance_sampling(self, target_function: Callable, sample_size: int, 
                          strategy: str = 'proportional', temperature: float = 1.0, 
                          max_bucket_size: Optional[int] = None,
                          precompute_batch_size: int = 1000) -> Tuple[List[int], List[float]]:
        """
        Importance Sampling на основе LSH бакетов с поддержкой кэшированных данных.
        
        Стратегии выборки (IS_LSH.pdf, раздел 3.2):
        - Пропорциональная: n_i ∝ |B_i| * S/N
        - Сбалансированная: n_i = min(c, |B_i|) * S/N  
        - На основе дисперсии: n_i ∝ |B_i| * σ_B_i
        
        Args:
            target_function: Функция f(x) для оценки важности
            sample_size: Размер выборки
            strategy: Стратегия выборки ('proportional', 'balanced', 'variance_based')
            temperature: Параметр для смягчения распределения
            max_bucket_size: Максимальный размер выборки из одного бакета
            precompute_batch_size: Размер батча для предварительных вычислений
            
        Returns:
            Кортеж (индексы выбранных точек, веса для importance sampling)
            
        Raises:
            ValueError: Если нет данных для семплинга
            ValueError: Если указана неизвестная стратегия
        """
        if max_bucket_size is None:
            max_bucket_size = max(1, sample_size // 10)
            
        if not self.bucket_stats:
            raise ValueError("Нет данных для семплинга. Сначала добавьте векторы.")
        
        print(f"Начинаем оптимизированный семплинг для {len(self.bucket_stats)} бакетов...")
        
        # Предварительное вычисление статистики для стратегии на основе дисперсии
        if strategy == 'variance_based':
            print("Предварительное вычисление статистики бакетов...")
            bucket_target_stats = self._precompute_target_function(target_function, precompute_batch_size)
        else:
            bucket_target_stats = None
        
        # Вычисляем важность для каждого бакета согласно выбранной стратегии
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
                if bucket_target_stats and bucket_id in bucket_target_stats:
                    importance = bucket_target_stats[bucket_id]['variance'] * bucket_size
                else:
                    # Запасной вариант если статистика не предвычислена
                    bucket_vectors = self._batch_get_data_points(stats['ids'])
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
        
        # Нормализуем важности с температурой (softmax с температурой)
        importances = torch.tensor(list(bucket_importances.values()))
        probs = F.softmax(importances / temperature, dim=0)
        
        bucket_ids = list(bucket_importances.keys())
        
        # Эффективное распределение размера выборки по бакетам
        selected_buckets = self._distribute_samples_optimized(
            bucket_ids, probs.cpu().numpy(), 
            [bucket_sizes[b_id] for b_id in bucket_ids],
            sample_size, max_bucket_size
        )
        
        # Батчевый сбор семплов из выбранных бакетов
        sampled_indices, sampling_weights = self._collect_samples_optimized(
            selected_buckets, bucket_importances, bucket_sizes
        )
        
        print(f"Семплинг завершен: отобрано {len(sampled_indices)} samples")
        return sampled_indices, sampling_weights
    
    def _distribute_samples_optimized(self, bucket_ids: List[str], probs: np.ndarray,
                                    bucket_sizes: List[int], total_size: int,
                                    max_bucket_size: int) -> List[Tuple[str, int]]:
        """
        Оптимизированное распределение семплов по бакетам.
        
        Args:
            bucket_ids: Список идентификаторов бакетов
            probs: Вероятности выборки для каждого бакета
            bucket_sizes: Размеры бакетов
            total_size: Общий размер выборки
            max_bucket_size: Максимальный размер выборки из одного бакета
            
        Returns:
            Список пар (bucket_id, sample_size) для выбранных бакетов
        """
        selected_buckets = []
        remaining_size = total_size
        
        # Сортируем бакеты по вероятности (по убыванию) для эффективного распределения
        sorted_indices = np.argsort(probs)[::-1]
        
        for idx in sorted_indices:
            if remaining_size <= 0:
                break
                
            bucket_id = bucket_ids[idx]
            bucket_size = bucket_sizes[idx]
            prob = probs[idx]
            
            # Вычисляем размер выборки для бакета
            bucket_sample_size = min(
                math.ceil(prob * total_size),
                bucket_size,
                max_bucket_size,
                remaining_size
            )
            
            if bucket_sample_size > 0:
                selected_buckets.append((bucket_id, bucket_sample_size))
                remaining_size -= bucket_sample_size
        
        # Если остались нераспределенные семплы, распределяем равномерно по оставшимся бакетам
        if remaining_size > 0:
            for bucket_id, bucket_size in zip(bucket_ids, bucket_sizes):
                if remaining_size <= 0:
                    break
                    
                # Пропускаем уже выбранные бакеты
                if bucket_id not in [b[0] for b in selected_buckets]:
                    additional_size = min(bucket_size, remaining_size)
                    if additional_size > 0:
                        selected_buckets.append((bucket_id, additional_size))
                        remaining_size -= additional_size
        
        return selected_buckets
    
    def _collect_samples_optimized(self, selected_buckets: List[Tuple[str, int]],
                                 bucket_importances: Dict[str, float],
                                 bucket_sizes: Dict[str, int]) -> Tuple[List[int], List[float]]:
        """
        Оптимизированный сбор семплов из выбранных бакетов.
        
        Args:
            selected_buckets: Список выбранных бакетов и размеров выборки
            bucket_importances: Важности бакетов
            bucket_sizes: Размеры бакетов
            
        Returns:
            Кортеж (индексы выбранных точек, веса для importance sampling)
        """
        sampled_indices = []
        sampling_weights = []
        
        total_importance = sum(bucket_importances.values())
        
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
        """
        Создание PyTorch Sampler для DataLoader.
        
        Args:
            target_function: Функция для оценки важности
            sample_size: Размер выборки
            strategy: Стратегия выборки
            **kwargs: Дополнительные аргументы для importance_sampling
            
        Returns:
            WeightedRandomSampler для использования в DataLoader
            
        Raises:
            ValueError: Если не удалось создать выборку
        """
        indices, weights = self.importance_sampling(
            target_function, sample_size, strategy, **kwargs
        )
        
        if not indices:
            raise ValueError("Не удалось создать выборку")
            
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        
        return WeightedRandomSampler(weights_tensor, len(indices), replacement=True)