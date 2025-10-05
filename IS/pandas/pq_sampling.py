# pq_sampling.py
"""
Модуль Product Quantization с поддержкой Importance Sampling.

Реализует:
- Базовый Product Quantizer для эффективного сжатия векторов
- PQ с поддержкой Importance Sampling для интеллектуальной выборки данных
- Оптимизированные алгоритмы для работы с большими наборами данных

Математическая основа (PQ_IS.pdf, раздел 2):
Цель PQ - найти отображение q: ℝ^D → C, минимизирующее ошибку квантования:
min_q Σ ||x_i - q(x_i)||^2

Пространство разбивается на M подпространств, каждое квантуется независимо.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Callable
import warnings
import math

from intelligent_caching import MemoryEfficientTensorStorage
from torch.utils.data import WeightedRandomSampler


class ProductQuantizer:
    """
    Product Quantization с поддержкой GPU для эффективного сжатия и поиска векторов.
    
    Attributes:
        input_dim (int): Размерность входных векторов
        num_subspaces (int): Количество подпространств (M)
        subspace_dim (int): Размерность каждого подпространства
        num_centroids (int): Количество центроидов в каждом подпространстве (K)
        device (str): Устройство вычислений
        cache_enabled (bool): Включено ли кэширование на диск
        max_memory_points (int): Максимальное количество точек в оперативной памяти
        codebooks (torch.Tensor): Кодбуки для каждого подпространства [M, K, subspace_dim]
        is_trained (bool): Обучена ли модель
        storage (MemoryEfficientTensorStorage): Хранилище для эффективного управления данными
    """
    
    def __init__(self, input_dim: int, num_subspaces: int = 8, 
                 num_centroids: int = 256, device: str = 'auto',
                 cache_enabled: bool = True, max_memory_points: int = 10000):
        """
        Инициализация Product Quantizer.
        
        Args:
            input_dim: Размерность входных векторов
            num_subspaces: Количество подпространств (M)
            num_centroids: Количество центроидов в каждом подпространстве (K)
            device: Устройство вычислений
            cache_enabled: Включено ли кэширование на диск
            max_memory_points: Максимальное количество точек в оперативной памяти
            
        Raises:
            AssertionError: Если input_dim не делится на num_subspaces
        """
        assert input_dim % num_subspaces == 0, "input_dim должен делиться на num_subspaces"
        
        self.input_dim = input_dim
        self.num_subspaces = num_subspaces
        self.subspace_dim = input_dim // num_subspaces
        self.num_centroids = num_centroids
        self.cache_enabled = cache_enabled
        self.max_memory_points = max_memory_points
        
        # Автоматическое определение устройства вычислений
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.codebooks = None
        self.is_trained = False
        
        # Инициализация системы хранения данных
        self.storage = MemoryEfficientTensorStorage(
            cache_enabled=cache_enabled,
            max_memory_points=max_memory_points
        )
        
    def train(self, data: torch.Tensor, max_iter: int = 100, 
              batch_size: int = 1000) -> torch.Tensor:
        """
        Обучение кодбуков с помощью k-means с поддержкой кэширования.
        
        Args:
            data: Данные для обучения [N, input_dim]
            max_iter: Максимальное количество итераций k-means
            batch_size: Размер батча для мини-батч k-means
            
        Returns:
            Обученные кодбуки [num_subspaces, num_centroids, subspace_dim]
            
        Raises:
            ValueError: Если размерность данных не совпадает с ожидаемой
        """
        if len(data.shape) != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"Ожидалась размерность (N, {self.input_dim})")
            
        data = data.to(self.device)
        num_samples = data.shape[0]
        
        # Сохраняем данные в хранилище для эффективного доступа
        tensor_list = [data[i] for i in range(num_samples)]
        self.storage.add_tensors(tensor_list)
        
        # Разделяем данные на подпространства
        data_split = data.view(num_samples, self.num_subspaces, self.subspace_dim)
        
        # Инициализируем кодбуки
        self.codebooks = torch.zeros(self.num_subspaces, self.num_centroids, 
                                   self.subspace_dim, device=self.device)
        
        # Обучаем k-means для каждого подпространства независимо
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
        """
        Приближенный k-means с мини-батчами для больших datasets.
        
        Args:
            data: Данные для кластеризации
            max_iter: Максимальное количество итераций
            batch_size: Размер мини-батча
            
        Returns:
            Центроиды кластеров
        """
        num_samples = data.shape[0]
        
        # Инициализация центроидов случайными точками из данных
        indices = torch.randperm(num_samples)[:self.num_centroids]
        centroids = data[indices].clone()
        
        for iteration in range(max_iter):
            # Случайный мини-батч из данных
            batch_indices = torch.randint(0, num_samples, (batch_size,))
            batch_data = data[batch_indices]
            
            # Находим ближайшие центроиды для каждой точки батча
            distances = torch.cdist(batch_data, centroids)
            closest = torch.argmin(distances, dim=1)
            
            # Обновляем центроиды на основе точек батча
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.num_centroids, device=self.device)
            
            for i in range(batch_size):
                centroid_idx = closest[i]
                new_centroids[centroid_idx] += batch_data[i]
                counts[centroid_idx] += 1
            
            # Избегаем деления на ноль для пустых кластеров
            mask = counts > 0
            if mask.any():
                new_centroids[mask] /= counts[mask].unsqueeze(1)
            
            # Заменяем пустые центроиды случайными точками из данных
            empty_mask = counts == 0
            if empty_mask.any():
                new_indices = torch.randint(0, num_samples, (empty_mask.sum(),))
                new_centroids[empty_mask] = data[new_indices]
            
            # Проверяем сходимость (изменение центроидов незначительно)
            if torch.allclose(centroids, new_centroids, rtol=1e-6):
                break
                
            centroids = new_centroids
            
        return centroids
    
    def _exact_kmeans(self, data: torch.Tensor, max_iter: int) -> torch.Tensor:
        """
        Точный k-means через sklearn (для CPU).
        
        Args:
            data: Данные для кластеризации
            max_iter: Максимальное количество итераций
            
        Returns:
            Центроиды кластеров
        """
        data_cpu = data.cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.num_centroids, max_iter=max_iter, n_init=1)
        kmeans.fit(data_cpu)
        
        return torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32)
    
    def _get_data_point(self, idx: int) -> torch.Tensor:
        """
        Получение точки данных с автоматической загрузкой из хранилища.
        
        Args:
            idx: Индекс точки данных
            
        Returns:
            Тензор точки данных на нужном устройстве
        """
        return self.storage.get_tensor(idx, self.device)
    
    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Кодирование векторов в PQ-коды.
        
        Args:
            vectors: Векторы для кодирования [N, input_dim]
            
        Returns:
            PQ-коды [N, num_subspaces]
            
        Raises:
            ValueError: Если модель не обучена
            ValueError: Если размерность векторов не совпадает с ожидаемой
        """
        if not self.is_trained:
            raise ValueError("Сначала обучите модель с помощью train()")
            
        if len(vectors.shape) != 2 or vectors.shape[1] != self.input_dim:
            raise ValueError(f"Ожидалась размерность (N, {self.input_dim})")
            
        vectors = vectors.to(self.device)
        batch_size = vectors.shape[0]
        
        # Разделяем векторы на подпространства
        vectors_split = vectors.view(batch_size, self.num_subspaces, self.subspace_dim)
        
        # Инициализируем матрицу кодов
        codes = torch.zeros(batch_size, self.num_subspaces, dtype=torch.long, device=self.device)
        
        # Кодируем каждое подпространство независимо
        for subspace_idx in range(self.num_subspaces):
            subspace_vectors = vectors_split[:, subspace_idx, :]
            subspace_codebook = self.codebooks[subspace_idx]
            
            # Вычисляем расстояния до всех центроидов подпространства
            distances = torch.cdist(subspace_vectors, subspace_codebook)
            
            # Находим индексы ближайших центроидов
            codes[:, subspace_idx] = torch.argmin(distances, dim=1)
        
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Декодирование PQ-кодов обратно в векторы.
        
        Args:
            codes: PQ-коды для декодирования [N, num_subspaces]
            
        Returns:
            Реконструированные векторы [N, input_dim]
        """
        batch_size = codes.shape[0]
        
        # Инициализируем матрицу реконструированных векторов
        reconstructed = torch.zeros(batch_size, self.input_dim, device=self.device)
        
        # Декодируем каждое подпространство независимо
        for subspace_idx in range(self.num_subspaces):
            subspace_codes = codes[:, subspace_idx]
            # Извлекаем соответствующие центроиды из кодбука
            subspace_vectors = self.codebooks[subspace_idx][subspace_codes]
            
            # Записываем декодированные подпространства в результирующий вектор
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
            query_vector: Вектор запроса [input_dim]
            encoded_data: Закодированные данные [N, num_subspaces]
            k: Количество ближайших соседей
            
        Returns:
            Кортеж (индексы ближайших соседей, расстояния)
            
        Raises:
            ValueError: Если модель не обучена
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
            
            # Евклидово расстояние от запроса до всех центроидов подпространства
            distances = torch.norm(codebook_subspace - query_subspace, dim=1)
            distance_tables[subspace_idx] = distances
        
        # Вычисляем расстояния до всех закодированных векторов
        total_distances = torch.zeros(num_samples, device=self.device)
        
        for subspace_idx in range(self.num_subspaces):
            subspace_codes = encoded_data[:, subspace_idx]
            # Суммируем расстояния по всем подпространствам
            total_distances += distance_tables[subspace_idx, subspace_codes]
        
        # Находим k ближайших соседей
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
    
    Attributes:
        cluster_stats (dict): Статистика по кластерам PQ
        encoded_data (torch.Tensor): Закодированные данные
        total_points (int): Общее количество точек данных
    """
    
    def __init__(self, input_dim: int, num_subspaces: int = 8, 
                 num_centroids: int = 256, device: str = 'auto',
                 cache_enabled: bool = True, max_memory_points: int = 10000):
        """
        Инициализация PQ с поддержкой семплинга.
        
        Args:
            input_dim: Размерность входных векторов
            num_subspaces: Количество подпространств
            num_centroids: Количество центроидов в каждом подпространстве
            device: Устройство вычислений
            cache_enabled: Включено ли кэширование на диск
            max_memory_points: Максимальное количество точек в оперативной памяти
        """
        super().__init__(input_dim, num_subspaces, num_centroids, device, cache_enabled, max_memory_points)
        self.cluster_stats = None
        self.encoded_data = None
        self.total_points = 0
        
    def train(self, data: torch.Tensor, max_iter: int = 100, 
              batch_size: int = 1000) -> torch.Tensor:
        """
        Обучение с сохранением статистики кластеров и поддержкой кэширования.
        
        Args:
            data: Данные для обучения
            max_iter: Максимальное количество итераций k-means
            batch_size: Размер батча для мини-батч k-means
            
        Returns:
            Обученные кодбуки
        """
        # Вызываем обучение родительского класса
        result = super().train(data, max_iter, batch_size)
        
        # Сохраняем закодированные данные для семплинга
        self.encoded_data = self.encode(data)
        self.total_points = data.shape[0]
        
        # Вычисляем статистику кластеров для семплинга
        self._compute_cluster_stats()
        
        return result
    
    def _compute_cluster_stats(self) -> None:
        """Вычисление статистики по кластерам PQ для семплинга."""
        if self.encoded_data is None:
            return
            
        num_samples = self.encoded_data.shape[0]
        self.cluster_stats = {}
        
        # Для каждого подпространства считаем статистику кластеров
        for subspace_idx in range(self.num_subspaces):
            subspace_codes = self.encoded_data[:, subspace_idx]
            # Находим уникальные коды и их частоты
            unique_codes, counts = torch.unique(subspace_codes, return_counts=True)
            
            self.cluster_stats[subspace_idx] = {
                'codes': unique_codes,
                'sizes': counts,
                'centroids': self.codebooks[subspace_idx]
            }
    
    def importance_sampling(self, target_function: Callable, sample_size: int,
                          strategy: str = 'proportional', use_residuals: bool = True,
                          temperature: float = 1.0,
                          precompute_batch_size: int = 1000) -> Tuple[List[int], List[float]]:
        """
        Importance Sampling на основе PQ кластеров.
        
        Алгоритм соответствует PQ_IS.pdf, Algorithm 1:
        1. Оценить μ_l, Σ_l для каждого кластера
        2. Вычислить важность I_l = Var_{x∼q_l}[f(x)]
        3. Нормировать веса P(l) = (w_l * I_l) / Σ_j w_j * I_j
        4. Выбирать кластер l ∼ P(l) и генерировать x_s ∼ q_l(x)
        5. Вычислять вес w_s = p(x_s)/q(x_s)
        
        Args:
            target_function: Функция f(x) для оценки важности
            sample_size: Размер выборки
            strategy: Стратегия выборки ('proportional', 'residual_variance', 'cluster_density')
            use_residuals: Использовать ли остатки для вычисления важности
            temperature: Параметр для смягчения распределения
            precompute_batch_size: Размер батча для предварительных вычислений
            
        Returns:
            Кортеж (индексы выбранных точек, веса для importance sampling)
            
        Raises:
            ValueError: Если модель не обучена
        """
        if self.cluster_stats is None:
            raise ValueError("Сначала обучите модель с помощью train()")
            
        print(f"Начинаем оптимизированный PQ семплинг...")
        
        # Используем первое подпространство для семплинга (можно расширить на все)
        selected_subspace = 0
        stats = self.cluster_stats[selected_subspace]
        
        # Оптимизированное вычисление важностей кластеров
        cluster_importances = self._compute_cluster_importances_optimized(
            selected_subspace, target_function, strategy, use_residuals, precompute_batch_size
        )
        
        # Обработка случая когда не удалось вычислить важности
        if cluster_importances is None or len(cluster_importances) == 0:
            warnings.warn("Не удалось вычислить важности кластеров, используется равномерное распределение")
            cluster_importances = torch.ones(len(stats['codes']), device=self.device)
        
        # Обработка некорректных значений (NaN, infinity)
        cluster_importances = torch.nan_to_num(cluster_importances, nan=1.0, posinf=1.0, neginf=0.0)
        cluster_importances = torch.clamp(cluster_importances, min=0.0)
        
        # Если все важности нулевые, используем равномерное распределение
        if cluster_importances.sum() == 0:
            cluster_importances = torch.ones_like(cluster_importances)
        
        # Нормализация с температурой (softmax с температурой)
        probs = F.softmax(cluster_importances / temperature, dim=0)
        
        # Распределение размера выборки по кластерам
        cluster_sizes = stats['sizes'].cpu().numpy()
        cluster_sample_sizes = self._distribute_sample_size_optimized(
            probs.cpu().numpy(), cluster_sizes, sample_size
        )
        
        # Оптимизированный сбор семплов из кластеров
        sampled_indices, sampling_weights = self._collect_cluster_samples_optimized(
            selected_subspace, stats, cluster_sample_sizes, probs, cluster_importances
        )
        
        print(f"PQ семплинг завершен: отобрано {len(sampled_indices)} samples")
        return sampled_indices, sampling_weights
    
    def _compute_cluster_importances_optimized(self, subspace_idx: int, target_function: Callable,
                                             strategy: str, use_residuals: bool, 
                                             batch_size: int) -> Optional[torch.Tensor]:
        """
        Оптимизированное вычисление важностей кластеров.
        
        Args:
            subspace_idx: Индекс подпространства
            target_function: Функция для оценки важности
            strategy: Стратегия выборки
            use_residuals: Использовать ли остатки
            batch_size: Размер батча для обработки
            
        Returns:
            Тензор важностей кластеров или None в случае ошибки
        """
        stats = self.cluster_stats[subspace_idx]
        importances = []
        
        # Кэш для данных кластеров (оптимизация повторных загрузок)
        cluster_data_cache = {}
        
        for code in stats['codes']:
            # Находим точки в кластере
            mask = self.encoded_data[:, subspace_idx] == code
            cluster_indices = torch.where(mask)[0]
            
            if len(cluster_indices) == 0:
                importances.append(0.0)
                continue
            
            # Батчевая загрузка данных кластера с кэшированием
            cluster_points = []
            for i in range(0, len(cluster_indices), batch_size):
                batch_indices = cluster_indices[i:i + batch_size]
                batch_points = []
                for idx in batch_indices:
                    batch_points.append(self._get_data_point(idx.item()))
                cluster_points.append(torch.stack(batch_points))
            
            if cluster_points:
                cluster_data = torch.cat(cluster_points)
                cluster_data_cache[code.item()] = cluster_data
            else:
                cluster_data_cache[code.item()] = None
                
            # Вычисляем важность в зависимости от стратегии
            if strategy == 'proportional':
                # Важность пропорциональна размеру кластера
                importance = len(cluster_indices)
            elif strategy == 'residual_variance' and use_residuals:
                # Дисперсия остатков относительно центроида
                if cluster_data_cache[code.item()] is not None:
                    importance = self._compute_residual_variance_optimized(
                        cluster_data_cache[code.item()], stats['centroids'][code], 
                        subspace_idx, target_function
                    )
                else:
                    importance = len(cluster_indices)
            elif strategy == 'cluster_density':
                # Плотность кластера
                if cluster_data_cache[code.item()] is not None:
                    importance = self._compute_cluster_density(cluster_data_cache[code.item()])
                else:
                    importance = len(cluster_indices)
            else:
                # По умолчанию используем размер кластера
                importance = len(cluster_indices)
                
            importances.append(importance)
        
        return torch.tensor(importances, device=self.device)

    def _compute_residual_variance_optimized(self, cluster_points: torch.Tensor, centroid: torch.Tensor,
                                           subspace_idx: int, target_function: Callable) -> float:
        """
        Оптимизированное вычисление дисперсии остатков в кластере.
        
        Args:
            cluster_points: Точки кластера
            centroid: Центроид кластера
            subspace_idx: Индекс подпространства
            target_function: Функция для оценки важности
            
        Returns:
            Важность кластера на основе дисперсии остатков
        """
        if len(cluster_points) < 2:
            return len(cluster_points)  # Для кластеров с 1 точкой используем размер
        
        # Вычисляем индексы подпространства в полном векторе
        start_idx = subspace_idx * self.subspace_dim
        end_idx = start_idx + self.subspace_dim
        
        # Берем только часть векторов, соответствующую подпространству
        cluster_points_subspace = cluster_points[:, start_idx:end_idx]
        
        # Вычисляем остатки (расстояние до центроида в подпространстве)
        residuals = torch.norm(cluster_points_subspace - centroid, dim=1)
        
        # Оцениваем важность через target_function (на полных векторах)
        with torch.no_grad():
            target_vals = target_function(cluster_points)
            
            # Комбинируем дисперсию остатков и целевой функции
            if len(target_vals) > 1:
                residual_var = residuals.var().item()
                target_var = target_vals.var().item()
                importance = residual_var * target_var * len(cluster_points)
            else:
                importance = len(cluster_points)
            
        return importance
    
    def _compute_cluster_density(self, cluster_points: torch.Tensor) -> float:
        """
        Вычисление плотности кластера через среднее расстояние до центроида.
        
        Args:
            cluster_points: Точки кластера
            
        Returns:
            Плотность кластера (количество точек / (1 + среднее расстояние))
        """
        if len(cluster_points) == 0:
            return 0.0
            
        # Вычисляем центроид кластера
        centroid = cluster_points.mean(dim=0)
        
        # Вычисляем среднее расстояние до центроида
        avg_distance = torch.norm(cluster_points - centroid, dim=1).mean().item()
        
        # Чем меньше среднее расстояние, тем выше плотность
        density = len(cluster_points) / (1.0 + avg_distance)
        return density
    
    def _distribute_sample_size_optimized(self, probs: np.ndarray, cluster_sizes: np.ndarray,
                                        total_size: int) -> np.ndarray:
        """
        Оптимизированное распределение размера выборки по кластерам.
        
        Args:
            probs: Вероятности выборки для каждого кластера
            cluster_sizes: Размеры кластеров
            total_size: Общий размер выборки
            
        Returns:
            Массив размеров выборки для каждого кластера
        """
        # Обработка некорректных значений вероятностей
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Если все вероятности нулевые, распределяем равномерно
        if probs.sum() == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()
        
        sample_sizes = np.zeros_like(cluster_sizes, dtype=int)
        remaining_size = total_size
        
        # Оптимизированное распределение - начинаем с самых вероятных кластеров
        sorted_indices = np.argsort(probs)[::-1]
        
        for idx in sorted_indices:
            if remaining_size <= 0:
                break
                
            allocation = min(
                int(probs[idx] * total_size),
                cluster_sizes[idx],
                remaining_size
            )
            
            if allocation > 0:
                sample_sizes[idx] = allocation
                remaining_size -= allocation
        
        return sample_sizes
    
    def _collect_cluster_samples_optimized(self, subspace_idx: int, stats: dict,
                                         cluster_sample_sizes: np.ndarray, probs: torch.Tensor,
                                         cluster_importances: torch.Tensor) -> Tuple[List[int], List[float]]:
        """
        Оптимизированный сбор семплов из кластеров.
        
        Args:
            subspace_idx: Индекс подпространства
            stats: Статистика кластеров
            cluster_sample_sizes: Размеры выборки для каждого кластера
            probs: Вероятности кластеров
            cluster_importances: Важности кластеров
            
        Returns:
            Кортеж (индексы выбранных точек, веса для importance sampling)
        """
        sampled_indices = []
        sampling_weights = []
        
        total_importance = cluster_importances.sum().item()
        
        for cluster_idx, (code, size) in enumerate(zip(stats['codes'], cluster_sample_sizes)):
            if size == 0:
                continue
                
            # Находим индексы точек в этом кластере
            mask = self.encoded_data[:, subspace_idx] == code
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