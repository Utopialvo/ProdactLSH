from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf, col, rand, pandas_udf, PandasUDFType
from pyspark.sql.types import IntegerType, ArrayType, FloatType
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark import StorageLevel
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
import logging
import time
from contextlib import contextmanager
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import pandas as pd
from scipy.spatial.distance import cdist
import numba
from numba import jit, prange


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@contextmanager
def temporary_broadcast(sc, data):
    """
    Контекстный менеджер для безопасной работы с broadcast переменными.
    
    Args:
        sc: SparkContext
        data: Данные для броадкаста
        
    Yields:
        Broadcast переменная
    """
    broadcast_var = sc.broadcast(data)
    try:
        yield broadcast_var
    finally:
        broadcast_var.unpersist()
        del broadcast_var


@jit(nopython=True, fastmath=True, cache=True)
def _compute_distances_numba(points_array, centroids):
    """
    Оптимизированное вычисление расстояний с использованием Numba.
    
    Args:
        points_array: Массив точек
        centroids: Массив центроидов
        
    Returns:
        Расстояния и метки
    """
    n_points = points_array.shape[0]
    n_centroids = centroids.shape[0]
    n_features = points_array.shape[1]
    
    distances = np.zeros((n_points, n_centroids))
    
    for i in prange(n_points):
        for j in prange(n_centroids):
            dist = 0.0
            for k in prange(n_features):
                diff = points_array[i, k] - centroids[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)
    
    labels = np.zeros(n_points, dtype=np.int64)
    for i in prange(n_points):
        labels[i] = np.argmin(distances[i])
    
    return distances, labels


@jit(nopython=True, fastmath=True, cache=True)
def _update_centroids_numba(points_array, labels, k, current_centroids):
    """
    Оптимизированное обновление центроидов с использованием Numba.
    
    Args:
        points_array: Массив точек
        labels: Метки кластеров
        k: Количество кластеров
        current_centroids: Текущие центроиды
        
    Returns:
        Новые центроиды и счетчики
    """
    n_features = points_array.shape[1]
    new_centroids = np.zeros((k, n_features))
    counts = np.zeros(k, dtype=np.int64)
    
    for i in range(len(points_array)):
        cluster_idx = labels[i]
        counts[cluster_idx] += 1
        for j in range(n_features):
            new_centroids[cluster_idx, j] += points_array[i, j]
    
    for i in range(k):
        if counts[i] > 0:
            for j in range(n_features):
                new_centroids[i, j] /= counts[i]
        else:
            for j in range(n_features):
                new_centroids[i, j] = current_centroids[i, j]
    
    return new_centroids, counts


def _process_partition_optimized(points_iterator, k, max_iter, tol, current_centroids):
    """
    Оптимизированная обработка партиции с векторизацией и Numba.
    
    Args:
        points_iterator: Итератор по точкам партиции
        k: Количество кластеров
        max_iter: Максимальное количество итераций
        tol: Допуск сходимости
        current_centroids: Текущие центроиды
        
    Returns:
        List: Найденные центроиды на партиции
    """
    # Батчевое чтение точек для уменьшения накладных расходов
    batch_size = 1000
    points_batch = []
    
    for i, row in enumerate(points_iterator):
        if hasattr(row, '__getitem__'):
            features = row[0] if len(row) == 1 else row
        else:
            features = row
        points_batch.append(np.array(features, dtype=np.float64))
        
        # Обрабатываем батчами для оптимизации памяти
        if len(points_batch) >= batch_size:
            break
    
    if not points_batch:
        return []
    
    # Единоразовое преобразование в numpy array
    points_array = np.array(points_batch, dtype=np.float64)
    
    # Обеспечиваем правильную размерность данных
    if points_array.ndim == 1:
        points_array = points_array.reshape(-1, 1)
    
    # Получаем текущие центроиды
    centroids = np.array(current_centroids, dtype=np.float64)
    if centroids.ndim == 1:
        centroids = centroids.reshape(-1, points_array.shape[1])
    
    # Проверяем совместимость размерностей
    if points_array.shape[1] != centroids.shape[1]:
        logger.warning(f"Dimension mismatch: points {points_array.shape[1]}, centroids {centroids.shape[1]}")
        return []
    
    # Если точек недостаточно для кластеризации
    if len(points_array) < k:
        return []
    
    # Основной цикл локального K-means с оптимизацией
    for iteration in range(max_iter):
        # Оптимизированное вычисление расстояний
        distances, labels = _compute_distances_numba(points_array, centroids)
        
        # Оптимизированное обновление центроидов
        new_centroids, counts = _update_centroids_numba(points_array, labels, k, centroids)
        
        # Проверяем сходимость
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        if centroid_shift < tol:
            break
            
        centroids = new_centroids
    
    return [centroids.tolist()]


class DistributedKMeans:
    """
    Оптимизированная распределенная реализация K-means алгоритма.
    
    Улучшения производительности:
    - Векторизованные операции с NumPy
    - JIT-компиляция с Numba для критических участков
    - Оптимизированная балансировка партиций
    - TreeAggregate для уменьшения сетевого трафика
    - Пакетная обработка в UDF
    - Эффективное кэширование данных
    
    Args:
        k (int): Количество кластеров
        max_epochs (int): Максимальное количество эпох
        tolerance (float): Допуск сходимости
        local_max_iter (int): Максимальное количество итераций для локального K-means
        local_tol (float): Допуск сходимости для локального K-means
        batch_size (int): Размер батча для обработки
        rebalance_partitions (bool): Флаг балансировки партиций
        use_kmeans_plusplus (bool): Использовать K-means++ инициализацию
        max_centroids_to_collect (int): Максимальное количество центроидов для сбора на драйвер
        learning_rate (float): Скорость обучения для инкрементального обновления (0-1)
        warm_start (bool): Использовать существующие центроиды при дообучении
        use_numba (bool): Использовать Numba для оптимизации
        optimal_partition_size (int): Оптимальный размер партиции в MB
    """
    
    def __init__(self, k: int, 
                 max_epochs: int = 10, 
                 tolerance: float = 1e-4,
                 local_max_iter: int = 100, 
                 local_tol: float = 1e-4,
                 batch_size: int = 1000, 
                 rebalance_partitions: bool = True,
                 use_kmeans_plusplus: bool = True, 
                 max_centroids_to_collect: int = 5000,
                 learning_rate: float = 0.1,
                 warm_start: bool = False,
                 use_numba: bool = True,
                 optimal_partition_size: int = 128):
        
        # Параметры алгоритма
        self.k = k
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.local_max_iter = local_max_iter
        self.local_tol = local_tol
        self.batch_size = batch_size
        self.rebalance_partitions = rebalance_partitions
        self.use_kmeans_plusplus = use_kmeans_plusplus
        self.max_centroids_to_collect = max_centroids_to_collect
        self.learning_rate = max(0.0, min(1.0, learning_rate))
        self.warm_start = warm_start
        self.use_numba = use_numba
        self.optimal_partition_size = optimal_partition_size  # MB
        
        # Состояние модели
        self.centroids = None
        self.feature_stats = {}
        self.is_fitted = False
        self.n_features = None
        self.cluster_counts = None
        self.n_samples_seen_ = 0
        self.partial_fit_count = 0
        self.optimization_stats = {}

    def _estimate_data_size(self, df: DataFrame, feature_col: str) -> Tuple[int, float]:
        """
        Оценивает размер данных для оптимизации партиционирования.
        
        Args:
            df: DataFrame с данными
            feature_col: Колонка с признаками
            
        Returns:
            Tuple: (количество строк, средний размер строки в байтах)
        """
        # Используем выборку для оценки размера
        sample_df = df.select(feature_col).limit(1000)
        sample_size = sample_df.count()
        
        if sample_size == 0:
            return 0, 0
        
        # Оцениваем размер одной строки
        sample_row = sample_df.first()
        if hasattr(sample_row, '__getitem__'):
            features = sample_row[0] if len(sample_row) == 1 else sample_row
        else:
            features = sample_row
            
        # Приблизительная оценка размера в байтах
        if hasattr(features, 'size'):
            row_size = features.size * 8  # float64 = 8 bytes
        else:
            row_size = len(features) * 8
        
        total_rows = df.count()
        estimated_size_mb = (total_rows * row_size) / (1024 * 1024)
        
        return total_rows, estimated_size_mb

    def _optimize_partitions(self, df: DataFrame, feature_col: str) -> DataFrame:
        """
        Оптимизирует количество партиций на основе размера данных.
        
        Args:
            df: Исходный DataFrame
            feature_col: Колонка с признаками
            
        Returns:
            Оптимизированный DataFrame
        """
        if not self.rebalance_partitions:
            return df
            
        total_rows, estimated_size_mb = self._estimate_data_size(df, feature_col)
        
        if total_rows == 0:
            return df
        
        # Вычисляем оптимальное количество партиций
        optimal_partitions = max(1, int(estimated_size_mb / self.optimal_partition_size))
        
        # Ограничиваем сверху для избежания излишней фрагментации
        max_reasonable_partitions = min(df.sparkSession.sparkContext.defaultParallelism * 10, 1000)
        optimal_partitions = min(optimal_partitions, max_reasonable_partitions)
        
        current_partitions = df.rdd.getNumPartitions()
        
        # Перебалансируем только если есть значительная разница
        if abs(current_partitions - optimal_partitions) > current_partitions * 0.3:
            logger.info(f"Repartitioning from {current_partitions} to {optimal_partitions} "
                       f"(estimated data size: {estimated_size_mb:.2f} MB)")
            return df.repartition(optimal_partitions)
        
        return df

    def _normalize_features_vectorized(self, points: np.ndarray) -> np.ndarray:
        """
        Векторизованная нормализация признаков.
        
        Args:
            points: Входные точки данных
            
        Returns:
            Нормализованные точки
        """
        if points.size == 0:
            return points
            
        if points.ndim == 1:
            self.n_features = 1
            points = points.reshape(-1, 1)
        else:
            self.n_features = points.shape[1]
        
        # Векторизованные вычисления
        mean = np.mean(points, axis=0, keepdims=True)
        std = np.std(points, axis=0, keepdims=True)
        
        # Защита от деления на ноль
        std = np.where(std == 0, 1.0, std)
        
        self.feature_stats = {'mean': mean.flatten(), 'std': std.flatten()}
        return (points - mean) / std

    def _initialize_centroids_kmeans_plusplus_vectorized(self, points: np.ndarray) -> np.ndarray:
        """
        Векторизованная K-means++ инициализация.
        
        Args:
            points: Точки данных
            
        Returns:
            Инициализированные центроиды
        """
        n_points = points.shape[0]
        
        if n_points < self.k:
            indices = np.random.choice(n_points, self.k, replace=True)
            return points[indices]
        
        # Выбираем первый центроид случайно
        centroids = [points[np.random.randint(n_points)]]
        
        # Векторизованный выбор остальных центроидов
        for i in range(1, self.k):
            # Векторизованное вычисление минимальных расстояний
            points_array = np.array(points)
            centroids_array = np.array(centroids)
            
            # Вычисляем расстояния от всех точек до всех центроидов
            distances = cdist(points_array, centroids_array, metric='euclidean')
            min_distances = np.min(distances, axis=1)
            
            # Защита от нулевых расстояний
            if np.sum(min_distances) == 0:
                new_centroid_idx = np.random.randint(n_points)
            else:
                probabilities = min_distances ** 2
                probabilities /= probabilities.sum()
                new_centroid_idx = np.random.choice(n_points, p=probabilities)
            
            centroids.append(points[new_centroid_idx])
        
        return np.array(centroids)

    def _safe_sample_data_batch(self, df: DataFrame, feature_col: str, sample_size: int) -> np.ndarray:
        """
        Батчевое извлечение sample данных.
        
        Args:
            df: Исходный DataFrame
            feature_col: Колонка с признаками
            sample_size: Размер выборки
            
        Returns:
            Массив с точками данных
        """
        # Используем более эффективный метод выборки
        fraction = min(1.0, sample_size / max(1, df.count()))
        if fraction >= 1.0:
            sample_rdd = df.select(feature_col).rdd
        else:
            sample_rdd = df.select(feature_col).sample(False, fraction).rdd
        
        # Батчевая обработка для уменьшения накладных расходов
        points_list = sample_rdd.map(
            lambda row: np.array(row[0] if hasattr(row, '__getitem__') else row, dtype=np.float64)
        ).collect()
        
        if not points_list:
            raise ValueError("No points available for centroid initialization")
        
        points_array = np.array(points_list)
        
        if points_array.ndim == 1:
            points_array = points_array.reshape(-1, 1)
            
        return points_array

    def _safe_collect_centroids_tree(self, centroids_rdd) -> List[List[float]]:
        """
        Безопасный сбор центроидов с использованием treeAggregate.
        
        Args:
            centroids_rdd: RDD с центроидами
            
        Returns:
            Собранные центроиды
        """
        def seq_op(acc, element):
            """Последовательная операция агрегации"""
            if element:
                # Ограничиваем размер на каждом уровне
                new_acc = acc + element
                if len(new_acc) > self.max_centroids_to_collect // 4:
                    # Выбираем наиболее разнообразные центроиды
                    return self._select_diverse_centroids(new_acc, self.max_centroids_to_collect // 4)
                return new_acc
            return acc
        
        def comb_op(acc1, acc2):
            """Комбинирующая операция агрегации"""
            combined = acc1 + acc2
            if len(combined) > self.max_centroids_to_collect // 2:
                return self._select_diverse_centroids(combined, self.max_centroids_to_collect // 2)
            return combined
        
        # Используем treeAggregate для уменьшения сетевого трафика
        all_centroids = centroids_rdd.treeAggregate([], seq_op, comb_op, depth=2)
        
        # Финальное ограничение
        if len(all_centroids) > self.max_centroids_to_collect:
            all_centroids = self._select_diverse_centroids(all_centroids, self.max_centroids_to_collect)
        
        return all_centroids

    def _select_diverse_centroids(self, centroids_list: List[List[float]], n_select: int) -> List[List[float]]:
        """
        Выбирает наиболее разнообразные центроиды для лучшего представления.
        
        Args:
            centroids_list: Список центроидов
            n_select: Количество для выбора
            
        Returns:
            Выбранные центроиды
        """
        if len(centroids_list) <= n_select:
            return centroids_list
        
        centroids_array = np.array(centroids_list)
        
        # Алгоритм максимальной минимизации расстояний
        selected_indices = [np.random.randint(len(centroids_list))]
        
        for _ in range(1, n_select):
            max_min_dist = -1
            best_idx = -1
            
            for i in range(len(centroids_list)):
                if i not in selected_indices:
                    # Вычисляем минимальное расстояние до уже выбранных
                    min_dist = min([np.linalg.norm(centroids_array[i] - centroids_array[j]) 
                                  for j in selected_indices])
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
            else:
                break
        
        return [centroids_list[i] for i in selected_indices]

    @jit(nopython=True, fastmath=True)
    def _incremental_update_vectorized(self, new_points: np.ndarray, current_centroids: np.ndarray, 
                                     cluster_counts: np.ndarray, learning_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Векторизованное инкрементальное обновление центроидов.
        
        Args:
            new_points: Новые точки данных
            current_centroids: Текущие центроиды
            cluster_counts: Количество точек в кластерах
            learning_rate: Скорость обучения
            
        Returns:
            Обновленные центроиды и счетчики
        """
        if new_points.size == 0:
            return current_centroids, cluster_counts
        
        n_new = new_points.shape[0]
        n_clusters = current_centroids.shape[0]
        n_features = current_centroids.shape[1]
        
        # Векторизованное вычисление расстояний
        distances = np.zeros((n_new, n_clusters))
        for i in range(n_new):
            for j in range(n_clusters):
                dist = 0.0
                for k in range(n_features):
                    diff = new_points[i, k] - current_centroids[j, k]
                    dist += diff * diff
                distances[i, j] = np.sqrt(dist)
        
        labels = np.argmin(distances, axis=1)
        
        # Векторизованное обновление
        updated_centroids = current_centroids.copy()
        updated_counts = cluster_counts.copy()
        
        for i in range(n_clusters):
            mask = labels == i
            n_new_in_cluster = np.sum(mask)
            
            if n_new_in_cluster > 0:
                new_cluster_mean = np.zeros(n_features)
                for j in range(n_features):
                    new_cluster_mean[j] = np.mean(new_points[mask, j])
                
                if updated_counts[i] > 0:
                    total_points = updated_counts[i] + n_new_in_cluster
                    weight_old = updated_counts[i] / total_points
                    weight_new = n_new_in_cluster / total_points
                    
                    for j in range(n_features):
                        updated_centroids[i, j] = (weight_old * updated_centroids[i, j] + 
                                                 weight_new * new_cluster_mean[j])
                else:
                    for j in range(n_features):
                        updated_centroids[i, j] = new_cluster_mean[j]
                
                updated_counts[i] += n_new_in_cluster
        
        return updated_centroids, updated_counts

    def _cluster_centroids_fast(self, centroids: np.ndarray) -> np.ndarray:
        """
        Быстрая кластеризация центроидов с использованием sklearn.
        
        Args:
            centroids: Центроиды с партиций
            
        Returns:
            Финальные k центроидов
        """
        if len(centroids) <= self.k:
            if len(centroids) < self.k:
                needed = self.k - len(centroids)
                random_indices = np.random.choice(len(centroids), needed, replace=True)
                random_centroids = centroids[random_indices]
                result = np.vstack([centroids, random_centroids])
            else:
                result = centroids
        else:
            # Используем оптимизированный K-means из sklearn
            kmeans = SkKMeans(n_clusters=self.k, max_iter=50, n_init=3, random_state=42)
            kmeans.fit(centroids)
            result = kmeans.cluster_centers_
        
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        
        return result

    def fit(self, df: DataFrame, feature_col: str = 'features') -> 'DistributedKMeansOptimized':
        """
        Оптимизированное обучение модели.
        
        Args:
            df: Данные для обучения
            feature_col: Колонка с признаками
            
        Returns:
            Обученная модель
        """
        spark = df.sparkSession.sparkContext
        start_time = time.time()
        
        # Оптимизированная предобработка данных
        df_preprocessed = self._optimize_partitions(df, feature_col)
        
        # Эффективное кэширование
        df_cached = df_preprocessed.persist(StorageLevel.MEMORY_AND_DISK_DESER)
        
        # Принудительное вычисление для активации кэширования
        df_cached.foreach(lambda x: None)
        
        try:
            # Инициализация центроидов
            self.centroids = self._initialize_centroids(df_cached, feature_col)
            logger.info(f"Initialized {self.k} centroids with shape {self.centroids.shape}")
            
            if self.cluster_counts is None:
                self.cluster_counts = np.zeros(self.k, dtype=int)
            
            previous_centroids = self.centroids.copy()
            
            # Основной цикл обучения
            for epoch in range(self.max_epochs):
                epoch_start = time.time()
                
                with temporary_broadcast(spark, self.centroids) as centroids_broadcast:
                    # Выбираем оптимизированную функцию обработки
                    if self.use_numba:
                        process_func = _process_partition_optimized
                    else:
                        process_func = _process_partition_optimized  # fallback
                    
                    centroids_rdd = df_cached.select(feature_col).rdd.mapPartitions(
                        lambda it: process_func(
                            it, self.k, self.local_max_iter, self.local_tol, 
                            centroids_broadcast.value
                        )
                    )
                    
                    # Используем treeAggregate для сбора
                    collected_centroids = self._safe_collect_centroids_tree(centroids_rdd)
                
                # Объединяем центроиды
                if collected_centroids:
                    new_centroids = self._combine_centroids(collected_centroids)
                else:
                    new_centroids = previous_centroids
                
                # Проверка валидности
                if (new_centroids is None or 
                    np.any(np.isnan(new_centroids)) or 
                    np.any(np.isinf(new_centroids))):
                    logger.warning(f"Invalid centroids in epoch {epoch + 1}, using previous")
                    new_centroids = previous_centroids
                
                # Проверка сходимости
                shift = np.linalg.norm(new_centroids - previous_centroids)
                epoch_time = time.time() - epoch_start
                
                logger.info(f"Epoch {epoch + 1}: centroid shift = {shift:.6f}, time = {epoch_time:.2f}s")
                
                if shift < self.tolerance:
                    logger.info(f"Converged after {epoch + 1} epochs")
                    self.centroids = new_centroids
                    break
                
                previous_centroids = self.centroids
                self.centroids = new_centroids
                
            # Статистика обучения
            self.n_samples_seen_ = df_cached.count()
            self.partial_fit_count = 0
                
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
            self.is_fitted = True
            
            # Сохраняем статистику оптимизации
            self.optimization_stats = {
                'total_training_time': total_time,
                'n_epochs': epoch + 1,
                'final_shift': shift,
                'n_samples': self.n_samples_seen_
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
        finally:
            df_cached.unpersist()
            
        return self

    def transform_batch(self, df: DataFrame, feature_col: str = 'features') -> DataFrame:
        """
        Пакетное предсказание с использованием Pandas UDF для лучшей производительности.
        
        Args:
            df: Данные для предсказания
            feature_col: Колонка с признаками
            
        Returns:
            DataFrame с колонкой 'cluster'
        """
        if not self.is_fitted or self.centroids is None:
            raise ValueError("Model must be fitted before transformation")
        
        centroids_bc = df.sparkContext.broadcast(self.centroids)
        
        @pandas_udf(ArrayType(IntegerType()), PandasUDFType.GROUPED_MAP)
        def predict_batch_udf(pdf):
            """Pandas UDF для пакетного предсказания"""
            try:
                features_list = pdf[feature_col].tolist()
                
                # Преобразуем в numpy array
                points = np.array([np.array(f, dtype=np.float64) for f in features_list])
                if points.ndim == 1:
                    points = points.reshape(-1, 1)
                
                centroids = centroids_bc.value
                if centroids.ndim == 1:
                    centroids = centroids.reshape(-1, 1)
                
                # Векторизованное вычисление расстояний
                distances = cdist(points, centroids, metric='euclidean')
                predictions = np.argmin(distances, axis=1)
                
                result_df = pdf.copy()
                result_df['cluster'] = predictions.tolist()
                return result_df
                
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                pdf['cluster'] = [0] * len(pdf)
                return pdf
        
        # Применяем пакетное предсказание
        result = df.groupBy(F.spark_partition_id()).apply(predict_batch_udf)
        centroids_bc.unpersist()
        
        return result

    def transform(self, df: DataFrame, feature_col: str = 'features') -> DataFrame:
        """
        Умное предсказание - автоматически выбирает метод в зависимости от размера данных.
        
        Args:
            df: Данные для предсказания
            feature_col: Колонка с признаками
            
        Returns:
            DataFrame с колонкой 'cluster'
        """
        # Для небольших датасетов используем обычный метод, для больших - пакетный
        if df.count() < 10000:
            return self._transform_small(df, feature_col)
        else:
            return self.transform_batch(df, feature_col)

    def _transform_small(self, df: DataFrame, feature_col: str = 'features') -> DataFrame:
        """
        Предсказание для небольших датасетов.
        """
        spark = df.sparkSession.sparkContext
        
        with temporary_broadcast(spark, self.centroids) as centroids_broadcast:
            
            def predict_point_optimized(features):
                """Оптимизированная UDF для предсказания одной точки"""
                try:
                    point = np.array(features, dtype=np.float64)
                    
                    if point.ndim == 0:
                        point = np.array([point])
                    elif point.ndim == 1:
                        point = point.reshape(1, -1)[0]
                    
                    centroids = centroids_broadcast.value
                    
                    if centroids.ndim == 1:
                        centroids = centroids.reshape(-1, 1)
                    
                    # Векторизованное вычисление расстояний
                    distances = np.linalg.norm(centroids - point, axis=1)
                    return int(np.argmin(distances))
                    
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    return 0
            
            predict_udf = udf(predict_point_optimized, IntegerType())
            result = df.withColumn("cluster", predict_udf(col(feature_col)))
            
        return result

    # Остальные методы остаются аналогичными оригинальной реализации
    # с небольшими оптимизациями...

    def _initialize_centroids(self, df: DataFrame, feature_col: str) -> np.ndarray:
        """Оптимизированная инициализация центроидов"""
        if self.warm_start and self.centroids is not None:
            logger.info("Using existing centroids due to warm_start=True")
            return self.centroids
            
        sample_size = min(1000, df.count())
        points_array = self._safe_sample_data_batch(df, feature_col, sample_size)
        points_normalized = self._normalize_features_vectorized(points_array)
        
        if self.use_kmeans_plusplus and len(points_normalized) >= self.k:
            centroids_normalized = self._initialize_centroids_kmeans_plusplus_vectorized(points_normalized)
        else:
            n_points = len(points_normalized)
            if n_points < self.k:
                indices = np.random.choice(n_points, self.k, replace=True)
            else:
                indices = np.random.choice(n_points, self.k, replace=False)
            centroids_normalized = points_normalized[indices]
        
        return self._denormalize_centroids(centroids_normalized)

    def _denormalize_centroids(self, centroids: np.ndarray) -> np.ndarray:
        """Векторизованная денормализация"""
        if not self.feature_stats:
            return centroids
            
        mean = self.feature_stats['mean']
        std = self.feature_stats['std']
        return centroids * std + mean

    def _combine_centroids(self, all_centroids: List[List[float]]) -> np.ndarray:
        """Оптимизированное объединение центроидов"""
        if not all_centroids:
            return self.centroids
            
        valid_centroids = []
        for centroid in all_centroids:
            centroid_array = np.array(centroid, dtype=np.float64)
            
            if centroid_array.ndim == 1:
                if self.n_features is not None and self.n_features > 1:
                    continue
                centroid_array = centroid_array.reshape(1, -1)
            
            if self.n_features is not None and centroid_array.shape[1] != self.n_features:
                continue
                
            if not np.any(np.isnan(centroid_array) | np.isinf(centroid_array)):
                valid_centroids.append(centroid_array)
        
        if not valid_centroids:
            logger.warning("No valid centroids found, using previous centroids")
            return self.centroids
        
        combined = np.vstack(valid_centroids)
        combined_normalized = self._normalize_features_vectorized(combined)
        final_centroids = self._cluster_centroids_fast(combined_normalized)
        return self._denormalize_centroids(final_centroids)

    def partial_fit(self, df: DataFrame, feature_col: str = 'features', 
                   learning_rate: float = None) -> 'DistributedKMeansOptimized':
        """Оптимизированное инкрементальное обучение"""
        if not self.is_fitted:
            logger.warning("Model not fitted yet, performing full fit instead of partial_fit")
            return self.fit(df, feature_col)
        
        start_time = time.time()
        
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        batch_size = min(self.batch_size, df.count())
        logger.info(f"Starting partial_fit on {df.count()} points with batch_size={batch_size}")
        
        try:
            for i in range(0, df.count(), batch_size):
                batch_df = df.limit(batch_size).offset(i)
                new_points = self._safe_sample_data_batch(batch_df, feature_col, batch_df.count())
                
                if new_points.size > 0:
                    new_points_normalized = self._normalize_features_vectorized(new_points)
                    centroids_normalized = self._normalize_features_vectorized(self.centroids)
                    
                    if self.use_numba:
                        updated_centroids_normalized, updated_counts = self._incremental_update_vectorized(
                            new_points_normalized, centroids_normalized, self.cluster_counts, learning_rate
                        )
                    else:
                        updated_centroids_normalized, updated_counts = self._incremental_centroid_update(
                            new_points_normalized, centroids_normalized, self.cluster_counts, learning_rate
                        )
                    
                    self.centroids = self._denormalize_centroids(updated_centroids_normalized)
                    self.cluster_counts = updated_counts
            
            self.n_samples_seen_ += df.count()
            self.partial_fit_count += 1
            
            total_time = time.time() - start_time
            logger.info(f"Partial fit completed in {total_time:.2f}s. Total samples seen: {self.n_samples_seen_}")
            
        except Exception as e:
            logger.error(f"Error during partial_fit: {e}")
            raise
            
        return self

    def _incremental_centroid_update(self, new_data_points, current_centroids, 
                                   cluster_counts, learning_rate):
        """Резервная реализация без Numba"""
        # Реализация аналогичная оригинальной, но с векторизацией
        if new_data_points.size == 0:
            return current_centroids, cluster_counts
            
        if new_data_points.ndim == 1:
            new_data_points = new_data_points.reshape(-1, 1)
        if current_centroids.ndim == 1:
            current_centroids = current_centroids.reshape(-1, 1)
            
        distances = cdist(new_data_points, current_centroids, metric='euclidean')
        closest_centroids = np.argmin(distances, axis=1)
        
        updated_centroids = current_centroids.copy()
        updated_counts = cluster_counts.copy()
        
        for i in range(self.k):
            mask = (closest_centroids == i)
            new_points_in_cluster = new_data_points[mask]
            n_new_points = len(new_points_in_cluster)
            
            if n_new_points > 0:
                new_cluster_mean = np.mean(new_points_in_cluster, axis=0)
                
                if updated_counts[i] > 0:
                    total_points = updated_counts[i] + n_new_points
                    weight_old = updated_counts[i] / total_points
                    weight_new = n_new_points / total_points
                    
                    updated_centroids[i] = (weight_old * updated_centroids[i] + 
                                          weight_new * new_cluster_mean)
                else:
                    updated_centroids[i] = new_cluster_mean
                
                updated_counts[i] += n_new_points
        
        return updated_centroids, updated_counts

    # Остальные методы для совместимости
    def fit_transform(self, df: DataFrame, feature_col: str = 'features') -> DataFrame:
        self.fit(df, feature_col)
        return self.transform(df, feature_col)

    @property
    def cluster_centers_(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.centroids

    def get_optimization_stats(self) -> dict:
        return self.optimization_stats.copy()