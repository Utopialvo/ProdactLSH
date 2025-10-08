# IS_PQ.py - Fixed version with corrected weights and all subspaces usage

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Union, Optional, Callable
import warnings
import math


class ProductQuantizer:
    """
    Product Quantization с поддержкой GPU для эффективного сжатия и поиска векторов.
    
    Математическая основа (PQ_IS.pdf, раздел 2):
    Цель PQ - найти отображение q: ℝ^D → C, минимизирующее ошибку квантования:
    min_q Σ ||x_i - q(x_i)||^2
    
    Пространство разбивается на M подпространств, каждое квантуется независительно.
    
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
    
    Рекомендации по использованию:
    1. Для высокоразмерных данных используйте strategy='residual_variance'
    2. Температуру (temperature) настройте для контроля дисперсии
    3. Проверяйте распределение весов для избежания выбросов
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
                          strategy: str = 'proportional',
                          temperature: float = 1.0) -> Tuple[List[int], List[float]]:
        """
        Importance Sampling на основе PQ кластеров.
        
        Args:
            target_function: целевая функция для оценки
            sample_size: размер выборки
            strategy: стратегия вычисления важности ('proportional', 'residual_variance', etc.)
            use_residuals: использовать ли остатки для residual_variance стратегии
            temperature: температура для softmax
            
        Returns:
            Кортеж (индексы семплов, веса семплов)
        """
        if self.cluster_stats is None:
            raise ValueError("Сначала обучите модель с помощью train()")
            
        # Шаг 1: Вычисляем важности для всех подпространств
        subspace_importances = []
        subspace_cluster_sizes = []
        subspace_cluster_indices = []
        
        for subspace_idx in range(self.num_subspaces):
            stats = self.cluster_stats[subspace_idx]
            cluster_importances = self._compute_cluster_importances(
                subspace_idx, target_function, strategy)
            
            if cluster_importances is None or len(cluster_importances) == 0:
                # Используем равномерное распределение при проблемах
                cluster_importances = torch.ones(len(stats['codes']), device=self.device)
            
            # ИСПРАВЛЕНИЕ: Более агрессивная обработка выбросов
            cluster_importances = torch.nan_to_num(cluster_importances, nan=1.0, posinf=1.0, neginf=0.0)
            cluster_importances = torch.clamp(cluster_importances, min=1e-6)  # Избегаем нулей
            
            # НОРМАЛИЗАЦИЯ: Логарифмическое масштабирование для борьбы с выбросами
            if cluster_importances.max() > 0:
                cluster_importances = torch.log1p(cluster_importances)  # log(1 + x)
                cluster_importances = cluster_importances / cluster_importances.max()
            
            # ЗАЩИТА: Гарантируем, что сумма не нулевая
            if cluster_importances.sum() == 0:
                cluster_importances = torch.ones_like(cluster_importances)
            
            subspace_importances.append(cluster_importances)
            subspace_cluster_sizes.append(stats['sizes'])
            
            # Сохраняем индексы точек для каждого кластера
            cluster_indices_list = []
            for code in stats['codes']:
                mask = self.encoded_data[:, subspace_idx] == code
                cluster_indices = torch.where(mask)[0]
                cluster_indices_list.append(cluster_indices)
            subspace_cluster_indices.append(cluster_indices_list)
        
        # Шаг 2: Агрегируем важности через softmax с температурой
        point_importances = torch.zeros(self.total_points, device=self.device)
        point_counts = torch.zeros(self.total_points, device=self.device)  # Счетчик вхождений
        
        for subspace_idx in range(self.num_subspaces):
            stats = self.cluster_stats[subspace_idx]
            cluster_importances = subspace_importances[subspace_idx]
            cluster_sizes = subspace_cluster_sizes[subspace_idx]
            
            # Нормализуем важности кластеров через softmax
            cluster_probs = F.softmax(cluster_importances / temperature, dim=0)
            
            for cluster_idx, code in enumerate(stats['codes']):
                cluster_prob = cluster_probs[cluster_idx]
                cluster_size = cluster_sizes[cluster_idx]
                cluster_indices = subspace_cluster_indices[subspace_idx][cluster_idx]
                
                if len(cluster_indices) > 0:
                    # Важность точки = вероятность кластера / размер кластера
                    point_importance = cluster_prob / cluster_size
                    point_importances[cluster_indices] += point_importance
                    point_counts[cluster_indices] += 1
        
        # ИСПРАВЛЕНИЕ: Усредняем важности по всем подпространствам
        mask = point_counts > 0
        point_importances[mask] = point_importances[mask] / point_counts[mask]
        
        # Нормализуем важности точек
        if point_importances.sum() > 0:
            point_importances = point_importances / point_importances.sum()
        else:
            point_importances = torch.ones_like(point_importances) / self.total_points
        
        # Шаг 3: Семплируем точки на основе их важностей
        if sample_size >= self.total_points:
            # Если нужна вся выборка
            sampled_indices = list(range(self.total_points))
            sampling_weights = [1.0] * self.total_points
        else:
            # Многократное семплирование с заменой
            probs = point_importances.cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.0)
            
            # ЗАЩИТА: Проверяем, что распределение не вырождено
            if (probs > 0).sum() < sample_size:
                # Если слишком много нулевых вероятностей, добавляем равномерный шум
                uniform_component = np.ones_like(probs) / len(probs)
                probs = 0.7 * probs + 0.3 * uniform_component
            
            if probs.sum() == 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
            
            sampled_indices = np.random.choice(
                self.total_points, size=sample_size, replace=True, p=probs
            ).tolist()
            
            # КОРРЕКТНЫЙ РАСЧЕТ ВЕСОВ согласно теории Importance Sampling
            sampling_weights = []
            for idx in sampled_indices:
                # p(x) - истинное распределение (предполагаем равномерное)
                p_x = 1.0 / self.total_points
                # q(x) - proposal распределение (на основе важностей)
                q_x = point_importances[idx].item()
                # Вес = p(x) / q(x)
                weight = p_x / q_x if q_x > 0 else 1.0
                sampling_weights.append(weight)
        
        return sampled_indices, sampling_weights
    
    def _compute_cluster_importances(self, subspace_idx: int, target_function: Callable,
                                  strategy: str) -> Optional[torch.Tensor]:
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
            elif strategy == 'residual_variance':
                # Дисперсия остатков относительно центроида (только в подпространстве)
                importance = self._compute_residual_variance(
                    cluster_points, stats['centroids'][code], subspace_idx, target_function)
            elif strategy == 'cluster_density':
                # Плотность кластера (можно использовать более сложные метрики)
                importance = self._compute_cluster_density(cluster_points)
            elif strategy == 'target_variance':
                # Дисперсия целевой функции в кластере (новая стратегия)
                importance = self._compute_target_variance(cluster_points, target_function)
            else:
                # По умолчанию используем размер кластера
                importance = len(cluster_points)
                
            importances.append(importance)
        
        return torch.tensor(importances, device=self.device)

    def _compute_residual_variance(self, cluster_points: torch.Tensor, centroid: torch.Tensor,
                                subspace_idx: int, target_function: Callable) -> float:
        """
        ИСПРАВЛЕННОЕ вычисление дисперсии остатков в кластере
        Добавлены защиты от выбросов и нормализация
        """
        if len(cluster_points) < 2:
            return float(len(cluster_points))  # Для кластеров с 1 точкой используем размер
        
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
            
            if len(target_vals) > 1:
                residual_var = residuals.var().item()
                target_var = target_vals.var().item()
                
                residual_var = max(residual_var, 1e-6)
                target_var = max(target_var, 1e-6)
                
                importance = math.sqrt(residual_var * target_var) * len(cluster_points)
                
                max_importance = len(cluster_points) * 10
                importance = min(importance, max_importance)
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
        
        density = len(cluster_points) / (1.0 + avg_distance)
        return density
    
    def _compute_target_variance(self, cluster_points: torch.Tensor, target_function: Callable) -> float:
        """
        НОВАЯ СТРАТЕГИЯ: Дисперсия целевой функции в кластере
        Стратегия основана на теории Importance Sampling
        """
        if len(cluster_points) < 2:
            return float(len(cluster_points))
            
        with torch.no_grad():
            target_vals = target_function(cluster_points)
            variance = target_vals.var().item()
            
            variance = max(variance, 1e-6)
            importance = variance * len(cluster_points)
            
            max_importance = len(cluster_points) * 10
            importance = min(importance, max_importance)
            
        return importance
    
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
    
    def get_sampling_stats(self) -> Dict:
        """
        ДОПОЛНИТЕЛЬНЫЙ МЕТОД: Статистика по семплингу
        Возвращает информацию для диагностики качества семплинга
        """
        if self.cluster_stats is None:
            return {}
            
        stats = {}
        stats['total_clusters'] = sum(len(subspace_stats['codes']) for subspace_stats in self.cluster_stats.values())
        stats['cluster_sizes'] = {}
        for subspace_idx, subspace_stats in self.cluster_stats.items():
            stats['cluster_sizes'][subspace_idx] = subspace_stats['sizes'].cpu().numpy()
        stats['total_points'] = self.total_points
        stats['num_subspaces'] = self.num_subspaces
        
        return stats


class SamplingEvaluator:
    """Утилиты для оценки качества семплинга."""
    
    @staticmethod
    def estimate_expectation(samples: torch.Tensor, weights: List[float], 
                           target_function: Callable) -> float:
        """
        Оценка матожидания с помощью Importance Sampling.
        
        Формула из IS_LSH.pdf: E_p[f] ≈ (Σ w_i * f(x_i)) / (Σ w_i)
        Нормализация весов согласно теории
        """
        if not weights or len(weights) == 0:
            return 0.0
            
        with torch.no_grad():
            target_values = target_function(samples)
            weights_tensor = torch.tensor(weights, device=samples.device)
            
            # Нормализация весов для устойчивости
            if weights_tensor.sum() == 0:
                return 0.0
                
            normalized_weights = weights_tensor / weights_tensor.sum()
            
            weighted_sum = (target_values * normalized_weights).sum()
            
            return weighted_sum.item()
    
    @staticmethod
    def estimate_variance(samples: torch.Tensor, weights: List[float],
                         target_function: Callable) -> float:
        """
        Оценка дисперсии взвешенной выборки
        """
        if not weights or len(weights) == 0:
            return 0.0
            
        with torch.no_grad():
            target_values = target_function(samples)
            weights_tensor = torch.tensor(weights, device=samples.device)
            
            if weights_tensor.sum() == 0:
                return 0.0
                
            normalized_weights = weights_tensor / weights_tensor.sum()
            
            # Оценка матожидания
            expectation = (target_values * normalized_weights).sum()
            
            # Оценка дисперсии
            variance = (normalized_weights * (target_values - expectation)**2).sum()
            
            return variance.item()
    
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
            'covariance_difference': cov_diff,
            'effective_sample_size': SamplingEvaluator.effective_sample_size(sampled_weights) if sampled_weights else len(sampled_data)
        }
    
    @staticmethod
    def effective_sample_size(weights: List[float]) -> float:
        """
        Оценка эффективного размера выборки по весам
        ESS = (Σ w_i)^2 / (Σ w_i^2)
        """
        if not weights:
            return 0.0
            
        weights_tensor = torch.tensor(weights)
        weight_sum = weights_tensor.sum()
        weight_sq_sum = (weights_tensor ** 2).sum()
        
        if weight_sq_sum == 0:
            return 0.0
            
        return (weight_sum ** 2 / weight_sq_sum).item()