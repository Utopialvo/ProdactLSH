import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any, Tuple
import warnings
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import silhouette_score
import numpy as np

class BaseDistanceMetric(nn.Module):
    """
    Базовый класс для всех метрик расстояния.
    
    Все пользовательские метрики расстояния должны наследовать от этого класса
    и реализовывать метод forward.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет матрицу расстояний между x и y.
        
        Args:
            x: Тензор формы (N, D)
            y: Тензор формы (M, D)
            
        Returns:
            Матрица расстояний формы (N, M)
        """
        raise NotImplementedError("Метод forward должен быть реализован в дочернем классе")

class EuclideanDistance(BaseDistanceMetric):
    """Евклидово расстояние (L2 норма)."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y, p=2.0)

class ManhattanDistance(BaseDistanceMetric):
    """Манхеттенское расстояние (L1 норма)."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y, p=1.0)

class CosineDistance(BaseDistanceMetric):
    """Косинусное расстояние (1 - косинусная схожесть)."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        y_norm = F.normalize(y, p=2, dim=1, eps=1e-8)
        cosine_sim = torch.mm(x_norm, y_norm.t())
        return 1 - torch.clamp(cosine_sim, -1.0 + 1e-8, 1.0 - 1e-8)

class HammingDistance(BaseDistanceMetric):
    """Расстояние Хемминга для категориальных данных."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("Для расстояния Хемминга ожидаются 2D тензоры")
        x_expanded = x.unsqueeze(1).expand(-1, y.size(0), -1)
        y_expanded = y.unsqueeze(0).expand(x.size(0), -1, -1)
        return (x_expanded != y_expanded).float().mean(dim=2)

class BaseLossFunction(nn.Module):
    """Базовый класс для функций потерь кластеризации."""
    def __init__(self):
        super().__init__()
    
    def forward(self, distances: torch.Tensor, assignments: torch.Tensor, 
                embeddings: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Вычисляет значение функции потерь.
        
        Args:
            distances: Матрица расстояний формы (N, K)
            assignments: Назначения кластеров формы (N,)
            embeddings: Исходные эмбеддинги формы (N, D)
            
        Returns:
            Скалярное значение потерь
        """
        raise NotImplementedError("Метод forward должен быть реализован в дочернем классе")

class StandardKMeansLoss(BaseLossFunction):
    """Стандартная функция потерь K-means (within-cluster sum of squares)."""
    def forward(self, distances: torch.Tensor, assignments: torch.Tensor, 
                embeddings: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        min_distances = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
        return min_distances.mean()

class WithinClusterVarianceLoss(BaseLossFunction):
    """Функция потерь на основе внутрикластерной дисперсии."""
    def forward(self, distances: torch.Tensor, assignments: torch.Tensor,
                embeddings: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        n_clusters = distances.shape[1]
        
        # Векторизованная реализация вместо цикла
        mask = F.one_hot(assignments, num_classes=n_clusters).bool()
        cluster_distances = distances[mask].view(-1, n_clusters)
        cluster_sizes = mask.sum(dim=0)
        
        valid_clusters = cluster_sizes > 0
        if valid_clusters.sum() == 0:
            return torch.tensor(0.0, device=distances.device)
        
        # Вычисляем средние расстояния только для валидных кластеров
        cluster_means = torch.zeros(n_clusters, device=distances.device)
        cluster_means[valid_clusters] = cluster_distances[:, valid_clusters].sum(dim=0) / cluster_sizes[valid_clusters]
        
        return cluster_means[valid_clusters].mean()

class ContrastiveClusteringLoss(BaseLossFunction):
    """
    Контрастные потери для кластеризации.
    
    Args:
        temperature: Температурный параметр для масштабирования сходств
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, distances: torch.Tensor, assignments: torch.Tensor,
                embeddings: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if embeddings is None:
            raise ValueError("Контрастные потери требуют embeddings")
        
        batch_size = embeddings.size(0)
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        assignment_matrix = (assignments.unsqueeze(1) == assignments.unsqueeze(0)).float()
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        assignment_matrix.masked_fill_(mask, 0)
        
        exp_sim = torch.exp(similarity_matrix)
        exp_sim.masked_fill_(mask, 0)
        
        positive_pairs = (assignment_matrix * similarity_matrix).sum(dim=1)
        positive_exp = (assignment_matrix * exp_sim).sum(dim=1)
        all_exp = exp_sim.sum(dim=1)
        
        # Защита от численной нестабильности
        loss = -torch.log(positive_exp / (all_exp + 1e-8) + 1e-8).mean()
        return loss

class EntropyRegularizedLoss(BaseLossFunction):
    """
    Функция потерь с регуляризацией энтропией для балансировки кластеров.
    
    Args:
        alpha: Коэффициент регуляризации энтропии
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, distances: torch.Tensor, assignments: torch.Tensor,
                embeddings: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Основные потери
        min_distances = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
        main_loss = min_distances.mean()
        
        # Регуляризация энтропией для балансировки кластеров
        cluster_probs = torch.zeros(distances.shape[1], device=distances.device)
        for k in range(distances.shape[1]):
            cluster_probs[k] = (assignments == k).float().mean()
        
        entropy = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8))
        
        return main_loss - self.alpha * entropy

class CentroidRegularizationLoss(BaseLossFunction):
    """
    Функция потерь с регуляризацией центроидов для предотвращения коллапса кластеров.
    
    Args:
        lambda_reg: Коэффициент регуляризации
        min_distance: Минимальное желаемое расстояние между центроидами
    """
    def __init__(self, lambda_reg: float = 0.01, min_distance: float = 1.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.min_distance = min_distance
    
    def forward(self, distances: torch.Tensor, assignments: torch.Tensor,
                embeddings: Optional[torch.Tensor] = None, centroids: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Основные потери
        min_distances = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
        main_loss = min_distances.mean()
        
        # Регуляризация для предотвращения схлопывания центроидов
        if centroids is not None:
            centroid_distances = torch.cdist(centroids, centroids, p=2.0)
            mask = ~torch.eye(centroids.shape[0], dtype=torch.bool, device=centroids.device)
            
            if mask.sum() > 0:  # Если есть пары для сравнения
                pairwise_distances = centroid_distances[mask]
                # Штрафуем за слишком близкие центроиды
                repulsion_loss = torch.exp(-pairwise_distances / self.min_distance).mean()
                return main_loss + self.lambda_reg * repulsion_loss
        
        return main_loss

class GradientKMeans:
    """
    Унифицированный класс для градиентной кластеризации K-means с поддержкой 
    инкрементного обучения и различных метрик расстояния.
    
    Args:
        n_clusters: Количество кластеров
        n_features: Размерность признакового пространства
        distance_metric: Метрика расстояния ('euclidean', 'manhattan', 'cosine', 'hamming')
        loss_function: Функция потерь ('standard', 'variance', 'contrastive', 'entropy', 'centroid_reg')
        init_method: Метод инициализации ('random', 'kmeans++')
        optimizer: Оптимизатор ('SGD', 'Adam', 'RMSprop')
        lr: Скорость обучения
        temperature: Температура для контрастных потерь
        entropy_alpha: Коэффициент энтропии для регуляризации
        reg_lambda: Коэффициент регуляризации центроидов
        reg_min_distance: Минимальное расстояние между центроидами для регуляризации
        max_iters: Максимальное количество итераций
        tol: Порог сходимости
        verbose: Вывод информации о процессе
        chunk_size: Размер чанка для инкрементного обучения
        incremental_frequency: Частота обновления центроидов при инкрементном обучении
        early_stopping_patience: Терпение для ранней остановки
        handle_empty_clusters: Стратегия обработки пустых кластеров ('reinitialize', 'ignore')
        use_scheduler: Использовать ли планировщик learning rate
        scheduler_step_size: Шаг для планировщика
        scheduler_gamma: Коэффициент уменьшения learning rate
    """
    
    def __init__(self, n_clusters: int, n_features: int, **kwargs):
        # Валидация входных параметров
        if n_clusters <= 0:
            raise ValueError("n_clusters должен быть положительным числом")
        if n_features <= 0:
            raise ValueError("n_features должен быть положительным числом")
        
        self.n_clusters = n_clusters
        self.n_features = n_features
        
        # Параметры с значениями по умолчанию
        self.distance_metric = kwargs.get('distance_metric', 'euclidean')
        self.loss_function = kwargs.get('loss_function', 'standard')
        self.init_method = kwargs.get('init_method', 'random')
        self.optimizer_name = kwargs.get('optimizer', 'Adam')
        self.lr = kwargs.get('lr', 0.01)
        self.temperature = kwargs.get('temperature', 0.5)
        self.entropy_alpha = kwargs.get('entropy_alpha', 0.1)
        self.reg_lambda = kwargs.get('reg_lambda', 0.01)
        self.reg_min_distance = kwargs.get('reg_min_distance', 1.0)
        self.max_iters = kwargs.get('max_iters', 100)
        self.tol = kwargs.get('tol', 1e-4)
        self.verbose = kwargs.get('verbose', False)
        self.chunk_size = kwargs.get('chunk_size', 1000)
        self.incremental_frequency = kwargs.get('incremental_frequency', 10)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.handle_empty_clusters = kwargs.get('handle_empty_clusters', 'reinitialize')
        self.use_scheduler = kwargs.get('use_scheduler', True)
        self.scheduler_step_size = kwargs.get('scheduler_step_size', 30)
        self.scheduler_gamma = kwargs.get('scheduler_gamma', 0.5)
        
        # Валидация гиперпараметров
        if self.lr <= 0:
            raise ValueError("Learning rate должен быть положительным")
        if self.temperature <= 0:
            raise ValueError("Temperature должна быть положительной")
        if self.entropy_alpha < 0:
            raise ValueError("entropy_alpha должен быть неотрицательным")
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda должен быть неотрицательным")
        if self.reg_min_distance <= 0:
            raise ValueError("reg_min_distance должен быть положительным")
        if self.max_iters <= 0:
            raise ValueError("max_iters должен быть положительным")
        if self.tol <= 0:
            raise ValueError("tol должен быть положительным")
        
        # Инициализация компонентов
        self._init_distance_metric()
        self._init_loss_function()
        
        # Отложенная инициализация центроидов (требует данные)
        self.centroids = None
        self.scheduler = None
        
        self.history = []
        self.incremental_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.cluster_sizes_history = []
    
    def _init_distance_metric(self):
        """Инициализация метрики расстояния на основе параметров."""
        metrics = {
            'euclidean': EuclideanDistance,
            'manhattan': ManhattanDistance,
            'cosine': CosineDistance,
            'hamming': HammingDistance
        }
        
        if self.distance_metric not in metrics:
            raise ValueError(f"Неподдерживаемая метрика расстояния: {self.distance_metric}")
        
        self.distance_fn = metrics[self.distance_metric]()
    
    def _init_loss_function(self):
        """Инициализация функции потерь на основе параметров."""
        losses = {
            'standard': StandardKMeansLoss,
            'variance': WithinClusterVarianceLoss,
            'contrastive': lambda: ContrastiveClusteringLoss(temperature=self.temperature),
            'entropy': lambda: EntropyRegularizedLoss(alpha=self.entropy_alpha),
            'centroid_reg': lambda: CentroidRegularizationLoss(
                lambda_reg=self.reg_lambda, 
                min_distance=self.reg_min_distance
            )
        }
        
        if self.loss_function not in losses:
            raise ValueError(f"Неподдерживаемая функция потерь: {self.loss_function}")
        
        self.loss_fn = losses[self.loss_function]()
    
    def _init_centroids(self, X: torch.Tensor):
        """Инициализация центроидов выбранным методом."""
        if self.init_method == 'random':
            # Добавляем небольшой шум для разнообразия
            self.centroids = nn.Parameter(
                X[:self.n_clusters].clone() + 
                0.1 * torch.randn(self.n_clusters, self.n_features, device=X.device)
            )
        elif self.init_method == 'kmeans++':
            self.centroids = nn.Parameter(self._kmeans_plus_plus_init(X))
        else:
            raise ValueError(f"Неподдерживаемый метод инициализации: {self.init_method}")
    
    def _kmeans_plus_plus_init(self, X: torch.Tensor) -> torch.Tensor:
        """Улучшенная k-means++ инициализация с защитой от вырожденных случаев."""
        n_samples = X.shape[0]
        
        if n_samples < self.n_clusters:
            raise ValueError(f"Количество образцов ({n_samples}) должно быть >= n_clusters ({self.n_clusters})")
        
        centroids = torch.zeros((self.n_clusters, self.n_features), device=X.device)
        
        # Первый центроид выбирается случайно
        first_idx = torch.randint(0, n_samples, (1,))
        centroids[0] = X[first_idx].clone()
        
        for k in range(1, self.n_clusters):
            # Вычисляем расстояния до ближайшего центроида
            distances = self.distance_fn(X, centroids[:k])
            min_distances = distances.min(dim=1).values
            
            # Защита от нулевых расстояний
            if min_distances.sum() == 0:
                # Выбираем случайную точку
                next_idx = torch.randint(0, n_samples, (1,))
            else:
                # Вероятность пропорциональна квадрату расстояния
                probabilities = min_distances ** 2
                probabilities /= probabilities.sum()
                
                # Проверка на численную стабильность
                if torch.any(torch.isnan(probabilities)) or torch.any(torch.isinf(probabilities)):
                    # Равномерное распределение в случае проблем
                    probabilities = torch.ones(n_samples, device=X.device) / n_samples
                
                # Выбираем следующий центроид
                next_idx = torch.multinomial(probabilities, 1)
            
            centroids[k] = X[next_idx].clone()
        
        return centroids
    
    def _init_optimizer(self):
        """Инициализация оптимизатора для центроидов."""
        if self.centroids is None:
            # Создаем временные центроиды для инициализации оптимизатора
            self.centroids = nn.Parameter(torch.randn(self.n_clusters, self.n_features))
        
        optimizers = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'RMSprop': torch.optim.RMSprop
        }
        
        if self.optimizer_name not in optimizers:
            raise ValueError(f"Неподдерживаемый оптимизатор: {self.optimizer_name}")
        
        self.optimizer = optimizers[self.optimizer_name]([self.centroids], lr=self.lr)
        
        # Инициализация планировщика learning rate
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.scheduler_step_size, 
                gamma=self.scheduler_gamma
            )
    
    def _compute_assignments(self, distances: torch.Tensor) -> torch.Tensor:
        """Вычисление назначений кластеров на основе расстояний."""
        return distances.argmin(dim=1)
    
    def _handle_empty_clusters(self, X: torch.Tensor, assignments: torch.Tensor):
        """
        Обработка пустых кластеров путем переинициализации.
        
        Args:
            X: Входные данные
            assignments: Текущие назначения кластеров
        """
        if self.handle_empty_clusters == 'ignore':
            return
        
        cluster_sizes = torch.bincount(assignments, minlength=self.n_clusters)
        empty_clusters = torch.where(cluster_sizes == 0)[0]
        
        if len(empty_clusters) > 0 and self.verbose:
            print(f"Обнаружены пустые кластеры: {empty_clusters.tolist()}")
        
        for cluster_idx in empty_clusters:
            if self.handle_empty_clusters == 'reinitialize':
                # Находим точку с максимальным расстоянием до ближайшего центроида
                distances = self.distance_fn(X, self.centroids)
                min_distances = distances.min(dim=1).values
                farthest_idx = min_distances.argmax()
                
                # Переинициализируем центроид
                with torch.no_grad():
                    self.centroids.data[cluster_idx] = X[farthest_idx].clone() + \
                        0.01 * torch.randn_like(self.centroids.data[cluster_idx])
                
                if self.verbose:
                    print(f"Переинициализирован центроид кластера {cluster_idx}")
    
    def _compute_cluster_sizes(self, assignments: torch.Tensor) -> List[int]:
        """Вычисление размеров кластеров."""
        cluster_sizes = []
        for k in range(self.n_clusters):
            size = (assignments == k).sum().item()
            cluster_sizes.append(size)
        return cluster_sizes
    
    def _check_early_stopping(self, loss: float) -> bool:
        """Проверка условия ранней остановки."""
        if loss < self.best_loss - self.tol:
            self.best_loss = loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience
    
    def fit(self, X: torch.Tensor, embeddings: Optional[torch.Tensor] = None) -> 'GradientKMeans':
        """
        Полное обучение модели на всех данных.
        
        Args:
            X: Входные данные формы (n_samples, n_features)
            embeddings: Эмбеддинги для контрастных потерь
            
        Returns:
            Обученная модель
        """
        # Валидация входных данных
        if X.dim() != 2:
            raise ValueError("Ожидается 2D тензор формы (n_samples, n_features)")
        
        if X.size(1) != self.n_features:
            raise ValueError(f"Ожидается {self.n_features} признаков, получено {X.size(1)}")
        
        if X.size(0) < self.n_clusters:
            raise ValueError(f"Количество образцов ({X.size(0)}) должно быть >= n_clusters ({self.n_clusters})")
        
        # Инициализация центроидов, если еще не инициализированы
        if self.centroids is None:
            self._init_centroids(X)
            self._init_optimizer()
        
        self.history = []
        self.cluster_sizes_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        prev_centroids = self.centroids.data.clone()
        
        for iteration in range(self.max_iters):
            # Прямой проход
            distances = self.distance_fn(X, self.centroids)
            assignments = self._compute_assignments(distances)
            
            # Отслеживание размеров кластеров
            cluster_sizes = self._compute_cluster_sizes(assignments)
            self.cluster_sizes_history.append(cluster_sizes)
            
            # Обработка пустых кластеров
            if self.handle_empty_clusters != 'ignore':
                self._handle_empty_clusters(X, assignments)
                # Пересчитываем расстояния после возможной переинициализации центроидов
                distances = self.distance_fn(X, self.centroids)
                assignments = self._compute_assignments(distances)
            
            # Вычисление потерь
            loss_kwargs = {}
            if self.loss_function == 'centroid_reg':
                loss_kwargs['centroids'] = self.centroids
            
            loss = self.loss_fn(
                distances=distances, 
                assignments=assignments,
                embeddings=embeddings,
                **loss_kwargs
            )
            
            # Обратное распространение
            self.optimizer.zero_grad()
            loss.backward()
            
            # Градиентный clipping для стабильности
            torch.nn.utils.clip_grad_norm_([self.centroids], max_norm=1.0)
            
            self.optimizer.step()
            
            # Обновление планировщика
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Проверка сходимости
            centroid_change = torch.norm(self.centroids.data - prev_centroids)
            
            # Сохранение истории
            history_entry = {
                'iteration': iteration,
                'loss': loss.item(),
                'assignments': assignments.detach().clone(),
                'centroids': self.centroids.data.detach().clone(),
                'inertia': self._inertia(X, assignments).item(),
                'centroid_change': centroid_change.item(),
                'cluster_sizes': cluster_sizes,
                'empty_clusters': cluster_sizes.count(0),
                'learning_rate': self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else self.lr
            }
            self.history.append(history_entry)
            
            # Проверка ранней остановки
            if self._check_early_stopping(loss.item()):
                if self.verbose:
                    print(f"Ранняя остановка на итерации {iteration}")
                break
            
            if centroid_change < self.tol:
                if self.verbose:
                    print(f"Сходимость достигнута на итерации {iteration}")
                break
            
            prev_centroids = self.centroids.data.clone()
            
            if self.verbose and iteration % 10 == 0:
                empty_count = cluster_sizes.count(0)
                print(f"Iteration {iteration}: Loss = {loss.item():.4f}, "
                      f"Centroid change = {centroid_change:.4f}, "
                      f"Inertia = {history_entry['inertia']:.4f}, "
                      f"Empty clusters = {empty_count}/{self.n_clusters}, "
                      f"LR = {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if empty_count == 0:
                    min_size, max_size = min(cluster_sizes), max(cluster_sizes)
                    print(f"  Cluster sizes: min={min_size}, max={max_size}, balance={min_size/max_size:.3f}")
        
        if self.verbose:
            final_cluster_sizes = self._compute_cluster_sizes(assignments)
            empty_count = final_cluster_sizes.count(0)
            print(f"Обучение завершено. Пустых кластеров: {empty_count}/{self.n_clusters}")
            
            if empty_count == 0:
                min_size, max_size = min(final_cluster_sizes), max(final_cluster_sizes)
                balance_ratio = min_size / max_size
                print(f"Баланс кластеров: {balance_ratio:.3f} (min={min_size}, max={max_size})")
        
        return self
    
    def fit_incremental(self, X_chunk: torch.Tensor, n_epochs: int = 1, 
                       embeddings: Optional[torch.Tensor] = None,
                       learning_ratio: float = 0.1) -> 'GradientKMeans':
        """
        Инкрементное обучение на чанке данных.
        
        Args:
            X_chunk: Чанк данных формы (chunk_size, n_features)
            n_epochs: Количество эпох обучения на чанке
            embeddings: Эмбеддинги для контрастных потерь
            learning_ratio: Коэффициент уменьшения learning rate для инкрементного обучения
            
        Returns:
            Обновленная модель
        """
        if X_chunk.dim() != 2:
            raise ValueError("Ожидается 2D тензор формы (chunk_size, n_features)")
        
        if X_chunk.size(1) != self.n_features:
            raise ValueError(f"Ожидается {self.n_features} признаков, получено {X_chunk.size(1)}")
        
        # Инициализация центроидов, если еще не инициализированы
        if self.centroids is None:
            self._init_centroids(X_chunk)
            self._init_optimizer()
        
        # Сохраняем оригинальный learning rate
        original_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = original_lr * learning_ratio
        
        for epoch in range(n_epochs):
            # Прямой проход на чанке
            distances = self.distance_fn(X_chunk, self.centroids)
            assignments = self._compute_assignments(distances)
            
            # Обработка пустых кластеров
            if self.handle_empty_clusters != 'ignore':
                self._handle_empty_clusters(X_chunk, assignments)
                distances = self.distance_fn(X_chunk, self.centroids)
                assignments = self._compute_assignments(distances)
            
            # Вычисление потерь
            loss_kwargs = {}
            if self.loss_function == 'centroid_reg':
                loss_kwargs['centroids'] = self.centroids
            
            loss = self.loss_fn(
                distances=distances, 
                assignments=assignments,
                embeddings=embeddings,
                **loss_kwargs
            )
            
            # Обратное распространение и обновление на каждом шаге
            self.optimizer.zero_grad()
            loss.backward()
            
            # Градиентный clipping
            torch.nn.utils.clip_grad_norm_([self.centroids], max_norm=1.0)
            
            self.optimizer.step()
            
            self.incremental_step += 1
            
            if self.verbose and epoch % 5 == 0:
                cluster_sizes = self._compute_cluster_sizes(assignments)
                empty_count = cluster_sizes.count(0)
                print(f"Incremental step {self.incremental_step}, Epoch {epoch}: "
                      f"Loss = {loss.item():.4f}, Empty clusters = {empty_count}")
        
        # Восстанавливаем оригинальный learning rate
        self.optimizer.param_groups[0]['lr'] = original_lr
        
        return self
    
    def process_large_dataset(self, data_loader: DataLoader, 
                            n_epochs_per_chunk: int = 1) -> 'GradientKMeans':
        """
        Обработка очень большого датасета с использованием DataLoader.
        
        Args:
            data_loader: Загрузчик данных, возвращающий чанки
            n_epochs_per_chunk: Количество эпох на каждый чанк
            
        Returns:
            Обученная модель
        """
        for chunk_idx, (X_chunk,) in enumerate(data_loader):
            if self.verbose:
                print(f"Обработка чанка {chunk_idx + 1}")
            
            self.fit_incremental(X_chunk, n_epochs=n_epochs_per_chunk)
            
            # Периодическая валидация на подмножестве данных
            if chunk_idx % 10 == 0 and self.verbose:
                current_loss = self._validate_on_subset(data_loader)
                silhouette = self.silhouette_score_from_loader(data_loader)
                cluster_sizes = self._compute_cluster_sizes_from_loader(data_loader)
                empty_count = cluster_sizes.count(0)
                
                print(f"Чанк {chunk_idx}, Валидационные потери: {current_loss:.4f}, "
                      f"Silhouette score: {silhouette:.4f}, Empty clusters: {empty_count}")
        
        return self
    
    def _validate_on_subset(self, data_loader: DataLoader, 
                          n_batches: int = 5) -> float:
        """
        Валидация модели на подмножестве данных.
        
        Args:
            data_loader: Загрузчик данных
            n_batches: Количество батчей для валидации
            
        Returns:
            Средние потери на валидационном подмножестве
        """
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for i, (X_batch,) in enumerate(data_loader):
                if i >= n_batches:
                    break
                
                distances = self.distance_fn(X_batch, self.centroids)
                assignments = self._compute_assignments(distances)
                
                loss_kwargs = {}
                if self.loss_function == 'centroid_reg':
                    loss_kwargs['centroids'] = self.centroids
                
                loss = self.loss_fn(
                    distances=distances, 
                    assignments=assignments,
                    **loss_kwargs
                )
                
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def _compute_cluster_sizes_from_loader(self, data_loader: DataLoader, n_batches: int = 10) -> List[int]:
        """Вычисление размеров кластеров на подмножестве данных."""
        cluster_counts = torch.zeros(self.n_clusters, dtype=torch.long)
        total_samples = 0
        
        with torch.no_grad():
            for i, (X_batch,) in enumerate(data_loader):
                if i >= n_batches:
                    break
                
                assignments = self.predict(X_batch)
                counts = torch.bincount(assignments, minlength=self.n_clusters)
                cluster_counts += counts
                total_samples += X_batch.size(0)
        
        return cluster_counts.tolist()
    
    def _inertia(self, X: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
        """Вычисляет инерцию (сумму квадратов расстояний до ближайших центроидов)."""
        distances = self.distance_fn(X, self.centroids)
        min_distances = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
        return min_distances.sum()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Предсказание кластеров для новых данных.
        
        Args:
            X: Входные данные формы (n_samples, n_features)
            
        Returns:
            Назначения кластеров формы (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите fit().")
        
        with torch.no_grad():
            distances = self.distance_fn(X, self.centroids)
            return self._compute_assignments(distances)
    
    def fit_predict(self, X: torch.Tensor, embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Обучение и предсказание на одних данных.
        
        Args:
            X: Входные данные
            embeddings: Эмбеддинги для контрастных потерь
            
        Returns:
            Назначения кластеров
        """
        self.fit(X, embeddings)
        return self.predict(X)
    
    def get_centroids(self) -> torch.Tensor:
        """Получение текущих центроидов кластеров."""
        if self.centroids is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите fit().")
        
        return self.centroids.data.detach().clone()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Получение истории обучения."""
        return self.history
    
    def get_cluster_sizes_history(self) -> List[List[int]]:
        """Получение истории размеров кластеров."""
        return self.cluster_sizes_history
    
    def score(self, X: torch.Tensor) -> float:
        """
        Оценка качества кластеризации (within-cluster sum of squares).
        
        Args:
            X: Входные данные
            
        Returns:
            WCSS score (меньше = лучше)
        """
        assignments = self.predict(X)
        distances = self.distance_fn(X, self.centroids)
        min_distances = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
        return min_distances.mean().item()
    
    def inertia_(self, X: torch.Tensor) -> float:
        """
        Вычисляет инерцию (сумму квадратов расстояний до ближайших центроидов).
        
        Args:
            X: Входные данные
            
        Returns:
            Inertia value (меньше = лучше)
        """
        assignments = self.predict(X)
        return self._inertia(X, assignments).item()
    
    def silhouette_score(self, X: torch.Tensor) -> float:
        """
        Вычисляет silhouette score для оценки качества кластеризации.
        
        Args:
            X: Входные данные
            
        Returns:
            Silhouette score (-1 до 1, больше = лучше)
        """
        assignments = self.predict(X).cpu().numpy()
        X_np = X.cpu().numpy()
        
        if len(np.unique(assignments)) < 2:
            return -1.0  # Нельзя вычислить для одного кластера
        
        return silhouette_score(X_np, assignments)
    
    def silhouette_score_from_loader(self, data_loader: DataLoader, n_samples: int = 1000) -> float:
        """
        Вычисляет silhouette score на подмножестве данных из DataLoader.
        
        Args:
            data_loader: Загрузчик данных
            n_samples: Количество семплов для оценки
            
        Returns:
            Silhouette score
        """
        all_data = []
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, in data_loader:
                all_data.append(X_batch)
                total_samples += X_batch.size(0)
                if total_samples >= n_samples:
                    break
        
        if not all_data:
            return -1.0
        
        X_subset = torch.cat(all_data, dim=0)[:n_samples]
        return self.silhouette_score(X_subset)
    
    def get_cluster_balance(self) -> float:
        """
        Вычисляет баланс кластеров (отношение минимального размера к максимальному).
        
        Returns:
            Коэффициент баланса (0-1, где 1 - идеальный баланс)
        """
        if not self.cluster_sizes_history:
            return 0.0
        
        current_sizes = self.cluster_sizes_history[-1]
        non_empty_sizes = [size for size in current_sizes if size > 0]
        
        if len(non_empty_sizes) < 2:
            return 0.0
        
        return min(non_empty_sizes) / max(non_empty_sizes)
    
    def copy(self) -> 'GradientKMeans':
        """Создает копию модели."""
        if self.centroids is None:
            raise ValueError("Модель еще не обучена. Нельзя создать копию.")
        
        # Создаем новую модель с теми же параметрами
        new_model = GradientKMeans(
            n_clusters=self.n_clusters,
            n_features=self.n_features,
            distance_metric=self.distance_metric,
            loss_function=self.loss_function,
            init_method=self.init_method,
            optimizer=self.optimizer_name,
            lr=self.lr,
            temperature=self.temperature,
            entropy_alpha=self.entropy_alpha,
            reg_lambda=self.reg_lambda,
            reg_min_distance=self.reg_min_distance,
            max_iters=self.max_iters,
            tol=self.tol,
            verbose=self.verbose
        )
        
        new_model.centroids = nn.Parameter(self.centroids.data.clone())
        new_model._init_optimizer()
        
        # Копируем историю
        new_model.history = self.history.copy()
        new_model.cluster_sizes_history = self.cluster_sizes_history.copy()
        
        return new_model
    
    def save(self, path: str):
        """
        Сохраняет модель в файл.
        
        Args:
            path: Путь для сохранения
        """
        if self.centroids is None:
            raise ValueError("Модель еще не обучена. Нечего сохранять.")
        
        torch.save({
            'n_clusters': self.n_clusters,
            'n_features': self.n_features,
            'centroids': self.centroids.data,
            'history': self.history,
            'cluster_sizes_history': self.cluster_sizes_history,
            'distance_metric': self.distance_metric,
            'loss_function': self.loss_function,
            'init_method': self.init_method,
            'optimizer_state': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'GradientKMeans':
        """
        Загружает модель из файла.
        
        Args:
            path: Путь к файлу модели
            
        Returns:
            Загруженная модель
        """
        checkpoint = torch.load(path)
        
        model = cls(
            n_clusters=checkpoint['n_clusters'],
            n_features=checkpoint['n_features'],
            distance_metric=checkpoint['distance_metric'],
            loss_function=checkpoint['loss_function'],
            init_method=checkpoint['init_method'],
        )
        
        model.centroids = nn.Parameter(checkpoint['centroids'])
        model._init_optimizer()
        model.history = checkpoint['history']
        model.cluster_sizes_history = checkpoint.get('cluster_sizes_history', [])
        
        if hasattr(model, 'optimizer') and checkpoint.get('optimizer_state'):
            model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        return model

class KModes:
    """
    K-modes для категориальных данных с использованием расстояния Хемминга.
    
    Реализует классический алгоритм K-modes (без градиентной оптимизации).
    
    Args:
        n_clusters: Количество кластеров
        n_features: Размерность признакового пространства
        init_method: Метод инициализации ('random', 'kmeans++')
        max_iters: Максимальное количество итераций
        tol: Порог сходимости
        verbose: Вывод информации о процессе
        chunk_size: Размер чанка для инкрементного обучения
        early_stopping_patience: Терпение для ранней остановки
        handle_empty_clusters: Стратегия обработки пустых кластеров
    """
    
    def __init__(self, n_clusters: int, n_features: int, **kwargs):
        # Валидация входных параметров
        if n_clusters <= 0:
            raise ValueError("n_clusters должен быть положительным числом")
        if n_features <= 0:
            raise ValueError("n_features должен быть положительным числом")
        
        self.n_clusters = n_clusters
        self.n_features = n_features
        
        self.init_method = kwargs.get('init_method', 'random')
        self.max_iters = kwargs.get('max_iters', 100)
        self.tol = kwargs.get('tol', 1e-4)
        self.verbose = kwargs.get('verbose', False)
        self.chunk_size = kwargs.get('chunk_size', 1000)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.handle_empty_clusters = kwargs.get('handle_empty_clusters', 'reinitialize')
        
        # Валидация гиперпараметров
        if self.max_iters <= 0:
            raise ValueError("max_iters должен быть положительным")
        if self.tol <= 0:
            raise ValueError("tol должен быть положительным")
        
        self.distance_fn = HammingDistance()
        self.history = []
        self.cluster_sizes_history = []
        self.centroids = None
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def _init_centroids(self, X: torch.Tensor):
        """Инициализация центроидов для категориальных данных."""
        n_samples = X.shape[0]
        
        if n_samples < self.n_clusters:
            raise ValueError(f"Количество образцов ({n_samples}) должно быть >= n_clusters ({self.n_clusters})")
        
        if self.init_method == 'random':
            # Выбираем случайные точки как начальные центроиды
            indices = torch.randint(0, n_samples, (self.n_clusters,))
            self.centroids = X[indices].clone()
        elif self.init_method == 'kmeans++':
            self.centroids = self._kmodes_plus_plus_init(X)
        else:
            raise ValueError(f"Неподдерживаемый метод инициализации: {self.init_method}")
    
    def _kmodes_plus_plus_init(self, X: torch.Tensor) -> torch.Tensor:
        """Упрощенная k-modes++ инициализация."""
        n_samples = X.shape[0]
        centroids = torch.zeros((self.n_clusters, self.n_features), dtype=X.dtype, device=X.device)
        
        # Первый центроид выбирается случайно
        first_idx = torch.randint(0, n_samples, (1,))
        centroids[0] = X[first_idx].clone()
        
        for k in range(1, self.n_clusters):
            # Вычисляем расстояния до ближайшего центроида
            distances = self.distance_fn(X, centroids[:k])
            min_distances = distances.min(dim=1).values
            
            # Вероятность пропорциональна расстоянию (не квадрату, т.к. категориальные данные)
            probabilities = min_distances
            if probabilities.sum() == 0:
                # Если все расстояния нулевые, выбираем случайно
                next_idx = torch.randint(0, n_samples, (1,))
            else:
                probabilities /= probabilities.sum()
                next_idx = torch.multinomial(probabilities, 1)
            
            centroids[k] = X[next_idx].clone()
        
        return centroids
    
    def _compute_assignments(self, distances: torch.Tensor) -> torch.Tensor:
        """Вычисление назначений кластеров."""
        return distances.argmin(dim=1)
    
    def _compute_cluster_sizes(self, assignments: torch.Tensor) -> List[int]:
        """Вычисление размеров кластеров."""
        cluster_sizes = []
        for k in range(self.n_clusters):
            size = (assignments == k).sum().item()
            cluster_sizes.append(size)
        return cluster_sizes
    
    def _handle_empty_clusters(self, X: torch.Tensor, assignments: torch.Tensor):
        """Обработка пустых кластеров для KModes."""
        if self.handle_empty_clusters == 'ignore':
            return
        
        cluster_sizes = self._compute_cluster_sizes(assignments)
        empty_clusters = [i for i, size in enumerate(cluster_sizes) if size == 0]
        
        if empty_clusters and self.verbose:
            print(f"Обнаружены пустые кластеры: {empty_clusters}")
        
        for cluster_idx in empty_clusters:
            if self.handle_empty_clusters == 'reinitialize':
                # Находим точку с максимальным расстоянием до ближайшего центроида
                distances = self.distance_fn(X, self.centroids)
                min_distances = distances.min(dim=1).values
                farthest_idx = min_distances.argmax()
                
                # Переинициализируем центроид
                self.centroids[cluster_idx] = X[farthest_idx].clone()
                
                if self.verbose:
                    print(f"Переинициализирован центроид кластера {cluster_idx}")
    
    def _update_centroids(self, X: torch.Tensor, assignments: torch.Tensor):
        """Обновление центроидов (мод) для категориальных данных."""
        new_centroids = torch.zeros_like(self.centroids)
        
        for k in range(self.n_clusters):
            cluster_mask = (assignments == k)
            if cluster_mask.sum() > 0:
                cluster_data = X[cluster_mask]
                # Для каждого признака находим моду (наиболее частое значение)
                for j in range(self.n_features):
                    values, counts = cluster_data[:, j].unique(return_counts=True)
                    mode = values[counts.argmax()]
                    new_centroids[k, j] = mode
            else:
                # Если кластер пуст, оставляем старый центроид
                new_centroids[k] = self.centroids[k]
        
        return new_centroids
    
    def _check_early_stopping(self, loss: float) -> bool:
        """Проверка условия ранней остановки."""
        if loss < self.best_loss - self.tol:
            self.best_loss = loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience
    
    def fit(self, X: torch.Tensor) -> 'KModes':
        """
        Обучение K-modes на данных.
        
        Args:
            X: Категориальные данные формы (n_samples, n_features)
            
        Returns:
            Обученная модель
        """
        if X.dim() != 2:
            raise ValueError("Ожидается 2D тензор формы (n_samples, n_features)")
        
        if X.size(1) != self.n_features:
            raise ValueError(f"Ожидается {self.n_features} признаков, получено {X.size(1)}")
        
        if X.size(0) < self.n_clusters:
            raise ValueError(f"Количество образцов ({X.size(0)}) должно быть >= n_clusters ({self.n_clusters})")
        
        # Инициализация центроидов
        if self.centroids is None:
            self._init_centroids(X)
        
        self.history = []
        self.cluster_sizes_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        prev_centroids = self.centroids.clone()
        
        for iteration in range(self.max_iters):
            # Вычисление расстояний и назначений
            distances = self.distance_fn(X, self.centroids)
            assignments = self._compute_assignments(distances)
            
            # Отслеживание размеров кластеров
            cluster_sizes = self._compute_cluster_sizes(assignments)
            self.cluster_sizes_history.append(cluster_sizes)
            
            # Обработка пустых кластеров
            if self.handle_empty_clusters != 'ignore':
                self._handle_empty_clusters(X, assignments)
                # Пересчитываем после возможной переинициализации
                distances = self.distance_fn(X, self.centroids)
                assignments = self._compute_assignments(distances)
            
            # Обновление центроидов
            new_centroids = self._update_centroids(X, assignments)
            
            # Проверка сходимости
            centroid_change = (new_centroids != prev_centroids).float().mean()
            
            # Вычисление потерь
            loss = distances.gather(1, assignments.unsqueeze(1)).mean()
            
            # Сохранение истории
            history_entry = {
                'iteration': iteration,
                'loss': loss.item(),
                'assignments': assignments.clone(),
                'centroid_change': centroid_change.item(),
                'cluster_sizes': cluster_sizes,
                'empty_clusters': cluster_sizes.count(0)
            }
            self.history.append(history_entry)
            
            # Проверка ранней остановки
            if self._check_early_stopping(loss.item()):
                if self.verbose:
                    print(f"Ранняя остановка на итерации {iteration}")
                break
            
            self.centroids = new_centroids
            prev_centroids = self.centroids.clone()
            
            if centroid_change < self.tol:
                if self.verbose:
                    print(f"Сходимость достигнута на итерации {iteration}")
                break
            
            if self.verbose and iteration % 10 == 0:
                empty_count = cluster_sizes.count(0)
                print(f"Iteration {iteration}: Loss = {loss.item():.4f}, "
                      f"Centroid change = {centroid_change:.4f}, "
                      f"Empty clusters = {empty_count}/{self.n_clusters}")
        
        if self.verbose:
            final_cluster_sizes = self._compute_cluster_sizes(assignments)
            empty_count = final_cluster_sizes.count(0)
            print(f"Обучение завершено. Пустых кластеров: {empty_count}/{self.n_clusters}")
        
        return self
    
    def fit_incremental(self, X_chunk: torch.Tensor) -> 'KModes':
        """
        Инкрементное обучение K-modes на чанке данных.
        
        Args:
            X_chunk: Чанк категориальных данных
            
        Returns:
            Обновленная модель
        """
        return self.fit(X_chunk)  # K-modes не поддерживает true инкрементное обучение
    
    def process_large_dataset(self, data_loader: DataLoader) -> 'KModes':
        """
        Обработка большого датасета категориальных данных.
        
        Args:
            data_loader: Загрузчик категориальных данных
            
        Returns:
            Обученная модель
        """
        # Для K-modes обрабатываем все данные сразу
        all_data = []
        for (X_batch,) in data_loader:
            all_data.append(X_batch)
        
        X_all = torch.cat(all_data, dim=0)
        return self.fit(X_all)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Предсказание кластеров для новых категориальных данных.
        
        Args:
            X: Категориальные данные
            
        Returns:
            Назначения кластеров
        """
        if self.centroids is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите fit().")
        
        with torch.no_grad():
            distances = self.distance_fn(X, self.centroids)
            return self._compute_assignments(distances)
    
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Обучение и предсказание на одних данных.
        
        Args:
            X: Категориальные данные
            
        Returns:
            Назначения кластеров
        """
        self.fit(X)
        return self.predict(X)
    
    def get_centroids(self) -> torch.Tensor:
        """Получение текущих центроидов (мод)."""
        if self.centroids is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите fit().")
        
        return self.centroids.clone()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Получение истории обучения."""
        return self.history
    
    def get_cluster_sizes_history(self) -> List[List[int]]:
        """Получение истории размеров кластеров."""
        return self.cluster_sizes_history
    
    def score(self, X: torch.Tensor) -> float:
        """
        Оценка качества кластеризации.
        
        Args:
            X: Категориальные данные
            
        Returns:
            Среднее расстояние Хемминга до центроидов
        """
        assignments = self.predict(X)
        distances = self.distance_fn(X, self.centroids)
        min_distances = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
        return min_distances.mean().item()
    
    def get_cluster_balance(self) -> float:
        """
        Вычисляет баланс кластеров.
        
        Returns:
            Коэффициент баланса (0-1, где 1 - идеальный баланс)
        """
        if not self.cluster_sizes_history:
            return 0.0
        
        current_sizes = self.cluster_sizes_history[-1]
        non_empty_sizes = [size for size in current_sizes if size > 0]
        
        if len(non_empty_sizes) < 2:
            return 0.0
        
        return min(non_empty_sizes) / max(non_empty_sizes)
    
    def copy(self) -> 'KModes':
        """Создает копию модели."""
        if self.centroids is None:
            raise ValueError("Модель еще не обучена. Нельзя создать копию.")
        
        new_model = KModes(self.n_clusters, self.n_features)
        new_model.centroids = self.centroids.clone()
        new_model.history = self.history.copy()
        new_model.cluster_sizes_history = self.cluster_sizes_history.copy()
        
        return new_model
    
    def save(self, path: str):
        """
        Сохраняет модель в файл.
        
        Args:
            path: Путь для сохранения
        """
        if self.centroids is None:
            raise ValueError("Модель еще не обучена. Нечего сохранять.")
        
        torch.save({
            'n_clusters': self.n_clusters,
            'n_features': self.n_features,
            'centroids': self.centroids,
            'history': self.history,
            'cluster_sizes_history': self.cluster_sizes_history,
            'init_method': self.init_method,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'KModes':
        """
        Загружает модель из файла.
        
        Args:
            path: Путь к файлу модели
            
        Returns:
            Загруженная модель
        """
        checkpoint = torch.load(path)
        
        model = cls(
            n_clusters=checkpoint['n_clusters'],
            n_features=checkpoint['n_features'],
            init_method=checkpoint['init_method'],
        )
        
        model.centroids = checkpoint['centroids']
        model.history = checkpoint['history']
        model.cluster_sizes_history = checkpoint.get('cluster_sizes_history', [])
        
        return model

# Дополнительные утилиты
def create_data_loader(X: torch.Tensor, batch_size: int = 1000, shuffle: bool = True) -> DataLoader:
    """
    Создает DataLoader для обработки больших датасетов.
    
    Args:
        X: Входные данные
        batch_size: Размер батча
        shuffle: Перемешивать ли данные
        
    Returns:
        DataLoader для итеративной обработки
    """
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def calculate_cluster_metrics(model: Union[GradientKMeans, KModes], X: torch.Tensor) -> Dict[str, float]:
    """
    Вычисляет различные метрики качества кластеризации.
    
    Args:
        model: Обученная модель кластеризации
        X: Входные данные
        
    Returns:
        Словарь с метриками качества
    """
    assignments = model.predict(X)
    
    metrics = {
        'inertia': model.inertia_(X) if hasattr(model, 'inertia_') else model.score(X),
        'n_clusters': len(torch.unique(assignments)),
        'n_empty_clusters': model.n_clusters - len(torch.unique(assignments))
    }
    
    # Вычисляем silhouette score только для GradientKMeans с непрерывными данными
    if isinstance(model, GradientKMeans) and not isinstance(model.distance_fn, HammingDistance):
        try:
            metrics['silhouette_score'] = model.silhouette_score(X)
        except:
            metrics['silhouette_score'] = -1.0
    
    # Вычисляем размеры кластеров
    cluster_sizes = []
    for k in range(model.n_clusters):
        cluster_size = (assignments == k).sum().item()
        if cluster_size > 0:
            cluster_sizes.append(cluster_size)
    
    if cluster_sizes:
        metrics['min_cluster_size'] = min(cluster_sizes)
        metrics['max_cluster_size'] = max(cluster_sizes)
        metrics['avg_cluster_size'] = sum(cluster_sizes) / len(cluster_sizes)
        metrics['cluster_balance'] = metrics['min_cluster_size'] / metrics['max_cluster_size']
    
    return metrics

def find_optimal_clusters(X: torch.Tensor, max_k: int = 15, n_init: int = 3, **kwargs) -> Dict[int, Dict[str, float]]:
    """
    Поиск оптимального количества кластеров с использованием метода локтя и silhouette score.
    
    Args:
        X: Входные данные
        max_k: Максимальное количество кластеров для проверки
        n_init: Количество инициализаций для каждого k
        **kwargs: Дополнительные параметры для GradientKMeans
        
    Returns:
        Словарь с метриками для каждого k
    """
    results = {}
    
    for k in range(2, max_k + 1):
        inertias = []
        silhouette_scores = []
        cluster_balances = []
        
        for init in range(n_init):
            try:
                model = GradientKMeans(n_clusters=k, n_features=X.shape[1], **kwargs)
                model.fit(X)
                
                inertias.append(model.inertia_(X))
                
                # Вычисляем silhouette score только если есть более 1 непустого кластера
                if len(torch.unique(model.predict(X))) > 1:
                    silhouette_scores.append(model.silhouette_score(X))
                
                cluster_balances.append(model.get_cluster_balance())
                
            except Exception as e:
                if kwargs.get('verbose', False):
                    print(f"Ошибка для k={k}, инициализация {init}: {e}")
        
        if inertias:
            results[k] = {
                'inertia': np.mean(inertias),
                'inertia_std': np.std(inertias),
                'silhouette_score': np.mean(silhouette_scores) if silhouette_scores else -1,
                'silhouette_std': np.std(silhouette_scores) if silhouette_scores else 0,
                'cluster_balance': np.mean(cluster_balances),
                'cluster_balance_std': np.std(cluster_balances)
            }
    
    return results