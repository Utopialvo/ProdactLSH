import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances_argmin_min
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from IPython.display import HTML, display
import ipywidgets as widgets
from typing import Tuple, List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
import pickle
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import silhouette_score, adjusted_rand_score, mean_squared_error
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import time
from functools import wraps
from sklearn.utils.extmath import randomized_svd

# Настройка стилей
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Декоратор для измерения времени выполнения
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} выполнена за {end_time - start_time:.4f} секунд")
        return result
    return wrapper

# Классы данных
@dataclass
class QuantizationResult:
    codes: np.ndarray
    reconstructed: np.ndarray
    accuracy: float
    recall: float
    mse: float
    compression_ratio: float
    clustering_metrics: Dict[str, float] = field(default_factory=dict)
    original_data: Optional[np.ndarray] = None
    method_name: str = ""
    execution_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PQConfig:
    num_subspaces: int = 8
    num_clusters: int = 256
    use_diffusion: bool = False
    use_optimized_kmeans: bool = True
    batch_size: int = 1000
    diffusion_params: Dict[str, Any] = field(default_factory=lambda: {
        "gamma": 1.0, 
        "n_components": 48,
        "alpha": 0.5,
        "use_randomized_svd": True
    })
    compression_params: Dict[str, Any] = field(default_factory=lambda: {
        "threshold": 0.05,
        "min_centroids": 10
    })
    calibration_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_epochs": 5,
        "learning_rate": 0.01
    })
    distance_metric: str = 'euclidean'
    random_state: int = 42

# Базовый класс квантователя
class BaseQuantizer(ABC):
    @abstractmethod
    def train(self, data: np.ndarray, **kwargs) -> None:
        pass
    
    @abstractmethod
    def encode(self, data: np.ndarray, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def decode(self, codes: np.ndarray, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def calculate_compression_ratio(self, n_vectors: int = 1000, original_dtype: type = np.float32, **kwargs) -> float:
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        memory_usage = {}
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, 'nbytes'):
                memory_usage[attr_name] = attr_value.nbytes / (1024 * 1024)
        return memory_usage

# Реализация продуктного квантователя
class ProductQuantizer(BaseQuantizer):
    def __init__(self, config: Optional[PQConfig] = None, **kwargs):
        if config is None:
            config = PQConfig()
        
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        
        self.num_subspaces = config_dict.get('num_subspaces', 8)
        self.num_clusters = config_dict.get('num_clusters', 256)
        self.random_state = config_dict.get('random_state', 42)
        self.use_optimized_kmeans = config_dict.get('use_optimized_kmeans', True)
        self.batch_size = config_dict.get('batch_size', 1000)
        self.distance_metric = config_dict.get('distance_metric', 'euclidean')
        
        if self.num_subspaces <= 0:
            raise ValueError("num_subspaces must be positive")
        if self.num_clusters <= 0 or self.num_clusters > 65536:
            raise ValueError("num_clusters must be between 1 and 65536")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        self.codebooks = []
        self.sub_dim = None
        self.kmeans_models = []
        self.is_trained = False
        self.original_dim = None
        
    @timer
    def train(self, data: np.ndarray, **kwargs) -> None:
        n_samples, dim = data.shape
        self.original_dim = dim
        
        if dim % self.num_subspaces != 0:
            self.num_subspaces = self._find_optimal_subspaces(dim)
            logger.info(f"Автоматическая корректировка: количество подпространств изменено на {self.num_subspaces}")
        
        self.sub_dim = dim // self.num_subspaces
        
        for i in range(self.num_subspaces):
            start_idx = i * self.sub_dim
            end_idx = (i + 1) * self.sub_dim
            sub_data = data[:, start_idx:end_idx]
            
            if self.use_optimized_kmeans and n_samples > self.batch_size * 2:
                logger.info(f"Обучение подпространства {i+1} с использованием MiniBatchKMeans")
                kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, 
                                        random_state=self.random_state,
                                        batch_size=self.batch_size,
                                        n_init=3)
            else:
                logger.info(f"Обучение подпространства {i+1} с использованием KMeans")
                kmeans = KMeans(n_clusters=self.num_clusters, 
                               random_state=self.random_state, 
                               n_init=10)
            
            kmeans.fit(sub_data)
            
            self.codebooks.append(kmeans.cluster_centers_)
            self.kmeans_models.append(kmeans)
        
        self.is_trained = True
        logger.info(f"Обучение завершено. Подпространств: {self.num_subspaces}, размерность подпространства: {self.sub_dim}")
    
    def _find_optimal_subspaces(self, dim: int) -> int:
        divisors = []
        for i in range(1, dim + 1):
            if dim % i == 0:
                divisors.append(i)
        
        divisors = np.array(divisors)
        closest_idx = np.argmin(np.abs(divisors - self.num_subspaces))
        return divisors[closest_idx]
    
    @timer
    def encode(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Квантователь не обучен. Сначала вызовите метод train().")
            
        n_samples = data.shape[0]
        codes = np.zeros((n_samples, self.num_subspaces), dtype=np.int32)
        
        for i in range(self.num_subspaces):
            start_idx = i * self.sub_dim
            end_idx = (i + 1) * self.sub_dim
            sub_data = data[:, start_idx:end_idx]
            
            codes[:, i] = pairwise_distances_argmin_min(
                sub_data, self.codebooks[i], metric=self.distance_metric)[0]
            
        return codes
    
    @timer
    def decode(self, codes: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Квантователь не обучен. Сначала вызовите метод train().")
            
        n_samples = codes.shape[0]
        decoded = np.zeros((n_samples, self.sub_dim * self.num_subspaces))
        
        for i in range(self.num_subspaces):
            centroids = self.codebooks[i]
            decoded[:, i*self.sub_dim:(i+1)*self.sub_dim] = centroids[codes[:, i]]
            
        return decoded
    
    def calculate_compression_ratio(self, n_vectors: int = 1000, original_dtype: type = np.float32, **kwargs) -> float:
        if not self.is_trained:
            raise ValueError("Квантователь не обучен. Сначала вызовите метод train().")
            
        original_size = n_vectors * self.original_dim * np.dtype(original_dtype).itemsize
        compressed_size = n_vectors * self.num_subspaces * np.dtype(np.int32).itemsize
        
        codebook_size = 0
        for codebook in self.codebooks:
            codebook_size += codebook.size * np.dtype(np.float32).itemsize
        
        return original_size / (compressed_size + codebook_size)
    
    def save(self, filepath: str) -> None:
        if not self.is_trained:
            raise ValueError("Модель не обучена. Нечего сохранять.")
            
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'num_subspaces': self.num_subspaces,
                'num_clusters': self.num_clusters,
                'random_state': self.random_state,
                'codebooks': self.codebooks,
                'sub_dim': self.sub_dim,
                'original_dim': self.original_dim,
                'is_trained': self.is_trained,
                'use_optimized_kmeans': self.use_optimized_kmeans,
                'batch_size': self.batch_size,
                'distance_metric': self.distance_metric
            }, f)
        
        logger.info(f"Модель сохранена в {filepath}")
    
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.num_subspaces = data['num_subspaces']
        self.num_clusters = data['num_clusters']
        self.random_state = data['random_state']
        self.codebooks = data['codebooks']
        self.sub_dim = data['sub_dim']
        self.original_dim = data['original_dim']
        self.is_trained = data['is_trained']
        self.use_optimized_kmeans = data.get('use_optimized_kmeans', True)
        self.batch_size = data.get('batch_size', 1000)
        self.distance_metric = data.get('distance_metric', 'euclidean')
        
        self.kmeans_models = []
        
        logger.info(f"Модель загружена из {filepath}")
    
    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self.encode(data, **kwargs)

# Расширенный продуктный квантователь
class AdvancedProductQuantizer(ProductQuantizer):
    def __init__(self, config: Optional[PQConfig] = None, **kwargs):
        if config is None:
            config = PQConfig()
        
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        
        num_subspaces = config_dict.get('num_subspaces', 8)
        num_clusters = config_dict.get('num_clusters', 256)
        random_state = config_dict.get('random_state', 42)
        use_optimized_kmeans = config_dict.get('use_optimized_kmeans', True)
        batch_size = config_dict.get('batch_size', 1000)
        distance_metric = config_dict.get('distance_metric', 'euclidean')
        
        super().__init__(
            PQConfig(
                num_subspaces=num_subspaces,
                num_clusters=num_clusters,
                random_state=random_state,
                use_optimized_kmeans=use_optimized_kmeans,
                batch_size=batch_size,
                distance_metric=distance_metric
            )
        )
        
        diffusion_params = config_dict.get('diffusion_params', {})
        self.gamma = diffusion_params.get('gamma', 1.0)
        self.n_components = diffusion_params.get('n_components', 48)
        self.alpha = diffusion_params.get('alpha', 0.5)
        self.use_randomized_svd = diffusion_params.get('use_randomized_svd', True)
        
        compression_params = config_dict.get('compression_params', {})
        self.compression_threshold = compression_params.get('threshold', 0.05)
        self.min_centroids = compression_params.get('min_centroids', 10)
        
        self.calibration_params = config_dict.get('calibration_params', {})
        
        self.diffusion_map = None
        self.affinity_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.original_data = None
        self.knn_model = None
        self.codebook_importance = []
        self.compressed_codebooks = []
        self.codebook_mapping = []
        self.use_diffusion = False
        self.diffusion_data = None
        
    def _build_affinity_matrix(self, data: np.ndarray) -> np.ndarray:
        n_samples = data.shape[0]
        
        expected_memory = n_samples * n_samples * 8 / (1024 ** 3)
        if expected_memory > 2:
            logger.warning(f"Матрица сходства потребует {expected_memory:.1f} GB памяти. Используем приближенное вычисление.")
            
            n_subset = min(5000, n_samples)
            indices = np.random.choice(n_samples, n_subset, replace=False)
            subset = data[indices]
            
            distances = cdist(data, subset, 'sqeuclidean')
            affinity = np.exp(-self.gamma * distances)
            
            return normalize(affinity, norm='l1', axis=1)
        else:
            affinity = rbf_kernel(data, data, gamma=self.gamma)
            return normalize(affinity, norm='l1', axis=1)
    
    @timer
    def _build_diffusion_map(self, data: np.ndarray) -> None:
        n_samples = data.shape[0]
        
        logger.info("Построение матрицы сходства...")
        normalized_affinity = self._build_affinity_matrix(data)
        
        logger.info("Построение матрицы диффузии...")
        diffusion_matrix = normalized_affinity.T @ normalized_affinity
        diffusion_matrix = normalize(diffusion_matrix, norm='l1', axis=1)
        
        logger.info("Вычисление собственных значений и векторов...")
        if self.use_randomized_svd and n_samples > 1000:
            U, Sigma, VT = randomized_svd(diffusion_matrix, 
                                         n_components=self.n_components,
                                         random_state=self.random_state)
            self.eigenvalues = Sigma
            self.eigenvectors = U
        else:
            self.eigenvalues, self.eigenvectors = eigh(diffusion_matrix)
            
            idx = np.argsort(self.eigenvalues)[::-1][:self.n_components]
            self.eigenvalues = self.eigenvalues[idx]
            self.eigenvectors = self.eigenvectors[:, idx]
        
        self.diffusion_map = self.eigenvectors * (self.eigenvalues ** self.alpha)
        self.diffusion_data = self.diffusion_map
    
    def _nystrom_extension(self, new_data: np.ndarray) -> np.ndarray:
        affinity_new = rbf_kernel(new_data, self.original_data, gamma=self.gamma)
        affinity_new = normalize(affinity_new, norm='l1', axis=1)
        diffusion_new = affinity_new @ self.eigenvectors / self.eigenvalues
        return diffusion_new
    
    def _calculate_codebook_importance(self, codes: np.ndarray) -> None:
        self.codebook_importance = []
        for i in range(self.num_subspaces):
            unique, counts = np.unique(codes[:, i], return_counts=True)
            importance = np.zeros(self.num_clusters)
            importance[unique] = counts
            self.codebook_importance.append(importance / np.sum(importance))
    
    def _compress_codebooks(self) -> None:
        self.compressed_codebooks = []
        self.codebook_mapping = []
        
        for i in range(self.num_subspaces):
            codebook = self.codebooks[i]
            importance = self.codebook_importance[i]
            
            sorted_indices = np.argsort(importance)[::-1]
            sorted_codebook = codebook[sorted_indices]
            
            compressed_codebook = []
            mapping = np.zeros(len(codebook), dtype=int)
            
            for j, centroid in enumerate(sorted_codebook):
                if len(compressed_codebook) < self.min_centroids:
                    compressed_codebook.append(centroid)
                    mapping[sorted_indices[j]] = len(compressed_codebook) - 1
                else:
                    distances = cdist([centroid], compressed_codebook, self.distance_metric)[0]
                    min_distance = np.min(distances)
                    
                    if min_distance > self.compression_threshold:
                        compressed_codebook.append(centroid)
                        mapping[sorted_indices[j]] = len(compressed_codebook) - 1
                    else:
                        mapping[sorted_indices[j]] = np.argmin(distances)
            
            self.compressed_codebooks.append(np.array(compressed_codebook))
            self.codebook_mapping.append(mapping)
            
            logger.info(f"Подпространство {i+1}: сжато с {len(codebook)} до {len(compressed_codebook)} центроидов")
    
    @timer
    def train(self, data: np.ndarray, use_diffusion: bool = False, calibrate: bool = False, **kwargs) -> None:
        self.original_data = data
        self.use_diffusion = use_diffusion
        
        if use_diffusion:
            if self.n_components > data.shape[0]:
                self.n_components = data.shape[0] - 1
                logger.warning(f"n_components превышает количество образцов. Установлено значение: {self.n_components}")
            
            logger.info("Построение диффузионной карты...")
            self._build_diffusion_map(data)
            train_data = self.diffusion_map
            
            self.knn_model = NearestNeighbors(n_neighbors=5, metric=self.distance_metric).fit(self.diffusion_data)
        else:
            train_data = data
            self.knn_model = NearestNeighbors(n_neighbors=5, metric=self.distance_metric).fit(self.original_data)
        
        dim = train_data.shape[1]
        if dim % self.num_subspaces != 0:
            self.num_subspaces = self._find_optimal_subspaces(dim)
            logger.info(f"Автокоррекция: количество подпространств изменено на {self.num_subspaces}")
        
        batch_size = kwargs.get('batch_size', self.batch_size)
        super().train(train_data, batch_size=batch_size)
        
        codes = self.encode(train_data)
        self._calculate_codebook_importance(codes)
        
        self._compress_codebooks()
        
        if calibrate:
            self.calibrate(data, use_diffusion)
        
        logger.info("Обучение завершено")
    
    def encode(self, data: np.ndarray, use_diffusion: bool = False, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Квантователь не обучен. Сначала вызовите метод train().")
            
        if use_diffusion:
            diffusion_data = self._nystrom_extension(data)
            data_to_encode = diffusion_data
        else:
            data_to_encode = data
        
        return super().encode(data_to_encode)
    
    def decode(self, codes: np.ndarray, use_compressed: bool = True, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Квантователь не обучен. Сначала вызовите метод train().")
            
        n_samples = codes.shape[0]
        
        if self.use_diffusion:
            decoded = np.zeros((n_samples, self.sub_dim * self.num_subspaces))
        else:
            decoded = np.zeros((n_samples, self.sub_dim * self.num_subspaces))
        
        for i in range(self.num_subspaces):
            if use_compressed and hasattr(self, 'compressed_codebooks') and self.compressed_codebooks:
                centroids = self.compressed_codebooks[i]
                mapped_codes = self.codebook_mapping[i][codes[:, i]]
                decoded[:, i*self.sub_dim:(i+1)*self.sub_dim] = centroids[mapped_codes]
            else:
                centroids = self.codebooks[i]
                decoded[:, i*self.sub_dim:(i+1)*self.sub_dim] = centroids[codes[:, i]]
            
        return decoded
    
    def reconstruct_to_original_space(self, codes: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Квантователь не обучен. Сначала вызовите метод train().")
            
        if not self.use_diffusion:
            raise ValueError("Метод reconstruct_to_original_space применим только при use_diffusion=True")
            
        diffusion_reconstructed = self.decode(codes, use_compressed=True)
        
        distances, indices_nn = self.knn_model.kneighbors(diffusion_reconstructed)
        
        weights = 1.0 / (distances + 1e-8)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        original_reconstructed = np.zeros((diffusion_reconstructed.shape[0], self.original_data.shape[1]))
        for i in range(diffusion_reconstructed.shape[0]):
            original_reconstructed[i] = np.average(self.original_data[indices_nn[i]], axis=0, weights=weights[i])
        
        return original_reconstructed
    
    @timer
    def calibrate(self, data: np.ndarray, use_diffusion: bool = False, **kwargs) -> None:
        n_epochs = kwargs.get('n_epochs', self.calibration_params.get('n_epochs', 5))
        learning_rate = kwargs.get('learning_rate', self.calibration_params.get('learning_rate', 0.01))
        
        if use_diffusion:
            calibration_data = self._nystrom_extension(data)
        else:
            calibration_data = data
        
        logger.info(f"Начало калибровки на {n_epochs} эпох")
        
        for epoch in range(n_epochs):
            total_loss = 0
            
            for i in range(self.num_subspaces):
                start_idx = i * self.sub_dim
                end_idx = (i + 1) * self.sub_dim
                sub_data = calibration_data[:, start_idx:end_idx]
                
                distances = cdist(sub_data, self.codebooks[i], metric=self.distance_metric)
                codes = np.argmin(distances, axis=1)
                
                centroids = self.codebooks[i]
                for j in range(len(centroids)):
                    mask = (codes == j)
                    if np.any(mask):
                        gradient = centroids[j] - np.mean(sub_data[mask], axis=0)
                        self.codebooks[i][j] -= learning_rate * gradient
                
                reconstructed = centroids[codes]
                loss = np.mean(np.square(sub_data - reconstructed))
                total_loss += loss
            
            logger.info(f"Эпоха {epoch+1}/{n_epochs}, Loss: {total_loss/self.num_subspaces:.6f}")
        
        codes = self.encode(calibration_data)
        self._calculate_codebook_importance(codes)
        self._compress_codebooks()
        
        logger.info("Калибровка завершена")
    
    def calculate_compression_ratio(self, n_vectors: int = 1000, original_dtype: type = np.float32, 
                                  use_compressed: bool = True, **kwargs) -> float:
        if not self.is_trained:
            raise ValueError("Квантователь не обучен. Сначала вызовите метод train().")
            
        original_size = n_vectors * self.original_dim * np.dtype(original_dtype).itemsize
        compressed_size = n_vectors * self.num_subspaces * np.dtype(np.int32).itemsize
        
        if use_compressed and hasattr(self, 'compressed_codebooks') and self.compressed_codebooks:
            codebook_size = 0
            for codebook in self.compressed_codebooks:
                codebook_size += codebook.size * np.dtype(np.float32).itemsize
            
            for mapping in self.codebook_mapping:
                codebook_size += mapping.size * np.dtype(np.int32).itemsize
        else:
            codebook_size = 0
            for codebook in self.codebooks:
                codebook_size += codebook.size * np.dtype(np.float32).itemsize
        
        return original_size / (compressed_size + codebook_size)
    
    def save(self, filepath: str) -> None:
        if not self.is_trained:
            raise ValueError("Модель не обучена. Нечего сохранять.")
            
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'num_subspaces': self.num_subspaces,
                'num_clusters': self.num_clusters,
                'random_state': self.random_state,
                'codebooks': self.codebooks,
                'compressed_codebooks': self.compressed_codebooks if hasattr(self, 'compressed_codebooks') else [],
                'codebook_mapping': self.codebook_mapping if hasattr(self, 'codebook_mapping') else [],
                'codebook_importance': self.codebook_importance if hasattr(self, 'codebook_importance') else [],
                'sub_dim': self.sub_dim,
                'original_dim': self.original_dim,
                'is_trained': self.is_trained,
                'gamma': self.gamma,
                'n_components': self.n_components,
                'alpha': self.alpha,
                'compression_threshold': self.compression_threshold,
                'min_centroids': self.min_centroids,
                'original_data': self.original_data if hasattr(self, 'original_data') else None,
                'eigenvalues': self.eigenvalues if hasattr(self, 'eigenvalues') else None,
                'eigenvectors': self.eigenvectors if hasattr(self, 'eigenvectors') else None,
                'use_diffusion': self.use_diffusion if hasattr(self, 'use_diffusion') else False,
                'diffusion_data': self.diffusion_data if hasattr(self, 'diffusion_data') else None,
                'calibration_params': self.calibration_params,
                'use_optimized_kmeans': self.use_optimized_kmeans,
                'batch_size': self.batch_size,
                'distance_metric': self.distance_metric,
                'use_randomized_svd': self.use_randomized_svd
            }, f)
        
        logger.info(f"Модель сохранена в {filepath}")
    
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.num_subspaces = data['num_subspaces']
        self.num_clusters = data['num_clusters']
        self.random_state = data['random_state']
        self.codebooks = data['codebooks']
        self.compressed_codebooks = data['compressed_codebooks']
        self.codebook_mapping = data['codebook_mapping']
        self.codebook_importance = data['codebook_importance']
        self.sub_dim = data['sub_dim']
        self.original_dim = data['original_dim']
        self.is_trained = data['is_trained']
        self.gamma = data['gamma']
        self.n_components = data['n_components']
        self.alpha = data['alpha']
        self.compression_threshold = data['compression_threshold']
        self.min_centroids = data.get('min_centroids', 10)
        self.original_data = data['original_data']
        self.eigenvalues = data['eigenvalues']
        self.eigenvectors = data['eigenvectors']
        self.use_diffusion = data['use_diffusion']
        self.diffusion_data = data['diffusion_data']
        self.calibration_params = data.get('calibration_params', {})
        self.use_optimized_kmeans = data.get('use_optimized_kmeans', True)
        self.batch_size = data.get('batch_size', 1000)
        self.distance_metric = data.get('distance_metric', 'euclidean')
        self.use_randomized_svd = data.get('use_randomized_svd', True)
        
        self.kmeans_models = []
        
        if self.use_diffusion and self.diffusion_data is not None:
            self.knn_model = NearestNeighbors(n_neighbors=5, metric=self.distance_metric).fit(self.diffusion_data)
        elif self.original_data is not None:
            self.knn_model = NearestNeighbors(n_neighbors=5, metric=self.distance_metric).fit(self.original_data)
        
        logger.info(f"Модель загружена из {filepath}")
    
    def transform(self, data: np.ndarray, use_diffusion: bool = False, **kwargs) -> np.ndarray:
        return self.encode(data, use_diffusion=use_diffusion)

# Улучшенный класс для оценки и визуализации
class EnhancedPQEvaluator:
    def __init__(self):
        self.results = {}
        self.figures = {}
    
    def add_result(self, name: str, result: QuantizationResult):
        self.results[name] = result
    
    def generate_summary_table(self):
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Method': name,
                'Accuracy (%)': f"{result.accuracy:.2f}",
                'Recall (%)': f"{result.recall:.2f}",
                'MSE': f"{result.mse:.4f}",
                'Compression Ratio': f"{result.compression_ratio:.1f}x",
                'Silhouette (orig)': f"{result.clustering_metrics.get('original_silhouette', 0):.3f}",
                'Silhouette (recon)': f"{result.clustering_metrics.get('reconstructed_silhouette', 0):.3f}",
                'Time (s)': f"{result.execution_time:.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        return df
    
    def plot_metrics_comparison(self):
        metrics = ['Accuracy (%)', 'Recall (%)', 'Compression Ratio', 'MSE']
        methods = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = []
            for method in methods:
                if metric == 'Accuracy (%)':
                    values.append(self.results[method].accuracy)
                elif metric == 'Recall (%)':
                    values.append(self.results[method].recall)
                elif metric == 'Compression Ratio':
                    values.append(self.results[method].compression_ratio)
                elif metric == 'MSE':
                    values.append(self.results[method].mse)
            
            axes[i].bar(methods, values, color=plt.cm.Set3(range(len(methods))))
            axes[i].set_title(metric, fontsize=14, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            
            for j, v in enumerate(values):
                axes[i].text(j, v + max(values)*0.01, f'{v:.2f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_3d_comparison(self, method_name: str):
        if method_name not in self.results:
            raise ValueError(f"Метод {method_name} не найден в результатах")
        
        result = self.results[method_name]
        if result.original_data is None:
            raise ValueError("Исходные данные не доступны для визуализации")
        
        pca = PCA(n_components=3)
        original_3d = pca.fit_transform(result.original_data)
        reconstructed_3d = pca.transform(result.reconstructed)
        
        fig = plt.figure(figsize=(15, 10))
        
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(original_3d[:, 0], original_3d[:, 1], original_3d[:, 2], 
                              c='blue', alpha=0.6, label='Original', s=30)
        ax1.set_title('Исходные данные', fontsize=14, fontweight='bold')
        ax1.legend()
        
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(reconstructed_3d[:, 0], reconstructed_3d[:, 1], reconstructed_3d[:, 2], 
                              c='red', alpha=0.6, label='Reconstructed', s=30)
        ax2.set_title('Восстановленные данные', fontsize=14, fontweight='bold')
        ax2.legend()
        
        plt.suptitle(f'3D сравнение для метода: {method_name}\nMSE: {result.mse:.4f}, Accuracy: {result.accuracy:.2f}%', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_error_distribution(self, method_name: str):
        if method_name not in self.results:
            raise ValueError(f"Метод {method_name} не найден в результатах")
        
        result = self.results[method_name]
        if result.original_data is None:
            raise ValueError("Исходные данные не доступны для визуализации")
        
        errors = np.sqrt(np.sum((result.original_data - result.reconstructed)**2, axis=1))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0].set_xlabel('Error magnitude', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Reconstruction Errors', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot(errors, vert=False)
        axes[1].set_xlabel('Error magnitude', fontsize=12)
        axes[1].set_title('Error Statistics', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Error Analysis for {method_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_interactive_plot(self):
        if not self.results:
            raise ValueError("Нет результатов для визуализации")
        
        methods = list(self.results.keys())
        metrics = ['Accuracy', 'Recall', 'Compression Ratio', 'MSE']
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=metrics)
        
        for i, method in enumerate(methods):
            result = self.results[method]
            
            fig.add_trace(go.Bar(x=[method], y=[result.accuracy], name=f"{method} Accuracy",
                                marker_color=px.colors.qualitative.Set1[i]),
                         row=1, col=1)
            
            fig.add_trace(go.Bar(x=[method], y=[result.recall], name=f"{method} Recall",
                                marker_color=px.colors.qualitative.Set1[i]),
                         row=1, col=2)
            
            fig.add_trace(go.Bar(x=[method], y=[result.compression_ratio], name=f"{method} Compression",
                                marker_color=px.colors.qualitative.Set1[i]),
                         row=2, col=1)
            
            fig.add_trace(go.Bar(x=[method], y=[result.mse], name=f"{method} MSE",
                                marker_color=px.colors.qualitative.Set1[i]),
                         row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="Сравнение методов квантования")
        return fig
    
    def generate_html_report(self, filename: str = "quantization_report.html"):
        if not self.results:
            raise ValueError("Нет результатов для отчета")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantization Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric-box { 
                    border: 1px solid #ccc; 
                    border-radius: 5px; 
                    padding: 10px; 
                    margin: 10px 0; 
                    background-color: #f9f9f9;
                }
                .good { color: green; }
                .medium { color: orange; }
                .bad { color: red; }
            </style>
        </head>
        <body>
            <h1>Отчет по квантованию данных</h1>
        """
        
        summary_df = self.generate_summary_table()
        html_content += "<h2>Сводные результаты</h2>"
        html_content += summary_df.to_html(index=False)
        
        html_content += "<h2>Интерпретация результатов</h2>"
        
        for method, result in self.results.items():
            html_content += f"<div class='metric-box'><h3>{method}</h3>"
            
            accuracy_rating = "good" if result.accuracy > 85 else "medium" if result.accuracy > 70 else "bad"
            html_content += f"<p>Точность: <span class='{accuracy_rating}'>{result.accuracy:.2f}%</span> "
            if result.accuracy > 85:
                html_content += "(Отличное сохранение структуры данных)"
            elif result.accuracy > 70:
                html_content += "(Хорошее сохранение структуры данных)"
            else:
                html_content += "(Низкое качество квантования)"
            html_content += "</p>"
            
            compression_rating = "good" if result.compression_ratio > 20 else "medium" if result.compression_ratio > 10 else "bad"
            html_content += f"<p>Коэффициент сжатия: <span class='{compression_rating}'>{result.compression_ratio:.1f}x</span> "
            if result.compression_ratio > 20:
                html_content += "(Высокий уровень сжатия)"
            elif result.compression_ratio > 10:
                html_content += "(Средний уровень сжатия)"
            else:
                html_content += "(Низкий уровень сжатия)"
            html_content += "</p>"
            
            mse_rating = "good" if result.mse < 0.1 else "medium" if result.mse < 0.5 else "bad"
            html_content += f"<p>Среднеквадратичная ошибка: <span class='{mse_rating}'>{result.mse:.4f}</span> "
            if result.mse < 0.1:
                html_content += "(Высокое качество восстановления)"
            elif result.mse < 0.5:
                html_content += "(Удовлетворительное качество восстановления)"
            else:
                html_content += "(Низкое качество восстановления)"
            html_content += "</p>"
            
            html_content += "</div>"
        
        html_content += """
            <h2>Рекомендации</h2>
            <ul>
                <li>Для максимального качества используйте методы с высокой точностью (>85%)</li>
                <li>Для максимального сжатия используйте методы с высоким коэффициентом сжатия (>20x)</li>
                <li>Для сбалансированного решения выбирайте методы с хорошими показателями и по точности, и по сжатию</li>
            </ul>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename

# Функции для генерации данных и оценки
def generate_synthetic_data(n_samples: int = 1000, data_type: str = 'blobs', 
                           random_state: int = 42, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    if data_type == 'blobs':
        centers = kwargs.get('centers', 4)
        X, y = make_blobs(n_samples=n_samples, centers=centers, 
                         random_state=random_state, cluster_std=1.2)
    elif data_type == 'circles':
        noise = kwargs.get('noise', 0.08)
        factor = kwargs.get('factor', 0.6)
        X, y = make_circles(n_samples=n_samples, noise=noise, 
                           factor=factor, random_state=random_state)
    elif data_type == 'moons':
        noise = kwargs.get('noise', 0.1)
        X, y = make_moons(n_samples=n_samples, noise=noise, 
                         random_state=random_state)
    elif data_type == 'complex':
        X1, y1 = make_blobs(n_samples=n_samples//2, centers=2, 
                           random_state=random_state, cluster_std=0.8)
        X2, y2 = make_circles(n_samples=n_samples//2, noise=0.05, 
                             factor=0.5, random_state=random_state+1)
        X2 = X2 * 5 + np.array([5, 5])
        y2 = y2 + 2
        
        X = np.vstack([X1, X2])
        y = np.concatenate([y1, y2])
    else:
        raise ValueError(f"Неизвестный тип данных: {data_type}")
    
    n_noise_features = kwargs.get('n_noise_features', 15)
    if n_noise_features > 0:
        noise_features = np.random.randn(n_samples, n_noise_features) * 0.5
        X = np.hstack([X, noise_features])
    
    return X, y

@timer
def evaluate_quantizer(quantizer: BaseQuantizer, train_data: np.ndarray, 
                      test_data: np.ndarray, true_nn_model: NearestNeighbors, 
                      n_neighbors: int = 5, true_labels: Optional[np.ndarray] = None,
                      **kwargs) -> QuantizationResult:
    start_time = time.time()
    
    quantizer.train(train_data, **kwargs)
    
    codes = quantizer.encode(test_data, **kwargs)
    
    use_diffusion = kwargs.get('use_diffusion', False)
    if use_diffusion and hasattr(quantizer, 'reconstruct_to_original_space'):
        reconstructed = quantizer.reconstruct_to_original_space(codes)
    else:
        reconstructed = quantizer.decode(codes, **kwargs)
    
    reconstructed_nn = NearestNeighbors(n_neighbors=n_neighbors, 
                                       metric=quantizer.distance_metric if hasattr(quantizer, 'distance_metric') else 'euclidean').fit(reconstructed)
    _, pred_indices = reconstructed_nn.kneighbors(reconstructed)
    
    _, true_indices = true_nn_model.kneighbors(test_data)
    
    accuracy = 0
    n_samples = true_indices.shape[0]
    for i in range(n_samples):
        common = np.intersect1d(true_indices[i], pred_indices[i])
        accuracy += len(common) / n_neighbors
    accuracy = accuracy / n_samples * 100
    
    recall = 0
    for i in range(n_samples):
        common = np.intersect1d(true_indices[i, :n_neighbors], pred_indices[i, :n_neighbors])
        recall += len(common) / n_neighbors
    recall = recall / n_samples * 100
    
    mse = np.mean(np.square(test_data - reconstructed))
    compression_ratio = quantizer.calculate_compression_ratio(n_vectors=test_data.shape[0], **kwargs)
    
    clustering_metrics = {}
    if true_labels is not None:
        n_clusters = len(np.unique(true_labels))
        kmeans_original = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_original = kmeans_original.fit_predict(test_data)
        
        kmeans_reconstructed = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_reconstructed = kmeans_reconstructed.fit_predict(reconstructed)
        
        if len(np.unique(pred_original)) > 1:
            clustering_metrics['original_silhouette'] = silhouette_score(test_data, pred_original)
        else:
            clustering_metrics['original_silhouette'] = 0.0
        
        clustering_metrics['original_ari'] = adjusted_rand_score(true_labels, pred_original)
        
        if len(np.unique(pred_reconstructed)) > 1:
            clustering_metrics['reconstructed_silhouette'] = silhouette_score(reconstructed, pred_reconstructed)
        else:
            clustering_metrics['reconstructed_silhouette'] = 0.0
        
        clustering_metrics['reconstructed_ari'] = adjusted_rand_score(true_labels, pred_reconstructed)
    
    execution_time = time.time() - start_time
    
    return QuantizationResult(
        codes=codes,
        reconstructed=reconstructed,
        accuracy=accuracy,
        recall=recall,
        mse=mse,
        compression_ratio=compression_ratio,
        clustering_metrics=clustering_metrics,
        original_data=test_data,
        method_name=quantizer.__class__.__name__,
        execution_time=execution_time,
        parameters=kwargs
    )

@timer
def compare_compressed_vs_full_training():
    n_samples = 1200
    data_types = ['blobs', 'circles', 'moons', 'complex']
    
    results = {}
    
    for data_type in data_types:
        logger.info(f"Генерация данных типа: {data_type}")
        
        if data_type == 'blobs':
            X, y = generate_synthetic_data(n_samples=n_samples, data_type=data_type, 
                                          centers=4, n_noise_features=15)
        elif data_type == 'circles':
            X, y = generate_synthetic_data(n_samples=n_samples, data_type=data_type, 
                                          noise=0.08, n_noise_features=12)
        elif data_type == 'moons':
            X, y = generate_synthetic_data(n_samples=n_samples, data_type=data_type, 
                                          noise=0.1, n_noise_features=10)
        else:
            X, y = generate_synthetic_data(n_samples=n_samples, data_type=data_type, 
                                          n_noise_features=20)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        n_neighbors = 5
        true_nn_model = NearestNeighbors(n_neighbors=n_neighbors).fit(X_test)
        
        config = PQConfig(
            num_subspaces=4,
            num_clusters=64,
            use_diffusion=True,
            use_optimized_kmeans=True,
            batch_size=500,
            diffusion_params={"gamma": 1.0, "n_components": 16, "alpha": 0.5, "use_randomized_svd": True},
            compression_params={"threshold": 0.1, "min_centroids": 5},
            calibration_params={"n_epochs": 3, "learning_rate": 0.01},
            distance_metric='euclidean'
        )
        
        logger.info(f"Обучение на полных данных ({data_type})...")
        apq_full = AdvancedProductQuantizer(config=config)
        full_result = evaluate_quantizer(
            apq_full, X_train, X_test, true_nn_model, n_neighbors, y_test,
            use_diffusion=True, use_compressed=True
        )
        
        logger.info(f"Сжатие данных с помощью PCA ({data_type})...")
        n_components = min(8, X_train.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_train_compressed = pca.fit_transform(X_train)
        X_test_compressed = pca.transform(X_test)
        
        compressed_nn_model = NearestNeighbors(n_neighbors=n_neighbors).fit(X_test_compressed)
        
        logger.info(f"Обучение на сжатых данных ({data_type})...")
        apq_compressed = AdvancedProductQuantizer(config=config)
        compressed_result = evaluate_quantizer(
            apq_compressed, X_train_compressed, X_test_compressed, compressed_nn_model, 
            n_neighbors, y_test, use_diffusion=True, use_compressed=True
        )
        
        results[data_type] = {
            'full_training': full_result,
            'compressed_training': compressed_result
        }
        
        print(f"\nРезультаты для {data_type}:")
        print(f"Полное обучение - Accuracy: {full_result.accuracy:.2f}%, MSE: {full_result.mse:.4f}, Время: {full_result.execution_time:.2f}с")
        print(f"Сжатое обучение - Accuracy: {compressed_result.accuracy:.2f}%, MSE: {compressed_result.mse:.4f}, Время: {compressed_result.execution_time:.2f}с")
        
        evaluator = EnhancedPQEvaluator()
        evaluator.add_result("Full Training", full_result)
        evaluator.add_result("Compressed Training", compressed_result)
        
        metrics_fig = evaluator.plot_metrics_comparison()
        metrics_fig.suptitle(f"Сравнение методов для {data_type}", fontsize=16, fontweight='bold')
        plt.show()
    
    return results

def demonstrate_enhanced_visualization():
    X, y = make_blobs(n_samples=1000, centers=4, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    evaluator = EnhancedPQEvaluator()
    
    result1 = QuantizationResult(
        codes=np.random.randint(0, 256, (X_test.shape[0], 8)),
        reconstructed=X_test + np.random.normal(0, 0.1, X_test.shape),
        accuracy=92.5,
        recall=89.3,
        mse=0.08,
        compression_ratio=18.5,
        clustering_metrics={'original_silhouette': 0.75, 'reconstructed_silhouette': 0.72},
        original_data=X_test,
        method_name="Basic Quantization",
        execution_time=5.3
    )
    
    result2 = QuantizationResult(
        codes=np.random.randint(0, 256, (X_test.shape[0], 8)),
        reconstructed=X_test + np.random.normal(0, 0.05, X_test.shape),
        accuracy=96.2,
        recall=93.7,
        mse=0.04,
        compression_ratio=15.2,
        clustering_metrics={'original_silhouette': 0.75, 'reconstructed_silhouette': 0.74},
        original_data=X_test,
        method_name="Advanced Quantization",
        execution_time=8.7
    )
    
    result3 = QuantizationResult(
        codes=np.random.randint(0, 256, (X_test.shape[0], 8)),
        reconstructed=X_test + np.random.normal(0, 0.03, X_test.shape),
        accuracy=98.1,
        recall=96.4,
        mse=0.02,
        compression_ratio=12.8,
        clustering_metrics={'original_silhouette': 0.75, 'reconstructed_silhouette': 0.73},
        original_data=X_test,
        method_name="Diffusion Quantization",
        execution_time=12.4
    )
    
    evaluator.add_result("Basic", result1)
    evaluator.add_result("Advanced", result2)
    evaluator.add_result("Diffusion", result3)
    
    summary_table = evaluator.generate_summary_table()
    print("Сводная таблица результатов:")
    print(summary_table.to_string(index=False))
    
    metrics_fig = evaluator.plot_metrics_comparison()
    metrics_fig.suptitle("Сравнение методов квантования", fontsize=16, fontweight='bold')
    plt.show()
    
    for method in evaluator.results.keys():
        try:
            fig_3d = evaluator.plot_3d_comparison(method)
            plt.show()
        except Exception as e:
            print(f"Не удалось создать 3D визуализацию для {method}: {e}")
    
    for method in evaluator.results.keys():
        try:
            error_fig = evaluator.plot_error_distribution(method)
            plt.show()
        except Exception as e:
            print(f"Не удалось создать визуализацию ошибок для {method}: {e}")
    
    try:
        interactive_fig = evaluator.create_interactive_plot()
        interactive_fig.show()
    except Exception as e:
        print(f"Не удалось создать интерактивную визуализацию: {e}")
    
    try:
        report_file = evaluator.generate_html_report()
        print(f"HTML отчет сохранен в файл: {report_file}")
    except Exception as e:
        print(f"Не удалось создать HTML отчет: {e}")

def create_parameter_widgets():
    style = {'description_width': 'initial'}
    
    widgets_dict = {
        'n_samples': widgets.IntSlider(
            value=1000, min=100, max=10000, step=100,
            description='Количество samples:', style=style
        ),
        'n_features': widgets.IntSlider(
            value=10, min=2, max=100, step=1,
            description='Количество features:', style=style
        ),
        'n_clusters': widgets.IntSlider(
            value=4, min=2, max=20, step=1,
            description='Количество clusters:', style=style
        ),
        'test_size': widgets.FloatSlider(
            value=0.2, min=0.1, max=0.5, step=0.05,
            description='Размер test set:', style=style
        ),
        'compression_method': widgets.Dropdown(
            options=['Basic', 'Advanced', 'Diffusion'],
            value='Advanced',
            description='Метод квантования:', style=style
        ),
        'use_diffusion': widgets.Checkbox(
            value=True,
            description='Использовать диффузионное преобразование', style=style
        ),
        'diffusion_components': widgets.IntSlider(
            value=50, min=10, max=200, step=10,
            description='Компоненты диффузии:', style=style,
            disabled=False
        )
    }
    
    def update_diffusion_components(change):
        widgets_dict['diffusion_components'].disabled = not change.new
    
    widgets_dict['use_diffusion'].observe(update_diffusion_components, names='value')
    
    return widgets_dict