from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf, col, lit, rand, array, explode, posexplode, monotonically_increasing_id
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructType, StructField
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
import logging
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import pyspark.sql.functions as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductQuantization:
    """
    Реализация Product Quantization с распределенным KMeans для кластеризации подпространств.
    """
    
    def __init__(self, M: int = 8, K: int = 256, kmeans_params: Dict[str, Any] = None,
                 use_opq: bool = False, use_ivf: bool = False, n_probe: int = 1):
        
        self.M = M
        self.K = K
        self.kmeans_params = kmeans_params or {}
        self.kmeans_params['k'] = K
        self.use_opq = use_opq
        self.use_ivf = use_ivf
        self.n_probe = n_probe
        
        self.codebooks = []
        self.subspace_dims = []
        self.is_fitted = False
        self.feature_dim = None
        self.rotation_matrix = None
        self.ivf_centroids = None
        self.ivf_assignments = None
        
    def _split_vector(self, vector: np.ndarray) -> List[np.ndarray]:
        if self.feature_dim is None:
            self.feature_dim = len(vector)
            
        subspace_size = self.feature_dim // self.M
        remainder = self.feature_dim % self.M
        
        subspaces = []
        start_idx = 0
        
        for i in range(self.M):
            end_idx = start_idx + subspace_size + (1 if i < remainder else 0)
            subspaces.append(vector[start_idx:end_idx])
            start_idx = end_idx
            
        return subspaces
    
    def _combine_subspaces(self, subspaces: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(subspaces)
    
    def _learn_opq_rotation(self, data: np.ndarray) -> np.ndarray:
        logger.info("Learning OPQ rotation matrix...")
        
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvectors.T
    
    def _apply_opq_rotation(self, data: np.ndarray, direction: str = 'forward') -> np.ndarray:
        if self.rotation_matrix is None:
            return data
            
        if direction == 'forward':
            return data @ self.rotation_matrix.T
        else:
            return data @ self.rotation_matrix
    
    def _build_ivf_index(self, df: DataFrame, feature_col: str) -> None:
        logger.info("Building IVF index...")
        
        from DistributedKMeans import DistributedKMeans
        
        n_coarse_clusters = max(2, self.M)
        coarse_kmeans = DistributedKMeans(k=n_coarse_clusters)
        coarse_kmeans.fit(df, feature_col)
        
        self.ivf_centroids = coarse_kmeans.cluster_centers_
        assigned_df = coarse_kmeans.transform(df)
        self.ivf_assignments = assigned_df.select('cluster').collect()
        
    def fit(self, df: DataFrame, feature_col: str = 'features') -> 'ProductQuantization':
        logger.info(f"Starting PQ training with M={self.M}, K={self.K}")
        
        sample_row = df.select(feature_col).first()
        if sample_row is None:
            raise ValueError("DataFrame is empty")
            
        sample_vector = sample_row[0]
        if hasattr(sample_vector, 'toArray'):
            sample_vector = sample_vector.toArray()
            
        self.feature_dim = len(sample_vector)
        logger.info(f"Feature dimension: {self.feature_dim}")
        
        subspace_size = self.feature_dim // self.M
        remainder = self.feature_dim % self.M
        
        self.subspace_dims = []
        start_idx = 0
        for i in range(self.M):
            dim = subspace_size + (1 if i < remainder else 0)
            self.subspace_dims.append((start_idx, start_idx + dim))
            start_idx += dim
        
        if self.use_opq:
            logger.info("Learning OPQ rotation...")
            sample_data = df.select(feature_col).limit(1000).collect()
            data_matrix = np.array([row[0].toArray() if hasattr(row[0], 'toArray') 
                                  else np.array(row[0]) for row in sample_data])
            self.rotation_matrix = self._learn_opq_rotation(data_matrix)
        
        if self.use_ivf:
            self._build_ivf_index(df, feature_col)
        
        self.codebooks = []
        
        for subspace_idx, (start, end) in enumerate(self.subspace_dims):
            logger.info(f"Training subspace {subspace_idx + 1}/{self.M} (dimensions {start}:{end})")
            
            @udf(VectorUDT())
            def extract_subspace_udf(vec):
                if hasattr(vec, 'toArray'):
                    vec_array = vec.toArray()
                else:
                    vec_array = np.array(vec)
                
                if self.rotation_matrix is not None:
                    vec_array = self._apply_opq_rotation(vec_array.reshape(1, -1), 'forward')[0]
                
                subspace_vec = vec_array[start:end]
                return Vectors.dense(subspace_vec)
            
            subspace_df = df.withColumn('subspace', extract_subspace_udf(col(feature_col)))
            
            from DistributedKMeans import DistributedKMeans
            kmeans = DistributedKMeans(**self.kmeans_params)
            kmeans.fit(subspace_df, feature_col='subspace')
            
            codebook = kmeans.cluster_centers_
            self.codebooks.append(codebook)
            
            logger.info(f"Subspace {subspace_idx + 1} codebook shape: {codebook.shape}")
        
        self.is_fitted = True
        logger.info("PQ training completed successfully")
        return self
    
    def _precompute_distance_tables(self, query: np.ndarray) -> List[np.ndarray]:
        distance_tables = []
        
        for subspace_idx, (start, end) in enumerate(self.subspace_dims):
            query_subspace = query[start:end]
            codebook = self.codebooks[subspace_idx]
            distances = np.linalg.norm(codebook - query_subspace, axis=1)
            distance_tables.append(distances)
        
        return distance_tables
    
    def encode(self, df: DataFrame, feature_col: str = 'features') -> DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
        
        @udf(ArrayType(IntegerType()))
        def encode_vector_udf(vec):
            if hasattr(vec, 'toArray'):
                vec_array = vec.toArray()
            else:
                vec_array = np.array(vec)
            
            if self.rotation_matrix is not None:
                vec_array = self._apply_opq_rotation(vec_array.reshape(1, -1), 'forward')[0]
            
            codes = []
            distance_tables = self._precompute_distance_tables(vec_array)
            
            for subspace_idx in range(self.M):
                distances = distance_tables[subspace_idx]
                code = int(np.argmin(distances))
                codes.append(code)
            
            return codes
        
        return df.withColumn('pq_code', encode_vector_udf(col(feature_col)))
    
    def decode(self, pq_codes: List[List[int]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
        
        reconstructed_vectors = []
        
        for codes in pq_codes:
            subspaces = []
            for subspace_idx, code in enumerate(codes):
                codebook = self.codebooks[subspace_idx]
                if code < len(codebook):
                    subspace_vec = codebook[code]
                else:
                    subspace_vec = np.zeros(codebook.shape[1])
                subspaces.append(subspace_vec)
            
            reconstructed_vec = self._combine_subspaces(subspaces)
            
            if self.rotation_matrix is not None:
                reconstructed_vec = self._apply_opq_rotation(
                    reconstructed_vec.reshape(1, -1), 'backward')[0]
            
            reconstructed_vectors.append(reconstructed_vec)
        
        return np.array(reconstructed_vectors)
    
    def asymmetric_distance(self, query: np.ndarray, pq_code: List[int]) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before distance computation")
        
        total_distance = 0.0
        
        for subspace_idx, (start, end) in enumerate(self.subspace_dims):
            query_subspace = query[start:end]
            codebook = self.codebooks[subspace_idx]
            code = pq_code[subspace_idx]
            
            if code < len(codebook):
                centroid = codebook[code]
                distance = np.linalg.norm(query_subspace - centroid)
                total_distance += distance ** 2
        
        return np.sqrt(total_distance)


class PQEvaluator:
    """
    Класс для оценки качества Product Quantization.
    """
    
    def __init__(self, pq_model: ProductQuantization):
        self.pq_model = pq_model
    
    def compute_quantization_error(self, original_df: DataFrame, 
                                 reconstructed_df: DataFrame, 
                                 feature_col: str = 'features') -> float:
        """
        Вычисление ошибки квантования без требования столбца index.
        """
        @udf(DoubleType())
        def squared_distance_udf(vec1, vec2):
            if hasattr(vec1, 'toArray'):
                arr1 = vec1.toArray()
            else:
                arr1 = np.array(vec1)
                
            if hasattr(vec2, 'toArray'):
                arr2 = vec2.toArray()
            else:
                arr2 = np.array(vec2)
            
            return float(np.sum((arr1 - arr2) ** 2))
        
        # Добавляем временные индексы для соединения
        original_with_idx = original_df.withColumn("temp_id", monotonically_increasing_id())
        reconstructed_with_idx = reconstructed_df.withColumn("temp_id", monotonically_increasing_id())
        
        # Соединяем по временным индексам
        joined_df = original_with_idx.alias('orig').join(
            reconstructed_with_idx.alias('recon'), 
            col('orig.temp_id') == col('recon.temp_id')
        )
        
        # Вычисляем MSE
        mse_result = joined_df.agg(
            F.mean(squared_distance_udf(col('orig.' + feature_col), 
                                      col('recon.' + feature_col))).alias('mse')
        ).collect()[0]
        
        return mse_result['mse']
    
    def evaluate_distance_preservation_fast(self, original_df: DataFrame, 
                                          reconstructed_df: DataFrame,
                                          feature_col: str = 'features', 
                                          sample_size: int = 100) -> Dict[str, float]:
        """
        Быстрая оценка сохранения расстояний на небольшой выборке.
        """
        # Берем небольшую выборку для быстрой оценки
        original_sample = original_df.limit(sample_size)
        reconstructed_sample = reconstructed_df.limit(sample_size)
        
        # Собираем векторы на драйвере (только для небольших выборок)
        original_vectors = [
            row[0].toArray() if hasattr(row[0], 'toArray') else np.array(row[0])
            for row in original_sample.select(feature_col).collect()
        ]
        
        reconstructed_vectors = [
            row[0].toArray() if hasattr(row[0], 'toArray') else np.array(row[0])
            for row in reconstructed_sample.select(feature_col).collect()
        ]
        
        if len(original_vectors) != len(reconstructed_vectors):
            raise ValueError("Sample sizes must match")
        
        # Вычисляем попарные расстояния
        original_distances = []
        reconstructed_distances = []
        
        n = len(original_vectors)
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):  # Ограничиваем количество пар для скорости
                dist_orig = np.linalg.norm(original_vectors[i] - original_vectors[j])
                dist_recon = np.linalg.norm(reconstructed_vectors[i] - reconstructed_vectors[j])
                original_distances.append(dist_orig)
                reconstructed_distances.append(dist_recon)
        
        if not original_distances:
            return {'distance_correlation': 0.0, 'distance_mape': 0.0}
        
        # Вычисляем корреляцию
        correlation = np.corrcoef(original_distances, reconstructed_distances)[0, 1]
        
        # Вычисляем MAPE
        mape = np.mean(np.abs(np.array(original_distances) - np.array(reconstructed_distances)) / 
                      (np.array(original_distances) + 1e-10))
        
        return {
            'distance_correlation': correlation if not np.isnan(correlation) else 0.0,
            'distance_mape': mape
        }
    
    def compute_compression_ratio(self, original_dim: int) -> float:
        bits_per_code = np.log2(self.pq_model.K)
        compressed_size_bits = self.pq_model.M * bits_per_code
        original_size_bits = original_dim * 32  # float32
        
        return original_size_bits / compressed_size_bits