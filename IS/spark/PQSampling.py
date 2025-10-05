from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf, col, lit, rand, array, explode, posexplode, row_number
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructType, StructField
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.window import Window
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
import logging
import pyspark.sql.functions as F
from PQ import ProductQuantization, PQEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PQDataSampler:
    """
    Класс для семплинга реальных данных и генерации искусственных данных из PQ-кластеров.
    """
    
    def __init__(self, pq_model: ProductQuantization):
        self.pq_model = pq_model
        self.cluster_stats = {}
    
    def _compute_cluster_statistics_optimized(self, encoded_df: DataFrame) -> DataFrame:
        """Оптимизированное вычисление статистик кластеров без collect()."""
        # Используем агрегацию для вычисления статистик
        stats_df = encoded_df.select(
            posexplode(col('pq_code')).alias('subspace_idx', 'cluster_idx')
        ).groupBy('subspace_idx', 'cluster_idx').agg(
            F.count('*').alias('count')
        )
        
        return stats_df.cache()
    
    def _compute_pq_code_statistics_optimized(self, encoded_df: DataFrame) -> DataFrame:
        """Оптимизированное вычисление статистик PQ-кодов без collect()."""
        stats_df = encoded_df.groupBy('pq_code').agg(
            F.count('*').alias('count')
        )
        return stats_df.cache()
    
    def uniform_sampling(self, encoded_df: DataFrame, sample_size: int) -> DataFrame:
        """Быстрый равномерный семплинг."""
        return encoded_df.orderBy(rand()).limit(sample_size)
    
    def stratified_sampling_optimized(self, encoded_df: DataFrame, sample_size: int, 
                                   strategy: str = 'balanced') -> DataFrame:
        """
        Оптимизированный стратифицированный семплинг.
        """
        if strategy == 'balanced':
            return self._balanced_sampling_optimized(encoded_df, sample_size)
        elif strategy == 'proportional':
            return self._proportional_sampling_optimized(encoded_df, sample_size)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _balanced_sampling_optimized(self, encoded_df: DataFrame, sample_size: int) -> DataFrame:
        """Оптимизированный балансированный семплинг с оконными функциями."""
        # Быстрое определение количества уникальных кодов
        n_clusters = encoded_df.select('pq_code').distinct().count()
        
        if n_clusters == 0:
            return encoded_df.limit(0)
        
        # Используем оконные функции для эффективного семплинга
        window = Window.partitionBy('pq_code').orderBy(rand())
        
        # Добавляем номер строки в каждом кластере
        ranked_df = encoded_df.withColumn('row_num', row_number().over(window))
        
        # Берем по 1 элементу из каждого кластера
        samples_per_cluster = max(1, sample_size // n_clusters)
        balanced_sample = ranked_df.filter(col('row_num') <= samples_per_cluster).drop('row_num')
        
        # Если нужно больше samples, добавляем случайные
        current_count = balanced_sample.count()
        if current_count < sample_size:
            remaining_needed = sample_size - current_count
            # Исключаем уже выбранные строки
            remaining_df = ranked_df.filter(col('row_num') > samples_per_cluster)
            additional_sample = remaining_df.orderBy(rand()).limit(remaining_needed).drop('row_num')
            balanced_sample = balanced_sample.union(additional_sample)
        
        return balanced_sample.limit(sample_size)
    
    def _proportional_sampling_optimized(self, encoded_df: DataFrame, sample_size: int) -> DataFrame:
        """Оптимизированный пропорциональный семплинг."""
        # Вычисляем веса для каждого PQ-кода
        pq_stats = self._compute_pq_code_statistics_optimized(encoded_df)
        total_count = encoded_df.count()
        
        # Присоединяем веса к основному DataFrame
        encoded_with_weights = encoded_df.alias('main').join(
            pq_stats.alias('stats'),
            col('main.pq_code') == col('stats.pq_code')
        ).select(
            col('main.*'),
            (col('stats.count') / total_count).alias('weight')
        )
        
        # Семплируем с учетом весов
        sampled_df = encoded_with_weights.orderBy(rand() / col('weight')).limit(sample_size)
        
        return sampled_df.drop('weight')
    
    def stratified_sampling(self, encoded_df: DataFrame, sample_size: int, 
                          strategy: str = 'balanced') -> DataFrame:
        """Совместимость со старым интерфейсом."""
        return self.stratified_sampling_optimized(encoded_df, sample_size, strategy)
    
    def cluster_based_sampling(self, encoded_df: DataFrame, sample_size: int, 
                             strategy: str = 'balanced') -> DataFrame:
        """Совместимость со старым интерфейсом."""
        return self.stratified_sampling_optimized(encoded_df, sample_size, strategy)
    
    def generate_synthetic_data_fast(self, generation_strategy: str = 'random_combination', 
                                   num_samples: int = 1000, noise_std: float = 0.1,
                                   real_data: Optional[DataFrame] = None) -> DataFrame:
        """
        Оптимизированная генерация синтетических данных.
        """
        if not self.pq_model.is_fitted:
            raise ValueError("PQ model must be fitted before data generation")
        
        spark = SparkSession.builder.getOrCreate()
        
        if generation_strategy == 'random_combination':
            # Генерируем данные в векторном виде
            synthetic_vectors = []
            
            for _ in range(min(num_samples, 10000)):  # Ограничиваем batch
                vector_parts = []
                for subspace_idx in range(self.pq_model.M):
                    codebook = self.pq_model.codebooks[subspace_idx]
                    random_idx = np.random.randint(0, len(codebook))
                    centroid = codebook[random_idx].copy()
                    
                    if noise_std > 0:
                        noise = np.random.normal(0, noise_std, centroid.shape)
                        centroid += noise
                    
                    vector_parts.append(centroid)
                
                full_vector = self.pq_model._combine_subspaces(vector_parts)
                synthetic_vectors.append(full_vector.tolist())
            
            # Создаем DataFrame одним batch
            schema = StructType([StructField("features", ArrayType(DoubleType()))])
            pdf = spark.createDataFrame(
                [(vec,) for vec in synthetic_vectors], 
                schema=schema
            )
            
            return pdf.withColumn("features", udf(Vectors.dense, VectorUDT())(col("features")))
        
        else:
            # Для других стратегий используем старую реализацию
            return self.generate_synthetic_data(generation_strategy, num_samples, noise_std, real_data)
    
    def generate_synthetic_data(self, generation_strategy: str = 'random_combination', 
                              num_samples: int = 1000, noise_std: float = 0.1,
                              real_data: Optional[DataFrame] = None) -> DataFrame:
        """Совместимость со старым интерфейсом."""
        return self.generate_synthetic_data_fast(generation_strategy, num_samples, noise_std, real_data)


class PQImportanceSamplingOptimized:
    """
    Оптимизированный Importance Sampling.
    """
    
    def __init__(self, pq_model: ProductQuantization):
        self.pq_model = pq_model
        self.cluster_weights = {}
    
    def _compute_importance_weights_optimized(self, encoded_df: DataFrame, 
                                           target_function: callable) -> DataFrame:
        """Оптимизированное вычисление весов важности."""
        
        @udf(DoubleType())
        def target_function_udf(features):
            if hasattr(features, 'toArray'):
                features_array = features.toArray()
            else:
                features_array = np.array(features)
            return float(target_function(features_array))
        
        # Вычисляем статистики в одном проходе
        importance_df = encoded_df.withColumn('target_value', target_function_udf(col('features'))) \
            .groupBy('pq_code') \
            .agg(
                F.avg('target_value').alias('avg_value'),
                F.variance('target_value').alias('variance_value'),
                F.count('*').alias('cluster_size')
            )
        
        return importance_df.cache()
    
    def importance_sampling_optimized(self, encoded_df: DataFrame, target_function: callable,
                                    num_samples: int = 1000) -> Tuple[float, DataFrame]:
        """
        Оптимизированный Importance Sampling.
        """
        # Вычисляем веса важности
        importance_df = self._compute_importance_weights_optimized(encoded_df, target_function)
        
        # Присоединяем веса к данным
        encoded_with_importance = encoded_df.alias('data').join(
            importance_df.alias('imp'),
            col('data.pq_code') == col('imp.pq_code')
        ).select(
            col('data.*'),
            F.coalesce(col('imp.variance_value'), lit(1.0)).alias('importance')
        )
        
        # Семплируем с учетом важности
        sampled_df = encoded_with_importance.orderBy(rand() / col('importance')).limit(num_samples)
        
        # Вычисляем целевую функцию для семпла
        @udf(DoubleType())
        def target_udf(features):
            if hasattr(features, 'toArray'):
                features_array = features.toArray()
            else:
                features_array = np.array(features)
            return float(target_function(features_array))
        
        result_df = sampled_df.withColumn('target_value', target_udf(col('features')))
        
        # Вычисляем оценку математического ожидания
        agg_result = result_df.agg(
            F.avg('target_value').alias('expectation')
        ).collect()[0]
        
        expectation = agg_result['expectation'] or 0.0
        
        return float(expectation), result_df
    
    def importance_sampling(self, encoded_df: DataFrame, target_function: callable,
                          num_samples: int = 1000) -> Tuple[float, DataFrame]:
        """Совместимость со старым интерфейсом."""
        return self.importance_sampling_optimized(encoded_df, target_function, num_samples)


# Сохраняем старый класс для совместимости
class PQImportanceSampling(PQImportanceSamplingOptimized):
    """Совместимость со старым кодом."""
    pass