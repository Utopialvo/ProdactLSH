from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, rand, explode, row_number, collect_list, size, sum as spark_sum, broadcast, mean as spark_mean
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window
from pyspark import AccumulatorParam
import pyspark.sql.functions as F

import numpy as np
import random
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
import threading
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.datasets import make_blobs, make_moons, make_circles

logger = logging.getLogger(__name__)

# Кастомный аккумулятор для сбора статистики
class DictAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue: Dict[str, Any]):
        return initialValue.copy()
    
    def addInPlace(self, v1: Dict[str, Any], v2: Dict[str, Any]):
        for key, value in v2.items():
            if key in v1:
                if isinstance(v1[key], (int, float)) and isinstance(value, (int, float)):
                    v1[key] += value
                elif isinstance(v1[key], list) and isinstance(value, list):
                    v1[key].extend(value)
                elif isinstance(v1[key], dict) and isinstance(value, dict):
                    for k, v in value.items():
                        if k in v1[key]:
                            v1[key][k] += v
                        else:
                            v1[key][k] = v
            else:
                v1[key] = value
        return v1

@dataclass
class LSHConfig:
    """Конфигурация для LSH"""
    m: int = 100
    k: int = 10
    L: int = 5
    w: float = 1.0
    distance_metric: str = 'euclidean'
    sampling_ratio: float = 0.1
    random_state: int = 42

@dataclass
class PQConfig:
    """Конфигурация для продуктного квантования"""
    num_subspaces: int = 8
    num_clusters: int = 256
    use_optimized_kmeans: bool = True
    batch_size: int = 1000
    random_state: int = 42

@dataclass 
class UnifiedConfig:
    """Унифицированная конфигурация системы"""
    lsh_config: LSHConfig = field(default_factory=LSHConfig)
    pq_config: PQConfig = field(default_factory=PQConfig)
    hybrid_mode: str = 'two_stage'
    hybrid_candidate_multiplier: int = 10
    optimization_interval: int = 10
    reservoir_size: int = 1000

# Вспомогательные функции для UDF (вынесены на уровень модуля)
def encode_vector_udf(vector, codebooks, sub_dim, num_subspaces):
    """Кодирование вектора с использованием кодбуков"""
    codes = []
    for i in range(num_subspaces):
        start_idx = i * sub_dim
        end_idx = (i + 1) * sub_dim
        subspace = vector[start_idx:end_idx]
        
        distances = cdist([subspace], codebooks[i], 'euclidean')[0]
        code = int(np.argmin(distances))
        codes.append(code)
    
    return codes

def decode_codes_udf(codes, codebooks, num_subspaces, sub_dim):
    """Декодирование кодов с использованием кодбуков"""
    decoded = []
    for i, code in enumerate(codes):
        decoded.extend(codebooks[i][code])
    return Vectors.dense(decoded)

def pq_distance_udf(query_vector, data_vector, codebooks, sub_dim, num_subspaces, distance_metric):
    """Вычисление расстояния между PQ кодами"""
    # Кодируем оба вектора
    query_code = []
    data_code = []
    
    for i in range(num_subspaces):
        start_idx = i * sub_dim
        end_idx = (i + 1) * sub_dim
        
        # Кодируем подпространство запроса
        query_subspace = query_vector[start_idx:end_idx]
        query_distances = cdist([query_subspace], codebooks[i], 'euclidean')[0]
        query_code.append(int(np.argmin(query_distances)))
        
        # Кодируем подпространство данных
        data_subspace = data_vector[start_idx:end_idx]
        data_distances = cdist([data_subspace], codebooks[i], 'euclidean')[0]
        data_code.append(int(np.argmin(data_distances)))
    
    # Вычисляем расстояние между кодами
    query_reconstructed = []
    data_reconstructed = []
    
    for i in range(len(query_code)):
        query_reconstructed.extend(codebooks[i][query_code[i]])
        data_reconstructed.extend(codebooks[i][data_code[i]])
    
    query_reconstructed = np.array(query_reconstructed)
    data_reconstructed = np.array(data_reconstructed)
    
    if distance_metric == 'euclidean':
        return float(np.linalg.norm(query_reconstructed - data_reconstructed))
    elif distance_metric == 'cosine':
        query_norm = np.linalg.norm(query_reconstructed)
        data_norm = np.linalg.norm(data_reconstructed)
        
        if query_norm == 0 or data_norm == 0:
            return 1.0
            
        query_normalized = query_reconstructed / query_norm
        data_normalized = data_reconstructed / data_norm
        return float(1 - np.dot(query_normalized, data_normalized))
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

class FastRoLSHsampler:
    """Улучшенная реализация FastLSH и roLSH с использованием DataFrame API"""
    
    def __init__(self, spark: SparkSession, config: LSHConfig = None):
        self.spark = spark
        self.config = config or LSHConfig()
        self.tables_df = None  # LSH таблицы в формате DataFrame
        self.hash_functions = []
        self.total_points = 0
        
        # Аккумуляторы для сбора статистики
        self.stats_acc = spark.sparkContext.accumulator(
            {"table_sizes": {}, "distance_stats": {}}, 
            DictAccumulatorParam()
        )
        
        # Инициализация хэш-функций
        self._init_hash_functions()
    
    def _init_hash_functions(self):
        """Инициализация хэш-функций"""
        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        
        for l in range(self.config.L):
            table_funcs = []
            for _ in range(self.config.k):
                # Генерируем индексы для семплирования
                indices = np.random.choice(self.config.m, size=self.config.m, replace=False).tolist()
                
                if self.config.distance_metric == 'euclidean':
                    a = np.random.randn(self.config.m).tolist()
                    b = random.uniform(0, self.config.w)
                    table_funcs.append(('euclidean', a, b, indices))
                elif self.config.distance_metric == 'cosine':
                    a = np.random.randn(self.config.m).tolist()
                    a_norm = np.linalg.norm(a)
                    a = [x / a_norm for x in a]  # normalize
                    table_funcs.append(('cosine', a, 0, indices))
            
            self.hash_functions.append(table_funcs)
    
    def _compute_hash_udf(self, table_idx, func_idx):
        """Создание UDF для вычисления хэша"""
        func_type, a, b, indices = self.hash_functions[table_idx][func_idx]
        w = self.config.w  # локальная переменная для захвата
        
        def hash_udf(vector):
            # Семплируем вектор по предварительно заданным индексам
            sampled_vector = [vector[i] for i in indices]
            
            if func_type == 'euclidean':
                projection = sum(a_i * v_i for a_i, v_i in zip(a, sampled_vector)) + b
                return int(math.floor(projection / w))
            else:
                dot_product = sum(a_i * v_i for a_i, v_i in zip(a, sampled_vector))
                return 1 if dot_product >= 0 else 0
        
        return udf(hash_udf, IntegerType())
    
    def add_data(self, data_df, id_col="id", vector_col="vector"):
        """Добавление данных в LSH индекс с использованием DataFrame API"""
        # Добавление ID если не существует
        if id_col not in data_df.columns:
            window = Window.orderBy(rand())
            data_df = data_df.withColumn(id_col, row_number().over(window) + self.total_points)
        
        # Вычисление хэшей для всех таблиц
        hash_dfs = []
        for table_idx in range(self.config.L):
            # Создание столбцов для каждого бита хэша
            hash_exprs = []
            for func_idx in range(self.config.k):
                hash_udf = self._compute_hash_udf(table_idx, func_idx)
                hash_exprs.append(hash_udf(col(vector_col)).alias(f"hash_{func_idx}"))
            
            # Вычисление хэшей
            table_hash_df = data_df.select(col(id_col), *hash_exprs)
            
            # Создание супер-хэша
            hash_cols = [col(f"hash_{i}") for i in range(self.config.k)]
            table_hash_df = table_hash_df.withColumn(
                "super_hash", 
                F.concat_ws(",", *hash_cols)
            ).withColumn("table_id", lit(table_idx))
            
            hash_dfs.append(table_hash_df.select("table_id", "super_hash", id_col))
        
        # Объединение всех таблиц
        all_hashes_df = hash_dfs[0]
        for df in hash_dfs[1:]:
            all_hashes_df = all_hashes_df.union(df)
        
        # Обновление LSH таблиц
        if self.tables_df is None:
            self.tables_df = all_hashes_df
        else:
            self.tables_df = self.tables_df.union(all_hashes_df)
        
        # Обновление счетчика точек
        self.total_points += data_df.count()
        
        # Сбор статистики - агрегирование без collect
        table_stats = all_hashes_df.groupBy("table_id", "super_hash").count()
        
        # Используем локальную ссылку на аккумулятор для избежания захвата self
        stats_acc = self.stats_acc
        
        def update_stats(partition):
            for row in partition:
                key = (row["table_id"], row["super_hash"])
                stats_acc.add({"table_sizes": {key: row["count"]}})
            return iter([])  # возвращаем пустой итератор
        
        # Применяем функцию для обновления статистики
        table_stats.rdd.mapPartitions(update_stats).count()  # trigger execution
        
        return True
    
    def query(self, query_vector, k=10, max_expansions=5):
        """Поиск ближайших соседей с использованием roLSH"""
        if self.tables_df is None:
            return []
        
        # Вычисление хэшей для запроса
        query_hashes = []
        for table_idx in range(self.config.L):
            super_hash = []
            for func_idx in range(self.config.k):
                # Используем ту же логику, что и в UDF
                func_type, a, b, indices = self.hash_functions[table_idx][func_idx]
                sampled_vector = [query_vector[i] for i in indices]
                
                if func_type == 'euclidean':
                    projection = sum(a_i * v_i for a_i, v_i in zip(a, sampled_vector)) + b
                    hash_val = int(math.floor(projection / self.config.w))
                else:
                    dot_product = sum(a_i * v_i for a_i, v_i in zip(a, sampled_vector))
                    hash_val = 1 if dot_product >= 0 else 0
                
                super_hash.append(str(hash_val))
            query_hashes.append((table_idx, ",".join(super_hash)))
        
        # Поиск кандидатов с адаптивным радиусом
        candidates = set()
        current_radius = self.config.w
        
        for expansion in range(max_expansions):
            for table_idx, query_hash in query_hashes:
                # Поиск в текущей таблице
                table_candidates = self.tables_df.filter(
                    (col("table_id") == table_idx) & (col("super_hash") == query_hash)
                ).select("id").collect()
                
                candidates.update([row["id"] for row in table_candidates])
                
                # Если нашли достаточно кандидатов, прекращаем поиск
                if len(candidates) >= k * 3:
                    break
            
            if len(candidates) >= k * 3:
                break
            
            # Расширение радиуса поиска
            current_radius *= 2.0
            
            # Генерация дополнительных хэшей для расширенного поиска
            if self.config.distance_metric == 'cosine':
                new_hashes = []
                for table_idx, base_hash in query_hashes:
                    hash_parts = base_hash.split(',')
                    for i in range(min(expansion + 1, len(hash_parts))):
                        new_hash = hash_parts.copy()
                        new_hash[i] = '1' if new_hash[i] == '0' else '0'
                        new_hashes.append((table_idx, ",".join(new_hash)))
                
                query_hashes.extend(new_hashes)
        
        return list(candidates)[:k]

class ProductQuantizer:
    """Улучшенная реализация продуктного квантования с использованием DataFrame API"""
    
    def __init__(self, spark: SparkSession, config: PQConfig = None):
        self.spark = spark
        self.config = config or PQConfig()
        self.codebooks = []
        self.sub_dim = None
        self.is_trained = False
    
    def train(self, data_df, vector_col="vector"):
        """Обучение продуктного квантователя с использованием DataFrame API"""
        # Получение размерности данных
        first_vector = data_df.select(vector_col).first()[0]
        dim = len(first_vector)
        
        # Определение размерности подпространств
        if dim % self.config.num_subspaces != 0:
            divisors = [i for i in range(1, dim + 1) if dim % i == 0]
            self.config.num_subspaces = min(divisors, key=lambda x: abs(x - self.config.num_subspaces))
        
        self.sub_dim = dim // self.config.num_subspaces
        
        # Обучаем квантователь для каждого подпространства
        for i in range(self.config.num_subspaces):
            start_idx = i * self.sub_dim
            end_idx = (i + 1) * self.sub_dim
            
            # Извлечение подпространства
            def extract_subspace(vector):
                return Vectors.dense(vector[start_idx:end_idx])
            
            extract_udf = udf(extract_subspace, VectorUDT())
            subspace_df = data_df.withColumn("subspace", extract_udf(col(vector_col)))
            
            # Сбор данных для кластеризации (ограниченный набор)
            # Используем reservoir sampling для ограничения размера данных
            reservoir_size = min(10000, subspace_df.count())
            reservoir_df = subspace_df.sample(False, reservoir_size / subspace_df.count(), seed=self.config.random_state)
            
            subspace_data = reservoir_df.select("subspace").rdd.map(lambda r: r[0].toArray()).collect()
            
            # Кластеризация с использованием MiniBatchKMeans
            if self.config.use_optimized_kmeans:
                kmeans = MiniBatchKMeans(
                    n_clusters=self.config.num_clusters,
                    random_state=self.config.random_state,
                    batch_size=self.config.batch_size
                )
                
                # Обучение на батчах
                for j in range(0, len(subspace_data), self.config.batch_size):
                    batch = subspace_data[j:j+self.config.batch_size]
                    if len(batch) > 0:
                        kmeans.partial_fit(batch)
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=self.config.num_clusters,
                    random_state=self.config.random_state
                ).fit(subspace_data)
            
            self.codebooks.append(kmeans.cluster_centers_)
        
        self.is_trained = True
    
    def encode(self, data_df, vector_col="vector"):
        """Кодирование данных с помощью продуктного квантования"""
        if not self.is_trained:
            raise ValueError("Квантователь не обучен")
        
        # Используем локальные переменные для избежания захвата self
        codebooks = self.codebooks
        sub_dim = self.sub_dim
        num_subspaces = self.config.num_subspaces
        
        def encode_vector(vector):
            return encode_vector_udf(vector, codebooks, sub_dim, num_subspaces)
        
        encode_udf = udf(encode_vector, ArrayType(IntegerType()))
        return data_df.withColumn("pq_codes", encode_udf(col(vector_col)))
    
    def decode(self, codes_df, codes_col="pq_codes"):
        """Декодирование данных из продуктного квантования"""
        # Используем локальные переменные для избежания захвата self
        codebooks = self.codebooks
        num_subspaces = self.config.num_subspaces
        sub_dim = self.sub_dim
        
        def decode_codes(codes):
            return decode_codes_udf(codes, codebooks, num_subspaces, sub_dim)
        
        decode_udf = udf(decode_codes, VectorUDT())
        return codes_df.withColumn("decoded_vector", decode_udf(col(codes_col)))

class UnifiedQuantizationEngine:
    """Улучшенная унифицированная система квантования с использованием DataFrame API"""
    
    def __init__(self, spark: SparkSession, config: UnifiedConfig = None):
        self.spark = spark
        self.config = config or UnifiedConfig()
        self.lsh_sampler = FastRoLSHsampler(spark, config.lsh_config)
        self.pq_quantizer = ProductQuantizer(spark, config.pq_config)
        self.is_trained = False
        self.total_points = 0
        self.data_df = None  # Хранилище всех данных
        
        # Резервуарная выборка для оптимизации
        self.reservoir_df = None
    
    def update(self, data_df, id_col="id", vector_col="vector"):
        """Обновление системы новыми данными"""
        # Сохраняем данные
        if self.data_df is None:
            self.data_df = data_df
        else:
            self.data_df = self.data_df.union(data_df)
        
        # Обновляем LSH семплер
        self.lsh_sampler.add_data(data_df, id_col, vector_col)
        
        # Обновляем резервуарную выборку
        if self.reservoir_df is None:
            self.reservoir_df = data_df.limit(self.config.reservoir_size)
        else:
            # Объединяем с существующей выборкой и ограничиваем размер
            combined_df = self.reservoir_df.union(data_df)
            reservoir_count = combined_df.count()
            
            if reservoir_count > self.config.reservoir_size:
                # Выбираем случайную подвыборку
                fraction = self.config.reservoir_size / reservoir_count
                self.reservoir_df = combined_df.sample(False, fraction, seed=self.config.lsh_config.random_state)
            else:
                self.reservoir_df = combined_df
        
        # Обучаем PQ на резервуарной выборке, если нужно
        if (self.config.hybrid_mode != 'lsh_only' and not self.is_trained and 
            self.reservoir_df is not None and self.reservoir_df.count() >= self.config.pq_config.num_clusters * 2):
            
            self.pq_quantizer.train(self.reservoir_df, vector_col)
            self.is_trained = True
        
        self.total_points += data_df.count()
        return True
    
    def query(self, query_df, k=10, id_col="id", vector_col="vector"):
        """Поиск ближайших соседей для DataFrame с запросами"""
        if self.config.hybrid_mode == 'lsh_only':
            # Только LSH - распределенный поиск
            return self._lsh_only_query(query_df, k, id_col, vector_col)
        elif self.config.hybrid_mode == 'pq_only':
            # Только продуктное квантование - распределенный поиск
            return self._pq_only_query(query_df, k, id_col, vector_col)
        else:  # hybrid_mode == 'two_stage'
            # Двухэтапный поиск: LSH + PQ - распределенный поиск
            return self._two_stage_query(query_df, k, id_col, vector_col)
    
    def _lsh_only_query(self, query_df, k, id_col, vector_col):
        """Только LSH поиск - полностью распределенный"""
        # Собираем запросы на драйвере (их обычно немного)
        query_data = query_df.collect()
        
        results = []
        for row in query_data:
            query_id = row[id_col]
            query_vector = row[vector_col]
            
            # Используем LSH для поиска кандидатов
            candidates = self.lsh_sampler.query(query_vector, k)
            results.append((query_id, candidates))
        
        # Возвращаем результаты в виде DataFrame
        schema = StructType([
            StructField(id_col, LongType(), True),
            StructField("candidates", ArrayType(LongType()), True)
        ])
        
        return self.spark.createDataFrame(results, schema)
    
    def _pq_only_query(self, query_df, k, id_col, vector_col):
        """Только продуктное квантование - полностью распределенный"""
        if not self.is_trained:
            raise ValueError("PQ квантователь не обучен")
        
        # Броадкастим кодбуки для распределенного вычисления
        codebooks_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.codebooks)
        sub_dim_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.sub_dim)
        num_subspaces_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.config.num_subspaces)
        distance_metric_bc = self.spark.sparkContext.broadcast(self.config.lsh_config.distance_metric)
        
        # Для каждого запроса находим ближайших соседей
        query_data = query_df.collect()
        all_results = []
        
        for query_row in query_data:
            query_id = query_row[id_col]
            query_vector = query_row[vector_col]
            
            # Броадкастим вектор запроса
            query_vector_bc = self.spark.sparkContext.broadcast(query_vector)
            
            # Создаем UDF для вычисления расстояния
            def pq_distance_udf_wrapper(data_vector):
                return pq_distance_udf(
                    query_vector_bc.value, data_vector, 
                    codebooks_bc.value, sub_dim_bc.value, 
                    num_subspaces_bc.value, distance_metric_bc.value
                )
            
            pq_distance_udf_spark = udf(pq_distance_udf_wrapper, FloatType())
            
            # Вычисляем расстояния до всех точек
            distances_df = self.data_df.withColumn(
                "distance", 
                pq_distance_udf_spark(col(vector_col))
            )
            
            # Выбираем k ближайших
            window = Window.orderBy("distance")
            ranked_df = distances_df.withColumn("rank", row_number().over(window))
            top_k_df = ranked_df.filter(col("rank") <= k)
            
            # Собираем результаты
            top_k_ids = [row[id_col] for row in top_k_df.select(id_col).collect()]
            all_results.append((query_id, top_k_ids))
            
            # Очищаем broadcast переменную
            query_vector_bc.unpersist()
        
        # Очищаем broadcast переменные
        codebooks_bc.unpersist()
        sub_dim_bc.unpersist()
        num_subspaces_bc.unpersist()
        distance_metric_bc.unpersist()
        
        # Возвращаем результаты в виде DataFrame
        schema = StructType([
            StructField(id_col, LongType(), True),
            StructField("candidates", ArrayType(LongType()), True)
        ])
        
        return self.spark.createDataFrame(all_results, schema)
    
    def _two_stage_query(self, query_df, k, id_col, vector_col):
        """Двухэтапный поиск: LSH + PQ - полностью распределенный"""
        # Собираем запросы на драйвере (их обычно немного)
        query_data = query_df.collect()
        
        all_results = []
        
        for row in query_data:
            query_id = row[id_col]
            query_vector = row[vector_col]
            
            # Первый этап: LSH для поиска кандидатов
            candidate_multiplier = self.config.hybrid_candidate_multiplier
            candidates = self.lsh_sampler.query(query_vector, k * candidate_multiplier)
            
            if not self.is_trained or not candidates:
                # Если PQ не обучен или нет кандидатов, возвращаем результаты LSH
                all_results.append((query_id, candidates[:k]))
                continue
            
            # Второй этап: точное ранжирование кандидатов с помощью PQ
            # Создаем DataFrame с кандидатами
            candidates_df = self.spark.createDataFrame(
                [(cand_id,) for cand_id in candidates], 
                [id_col]
            )
            
            # Получаем векторы кандидатов
            candidates_with_data = candidates_df.join(
                self.data_df, 
                candidates_df[id_col] == self.data_df[id_col]
            ).select(self.data_df[id_col], vector_col)
            
            # Броадкастим кодбуки и вектор запроса для распределенного вычисления
            codebooks_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.codebooks)
            sub_dim_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.sub_dim)
            num_subspaces_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.config.num_subspaces)
            distance_metric_bc = self.spark.sparkContext.broadcast(self.config.lsh_config.distance_metric)
            query_vector_bc = self.spark.sparkContext.broadcast(query_vector)
            
            # Создаем UDF для вычисления расстояния
            def pq_distance_udf_wrapper(data_vector):
                return pq_distance_udf(
                    query_vector_bc.value, data_vector, 
                    codebooks_bc.value, sub_dim_bc.value, 
                    num_subspaces_bc.value, distance_metric_bc.value
                )
            
            pq_distance_udf_spark = udf(pq_distance_udf_wrapper, FloatType())
            
            # Вычисляем расстояния для кандидатов
            candidates_with_distances = candidates_with_data.withColumn(
                "distance", 
                pq_distance_udf_spark(col(vector_col))
            )
            
            # Выбираем k ближайших
            window = Window.orderBy("distance")
            ranked_candidates = candidates_with_distances.withColumn("rank", row_number().over(window))
            top_k_candidates = ranked_candidates.filter(col("rank") <= k)
            
            # Собираем результаты
            top_k_ids = [row[id_col] for row in top_k_candidates.select(id_col).collect()]
            all_results.append((query_id, top_k_ids))
            
            # Очищаем broadcast переменные
            codebooks_bc.unpersist()
            sub_dim_bc.unpersist()
            num_subspaces_bc.unpersist()
            distance_metric_bc.unpersist()
            query_vector_bc.unpersist()
        
        # Возвращаем результаты в виде DataFrame
        schema = StructType([
            StructField(id_col, LongType(), True),
            StructField("candidates", ArrayType(LongType()), True)
        ])
        
        return self.spark.createDataFrame(all_results, schema)
    
    def sample(self, strategy='proportional', size=1000, id_col="id"):
        """Семплирование данных с использованием LSH - полностью распределенный"""
        if self.lsh_sampler.tables_df is None:
            raise ValueError("Нет данных для семплирования")
        
        # Получаем информацию о бакетах только для table_id=0
        bucket_info = self.lsh_sampler.tables_df.filter(col("table_id") == 0) \
            .groupBy("super_hash").agg(collect_list(id_col).alias("points"), F.count(id_col).alias("count"))
        
        # Определяем количество точек из каждого бакета
        if strategy == 'proportional':
            total_count = bucket_info.agg(spark_sum("count")).first()[0]
            bucket_info = bucket_info.withColumn("samples", (col("count") / total_count * size).cast("int"))
        else:  # balanced
            avg_samples = size / bucket_info.count()
            bucket_info = bucket_info.withColumn("samples", lit(avg_samples).cast("int"))
        
        # Корректируем количество samples
        current_total = bucket_info.agg(spark_sum("samples")).first()[0]
        if current_total != size:
            diff = size - current_total
            # Добавляем недостающие samples к случайным бакетам
            bucket_info = bucket_info.withColumn(
                "samples", 
                col("samples") + F.when(row_number().over(Window.orderBy(rand())) <= diff, 1).otherwise(0)
            )
        
        # Выбираем точки из каждого бакета
        def sample_points(points, num_samples):
            if len(points) <= num_samples:
                return points
            return random.sample(points, num_samples)
        
        sample_udf = udf(sample_points, ArrayType(LongType()))
        sampled_df = bucket_info.withColumn("sampled_points", sample_udf(col("points"), col("samples")))
        
        # Преобразуем в плоский формат и добавляем вес
        if strategy == 'proportional':
            # Добавляем вес в sampled_df перед взрывом
            sampled_df = sampled_df.withColumn("weight", col("count") / col("samples"))
            result_df = sampled_df.select(explode(col("sampled_points")).alias(id_col), "weight")
        else:
            result_df = sampled_df.select(explode(col("sampled_points")).alias(id_col))
            result_df = result_df.withColumn("weight", lit(1.0))
        
        return result_df

def generate_synthetic_data(spark, n_samples=1000, n_features=2, pattern='blobs', random_state=42):
    """Генерация синтетических данных для тестирования"""
    if pattern == 'blobs':
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                         centers=3, cluster_std=0.8, random_state=random_state)
    elif pattern == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    elif pattern == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=random_state)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Создаем DataFrame
    data = [(i, Vectors.dense(X[i]), int(y[i])) for i in range(n_samples)]
    return spark.createDataFrame(data, ["id", "vector", "label"])

def evaluate_search_quality(engine, test_data, k=5):
    """Оценка качества поиска ближайших соседей"""
    # Подготовка тестовых запросов
    query_data = test_data.select("id", "vector").limit(10).collect()
    query_df = engine.spark.createDataFrame(query_data, ["id", "vector"])
    
    # Выполнение запросов
    results = engine.query(query_df, k=k).collect()
    
    # Сравнение с истинными метками
    true_labels = {}
    for row in test_data.select("id", "label").collect():
        true_labels[row["id"]] = row["label"]
    
    accuracies = []
    for result in results:
        query_id = result["id"]
        candidates = result["candidates"]
        
        if not candidates:
            continue
            
        # Определяем метку запроса
        query_label = true_labels[query_id]
        
        # Определяем метки кандидатов
        candidate_labels = [true_labels.get(cand_id, -1) for cand_id in candidates]
        
        # Вычисляем accuracy (доля кандидатов с той же меткой)
        same_label_count = sum(1 for label in candidate_labels if label == query_label)
        accuracy = same_label_count / len(candidates)
        accuracies.append(accuracy)
    
    return np.mean(accuracies) if accuracies else 0

def visualize_results(spark, original_data, sampled_data, title="Sampling Results"):
    """Визуализация исходных и семплированных данных"""
    # Собираем данные на драйвере для визуализации
    original_points = original_data.select("vector").rdd.map(lambda r: r[0].toArray()).collect()
    sampled_points = sampled_data.select("vector").rdd.map(lambda r: r[0].toArray()).collect()
    sampled_ids = sampled_data.select("id").rdd.map(lambda r: r[0]).collect()
    
    # Извлекаем координаты
    original_x = [p[0] for p in original_points]
    original_y = [p[1] for p in original_points]
    
    sampled_x = [p[0] for p in sampled_points]
    sampled_y = [p[1] for p in sampled_points]
    
    # Создаем визуализацию
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(original_x, original_y, alpha=0.5, s=10, label="Original")
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.subplot(1, 2, 2)
    plt.scatter(original_x, original_y, alpha=0.2, s=10, label="Original")
    plt.scatter(sampled_x, sampled_y, alpha=0.8, s=30, color='red', label="Sampled")
    
    # Подписываем некоторые точки
    for i, (x, y) in enumerate(zip(sampled_x, sampled_y)):
        if i % 5 == 0:  # Подписываем каждую 5-ю точку
            plt.annotate(str(sampled_ids[i]), (x, y), xytext=(5, 5), 
                         textcoords='offset points', fontsize=8)
    
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("sampling_results.png")
    plt.show()

# Пример использования с оценкой качества
if __name__ == "__main__":
    # Инициализация Spark
    spark = SparkSession.builder \
        .appName("FastRoLSH-DataFrame") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Генерация синтетических данных
    print("Генерация синтетических данных...")
    data_df = generate_synthetic_data(spark, n_samples=1000, n_features=2, pattern='blobs', random_state=42)
    
    # Добавляем шумовые признаки для увеличения размерности
    def add_noise_features(vector, noise_level=0.1):
        original = vector.toArray()
        noise = np.random.normal(0, noise_level, 8)  # Добавляем 8 шумовых признаков
        return Vectors.dense(np.concatenate([original, noise]))
    
    add_noise_udf = udf(add_noise_features, VectorUDT())
    data_df = data_df.withColumn("vector", add_noise_udf(col("vector")))
    
    print(f"Размерность данных: {len(data_df.first()['vector'])}")
    print(f"Количество точек: {data_df.count()}")
    
    # Создание и настройка системы
    config = UnifiedConfig(
        lsh_config=LSHConfig(m=10, k=5, L=2, w=1.0),
        pq_config=PQConfig(num_subspaces=2, num_clusters=8),
        hybrid_mode='two_stage',
        reservoir_size=100
    )
    
    engine = UnifiedQuantizationEngine(spark, config)
    
    # Обновление системы данными
    print("Обновление системы данными...")
    engine.update(data_df)
    
    # Проверка размера LSH таблиц
    print("Размер LSH таблиц: ", engine.lsh_sampler.tables_df.count())
    
    # Выполнение запросов
    print("Выполнение тестовых запросов...")
    first_point = data_df.filter(col("id") == 0).select("vector").first()[0]
    query_data = [(0, first_point)]
    query_df = spark.createDataFrame(query_data, ["id", "vector"])
    
    results = engine.query(query_df, k=5).collect()
    print("Результаты поиска:")
    for row in results:
        print(f"Запрос {row['id']}: {len(row['candidates'])} кандидатов")
        if row['candidates']:
            print(f"Кандидаты: {row['candidates']}")
    
    # Семплирование данных
    print("Семплирование данных...")
    sampled_data = engine.sample('proportional', 50)
    print("Результаты семплирования:")
    sampled_data.show(10)
    
    # Оценка качества поиска
    print("Оценка качества поиска...")
    accuracy = evaluate_search_quality(engine, data_df, k=5)
    print(f"Средняя точность поиска: {accuracy:.3f}")
    
    # Визуализация результатов
    print("Визуализация результатов...")
    # Для визуализации используем только первые 2 признака (остальные - шум)
    def extract_first_two_features(vector):
        return Vectors.dense(vector[:2])
    
    extract_udf = udf(extract_first_two_features, VectorUDT())
    original_2d = data_df.withColumn("vector", extract_udf(col("vector")))
    
    # Получаем семплированные точки с исходными векторами
    sampled_with_vectors = sampled_data.join(original_2d, "id")
    
    visualize_results(spark, original_2d, sampled_with_vectors, "FastRoLSH Sampling Results")
    
    # Дополнительная оценка качества кластеризации
    print("Оценка качества кластеризации...")
    from sklearn.metrics import silhouette_score
    
    # Получаем все данные и метки
    all_data = data_df.select("vector", "label").collect()
    vectors = np.array([row["vector"].toArray() for row in all_data])
    labels = np.array([row["label"] for row in all_data])
    
    # Вычисляем метрики качества
    silhouette = silhouette_score(vectors, labels)
    print(f"Silhouette Score: {silhouette:.3f}")
    
    # Сравниваем распределение меток в исходных и семплированных данных
    original_labels = data_df.select("label").groupBy("label").count().orderBy("label")
    sampled_labels = sampled_data.join(data_df, "id").select("label").groupBy("label").count().orderBy("label")
    
    print("Распределение меток в исходных данных:")
    original_labels.show()
    
    print("Распределение меток в семплированных данных:")
    sampled_labels.show()
    
    # Остановка Spark
    spark.stop()