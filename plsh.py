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

# Custom accumulator for collecting statistics
class DictAccumulatorParam(AccumulatorParam):
    """Custom accumulator parameter for dictionary-based statistics collection."""
    
    def zero(self, initialValue: Dict[str, Any]):
        """Initialize with a copy of the initial value."""
        return initialValue.copy()
    
    def addInPlace(self, v1: Dict[str, Any], v2: Dict[str, Any]):
        """Merge two dictionaries by adding values for common keys."""
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
    """Configuration for Locality-Sensitive Hashing."""
    m: int = 100  # Number of dimensions in original space
    k: int = 10   # Number of hash functions per table
    L: int = 5    # Number of hash tables
    w: float = 1.0  # Bucket width
    distance_metric: str = 'euclidean'  # Distance metric
    sampling_ratio: float = 0.1  # Sampling ratio
    random_state: int = 42  # Random seed

@dataclass
class PQConfig:
    """Configuration for Product Quantization."""
    num_subspaces: int = 8  # Number of subspaces
    num_clusters: int = 256  # Number of clusters per subspace
    use_optimized_kmeans: bool = True  # Use optimized K-Means
    batch_size: int = 1000  # Batch size for MiniBatchKMeans
    random_state: int = 42  # Random seed

@dataclass 
class UnifiedConfig:
    """Unified configuration for the quantization system."""
    lsh_config: LSHConfig = field(default_factory=LSHConfig)  # LSH configuration
    pq_config: PQConfig = field(default_factory=PQConfig)  # PQ configuration
    hybrid_mode: str = 'two_stage'  # Hybrid mode: 'lsh_only', 'pq_only', 'two_stage'
    hybrid_candidate_multiplier: int = 10  # Candidate multiplier for hybrid search
    optimization_interval: int = 10  # Optimization interval
    reservoir_size: int = 1000  # Reservoir sampling size

# Helper UDF functions (module level)
def encode_vector_udf(vector, codebooks, sub_dim, num_subspaces):
    """Encode a vector using product quantization codebooks.
    
    Args:
        vector: Input vector to encode
        codebooks: List of codebooks for each subspace
        sub_dim: Dimension of each subspace
        num_subspaces: Number of subspaces
        
    Returns:
        List of quantization codes
    """
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
    """Decode quantization codes back to vector.
    
    Args:
        codes: List of quantization codes
        codebooks: List of codebooks for each subspace
        num_subspaces: Number of subspaces
        sub_dim: Dimension of each subspace
        
    Returns:
        Reconstructed vector
    """
    decoded = []
    for i, code in enumerate(codes):
        decoded.extend(codebooks[i][code])
    return Vectors.dense(decoded)

def pq_distance_udf(query_vector, data_vector, codebooks, sub_dim, num_subspaces, distance_metric):
    """Calculate distance between PQ-encoded vectors.
    
    Args:
        query_vector: Query vector
        data_vector: Data vector
        codebooks: List of codebooks
        sub_dim: Subspace dimension
        num_subspaces: Number of subspaces
        distance_metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Distance between vectors
    """
    # Encode both vectors
    query_code = []
    data_code = []
    
    for i in range(num_subspaces):
        start_idx = i * sub_dim
        end_idx = (i + 1) * sub_dim
        
        # Encode query subspace
        query_subspace = query_vector[start_idx:end_idx]
        query_distances = cdist([query_subspace], codebooks[i], 'euclidean')[0]
        query_code.append(int(np.argmin(query_distances)))
        
        # Encode data subspace
        data_subspace = data_vector[start_idx:end_idx]
        data_distances = cdist([data_subspace], codebooks[i], 'euclidean')[0]
        data_code.append(int(np.argmin(data_distances)))
    
    # Calculate distance between codes
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
    """Improved implementation of FastLSH and roLSH using DataFrame API."""
    
    def __init__(self, spark: SparkSession, config: LSHConfig = None):
        """Initialize LSH sampler.
        
        Args:
            spark: Spark session
            config: LSH configuration
        """
        self.spark = spark
        self.config = config or LSHConfig()
        self.tables_df = None  # LSH tables in DataFrame format
        self.hash_functions = []  # Hash functions
        self.total_points = 0  # Total points indexed
        
        # Accumulators for statistics
        self.stats_acc = spark.sparkContext.accumulator(
            {"table_sizes": {}, "distance_stats": {}}, 
            DictAccumulatorParam()
        )
        
        # Initialize hash functions
        self._init_hash_functions()
    
    def _init_hash_functions(self):
        """Initialize LSH hash functions."""
        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        
        for l in range(self.config.L):
            table_funcs = []
            for _ in range(self.config.k):
                # Generate indices for sampling
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
        """Create UDF for hash computation.
        
        Args:
            table_idx: Table index
            func_idx: Function index
            
        Returns:
            UDF function for hash computation
        """
        func_type, a, b, indices = self.hash_functions[table_idx][func_idx]
        w = self.config.w  # local variable for capture
        
        def hash_udf(vector):
            # Sample vector using predefined indices
            sampled_vector = [vector[i] for i in indices]
            
            if func_type == 'euclidean':
                projection = sum(a_i * v_i for a_i, v_i in zip(a, sampled_vector)) + b
                return int(math.floor(projection / w))
            else:
                dot_product = sum(a_i * v_i for a_i, v_i in zip(a, sampled_vector))
                return 1 if dot_product >= 0 else 0
        
        return udf(hash_udf, IntegerType())
    
    def add_data(self, data_df, id_col="id", vector_col="vector"):
        """Add data to LSH index using DataFrame API.
        
        Args:
            data_df: Input DataFrame
            id_col: ID column name
            vector_col: Vector column name
            
        Returns:
            True if successful
        """
        # Add ID if not exists
        if id_col not in data_df.columns:
            window = Window.orderBy(rand())
            data_df = data_df.withColumn(id_col, row_number().over(window) + self.total_points)
        
        # Calculate hashes for all tables
        hash_dfs = []
        for table_idx in range(self.config.L):
            # Create columns for each hash bit
            hash_exprs = []
            for func_idx in range(self.config.k):
                hash_udf = self._compute_hash_udf(table_idx, func_idx)
                hash_exprs.append(hash_udf(col(vector_col)).alias(f"hash_{func_idx}"))
            
            # Calculate hashes
            table_hash_df = data_df.select(col(id_col), *hash_exprs)
            
            # Create super hash
            hash_cols = [col(f"hash_{i}") for i in range(self.config.k)]
            table_hash_df = table_hash_df.withColumn(
                "super_hash", 
                F.concat_ws(",", *hash_cols)
            ).withColumn("table_id", lit(table_idx))
            
            hash_dfs.append(table_hash_df.select("table_id", "super_hash", id_col))
        
        # Combine all tables
        all_hashes_df = hash_dfs[0]
        for df in hash_dfs[1:]:
            all_hashes_df = all_hashes_df.union(df)
        
        # Update LSH tables
        if self.tables_df is None:
            self.tables_df = all_hashes_df
        else:
            self.tables_df = self.tables_df.union(all_hashes_df)
        
        # Update point counter
        self.total_points += data_df.count()
        
        # Collect statistics - aggregation without collect
        table_stats = all_hashes_df.groupBy("table_id", "super_hash").count()
        
        # Use local reference to accumulator to avoid capturing self
        stats_acc = self.stats_acc
        
        def update_stats(partition):
            """Update statistics accumulator."""
            for row in partition:
                key = (row["table_id"], row["super_hash"])
                stats_acc.add({"table_sizes": {key: row["count"]}})
            return iter([])  # return empty iterator
        
        # Apply function to update statistics
        table_stats.rdd.mapPartitions(update_stats).count()  # trigger execution
        
        return True
    
    def query(self, query_vector, k=10, max_expansions=5):
        """Find nearest neighbors using roLSH.
        
        Args:
            query_vector: Query vector
            k: Number of neighbors to find
            max_expansions: Maximum expansion steps
            
        Returns:
            List of candidate IDs
        """
        if self.tables_df is None:
            return []
        
        # Calculate query hashes
        query_hashes = []
        for table_idx in range(self.config.L):
            super_hash = []
            for func_idx in range(self.config.k):
                # Use same logic as in UDF
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
        
        # Find candidates with adaptive radius
        candidates = set()
        current_radius = self.config.w
        
        for expansion in range(max_expansions):
            for table_idx, query_hash in query_hashes:
                # Search in current table
                table_candidates = self.tables_df.filter(
                    (col("table_id") == table_idx) & (col("super_hash") == query_hash)
                ).select("id").collect()
                
                candidates.update([row["id"] for row in table_candidates])
                
                # Stop if enough candidates found
                if len(candidates) >= k * 3:
                    break
            
            if len(candidates) >= k * 3:
                break
            
            # Expand search radius
            current_radius *= 2.0
            
            # Generate additional hashes for expanded search
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
    """Improved product quantization implementation using DataFrame API."""
    
    def __init__(self, spark: SparkSession, config: PQConfig = None):
        """Initialize product quantizer.
        
        Args:
            spark: Spark session
            config: PQ configuration
        """
        self.spark = spark
        self.config = config or PQConfig()
        self.codebooks = []  # Codebooks for each subspace
        self.sub_dim = None  # Subspace dimension
        self.is_trained = False  # Training status
    
    def train(self, data_df, vector_col="vector"):
        """Train product quantizer using DataFrame API.
        
        Args:
            data_df: Training data
            vector_col: Vector column name
        """
        # Get data dimensionality
        first_vector = data_df.select(vector_col).first()[0]
        dim = len(first_vector)
        
        # Determine subspace dimensions
        if dim % self.config.num_subspaces != 0:
            divisors = [i for i in range(1, dim + 1) if dim % i == 0]
            self.config.num_subspaces = min(divisors, key=lambda x: abs(x - self.config.num_subspaces))
        
        self.sub_dim = dim // self.config.num_subspaces
        
        # Train quantizer for each subspace
        for i in range(self.config.num_subspaces):
            start_idx = i * self.sub_dim
            end_idx = (i + 1) * self.sub_dim
            
            # Extract subspace
            def extract_subspace(vector):
                return Vectors.dense(vector[start_idx:end_idx])
            
            extract_udf = udf(extract_subspace, VectorUDT())
            subspace_df = data_df.withColumn("subspace", extract_udf(col(vector_col)))
            
            # Collect data for clustering (limited set)
            # Use reservoir sampling to limit data size
            reservoir_size = min(10000, subspace_df.count())
            reservoir_df = subspace_df.sample(False, reservoir_size / subspace_df.count(), seed=self.config.random_state)
            
            subspace_data = reservoir_df.select("subspace").rdd.map(lambda r: r[0].toArray()).collect()
            
            # Clustering using MiniBatchKMeans
            if self.config.use_optimized_kmeans:
                kmeans = MiniBatchKMeans(
                    n_clusters=self.config.num_clusters,
                    random_state=self.config.random_state,
                    batch_size=self.config.batch_size
                )
                
                # Train in batches
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
        """Encode data using product quantization.
        
        Args:
            data_df: Data to encode
            vector_col: Vector column name
            
        Returns:
            DataFrame with encoded codes
        """
        if not self.is_trained:
            raise ValueError("Quantizer not trained")
        
        # Use local variables to avoid capturing self
        codebooks = self.codebooks
        sub_dim = self.sub_dim
        num_subspaces = self.config.num_subspaces
        
        def encode_vector(vector):
            return encode_vector_udf(vector, codebooks, sub_dim, num_subspaces)
        
        encode_udf = udf(encode_vector, ArrayType(IntegerType()))
        return data_df.withColumn("pq_codes", encode_udf(col(vector_col)))
    
    def decode(self, codes_df, codes_col="pq_codes"):
        """Decode data from product quantization.
        
        Args:
            codes_df: DataFrame with codes
            codes_col: Codes column name
            
        Returns:
            DataFrame with decoded vectors
        """
        # Use local variables to avoid capturing self
        codebooks = self.codebooks
        num_subspaces = self.config.num_subspaces
        sub_dim = self.sub_dim
        
        def decode_codes(codes):
            return decode_codes_udf(codes, codebooks, num_subspaces, sub_dim)
        
        decode_udf = udf(decode_codes, VectorUDT())
        return codes_df.withColumn("decoded_vector", decode_udf(col(codes_col)))

class UnifiedQuantizationEngine:
    """Improved unified quantization system using DataFrame API."""
    
    def __init__(self, spark: SparkSession, config: UnifiedConfig = None):
        """Initialize unified quantization engine.
        
        Args:
            spark: Spark session
            config: Unified configuration
        """
        self.spark = spark
        self.config = config or UnifiedConfig()
        self.lsh_sampler = FastRoLSHsampler(spark, config.lsh_config)
        self.pq_quantizer = ProductQuantizer(spark, config.pq_config)
        self.is_trained = False  # Training status
        self.total_points = 0  # Total points
        self.data_df = None  # Data storage
        
        # Reservoir sampling for optimization
        self.reservoir_df = None
    
    def update(self, data_df, id_col="id", vector_col="vector"):
        """Update system with new data.
        
        Args:
            data_df: New data
            id_col: ID column name
            vector_col: Vector column name
            
        Returns:
            True if successful
        """
        # Save data
        if self.data_df is None:
            self.data_df = data_df
        else:
            self.data_df = self.data_df.union(data_df)
        
        # Update LSH sampler
        self.lsh_sampler.add_data(data_df, id_col, vector_col)
        
        # Update reservoir sample
        if self.reservoir_df is None:
            self.reservoir_df = data_df.limit(self.config.reservoir_size)
        else:
            # Combine with existing sample and limit size
            combined_df = self.reservoir_df.union(data_df)
            reservoir_count = combined_df.count()
            
            if reservoir_count > self.config.reservoir_size:
                # Select random subset
                fraction = self.config.reservoir_size / reservoir_count
                self.reservoir_df = combined_df.sample(False, fraction, seed=self.config.lsh_config.random_state)
            else:
                self.reservoir_df = combined_df
        
        # Train PQ on reservoir sample if needed
        if (self.config.hybrid_mode != 'lsh_only' and not self.is_trained and 
            self.reservoir_df is not None and self.reservoir_df.count() >= self.config.pq_config.num_clusters * 2):
            
            self.pq_quantizer.train(self.reservoir_df, vector_col)
            self.is_trained = True
        
        self.total_points += data_df.count()
        return True
    
    def query(self, query_df, k=10, id_col="id", vector_col="vector"):
        """Find nearest neighbors for query DataFrame.
        
        Args:
            query_df: Query DataFrame
            k: Number of neighbors
            id_col: ID column name
            vector_col: Vector column name
            
        Returns:
            DataFrame with search results
        """
        if self.config.hybrid_mode == 'lsh_only':
            # LSH only - distributed search
            return self._lsh_only_query(query_df, k, id_col, vector_col)
        elif self.config.hybrid_mode == 'pq_only':
            # PQ only - distributed search
            return self._pq_only_query(query_df, k, id_col, vector_col)
        else:  # hybrid_mode == 'two_stage'
            # Two-stage search: LSH + PQ - distributed search
            return self._two_stage_query(query_df, k, id_col, vector_col)
    
    def _lsh_only_query(self, query_df, k, id_col, vector_col):
        """LSH-only search - fully distributed."""
        # Collect queries on driver (usually few)
        query_data = query_df.collect()
        
        results = []
        for row in query_data:
            query_id = row[id_col]
            query_vector = row[vector_col]
            
            # Use LSH to find candidates
            candidates = self.lsh_sampler.query(query_vector, k)
            results.append((query_id, candidates))
        
        # Return results as DataFrame
        schema = StructType([
            StructField(id_col, LongType(), True),
            StructField("candidates", ArrayType(LongType()), True)
        ])
        
        return self.spark.createDataFrame(results, schema)
    
    def _pq_only_query(self, query_df, k, id_col, vector_col):
        """PQ-only search - fully distributed."""
        if not self.is_trained:
            raise ValueError("PQ quantizer not trained")
        
        # Broadcast codebooks for distributed computation
        codebooks_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.codebooks)
        sub_dim_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.sub_dim)
        num_subspaces_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.config.num_subspaces)
        distance_metric_bc = self.spark.sparkContext.broadcast(self.config.lsh_config.distance_metric)
        
        # For each query find nearest neighbors
        query_data = query_df.collect()
        all_results = []
        
        for query_row in query_data:
            query_id = query_row[id_col]
            query_vector = query_row[vector_col]
            
            # Broadcast query vector
            query_vector_bc = self.spark.sparkContext.broadcast(query_vector)
            
            # Create UDF for distance calculation
            def pq_distance_udf_wrapper(data_vector):
                return pq_distance_udf(
                    query_vector_bc.value, data_vector, 
                    codebooks_bc.value, sub_dim_bc.value, 
                    num_subspaces_bc.value, distance_metric_bc.value
                )
            
            pq_distance_udf_spark = udf(pq_distance_udf_wrapper, FloatType())
            
            # Calculate distances to all points
            distances_df = self.data_df.withColumn(
                "distance", 
                pq_distance_udf_spark(col(vector_col))
            )
            
            # Select k nearest
            window = Window.orderBy("distance")
            ranked_df = distances_df.withColumn("rank", row_number().over(window))
            top_k_df = ranked_df.filter(col("rank") <= k)
            
            # Collect results
            top_k_ids = [row[id_col] for row in top_k_df.select(id_col).collect()]
            all_results.append((query_id, top_k_ids))
            
            # Cleanup broadcast variable
            query_vector_bc.unpersist()
        
        # Cleanup broadcast variables
        codebooks_bc.unpersist()
        sub_dim_bc.unpersist()
        num_subspaces_bc.unpersist()
        distance_metric_bc.unpersist()
        
        # Return results as DataFrame
        schema = StructType([
            StructField(id_col, LongType(), True),
            StructField("candidates", ArrayType(LongType()), True)
        ])
        
        return self.spark.createDataFrame(all_results, schema)
    
    def _two_stage_query(self, query_df, k, id_col, vector_col):
        """Two-stage search: LSH + PQ - fully distributed."""
        # Collect queries on driver (usually few)
        query_data = query_df.collect()
        
        all_results = []
        
        for row in query_data:
            query_id = row[id_col]
            query_vector = row[vector_col]
            
            # First stage: LSH for candidate search
            candidate_multiplier = self.config.hybrid_candidate_multiplier
            candidates = self.lsh_sampler.query(query_vector, k * candidate_multiplier)
            
            if not self.is_trained or not candidates:
                # If PQ not trained or no candidates, return LSH results
                all_results.append((query_id, candidates[:k]))
                continue
            
            # Second stage: exact ranking with PQ
            # Create DataFrame with candidates
            candidates_df = self.spark.createDataFrame(
                [(cand_id,) for cand_id in candidates], 
                [id_col]
            )
            
            # Get candidate vectors
            candidates_with_data = candidates_df.join(
                self.data_df, 
                candidates_df[id_col] == self.data_df[id_col]
            ).select(self.data_df[id_col], vector_col)
            
            # Broadcast codebooks and query vector for distributed computation
            codebooks_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.codebooks)
            sub_dim_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.sub_dim)
            num_subspaces_bc = self.spark.sparkContext.broadcast(self.pq_quantizer.config.num_subspaces)
            distance_metric_bc = self.spark.sparkContext.broadcast(self.config.lsh_config.distance_metric)
            query_vector_bc = self.spark.sparkContext.broadcast(query_vector)
            
            # Create UDF for distance calculation
            def pq_distance_udf_wrapper(data_vector):
                return pq_distance_udf(
                    query_vector_bc.value, data_vector, 
                    codebooks_bc.value, sub_dim_bc.value, 
                    num_subspaces_bc.value, distance_metric_bc.value
                )
            
            pq_distance_udf_spark = udf(pq_distance_udf_wrapper, FloatType())
            
            # Calculate distances for candidates
            candidates_with_distances = candidates_with_data.withColumn(
                "distance", 
                pq_distance_udf_spark(col(vector_col))
            )
            
            # Select k nearest
            window = Window.orderBy("distance")
            ranked_candidates = candidates_with_distances.withColumn("rank", row_number().over(window))
            top_k_candidates = ranked_candidates.filter(col("rank") <= k)
            
            # Collect results
            top_k_ids = [row[id_col] for row in top_k_candidates.select(id_col).collect()]
            all_results.append((query_id, top_k_ids))
            
            # Cleanup broadcast variables
            codebooks_bc.unpersist()
            sub_dim_bc.unpersist()
            num_subspaces_bc.unpersist()
            distance_metric_bc.unpersist()
            query_vector_bc.unpersist()
        
        # Return results as DataFrame
        schema = StructType([
            StructField(id_col, LongType(), True),
            StructField("candidates", ArrayType(LongType()), True)
        ])
        
        return self.spark.createDataFrame(all_results, schema)
    
    def sample(self, strategy='proportional', size=1000, id_col="id"):
        """Sample data using LSH - fully distributed.
        
        Args:
            strategy: Sampling strategy ('proportional' or 'balanced')
            size: Sample size
            id_col: ID column name
            
        Returns:
            Sampled DataFrame
        """
        if self.lsh_sampler.tables_df is None:
            raise ValueError("No data for sampling")
        
        # Get bucket information only for table_id=0
        bucket_info = self.lsh_sampler.tables_df.filter(col("table_id") == 0) \
            .groupBy("super_hash").agg(collect_list(id_col).alias("points"), F.count(id_col).alias("count"))
        
        # Determine number of points from each bucket
        if strategy == 'proportional':
            total_count = bucket_info.agg(spark_sum("count")).first()[0]
            bucket_info = bucket_info.withColumn("samples", (col("count") / total_count * size).cast("int"))
        else:  # balanced
            avg_samples = size / bucket_info.count()
            bucket_info = bucket_info.withColumn("samples", lit(avg_samples).cast("int"))
        
        # Adjust samples count
        current_total = bucket_info.agg(spark_sum("samples")).first()[0]
        if current_total != size:
            diff = size - current_total
            # Add missing samples to random buckets
            bucket_info = bucket_info.withColumn(
                "samples", 
                col("samples") + F.when(row_number().over(Window.orderBy(rand())) <= diff, 1).otherwise(0)
            )
        
        # Select points from each bucket
        def sample_points(points, num_samples):
            if len(points) <= num_samples:
                return points
            return random.sample(points, num_samples)
        
        sample_udf = udf(sample_points, ArrayType(LongType()))
        sampled_df = bucket_info.withColumn("sampled_points", sample_udf(col("points"), col("samples")))
        
        # Convert to flat format and add weight
        if strategy == 'proportional':
            # Add weight to sampled_df before explosion
            sampled_df = sampled_df.withColumn("weight", col("count") / col("samples"))
            result_df = sampled_df.select(explode(col("sampled_points")).alias(id_col), "weight")
        else:
            result_df = sampled_df.select(explode(col("sampled_points")).alias(id_col))
            result_df = result_df.withColumn("weight", lit(1.0))
        
        return result_df

def generate_synthetic_data(spark, n_samples=1000, n_features=2, pattern='blobs', random_state=42):
    """Generate synthetic data for testing.
    
    Args:
        spark: Spark session
        n_samples: Number of samples
        n_features: Number of features
        pattern: Data pattern ('blobs', 'moons', 'circles')
        random_state: Random seed
        
    Returns:
        DataFrame with synthetic data
    """
    if pattern == 'blobs':
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                         centers=3, cluster_std=0.8, random_state=random_state)
    elif pattern == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    elif pattern == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=random_state)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Create DataFrame
    data = [(i, Vectors.dense(X[i]), int(y[i])) for i in range(n_samples)]
    return spark.createDataFrame(data, ["id", "vector", "label"])

def evaluate_search_quality(engine, test_data, k=5):
    """Evaluate search quality.
    
    Args:
        engine: Quantization engine
        test_data: Test data
        k: Number of neighbors
        
    Returns:
        Average accuracy
    """
    # Prepare test queries
    query_data = test_data.select("id", "vector").limit(10).collect()
    query_df = engine.spark.createDataFrame(query_data, ["id", "vector"])
    
    # Execute queries
    results = engine.query(query_df, k=k).collect()
    
    # Compare with true labels
    true_labels = {}
    for row in test_data.select("id", "label").collect():
        true_labels[row["id"]] = row["label"]
    
    accuracies = []
    for result in results:
        query_id = result["id"]
        candidates = result["candidates"]
        
        if not candidates:
            continue
            
        # Get query label
        query_label = true_labels[query_id]
        
        # Get candidate labels
        candidate_labels = [true_labels.get(cand_id, -1) for cand_id in candidates]
        
        # Calculate accuracy (proportion of same-label candidates)
        same_label_count = sum(1 for label in candidate_labels if label == query_label)
        accuracy = same_label_count / len(candidates)
        accuracies.append(accuracy)
    
    return np.mean(accuracies) if accuracies else 0

def visualize_results(spark, original_data, sampled_data, title="Sampling Results"):
    """Visualize original and sampled data.
    
    Args:
        spark: Spark session
        original_data: Original data
        sampled_data: Sampled data
        title: Plot title
    """
    # Collect data on driver for visualization
    original_points = original_data.select("vector").rdd.map(lambda r: r[0].toArray()).collect()
    sampled_points = sampled_data.select("vector").rdd.map(lambda r: r[0].toArray()).collect()
    sampled_ids = sampled_data.select("id").rdd.map(lambda r: r[0]).collect()
    
    # Extract coordinates
    original_x = [p[0] for p in original_points]
    original_y = [p[1] for p in original_points]
    
    sampled_x = [p[0] for p in sampled_points]
    sampled_y = [p[1] for p in sampled_points]
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(original_x, original_y, alpha=0.5, s=10, label="Original")
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.subplot(1, 2, 2)
    plt.scatter(original_x, original_y, alpha=0.2, s=10, label="Original")
    plt.scatter(sampled_x, sampled_y, alpha=0.8, s=30, color='red', label="Sampled")
    
    # Label some points
    for i, (x, y) in enumerate(zip(sampled_x, sampled_y)):
        if i % 5 == 0:  # Label every 5th point
            plt.annotate(str(sampled_ids[i]), (x, y), xytext=(5, 5), 
                         textcoords='offset points', fontsize=8)
    
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("sampling_results.png")
    plt.show()

# Example usage with quality evaluation
if __name__ == "__main__":
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("FastRoLSH-DataFrame") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data_df = generate_synthetic_data(spark, n_samples=1000, n_features=2, pattern='blobs', random_state=42)
    
    # Add noise features to increase dimensionality
    def add_noise_features(vector, noise_level=0.1):
        original = vector.toArray()
        noise = np.random.normal(0, noise_level, 8)  # Add 8 noise features
        return Vectors.dense(np.concatenate([original, noise]))
    
    add_noise_udf = udf(add_noise_features, VectorUDT())
    data_df = data_df.withColumn("vector", add_noise_udf(col("vector")))
    
    print(f"Data dimensionality: {len(data_df.first()['vector'])}")
    print(f"Number of points: {data_df.count()}")
    
    # Create and configure system
    config = UnifiedConfig(
        lsh_config=LSHConfig(m=10, k=5, L=2, w=1.0),
        pq_config=PQConfig(num_subspaces=2, num_clusters=8),
        hybrid_mode='two_stage',
        reservoir_size=100
    )
    
    engine = UnifiedQuantizationEngine(spark, config)
    
    # Update system with data
    print("Updating system with data...")
    engine.update(data_df)
    
    # Check LSH table size
    print("LSH table size: ", engine.lsh_sampler.tables_df.count())
    
    # Execute queries
    print("Executing test queries...")
    first_point = data_df.filter(col("id") == 0).select("vector").first()[0]
    query_data = [(0, first_point)]
    query_df = spark.createDataFrame(query_data, ["id", "vector"])
    
    results = engine.query(query_df, k=5).collect()
    print("Search results:")
    for row in results:
        print(f"Query {row['id']}: {len(row['candidates'])} candidates")
        if row['candidates']:
            print(f"Candidates: {row['candidates']}")
    
    # Sample data
    print("Sampling data...")
    sampled_data = engine.sample('proportional', 50)
    print("Sampling results:")
    sampled_data.show(10)
    
    # Evaluate search quality
    print("Evaluating search quality...")
    accuracy = evaluate_search_quality(engine, data_df, k=5)
    print(f"Average search accuracy: {accuracy:.3f}")
    
    # Visualize results
    print("Visualizing results...")
    # For visualization use only first 2 features (rest are noise)
    def extract_first_two_features(vector):
        return Vectors.dense(vector[:2])
    
    extract_udf = udf(extract_first_two_features, VectorUDT())
    original_2d = data_df.withColumn("vector", extract_udf(col("vector")))
    
    # Get sampled points with original vectors
    sampled_with_vectors = sampled_data.join(original_2d, "id")
    
    visualize_results(spark, original_2d, sampled_with_vectors, "FastRoLSH Sampling Results")
    
    # Additional clustering quality evaluation
    print("Evaluating clustering quality...")
    from sklearn.metrics import silhouette_score
    
    # Get all data and labels
    all_data = data_df.select("vector", "label").collect()
    vectors = np.array([row["vector"].toArray() for row in all_data])
    labels = np.array([row["label"] for row in all_data])
    
    # Calculate quality metrics
    silhouette = silhouette_score(vectors, labels)
    print(f"Silhouette Score: {silhouette:.3f}")
    
    # Compare label distribution in original and sampled data
    original_labels = data_df.select("label").groupBy("label").count().orderBy("label")
    sampled_labels = sampled_data.join(data_df, "id").select("label").groupBy("label").count().orderBy("label")
    
    print("Label distribution in original data:")
    original_labels.show()
    
    print("Label distribution in sampled data:")
    sampled_labels.show()
    
    # Stop Spark
    spark.stop()
