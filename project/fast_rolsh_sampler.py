import torch
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Union, Any
import joblib
import hashlib
from datetime import datetime
import json
import asyncpg
from asyncpg.pool import Pool
import logging
import math
from scipy.spatial.distance import pdist
from scipy import stats
import itertools

logger = logging.getLogger(__name__)

class FastRoLSHsampler:
    """
    Combined implementation of FastLSH and roLSH for batch data processing.
    Supports online learning with continuous state updates.
    """
    
    def __init__(self, d: int, m: int = 100, k: int = 10, L: int = 10, w: float = 1.0,
                 dataset_id: Optional[int] = None, dataset_name: Optional[str] = None, 
                 db_pool: Optional[Pool] = None, distance_metric: str = "euclidean",
                 initial_radius: Optional[float] = None, radius_expansion: float = 2.0,
                 sampling_ratio: float = 0.1):
        # Core LSH parameters
        self.d = d  # Dimensionality of data vectors
        self.m = m  # Number of hash functions per table
        self.k = k  # Number of hash bits per function
        self.L = L  # Number of hash tables
        self.w = w  # Bucket width for Euclidean distance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device for computation
        self.dataset_id = dataset_id  # Database ID of the dataset
        self.dataset_name = dataset_name  # Name of the dataset
        self.db_pool = db_pool  # Database connection pool
        self.distance_metric = distance_metric  # Distance metric to use
        self.initial_radius = initial_radius  # Initial search radius for roLSH
        self.radius_expansion = radius_expansion  # Radius expansion factor for roLSH
        self.sampling_ratio = sampling_ratio  # Feature sampling ratio for FastLSH
        
        # Calculate number of features to sample (FastLSH)
        self.m_sampled = max(1, int(d * sampling_ratio))
        
        # Data structures for state management
        self.tables = [defaultdict(list) for _ in range(L)]  # Hash tables for storing point indices
        self.total_points = 0  # Total number of processed points
        self.batch_info = {}  # Information about processed batches
        self.last_update = None  # Time of last update
        self.radius_stats = {}  # Statistics for radius optimization
        
        # Flags for data-dependent initialization
        self.is_initialized = False  # Whether model has been initialized with data
        self.first_batch_data = None  # Storage for first batch of data
        
        # Initialize hash functions
        self._init_hash_functions()

    def _init_hash_functions(self):
        """Initialize hash functions based on distance metric"""
        if self.distance_metric == "euclidean":
            # For Euclidean distance: random projections with p-stable distribution
            self.A = torch.randn(self.L, self.k, self.m_sampled, device=self.device)
            self.B = torch.rand(self.L, self.k, device=self.device) * self.w
            self.indices = torch.randint(0, self.d, (self.L, self.k, self.m_sampled), device=self.device)
        elif self.distance_metric == "cosine":
            # For cosine distance: random hyperplanes with normalized vectors
            self.A = torch.randn(self.L, self.k, self.d, device=self.device)
            self.A = self.A / torch.norm(self.A, dim=2, keepdim=True)  # Normalize to unit length
            self.B = torch.zeros(self.L, self.k, device=self.device)  # Not needed for cosine
            self.indices = None  # Not needed for cosine
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    async def update(self, batch_data: torch.Tensor, batch_id: Optional[str] = None) -> bool:
        """
        Update LSH index state with new batch of data.
        
        Args:
            batch_data: Tensor containing batch data vectors
            batch_id: Optional identifier for the batch
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            batch_size = batch_data.shape[0]  # Number of vectors in batch
            
            # Store first batch for data-dependent initialization
            if not self.is_initialized and batch_size > 0:
                self.first_batch_data = batch_data.clone()
                await self._data_dependent_init()
                self.is_initialized = True
            
            # Generate batch ID if not provided
            if batch_id is None:
                batch_hash = hashlib.sha256(batch_data.cpu().numpy().tobytes()).hexdigest()
                batch_id = f"batch_{batch_hash[:16]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Compute hashes for the batch
            batch_hashes = self.hash(batch_data)
            
            # Update LSH tables
            for l in range(self.L):  # For each hash table
                for i in range(batch_size):  # For each vector in batch
                    global_idx = self.total_points + i  # Global index of the point
                    
                    # Form bucket key from hash values
                    hash_key = tuple(batch_hashes[i, l].cpu().numpy())
                    
                    # Add point to appropriate bucket in table
                    self.tables[l][hash_key].append(global_idx)
            
            # Update radius statistics for roLSH
            self._update_radius_stats(batch_data)
            
            # Save batch metadata to database
            features_hash = hashlib.sha256(batch_data.cpu().numpy().tobytes()).hexdigest()
            await self._save_batch_metadata(batch_id, batch_size, features_hash)
            
            # Save LSH table state
            await self._save_lsh_tables()
            
            # Update counters and metadata
            self.total_points += batch_size
            self.last_update = datetime.now()
            self.batch_info[batch_id] = {
                'size': batch_size,
                'processed_at': self.last_update.isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            return False

    async def _data_dependent_init(self):
        """Perform data-dependent initialization of parameters"""
        if self.first_batch_data is None:
            return
            
        X = self.first_batch_data.cpu().numpy()  # Convert to numpy for analysis
        
        if self.distance_metric == "euclidean":
            # Automatic bucket width selection based on data variance
            if self.w is None or self.w <= 0:
                std_dev = np.std(X)  # Calculate standard deviation
                self.w = max(0.1, std_dev * 0.5)  # Heuristic: half of standard deviation
            
            # Automatic initial radius selection
            if self.initial_radius is None or self.initial_radius <= 0:
                # Calculate pairwise distances for a sample of points
                sample_size = min(100, len(X))
                distances = pdist(X[:sample_size])  # Pairwise distances
                self.initial_radius = np.mean(distances) * 0.5  # Heuristic: half of mean distance
                
        elif self.distance_metric == "cosine":
            # Normalize data for cosine distance analysis
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_normalized = X / np.where(norms > 0, norms, 1)
            
            # Analyze distribution of cosine distances
            sample_size = min(100, len(X_normalized))
            cos_distances = pdist(X_normalized[:sample_size], metric='cosine')
            
            # Set parameters based on distance distribution
            if self.initial_radius is None or self.initial_radius <= 0:
                self.initial_radius = np.percentile(cos_distances, 25)  # 25th percentile
        
        # Reinitialize hash functions with new parameters
        self._init_hash_functions()
        
        logger.info(f"Data-dependent initialization completed: w={self.w}, initial_radius={self.initial_radius}")

    def _update_radius_stats(self, batch_data: torch.Tensor):
        """Update radius statistics for roLSH based on new batch data"""
        if self.initial_radius is not None:
            return  # Already set by user or data-dependent initialization
            
        # Calculate mean distance between points in batch
        if batch_data.shape[0] > 1:
            # Use subsample for computational efficiency
            sample_size = min(100, batch_data.shape[0])
            indices = torch.randperm(batch_data.shape[0])[:sample_size]
            sample_data = batch_data[indices]
            
            # Calculate pairwise distances
            if self.distance_metric == "euclidean":
                distances = torch.cdist(sample_data, sample_data, p=2)
            elif self.distance_metric == "cosine":
                # Normalize for cosine distance
                norms = torch.norm(sample_data, dim=1, keepdim=True)
                sample_data_normalized = sample_data / norms.clamp(min=1e-10)
                distances = 1 - torch.mm(sample_data_normalized, sample_data_normalized.t())
            
            mean_distance = distances[distances > 0].mean().item()
            
            # Update statistics
            if 'mean_distance' not in self.radius_stats:
                self.radius_stats['mean_distance'] = mean_distance
                self.radius_stats['count'] = 1
            else:
                total = self.radius_stats['mean_distance'] * self.radius_stats['count']
                self.radius_stats['count'] += 1
                self.radius_stats['mean_distance'] = (total + mean_distance) / self.radius_stats['count']
                
            # Set initial radius based on statistics
            if self.radius_stats['count'] >= 5:  # Set after 5 batches
                self.initial_radius = self.radius_stats['mean_distance'] * 0.5

    async def _save_batch_metadata(self, batch_id: str, batch_size: int, features_hash: str):
        """Save batch metadata to PostgreSQL database"""
        if self.db_pool is None or self.dataset_id is None:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO batches (dataset_id, batch_id, size, features_hash)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (batch_id) DO UPDATE SET
                        size = EXCLUDED.size,
                        features_hash = EXCLUDED.features_hash,
                        processed_at = NOW()
                """, self.dataset_id, batch_id, batch_size, features_hash)
        except Exception as e:
            logger.error(f"Error saving batch metadata {batch_id}: {e}")

    async def _save_lsh_tables(self):
        """Incrementally save LSH table state to PostgreSQL"""
        if self.db_pool is None or self.dataset_id is None:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # For each table, update only changed buckets
                for table_id, table in enumerate(self.tables):
                    for hash_value, indices in table.items():
                        indices_str = json.dumps(indices)  # Convert indices to JSON string
                        hash_str = str(hash_value)  # Convert hash key to string
                        
                        # Check if record already exists
                        existing = await conn.fetchrow(
                            "SELECT point_indices FROM lsh_tables WHERE dataset_id = $1 AND table_id = $2 AND hash_value = $3",
                            self.dataset_id, table_id, hash_str
                        )
                        
                        if existing:
                            # Update existing record if indices changed
                            if existing['point_indices'] != indices_str:
                                await conn.execute(
                                    "UPDATE lsh_tables SET point_indices = $1, updated_at = NOW() WHERE dataset_id = $2 AND table_id = $3 AND hash_value = $4",
                                    indices_str, self.dataset_id, table_id, hash_str
                                )
                        else:
                            # Insert new record
                            await conn.execute(
                                "INSERT INTO lsh_tables (dataset_id, table_id, hash_value, point_indices) VALUES ($1, $2, $3, $4)",
                                self.dataset_id, table_id, hash_str, indices_str
                            )
                
                # Delete buckets that no longer exist
                all_hashes = {str(hash_value) for table in self.tables for hash_value in table.keys()}
                await conn.execute(
                    "DELETE FROM lsh_tables WHERE dataset_id = $1 AND hash_value != ALL($2)",
                    self.dataset_id, list(all_hashes)
                )
        except Exception as e:
            logger.error(f"Error saving LSH tables: {e}")

    async def load_state_from_db(self):
        """Load LSH table state from PostgreSQL database"""
        if self.db_pool is None or self.dataset_id is None:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Load batch metadata
                batch_rows = await conn.fetch("""
                    SELECT batch_id, size, processed_at FROM batches WHERE dataset_id = $1
                """, self.dataset_id)
                
                for row in batch_rows:
                    self.batch_info[row['batch_id']] = {
                        'size': row['size'],
                        'processed_at': row['processed_at'].isoformat() if row['processed_at'] else None
                    }
                    self.total_points += row['size']
                
                # Load LSH tables
                table_rows = await conn.fetch("""
                    SELECT table_id, hash_value, point_indices FROM lsh_tables WHERE dataset_id = $1
                """, self.dataset_id)
                
                # Reset tables before loading
                self.tables = [defaultdict(list) for _ in range(self.L)]
                
                for row in table_rows:
                    table_id = row['table_id']
                    hash_value = row['hash_value']
                    indices = json.loads(row['point_indices'])
                    
                    # Convert string key back to tuple
                    if hash_value.startswith('(') and hash_value.endswith(')'):
                        hash_key = eval(hash_value)
                    else:
                        hash_key = hash_value
                    
                    self.tables[table_id][hash_key] = indices
                    
            logger.info(f"Loaded state from DB for dataset {self.dataset_name}")
                    
        except Exception as e:
            logger.error(f"Error loading state from DB: {e}")

    def hash(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute LSH hashes for a batch of data vectors.
        
        Args:
            X: Tensor of data vectors with shape [batch_size, d]
            
        Returns:
            Tensor of hash codes with shape [batch_size, L, k]
        """
        batch_size = X.shape[0]
        hashes = torch.zeros(batch_size, self.L, self.k, device=self.device, dtype=torch.int64)
        
        if self.distance_metric == "euclidean":
            # FastLSH: random feature sampling + random projection
            for l in range(self.L):
                for k_idx in range(self.k):
                    # Random feature sampling
                    inds = self.indices[l, k_idx]
                    selected_X = X[:, inds]
                    
                    # Random projection: a^T * x + b
                    proj = torch.einsum('nm,m->n', selected_X, self.A[l, k_idx])
                    
                    # Quantization with bucket width w
                    h_val = torch.floor((proj + self.B[l, k_idx]) / self.w).long()
                    hashes[:, l, k_idx] = h_val
                    
        elif self.distance_metric == "cosine":
            # Hyperplane hashing for cosine distance
            # Normalize input vectors for cosine distance
            X_norm = X / torch.norm(X, dim=1, keepdim=True).clamp(min=1e-10)
            
            for l in range(self.L):
                for k_idx in range(self.k):
                    # Apply random hyperplane projection
                    proj = torch.einsum('nd,d->n', X_norm, self.A[l, k_idx])
                    
                    # Quantization: 1 if positive, 0 otherwise
                    h_val = (proj > 0).long()
                    hashes[:, l, k_idx] = h_val
                    
        return hashes

    async def batched_query(self, queries: torch.Tensor, k: int = 10) -> List[List[int]]:
        """
        Find nearest neighbors for a batch of queries using roLSH.
        
        Args:
            queries: Tensor of query vectors with shape [n_queries, d]
            k: Number of neighbors to return for each query
            
        Returns:
            List of lists of neighbor indices for each query
        """
        results = []
        
        # Normalize queries for cosine distance
        if self.distance_metric == "cosine":
            queries = queries / torch.norm(queries, dim=1, keepdim=True).clamp(min=1e-10)
        
        query_hashes = self.hash(queries)
        
        for i in range(queries.shape[0]):
            candidates = set()
            current_radius = self.initial_radius or 1.0
            
            # roLSH: adaptive search with radius expansion
            for expansion_step in range(5):  # Limit number of expansions
                # Search for candidates in LSH tables
                for l in range(self.L):
                    query_hash = tuple(query_hashes[i, l].cpu().numpy())
                    
                    if query_hash in self.tables[l]:
                        candidates.update(self.tables[l][query_hash])
                
                # If enough candidates found, stop searching
                if len(candidates) >= k * 3:
                    break
                    
                # Expand search radius
                current_radius *= self.radius_expansion
                
                # Radius expansion for different metrics
                if self.distance_metric == "euclidean":
                    # Modify hash for expanded radius (Euclidean)
                    modified_hashes = self._get_expanded_hashes(queries[i:i+1], current_radius)
                    for l in range(self.L):
                        mod_hash = tuple(modified_hashes[0, l].cpu().numpy())
                        if mod_hash in self.tables[l]:
                            candidates.update(self.tables[l][mod_hash])
                            
                elif self.distance_metric == "cosine":
                    # Radius expansion for cosine distance via multi-probe search
                    expanded_hashes = self._get_cosine_expanded_hashes(queries[i:i+1], expansion_step)
                    for l in range(self.L):
                        for exp_hash in expanded_hashes[l]:
                            if exp_hash in self.tables[l]:
                                candidates.update(self.tables[l][exp_hash])
            
            # Convert to list and limit number of results
            candidate_list = list(candidates)[:k*3]
            results.append(candidate_list)
        
        return results

    def _get_expanded_hashes(self, query: torch.Tensor, radius: float) -> torch.Tensor:
        """Get hashes for expanded search radius (Euclidean distance)"""
        batch_size = query.shape[0]
        expanded_hashes = torch.zeros(batch_size, self.L, self.k, device=self.device, dtype=torch.int64)
        
        if self.distance_metric == "euclidean":
            for l in range(self.L):
                for k_idx in range(self.k):
                    # Random feature sampling
                    inds = self.indices[l, k_idx]
                    selected_X = query[:, inds]
                    
                    # Random projection with radius adjustment
                    proj = torch.einsum('nm,m->n', selected_X, self.A[l, k_idx])
                    
                    # Quantization with radius-adjusted bucket width
                    h_val = torch.floor((proj + self.B[l, k_idx]) / (self.w * radius)).long()
                    expanded_hashes[:, l, k_idx] = h_val
                    
        return expanded_hashes

    def _get_cosine_expanded_hashes(self, query: torch.Tensor, expansion_step: int) -> List[List[Any]]:
        """Get expanded hashes for cosine distance via multi-probe search"""
        batch_size = query.shape[0]
        expanded_hashes = [[] for _ in range(self.L)]
        
        # Generate base hashes
        base_hashes = self.hash(query)
        
        for l in range(self.L):
            for k_idx in range(self.k):
                base_hash = base_hashes[0, l, k_idx].item()
                
                # Flip more bits in later expansion steps
                bits_to_flip = min(expansion_step + 1, self.k)
                
                # Generate bit flip combinations
                for flip_indices in itertools.combinations(range(self.k), bits_to_flip):
                    modified_hash = list(base_hash) if isinstance(base_hash, (list, tuple)) else [base_hash]
                    
                    for idx in flip_indices:
                        if idx < len(modified_hash):
                            modified_hash[idx] = 1 - modified_hash[idx]  # Flip bit
                    
                    expanded_hashes[l].append(tuple(modified_hash))
        
        return expanded_hashes

    async def sample(self, strategy: str = 'proportional', size: int = 1000) -> Tuple[List[int], List[float]]:
        """
        Perform stratified sampling based on LSH buckets.
        
        Args:
            strategy: Sampling strategy ('proportional' or 'balanced')
            size: Total sample size
            
        Returns:
            Tuple of (sampled indices, sampling weights)
        """
        if not self.tables or self.total_points == 0:
            raise ValueError("No data available for sampling")
            
        table = self.tables[0]  # Use first table for sampling
        buckets = list(table.keys())
        bucket_sizes = [len(table[b]) for b in buckets]
        total_size = self.total_points
        
        # Determine number of points to sample from each bucket
        if strategy == 'proportional':
            p_bucket = np.array(bucket_sizes) / total_size
            n_i = (size * p_bucket).astype(int)
            
            # Adjust for rounding errors
            remainder = size - sum(n_i)
            if remainder > 0:
                fractional = size * p_bucket - n_i
                indices = np.argsort(fractional)[-remainder:]
                n_i[indices] += 1
                
        elif strategy == 'balanced':
            n_per_bucket = size // len(buckets)
            n_i = [min(n_per_bucket, sz) for sz in bucket_sizes]
            
            # Adjust for buckets with insufficient points
            remainder = size - sum(n_i)
            if remainder > 0:
                sorted_buckets = np.argsort(bucket_sizes)[::-1]
                for idx in sorted_buckets:
                    if n_i[idx] < bucket_sizes[idx]:
                        n_i[idx] += 1
                        remainder -= 1
                    if remainder == 0:
                        break
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        # Sample points from buckets
        sampled_indices = []
        weights = []
        for i, bucket in enumerate(buckets):
            indices_in_bucket = table[bucket]
            n_to_sample = n_i[i]
            if n_to_sample == 0:
                continue
                
            # Random sampling without replacement
            sampled = np.random.choice(indices_in_bucket, size=n_to_sample, replace=False)
            sampled_indices.extend(sampled.tolist())
            
            # Calculate weights for bias correction
            if strategy == 'proportional':
                p_select = len(indices_in_bucket) / total_size
            else:
                p_select = 1 / len(buckets)
                
            weight = 1.0 / (p_select * (n_to_sample / len(indices_in_bucket)))
            weights.extend([weight] * n_to_sample)
            
        return sampled_indices, weights

    async def get_stats(self) -> Dict:
        """
        Get statistics about current model state.
        
        Returns:
            Dictionary of model statistics including all parameters
        """
        # Calculate table sizes
        table_sizes = [sum(len(indices) for indices in table.values()) for table in self.tables]
        
        return {
            'dataset_id': self.dataset_id,
            'dataset_name': self.dataset_name,
            'total_points': self.total_points,
            'table_sizes': table_sizes,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'processed_batches': len(self.batch_info),
            'batch_info': self.batch_info,
            'distance_metric': self.distance_metric,
            'm': self.m,
            'k': self.k,
            'L': self.L,
            'w': self.w,
            'initial_radius': self.initial_radius,
            'radius_expansion': self.radius_expansion,
            'sampling_ratio': self.sampling_ratio,
            'radius_stats': self.radius_stats
        }

    async def optimize_parameters(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Optimize LSH parameters based on data characteristics.
        
        Args:
            sample_size: Number of points to use for parameter optimization
            
        Returns:
            Dictionary with optimized parameters
        """
        if self.total_points == 0:
            return {"error": "No data available for optimization"}
        
        # Get a sample of points for analysis
        sample_indices = torch.randint(0, self.total_points, (min(sample_size, self.total_points),))
        sample_points = [self._get_point(i) for i in sample_indices]
        
        if self.distance_metric == "euclidean":
            # Analyze Euclidean distance distribution
            distances = []
            for i in range(len(sample_points)):
                for j in range(i+1, len(sample_points)):
                    dist = torch.norm(sample_points[i] - sample_points[j])
                    distances.append(dist.item())
            
            # Optimize parameters based on distance distribution
            if distances:
                dist_array = np.array(distances)
                self.w = np.percentile(dist_array, 25)  # 25th percentile
                self.initial_radius = np.percentile(dist_array, 10)  # 10th percentile
                
        elif self.distance_metric == "cosine":
            # Analyze cosine distance distribution
            # Normalize points for cosine distance
            norms = [torch.norm(p) for p in sample_points]
            normalized_points = [p / n if n > 0 else p for p, n in zip(sample_points, norms)]
            
            cos_distances = []
            for i in range(len(normalized_points)):
                for j in range(i+1, len(normalized_points)):
                    cos_sim = torch.dot(normalized_points[i], normalized_points[j])
                    cos_dist = 1 - cos_sim.item()
                    cos_distances.append(cos_dist)
            
            # Optimize parameters based on cosine distance distribution
            if cos_distances:
                cos_dist_array = np.array(cos_distances)
                self.initial_radius = np.percentile(cos_dist_array, 25)  # 25th percentile
        
        # Reinitialize hash functions with new parameters
        self._init_hash_functions()
        
        return {
            "w": self.w,
            "initial_radius": self.initial_radius,
            "message": "Parameters optimized successfully"
        }
    
    def _get_point(self, index: int) -> torch.Tensor:
        """
        Get a data point by index (placeholder implementation).
        
        In a real implementation, this would retrieve the actual vector from storage.
        """
        # Placeholder implementation - returns zero vector
        # In production, this should retrieve the actual vector from database or storage
        return torch.zeros(self.d, device=self.device)
    
    def save_state(self, filepath: str) -> bool:
        """
        Save model state to a file.
        
        Args:
            filepath: Path to save model state
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Prepare state for saving
            state = {
                'd': self.d,
                'm': self.m,
                'k': self.k,
                'L': self.L,
                'w': self.w,
                'A': self.A.cpu(),
                'B': self.B.cpu(),
                'indices': self.indices.cpu() if self.indices is not None else None,
                'total_points': self.total_points,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'batch_info': self.batch_info,
                'dataset_id': self.dataset_id,
                'dataset_name': self.dataset_name,
                'distance_metric': self.distance_metric,
                'initial_radius': self.initial_radius,
                'radius_expansion': self.radius_expansion,
                'sampling_ratio': self.sampling_ratio,
                'radius_stats': self.radius_stats,
                'tables': {f"{i}": dict(table) for i, table in enumerate(self.tables)}
            }
            
            # Save to file
            joblib.dump(state, filepath)
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False

    def load_state(self, filepath: str) -> bool:
        """
        Load model state from a file.
        
        Args:
            filepath: Path to load model state from
            
        Returns:
            True if load was successful, False otherwise
        """
        try:
            # Load state from file
            state = joblib.load(filepath)
            
            # Restore parameters
            self.d = state['d']
            self.m = state['m']
            self.k = state['k']
            self.L = state['L']
            self.w = state['w']
            self.A = state['A'].to(self.device)
            self.B = state['B'].to(self.device)
            self.indices = state['indices'].to(self.device) if state['indices'] is not None else None
            self.total_points = state['total_points']
            self.last_update = datetime.fromisoformat(state['last_update']) if state['last_update'] else None
            self.batch_info = state['batch_info']
            self.dataset_id = state.get('dataset_id')
            self.dataset_name = state.get('dataset_name')
            self.distance_metric = state.get('distance_metric', 'euclidean')
            self.initial_radius = state.get('initial_radius')
            self.radius_expansion = state.get('radius_expansion', 2.0)
            self.sampling_ratio = state.get('sampling_ratio', 0.1)
            self.radius_stats = state.get('radius_stats', {})
            
            # Restore tables
            self.tables = [defaultdict(list) for _ in range(self.L)]
            for i, table_data in state['tables'].items():
                table_id = int(i)
                for hash_str, indices in table_data.items():
                    # Convert string key back to tuple
                    if hash_str.startswith('(') and hash_str.endswith(')'):
                        hash_key = eval(hash_str)
                    else:
                        hash_key = hash_str
                    self.tables[table_id][hash_key] = indices
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False