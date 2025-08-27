import requests
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import adjusted_rand_score, accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FastRoLSHTestClient:
    """
    Comprehensive test client for FastRoLSH API with support for various testing scenarios.
    This client allows testing of dataset creation, batch processing, querying, sampling,
    and parameter optimization.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        # Initialize the test client with base API URL
        self.base_url = base_url  # Base URL of the FastRoLSH API server
        self.session = requests.Session()  # HTTP session for making requests
        self.test_datasets = {}  # Dictionary to store test dataset information
    
    def create_dataset(self, name: str, dimension: int, m: int = 100, k: int = 10, L: int = 5, 
                      w: float = 1.0, distance_metric: str = "euclidean", 
                      initial_radius: Optional[float] = None, radius_expansion: float = 2.0,
                      sampling_ratio: float = 0.1) -> Dict[str, Any]:
        """
        Create a new dataset with custom parameters via the API.
        
        Args:
            name: Unique name for the dataset
            dimension: Dimensionality of the data vectors
            m: Number of hash functions per table
            k: Number of hash bits per function
            L: Number of hash tables
            w: Bucket width for Euclidean distance
            distance_metric: Distance metric ('euclidean' or 'cosine')
            initial_radius: Initial search radius for roLSH
            radius_expansion: Radius expansion factor for roLSH
            sampling_ratio: Feature sampling ratio for FastLSH
            
        Returns:
            API response containing dataset creation result
        """
        # Validate distance metric
        if distance_metric not in ["euclidean", "cosine"]:
            raise ValueError("Distance metric must be 'euclidean' or 'cosine'")
        
        # Make API request to create dataset
        url = f"{self.base_url}/datasets/"
        payload = {
            "name": name, 
            "dimension": dimension,
            "m": m,
            "k": k,
            "L": L,
            "w": w,
            "distance_metric": distance_metric,
            "initial_radius": initial_radius,
            "radius_expansion": radius_expansion,
            "sampling_ratio": sampling_ratio
        }
        response = self.session.post(url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    
    def process_batch(self, dataset_name: str, data: List[List[float]], batch_id: str = None) -> Dict[str, Any]:
        """
        Send a batch of data for processing via the API.
        
        Args:
            dataset_name: Name of the dataset to add data to
            data: List of data vectors to process
            batch_id: Optional custom identifier for the batch
            
        Returns:
            API response containing batch processing result
        """
        url = f"{self.base_url}/batches/"
        payload = {
            "dataset_name": dataset_name,
            "batch_id": batch_id,
            "data": data
        }
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def query_neighbors(self, dataset_name: str, queries: List[List[float]], k: int = 10) -> Dict[str, Any]:
        """
        Query for nearest neighbors of given vectors via the API.
        
        Args:
            dataset_name: Name of the dataset to query
            queries: List of query vectors
            k: Number of neighbors to return for each query
            
        Returns:
            API response containing query results
        """
        url = f"{self.base_url}/query/"
        payload = {
            "dataset_name": dataset_name,
            "queries": queries,
            "k": k
        }
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def sample_data(self, dataset_name: str, strategy: str = "proportional", size: int = 1000) -> Dict[str, Any]:
        """
        Sample data points using LSH-based sampling strategies via the API.
        
        Args:
            dataset_name: Name of the dataset to sample from
            strategy: Sampling strategy ('proportional' or 'balanced')
            size: Number of samples to return
            
        Returns:
            API response containing sampling results
        """
        url = f"{self.base_url}/sample/"
        payload = {
            "dataset_name": dataset_name,
            "strategy": strategy,
            "size": size
        }
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        Get a list of all datasets in the system via the API.
        
        Returns:
            List of dataset information dictionaries
        """
        url = f"{self.base_url}/datasets/"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_model_state(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get current state and statistics of a model via the API.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Model state and statistics
        """
        url = f"{self.base_url}/model/state/{dataset_name}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health status of the API server and database connection.
        
        Returns:
            Health status information
        """
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def wait_for_processing(self, dataset_name: str, expected_points: int, timeout: int = 30):
        """
        Wait for all batches to be processed by periodically checking model state.
        
        Args:
            dataset_name: Name of the dataset to monitor
            expected_points: Expected number of points after processing
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if processing completed, False if timeout reached
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                state = self.get_model_state(dataset_name)
                if state['total_points'] >= expected_points:
                    print(f"All batches processed. Points: {state['total_points']}/{expected_points}")
                    return True
                print(f"Waiting for processing... Points: {state['total_points']}/{expected_points}")
                time.sleep(1)
            except:
                print("Error getting model state, retrying...")
                time.sleep(1)
        print(f"Timeout waiting for batch processing")
        return False
    
    def generate_test_data(self, dataset_type: str, n_samples: int = 1000, n_features: int = 10, 
                          n_classes: int = 3, n_informative: int = 5, distance_metric: str = "euclidean"):
        """
        Generate test data of different types for testing purposes.
        
        Args:
            dataset_type: Type of data to generate ('classification', 'regression', or 'clustering')
            n_samples: Number of samples to generate
            n_features: Number of features for each sample
            n_classes: Number of classes (for classification and clustering)
            n_informative: Number of informative features
            distance_metric: Distance metric to optimize for
            
        Returns:
            Tuple of (data, labels) where data is a list of vectors and labels is a list of labels
        """
        # For all metrics, use standard data generation
        if dataset_type == "classification":
            X, y = make_classification(
                n_samples=n_samples, 
                n_features=n_features,
                n_classes=n_classes,
                n_informative=n_informative,
                random_state=42
            )
            # Normalize data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            return X.tolist(), y.tolist()
        
        elif dataset_type == "regression":
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                random_state=42
            )
            # Normalize data and target
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            y = scaler.fit_transform(y.reshape(-1, 1)).flatten().tolist()
            return X.tolist(), y
        
        elif dataset_type == "clustering":
            # Generate data for clustering
            X, y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=n_classes,
                cluster_std=1.5,
                random_state=42
            )
            # Normalize data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            return X.tolist(), y.tolist()
        
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def test_nearest_neighbors(self, dataset_name: str, n_samples: int = 1000, n_features: int = 10, 
                              n_queries: int = 5, k: int = 5, distance_metric: str = "euclidean"):
        """
        Test nearest neighbor search functionality with visualization.
        
        Args:
            dataset_name: Name for the test dataset
            n_samples: Number of samples to generate
            n_features: Number of features for each sample
            n_queries: Number of query points to test
            k: Number of neighbors to find for each query
            distance_metric: Distance metric to use
            
        Returns:
            Dictionary containing test results
        """
        print(f"\n=== Testing nearest neighbor search for dataset: {dataset_name} ===")
        
        # Generate test data
        X, _ = self.generate_test_data("clustering", n_samples, n_features, distance_metric=distance_metric)
        
        # Create dataset with specified distance metric
        self.create_dataset(dataset_name, n_features, distance_metric=distance_metric)
        
        # Split into batches and send
        batch_size = 200
        for i in range(0, len(X), batch_size):
            batch_data = X[i:i+batch_size]
            batch_response = self.process_batch(dataset_name, batch_data, f"batch_{i//batch_size}")
            print(f"Sent batch {i//batch_size}: {batch_response['message']}")
            time.sleep(0.1)  # Small delay between batches
        
        # Wait for all batches to be processed
        if not self.wait_for_processing(dataset_name, n_samples):
            return {"error": "Batch processing timeout"}
        
        # Generate test queries from the same data
        query_indices = np.random.choice(len(X), size=min(n_queries, len(X)), replace=False)
        query_data = [X[i] for i in query_indices]
        
        # Execute queries
        query_response = self.query_neighbors(dataset_name, query_data, k)
        
        # Analyze results
        found_neighbors = sum(len(result) > 0 for result in query_response['results'])
        print(f"Found neighbors for {found_neighbors}/{n_queries} queries")
        
        # Visualize results for 2D and 3D data
        if n_features in [2, 3]:
            self._visualize_neighbors(X, query_data, query_response, query_indices, dataset_name, n_features)
        
        # Get model state
        state_response = self.get_model_state(dataset_name)
        print(f"Model state: {state_response['total_points']} points, {len(state_response['batch_info'])} batches")
        
        return {
            "query_results": query_response,
            "model_state": state_response,
            "found_neighbors_ratio": found_neighbors / n_queries
        }
    
    def _visualize_neighbors(self, X, query_data, query_response, query_indices, dataset_name, n_features):
        """
        Visualize queries and found neighbors for 2D and 3D data.
        
        Args:
            X: All data points
            query_data: Query points
            query_response: Query results from API
            query_indices: Indices of query points in the dataset
            dataset_name: Name of the dataset
            n_features: Number of features (2 or 3 for visualization)
        """
        if n_features == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Display all points
            ax.scatter([x[0] for x in X], [x[1] for x in X], c='lightgray', alpha=0.5, label='All points')
            
            # Display queries and their neighbors
            for i, (query, neighbors) in enumerate(zip(query_data, query_response['results'])):
                # Query point
                ax.scatter(query[0], query[1], c=f'C{i}', marker='*', s=200, label=f'Query {i}')
                
                # Found neighbors
                if neighbors:
                    neighbor_points = [X[idx] for idx in neighbors if idx < len(X)]
                    ax.scatter(
                        [p[0] for p in neighbor_points],
                        [p[1] for p in neighbor_points],
                        c=f'C{i}', alpha=0.7, label=f'Neighbors of query {i}'
                    )
            
            ax.set_title(f'Neighbor search results ({dataset_name})')
            ax.legend()
            plt.savefig(f"{dataset_name}_neighbors_2d.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        elif n_features == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Display all points
            ax.scatter([x[0] for x in X], [x[1] for x in X], [x[2] for x in X], 
                      c='lightgray', alpha=0.3, label='All points')
            
            # Display queries and their neighbors
            for i, (query, neighbors) in enumerate(zip(query_data, query_response['results'])):
                # Query point
                ax.scatter([query[0]], [query[1]], [query[2]], 
                          c=f'C{i}', marker='*', s=200, label=f'Query {i}')
                
                # Found neighbors
                if neighbors:
                    neighbor_points = [X[idx] for idx in neighbors if idx < len(X)]
                    ax.scatter(
                        [p[0] for p in neighbor_points],
                        [p[1] for p in neighbor_points],
                        [p[2] for p in neighbor_points],
                        c=f'C{i}', alpha=0.7, label=f'Neighbors of query {i}'
                    )
            
            ax.set_title(f'Neighbor search results ({dataset_name})')
            ax.legend()
            plt.savefig(f"{dataset_name}_neighbors_3d.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def test_clustering(self, dataset_name: str, n_samples: int = 1000, n_features: int = 10, 
                       n_clusters: int = 3, sample_size: int = 300, distance_metric: str = "euclidean"):
        """
        Test clustering on sampled data.
        
        Args:
            dataset_name: Name for the test dataset
            n_samples: Number of samples to generate
            n_features: Number of features for each sample
            n_clusters: Number of clusters to generate
            sample_size: Number of samples to use for clustering
            distance_metric: Distance metric to use
            
        Returns:
            Dictionary containing clustering test results
        """
        print(f"\n=== Testing clustering for dataset: {dataset_name} ===")
        
        # Generate test data
        X, y_true = self.generate_test_data("clustering", n_samples, n_features, n_clusters, distance_metric=distance_metric)
        
        # Create dataset with specified distance metric
        self.create_dataset(dataset_name, n_features, distance_metric=distance_metric)
        
        # Send data
        batch_size = 200
        for i in range(0, len(X), batch_size):
            batch_data = X[i:i+batch_size]
            batch_response = self.process_batch(dataset_name, batch_data, f"batch_{i//batch_size}")
            print(f"Sent batch {i//batch_size}: {batch_response['message']}")
            time.sleep(0.1)
        
        # Wait for all batches to be processed
        if not self.wait_for_processing(dataset_name, n_samples):
            return {"error": "Batch processing timeout"}
        
        # Sample data
        sample_response = self.sample_data(dataset_name, "proportional", sample_size)
        sampled_indices = sample_response["indices"]
        
        # Get sampled data
        X_sampled = [X[i] for i in sampled_indices if i < len(X)]
        y_sampled = [y_true[i] for i in sampled_indices if i < len(y_true)]
        
        # Check if we have enough points for clustering
        if len(X_sampled) < n_clusters:
            print(f"Not enough points for clustering: {len(X_sampled)} < {n_clusters}")
            return {"error": "Not enough points for clustering"}
        
        # Clustering on full data
        kmeans_full = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_full = kmeans_full.fit_predict(X)
        
        # Clustering on sampled data
        kmeans_sample = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_sample = kmeans_sample.fit_predict(X_sampled)
        
        # Predict for all points using model trained on sample
        labels_all_from_sample = kmeans_sample.predict(X)
        
        # Evaluate quality
        ari_full = adjusted_rand_score(y_true, labels_full)
        ari_sample = adjusted_rand_score(y_sampled, labels_sample) if len(y_sampled) > 0 else 0
        ari_all_from_sample = adjusted_rand_score(y_true, labels_all_from_sample)
        
        print(f"Adjusted Rand Index - Full data: {ari_full:.4f}")
        print(f"Adjusted Rand Index - Sampled data: {ari_sample:.4f}")
        print(f"Adjusted Rand Index - All data from sampled model: {ari_all_from_sample:.4f}")
        
        # Visualization for 2D data
        if n_features == 2:
            self._plot_clustering_results(X, y_true, labels_full, labels_all_from_sample, 
                                         sampled_indices, dataset_name)
        
        return {
            "ari_full": ari_full,
            "ari_sample": ari_sample,
            "ari_all_from_sample": ari_all_from_sample,
            "sampled_indices": sampled_indices,
            "sampled_points_count": len(X_sampled)
        }
    
    def test_classification(self, dataset_name: str, n_samples: int = 1000, n_features: int = 10, 
                           n_classes: int = 3, sample_size: int = 300, distance_metric: str = "euclidean"):
        """
        Test classification on sampled data.
        
        Args:
            dataset_name: Name for the test dataset
            n_samples: Number of samples to generate
            n_features: Number of features for each sample
            n_classes: Number of classes to generate
            sample_size: Number of samples to use for classification
            distance_metric: Distance metric to use
            
        Returns:
            Dictionary containing classification test results
        """
        print(f"\n=== Testing classification for dataset: {dataset_name} ===")
        
        # Generate test data
        X, y_true = self.generate_test_data("classification", n_samples, n_features, n_classes, distance_metric=distance_metric)
        
        # Create dataset with specified distance metric
        self.create_dataset(dataset_name, n_features, distance_metric=distance_metric)
        
        # Send data
        batch_size = 200
        for i in range(0, len(X), batch_size):
            batch_data = X[i:i+batch_size]
            batch_response = self.process_batch(dataset_name, batch_data, f"batch_{i//batch_size}")
            print(f"Sent batch {i//batch_size}: {batch_response['message']}")
            time.sleep(0.1)
        
        # Wait for all batches to be processed
        if not self.wait_for_processing(dataset_name, n_samples):
            return {"error": "Batch processing timeout"}
        
        # Sample data
        sample_response = self.sample_data(dataset_name, "proportional", sample_size)
        sampled_indices = sample_response["indices"]
        
        # Get sampled data
        X_sampled = [X[i] for i in sampled_indices if i < len(X)]
        y_sampled = [y_true[i] for i in sampled_indices if i < len(y_true)]
        
        # Split into train and test sets
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_true[:split_idx], y_true[split_idx:]
        
        # Check if we have enough points for training
        if len(X_sampled) < n_classes * 2:
            print(f"Not enough points for classification: {len(X_sampled)} < {n_classes * 2}")
            return {"error": "Not enough points for classification"}
        
        # Classification on full data
        clf_full = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_full.fit(X_train, y_train)
        y_pred_full = clf_full.predict(X_test)
        accuracy_full = accuracy_score(y_test, y_pred_full)
        
        # Classification on sampled data
        # Take only points from sample that are in training part
        X_sampled_train = [X[i] for i in sampled_indices if i < split_idx]
        y_sampled_train = [y_true[i] for i in sampled_indices if i < split_idx]
        
        clf_sample = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_sample.fit(X_sampled_train, y_sampled_train)
        y_pred_sample = clf_sample.predict(X_test)
        accuracy_sample = accuracy_score(y_test, y_pred_sample)
        
        print(f"Accuracy - Full data: {accuracy_full:.4f}")
        print(f"Accuracy - Sampled data: {accuracy_sample:.4f}")
        
        return {
            "accuracy_full": accuracy_full,
            "accuracy_sample": accuracy_sample,
            "sampled_indices": sampled_indices,
            "sampled_points_count": len(X_sampled_train)
        }
    
    def test_regression(self, dataset_name: str, n_samples: int = 1000, n_features: int = 10, 
                       sample_size: int = 300, distance_metric: str = "euclidean"):
        """
        Test regression on sampled data.
        
        Args:
            dataset_name: Name for the test dataset
            n_samples: Number of samples to generate
            n_features: Number of features for each sample
            sample_size: Number of samples to use for regression
            distance_metric: Distance metric to use
            
        Returns:
            Dictionary containing regression test results
        """
        print(f"\n=== Testing regression for dataset: {dataset_name} ===")
        
        # Generate test data
        X, y_true = self.generate_test_data("regression", n_samples, n_features, distance_metric=distance_metric)
        
        # Create dataset with specified distance metric
        self.create_dataset(dataset_name, n_features, distance_metric=distance_metric)
        
        # Send data
        batch_size = 200
        for i in range(0, len(X), batch_size):
            batch_data = X[i:i+batch_size]
            batch_response = self.process_batch(dataset_name, batch_data, f"batch_{i//batch_size}")
            print(f"Sent batch {i//batch_size}: {batch_response['message']}")
            time.sleep(0.1)
        
        # Wait for all batches to be processed
        if not self.wait_for_processing(dataset_name, n_samples):
            return {"error": "Batch processing timeout"}
        
        # Sample data
        sample_response = self.sample_data(dataset_name, "proportional", sample_size)
        sampled_indices = sample_response["indices"]
        
        # Get sampled data
        X_sampled = [X[i] for i in sampled_indices if i < len(X)]
        y_sampled = [y_true[i] for i in sampled_indices if i < len(y_true)]
        
        # Split into train and test sets
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_true[:split_idx], y_true[split_idx:]
        
        # Check if we have enough points for training
        if len(X_sampled) < 10:
            print(f"Not enough points for regression: {len(X_sampled)} < 10")
            return {"error": "Not enough points for regression"}
        
        # Regression on full data
        reg_full = RandomForestRegressor(n_estimators=50, random_state=42)
        reg_full.fit(X_train, y_train)
        y_pred_full = reg_full.predict(X_test)
        mse_full = mean_squared_error(y_test, y_pred_full)
        
        # Regression on sampled data
        # Take only points from sample that are in training part
        X_sampled_train = [X[i] for i in sampled_indices if i < split_idx]
        y_sampled_train = [y_true[i] for i in sampled_indices if i < split_idx]
        
        reg_sample = RandomForestRegressor(n_estimators=50, random_state=42)
        reg_sample.fit(X_sampled_train, y_sampled_train)
        y_pred_sample = reg_sample.predict(X_test)
        mse_sample = mean_squared_error(y_test, y_pred_sample)
        
        print(f"MSE - Full data: {mse_full:.4f}")
        print(f"MSE - Sampled data: {mse_sample:.4f}")
        
        return {
            "mse_full": mse_full,
            "mse_sample": mse_sample,
            "sampled_indices": sampled_indices,
            "sampled_points_count": len(X_sampled_train)
        }
    
    def _plot_clustering_results(self, X, y_true, labels_full, labels_all_from_sample, 
                                sampled_indices, dataset_name):
        """
        Visualize clustering results for 2D data.
        
        Args:
            X: All data points
            y_true: True labels
            labels_full: Labels from clustering on full data
            labels_all_from_sample: Labels from clustering on all data using model trained on sample
            sampled_indices: Indices of sampled points
            dataset_name: Name of the dataset
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # True clusters
        axes[0, 0].scatter([x[0] for x in X], [x[1] for x in X], c=y_true, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('True clusters')
        
        # Clustering on full data
        axes[0, 1].scatter([x[0] for x in X], [x[1] for x in X], c=labels_full, cmap='viridis', alpha=0.6)
        axes[0, 1].set_title('Clustering on full data')
        
        # Clustering on all data using model trained on sample
        axes[1, 0].scatter([x[0] for x in X], [x[1] for x in X], c=labels_all_from_sample, cmap='viridis', alpha=0.6)
        axes[1, 0].set_title('Clustering on all data (model from sample)')
        
        # Highlight sampled points
        axes[1, 1].scatter([x[0] for x in X], [x[1] for x in X], c='gray', alpha=0.2)
        sampled_points = [X[i] for i in sampled_indices if i < len(X)]
        axes[1, 1].scatter([x[0] for x in sampled_points], [x[1] for x in sampled_points], 
                          c='red', alpha=0.6, s=30)
        axes[1, 1].set_title('Sampled points')
        
        plt.tight_layout()
        plt.savefig(f"{dataset_name}_clustering_results.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_different_metrics(self):
        """
        Test the system with different distance metrics.
        
        Returns:
            Dictionary containing test results for different metrics
        """
        print("\n=== Testing different distance metrics ===")
        
        timestamp = int(time.time())
        results = {}
        
        # Test different distance metrics
        for metric in ["euclidean", "cosine"]:
            print(f"\n--- Testing metric: {metric} ---")
            
            # Generate data
            X, y_true = self.generate_test_data("clustering", 1000, 10, 3, distance_metric=metric)
            
            # Create dataset with specified metric
            dataset_name = f"test_metric_{metric}_{timestamp}"
            self.create_dataset(dataset_name, 10, distance_metric=metric)
            
            # Send data
            batch_size = 200
            for i in range(0, len(X), batch_size):
                batch_data = X[i:i+batch_size]
                self.process_batch(dataset_name, batch_data, f"batch_{i//batch_size}")
            
            # Wait for processing
            if not self.wait_for_processing(dataset_name, len(X)):
                results[metric] = {"error": "Batch processing timeout"}
                continue
            
            # Query neighbors
            query_indices = np.random.choice(len(X), size=5, replace=False)
            query_data = [X[i] for i in query_indices]
            
            query_response = self.query_neighbors(dataset_name, query_data, 5)
            found_neighbors = sum(len(result) > 0 for result in query_response['results'])
            
            results[metric] = {
                "found_neighbors_ratio": found_neighbors / 5,
                "query_results": query_response
            }
            
            print(f"Metric {metric}: found neighbors for {found_neighbors}/5 queries")
        
        return results
    
    def test_different_parameters(self):
        """
        Test the system with different LSH parameters.
        
        Returns:
            Dictionary containing test results for different parameters
        """
        print("\n=== Testing different LSH parameters ===")
        
        timestamp = int(time.time())
        results = {}
        
        # Test different parameter sets
        test_cases = [
            {"m": 50, "k": 5, "L": 3, "name": "low_params"},
            {"m": 100, "k": 10, "L": 5, "name": "default_params"},
            {"m": 200, "k": 15, "L": 10, "name": "high_params"}
        ]
        
        for params in test_cases:
            print(f"\n--- Testing parameters: {params['name']} ---")
            
            # Generate data
            X, y_true = self.generate_test_data("clustering", 1000, 10, 3)
            
            # Create dataset with specified parameters
            dataset_name = f"test_params_{params['name']}_{timestamp}"
            self.create_dataset(
                dataset_name, 10, 
                m=params["m"], k=params["k"], L=params["L"]
            )
            
            # Send data
            batch_size = 200
            for i in range(0, len(X), batch_size):
                batch_data = X[i:i+batch_size]
                self.process_batch(dataset_name, batch_data, f"batch_{i//batch_size}")
            
            # Wait for processing
            if not self.wait_for_processing(dataset_name, len(X)):
                results[params['name']] = {"error": "Batch processing timeout"}
                continue
            
            # Query neighbors
            query_indices = np.random.choice(len(X), size=5, replace=False)
            query_data = [X[i] for i in query_indices]
            
            query_response = self.query_neighbors(dataset_name, query_data, 5)
            found_neighbors = sum(len(result) > 0 for result in query_response['results'])
            
            results[params['name']] = {
                "found_neighbors_ratio": found_neighbors / 5,
                "query_results": query_response
            }
            
            print(f"Parameters {params['name']}: found neighbors for {found_neighbors}/5 queries")
        
        return results
    
    def test_parameter_optimization(self, dataset_name: str, n_samples: int = 1000, n_features: int = 10):
        """Test parameter optimization functionality"""
        print(f"\n=== Testing parameter optimization for dataset: {dataset_name} ===")
        
        # Create dataset
        self.create_dataset(dataset_name, n_features, distance_metric="euclidean")
        
        # Generate test data
        X, _ = self.generate_test_data("clustering", n_samples, n_features)
        
        # Send data
        batch_size = 200
        for i in range(0, len(X), batch_size):
            batch_data = X[i:i+batch_size]
            self.process_batch(dataset_name, batch_data, f"batch_{i//batch_size}")
        
        # Wait for processing
        self.wait_for_processing(dataset_name, n_samples)
        
        # Optimize parameters
        optimization_url = f"{self.base_url}/model/optimize/{dataset_name}?sample_size=500"
        response = self.session.post(optimization_url)
        response.raise_for_status()
        optimization_result = response.json()
        
        print(f"Optimization result: w={optimization_result.get('w')}, initial_radius={optimization_result.get('initial_radius')}")
        
        # Verify parameters changed by checking model state
        state = self.get_model_state(dataset_name)
        
        # Check if optimization was successful by comparing values
        if 'w' in optimization_result and 'w' in state:
            print(f"Parameter 'w' optimized: {state['w']} -> {optimization_result['w']}")
        
        if 'initial_radius' in optimization_result and 'initial_radius' in state:
            print(f"Parameter 'initial_radius' optimized: {state['initial_radius']} -> {optimization_result['initial_radius']}")
        
        return optimization_result
    
    def run_comprehensive_test(self):
        """
        Run a comprehensive test of all system functions.
        
        Returns:
            Dictionary containing comprehensive test results
        """
        print("Starting comprehensive test of FastRoLSH system")
        
        # Check server health
        try:
            health = self.health_check()
            print(f"Server status: {health}")
        except:
            print("Server unavailable. Make sure the server is running.")
            return
        
        # Generate unique dataset names
        timestamp = int(time.time())
        
        # Test nearest neighbor search
        nn_results = self.test_nearest_neighbors(f"test_nn_dataset_{timestamp}", 
                                                n_samples=1000, n_features=10)
        
        # Test clustering
        clustering_results = self.test_clustering(f"test_clustering_dataset_{timestamp}", 
                                                n_samples=1000, n_features=2, n_clusters=3)
        
        # Test classification
        classification_results = self.test_classification(f"test_classification_dataset_{timestamp}", 
                                                        n_samples=1000, n_features=10, n_classes=3)
        
        # Test regression
        regression_results = self.test_regression(f"test_regression_dataset_{timestamp}", 
                                                n_samples=1000, n_features=10)
        
        # Test different metrics
        metrics_results = self.test_different_metrics()
        
        # Test different parameters
        params_results = self.test_different_parameters()
        
        # Test parameter optimization
        optimization_results = self.test_parameter_optimization(f"test_optimization_dataset_{timestamp}",
                                                               n_samples=1000, n_features=10)
        
        # Results summary
        print("\n" + "="*50)
        print("TEST RESULTS SUMMARY")
        print("="*50)
        
        if "error" in nn_results:
            print(f"Nearest neighbors test: ERROR - {nn_results['error']}")
        else:
            print(f"Nearest neighbors test: {nn_results['found_neighbors_ratio']*100:.1f}% queries found neighbors")
        
        if "error" in clustering_results:
            print(f"Clustering test: ERROR - {clustering_results['error']}")
        else:
            print(f"Clustering test - ARI Full data: {clustering_results['ari_full']:.4f}")
            print(f"Clustering test - ARI Sampled data: {clustering_results['ari_sample']:.4f}")
            print(f"Clustering test - ARI All from sample: {clustering_results['ari_all_from_sample']:.4f}")
        
        if "error" in classification_results:
            print(f"Classification test: ERROR - {classification_results['error']}")
        else:
            print(f"Classification test - Accuracy Full data: {classification_results['accuracy_full']:.4f}")
            print(f"Classification test - Accuracy Sampled data: {classification_results['accuracy_sample']:.4f}")
        
        if "error" in regression_results:
            print(f"Regression test: ERROR - {regression_results['error']}")
        else:
            print(f"Regression test - MSE Full data: {regression_results['mse_full']:.4f}")
            print(f"Regression test - MSE Sampled data: {regression_results['mse_sample']:.4f}")
        
        # Metrics test results
        print("\n--- Distance metrics test results ---")
        for metric, result in metrics_results.items():
            if "error" in result:
                print(f"Metric {metric}: ERROR - {result['error']}")
            else:
                print(f"Metric {metric}: {result['found_neighbors_ratio']*100:.1f}% queries found neighbors")
        
        # Parameters test results
        print("\n--- LSH parameters test results ---")
        for params, result in params_results.items():
            if "error" in result:
                print(f"Parameters {params}: ERROR - {result['error']}")
            else:
                print(f"Parameters {params}: {result['found_neighbors_ratio']*100:.1f}% queries found neighbors")
        
        # Parameter optimization results
        print("\n--- Parameter optimization results ---")
        if "error" in optimization_results:
            print(f"Parameter optimization: ERROR - {optimization_results['error']}")
        else:
            print(f"Parameter optimization: SUCCESS - w={optimization_results.get('w')}, initial_radius={optimization_results.get('initial_radius')}")
        
        return {
            "nearest_neighbors": nn_results,
            "clustering": clustering_results,
            "classification": classification_results,
            "regression": regression_results,
            "metrics": metrics_results,
            "parameters": params_results,
            "optimization": optimization_results
        }

# Example usage
if __name__ == "__main__":
    client = FastRoLSHTestClient()
    
    # Run comprehensive testing
    results = client.run_comprehensive_test()
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Testing completed. Results saved to test_results.json")