# full_demo.py
"""
Полная демонстрация всех функций библиотеки с примерами использования.

Содержит:
- Демонстрацию всех возможностей LSH и PQ
- Примеры Importance Sampling
- Визуализацию и оценку качества
- Интеграционные тесты с реальными данными
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.decomposition import KernelPCA
from torch.utils.data import WeightedRandomSampler, DataLoader
from typing import List, Tuple, Dict, Union, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

from intelligent_caching import MemoryEfficientTensorStorage
from lsh_sampling import LSH, LSHWithSampling
from pq_sampling import ProductQuantizer, PQWithSampling


class SamplingEvaluator:
    """
    Утилиты для оценки качества семплинга.
    
    Содержит методы для оценки математического ожидания с помощью Importance Sampling
    и сравнения распределений исходных данных и выборки.
    """
    
    @staticmethod
    def estimate_expectation(samples: torch.Tensor, weights: List[float], 
                           target_function: Callable) -> float:
        """
        Оценка матожидания с помощью Importance Sampling.
        
        Формула из IS_LSH.pdf: E_p[f] ≈ (Σ w_i * f(x_i)) / (Σ w_i)
        
        Args:
            samples: Выборочные данные
            weights: Веса для importance sampling
            target_function: Целевая функция f(x)
            
        Returns:
            Оценка математического ожидания
        """
        with torch.no_grad():
            target_values = target_function(samples)
            weighted_sum = (target_values * torch.tensor(weights, device=samples.device)).sum()
            weight_sum = torch.tensor(weights, device=samples.device).sum()
            
            if weight_sum == 0:
                return 0.0
                
            return (weighted_sum / weight_sum).item()
    
    @staticmethod
    def compare_distributions(original_data: torch.Tensor, sampled_data: torch.Tensor,
                            original_weights: Optional[List[float]] = None,
                            sampled_weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Сравнение распределений исходных данных и выборки.
        
        Метрики сравнения:
        - Разница средних: ||μ_original - μ_sampled||
        - Разница ковариационных матриц: ||Σ_original - Σ_sampled||_F
        
        Args:
            original_data: Исходные данные
            sampled_data: Выборочные данные
            original_weights: Веса для исходных данных (опционально)
            sampled_weights: Веса для выборочных данных (опционально)
            
        Returns:
            Словарь с метриками сравнения
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
            'covariance_difference': cov_diff
        }


def demo_all_features():
    """Демонстрация всех функций библиотеки"""
    torch.manual_seed(42)
    
    # Генерация тестовых данных
    data = torch.randn(1000, 128)
    query = torch.randn(128)
    
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ LSH С РАЗНЫМИ МЕТРИКАМИ И ПРОЕКЦИЯМИ")
    print("=" * 60)
    
    # Тестируем разные метрики и проекции
    configs = [
        ('euclidean', 'random', "Евклидово расстояние, случайные проекции"),
        ('cosine', 'orthogonal', "Косинусное расстояние, ортогональные проекции"),
        ('manhattan', 'random', "Манхеттенское расстояние, случайные проекции"),
    ]
    
    for distance_type, projection_type, description in configs:
        print(f"\n--- {description} ---")
        
        lsh = LSH(
            input_dim=128, 
            num_tables=5, 
            hash_size=8,
            distance_type=distance_type,
            projection_type=projection_type,
            cache_enabled=True,
            max_memory_points=500
        )
        lsh.add(data)
        
        indices, distances = lsh.query(query, k=5)
        print(f"Найдены соседи: {indices[:5]}")
        print(f"Расстояния: {distances[:5]}")
    
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ IMPORTANCE SAMPLING")
    print("=" * 60)
    
    # Определяем target function
    def target_function(x):
        return torch.norm(x, dim=1)
    
    # Демонстрация LSH с семплингом
    print("\n--- LSH с Importance Sampling ---")
    lsh_sampler = LSHWithSampling(
        input_dim=128, 
        num_tables=5, 
        hash_size=8,
        distance_type='cosine',
        projection_type='orthogonal',
        cache_enabled=True,
        max_memory_points=500
    )
    lsh_sampler.add(data)
    
    indices, weights = lsh_sampler.importance_sampling(
        target_function, 
        sample_size=100,
        strategy='variance_based'
    )
    
    print(f"Размер выборки: {len(indices)}")
    print(f"Пример весов: {weights[:5]}")
    
    # Оценка матожидания
    if len(indices) > 0:
        sampled_data = []
        for idx in indices:
            sampled_data.append(lsh_sampler._get_data_point(idx))
        sampled_data = torch.stack(sampled_data)
        expectation = SamplingEvaluator.estimate_expectation(sampled_data, weights, target_function)
        print(f"Оценка матожидания: {expectation:.4f}")
    
    # Демонстрация PQ с семплингом
    print("\n--- PQ с Importance Sampling ---")
    pq_sampler = PQWithSampling(
        input_dim=128, 
        num_subspaces=8, 
        num_centroids=256,
        cache_enabled=True,
        max_memory_points=500
    )
    pq_sampler.train(data)
    
    codes = pq_sampler.encode(data)
    print(f"Размер закодированных данных: {codes.shape}")
    
    indices, distances = pq_sampler.search(query, codes, k=5)
    print(f"Найдены соседи: {indices}")
    print(f"Расстояния: {distances}")
    
    # Проверка реконструкции
    reconstructed = pq_sampler.decode(codes[:5])
    reconstruction_error = torch.mean((data[:5] - reconstructed.cpu())**2)
    print(f"Ошибка реконструкции: {reconstruction_error:.6f}")
    
    # Демонстрация семплинга PQ
    indices, weights = pq_sampler.importance_sampling(
        target_function,
        sample_size=100,
        strategy='residual_variance'
    )
    
    print(f"Размер выборки PQ: {len(indices)}")
    if len(indices) > 0:
        print(f"Пример весов PQ: {weights[:5]}")
    
    # Сравнение распределений
    if len(indices) > 0:
        sampled_data_pq = []
        for idx in indices:
            sampled_data_pq.append(pq_sampler._get_data_point(idx))
        sampled_data_pq = torch.stack(sampled_data_pq)
        comparison = SamplingEvaluator.compare_distributions(data, sampled_data_pq)
        print(f"Сравнение распределений: {comparison}")
    
    # Очистка кэша
    for obj in [lsh_sampler, pq_sampler]:
        if hasattr(obj, 'storage'):
            obj.storage.cleanup()
    
    return lsh_sampler, pq_sampler


def load_data(parquet_path='encoded_blobs.parquet'):
    """Простая загрузка данных из parquet файла"""
    df = pd.read_parquet(parquet_path)
    
    # Разделяем на эмбеддинги и метки
    embedding_columns = [col for col in df.columns if col.startswith('emb_')]
    embeddings = df[embedding_columns].values
    labels = df['label'].values
    
    print(f"Загружено {len(embeddings)} векторов размерности {embeddings.shape[1]}")
    print(f"Количество классов: {len(np.unique(labels))}")
    
    return torch.tensor(embeddings, dtype=torch.float32), labels


def compress_with_pq(embeddings, num_subspaces=10, num_centroids=256):
    """Сжатие данных с помощью Product Quantization"""
    print(f"=== PQ СЖАТИЕ ===")
    
    pq = ProductQuantizer(
        input_dim=embeddings.shape[1],
        num_subspaces=num_subspaces,
        num_centroids=num_centroids
    )
    
    print("Обучение PQ...")
    pq.train(embeddings)
    
    # Кодирование и декодирование
    pq_codes = pq.encode(embeddings)
    compressed_embeddings = pq.decode(pq_codes)
    
    # Расчет коэффициента сжатия
    original_size = embeddings.shape[1] * 4  # float32 = 4 байта
    compressed_size = (num_subspaces * np.log2(num_centroids) / 8)
    compression_ratio = original_size / compressed_size
    
    print(f"Коэффициент сжатия: {compression_ratio:.2f}x")
    
    return compressed_embeddings.cpu().numpy(), compression_ratio


def compress_with_lsh(embeddings, num_tables=10, hash_size=16):
    """Сжатие данных с помощью LSH"""
    print(f"=== LSH СЖАТИЕ ===")
    
    lsh = LSH(
        input_dim=embeddings.shape[1],
        num_tables=num_tables,
        hash_size=hash_size,
        distance_type='cosine',
        projection_type='orthogonal'
    )
    
    print("Добавление данных в LSH...")
    lsh.add(embeddings)
    
    # Получение LSH представлений
    lsh_codes = []
    for i in range(len(embeddings)):
        vector = embeddings[i:i+1]
        hashes = lsh.compute_hashes(vector)
        code = hashes[0].flatten().cpu().numpy()
        lsh_codes.append(code)
    
    lsh_codes = np.array(lsh_codes)
    
    # Расчет коэффициента сжатия
    original_size = embeddings.shape[1] * 4
    compressed_size = (num_tables * hash_size) / 8  # бинарные коды
    compression_ratio = original_size / compressed_size
    
    print(f"Коэффициент сжатия: {compression_ratio:.2f}x")
    print(f"Размерность LSH: {lsh_codes.shape[1]}")
    
    return lsh_codes, compression_ratio


def evaluate_clustering(original_data, compressed_data, true_labels, method_name):
    """Оценка качества кластеризации"""
    print(f"=== ОЦЕНКА {method_name} ===")
    
    # Кластеризация на исходных данных
    clustering_orig = AgglomerativeClustering(n_clusters=64, metric='euclidean', linkage='ward')
    labels_orig = clustering_orig.fit_predict(original_data)
    ari_orig = adjusted_rand_score(true_labels, labels_orig)
    
    # Кластеризация на сжатых данных  
    clustering_comp = AgglomerativeClustering(n_clusters=64, metric='euclidean', linkage='ward')
    labels_comp = clustering_comp.fit_predict(compressed_data)
    ari_comp = adjusted_rand_score(true_labels, labels_comp)
    
    print(f"ARI исходные: {ari_orig:.4f}")
    print(f"ARI сжатые: {ari_comp:.4f}")
    print(f"Разница: {ari_comp - ari_orig:+.4f}")
    
    return ari_orig, ari_comp, labels_orig, labels_comp


def visualize_comparison(original_data, compressed_data, true_labels, 
                        labels_orig, labels_comp, method_name):
    """Визуализация сравнения кластеризации"""
    # Для исходных данных
    pca_orig = PCA(n_components=2)
    original_2d = pca_orig.fit_transform(original_data)
    
    # Для сжатых данных (разная размерность)
    pca_comp = PCA(n_components=2)
    compressed_2d = pca_comp.fit_transform(compressed_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Исходные данные
    scatter1 = axes[0, 0].scatter(original_2d[:, 0], original_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.6, s=10)
    axes[0, 0].set_title('Исходные данные (истинные метки)')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    scatter2 = axes[0, 1].scatter(original_2d[:, 0], original_2d[:, 1], c=labels_orig, cmap='tab10', alpha=0.6, s=10)
    axes[0, 1].set_title('Исходные данные (предсказанные)')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # Сжатые данные
    scatter3 = axes[1, 0].scatter(compressed_2d[:, 0], compressed_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.6, s=10)
    axes[1, 0].set_title(f'{method_name} данные (истинные метки)')
    plt.colorbar(scatter3, ax=axes[1, 0])
    
    scatter4 = axes[1, 1].scatter(compressed_2d[:, 0], compressed_2d[:, 1], c=labels_comp, cmap='tab10', alpha=0.6, s=10)
    axes[1, 1].set_title(f'{method_name} данные (предсказанные)')
    plt.colorbar(scatter4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f'{method_name.lower()}_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_pq():
    """Демонстрация работы с Product Quantization"""
    print("\n" + "="*50)
    print("DEMO: PRODUCT QUANTIZATION")
    print("="*50)
    
    # Загрузка данных
    embeddings, true_labels = load_data()
    
    # Сжатие с помощью PQ
    compressed_pq, compression_ratio = compress_with_pq(embeddings)
    
    # Оценка кластеризации
    ari_orig, ari_pq, labels_orig, labels_pq = evaluate_clustering(
        embeddings.numpy(), compressed_pq, true_labels, "PQ"
    )
    
    # Визуализация
    visualize_comparison(embeddings.numpy(), compressed_pq, true_labels, 
                        labels_orig, labels_pq, "PQ")
    
    return ari_orig, ari_pq, compression_ratio


def demo_lsh():
    """Демонстрация работы с LSH"""
    print("\n" + "="*50)
    print("DEMO: LOCALITY-SENSITIVE HASHING")  
    print("="*50)
    
    # Загрузка данных
    embeddings, true_labels = load_data()
    
    # Сжатие с помощью LSH
    compressed_lsh, compression_ratio = compress_with_lsh(embeddings)
    
    # Оценка кластеризации
    ari_orig, ari_lsh, labels_orig, labels_lsh = evaluate_clustering(
        embeddings.numpy(), compressed_lsh, true_labels, "LSH"
    )
    
    # Визуализация
    visualize_comparison(embeddings.numpy(), compressed_lsh, true_labels,
                        labels_orig, labels_lsh, "LSH")
    
    return ari_orig, ari_lsh, compression_ratio


def demo_sampling():
    """Демонстрация Importance Sampling"""
    print("\n" + "="*50)
    print("DEMO: IMPORTANCE SAMPLING")
    print("="*50)
    
    # Загрузка данных
    embeddings, true_labels = load_data()
    
    # Определяем target function
    def target_function(x):
        return torch.norm(x, dim=1) ** 2
    
    # Демонстрация LSH семплинга
    print("\n--- LSH Sampling ---")
    start_time = time.time()
    
    lsh_sampler = LSHWithSampling(
        input_dim=embeddings.shape[1],
        num_tables=8,
        hash_size=12,
        distance_type='cosine',
        cache_enabled=True,
        max_memory_points=20000  # Увеличиваем лимит памяти для производительности
    )
    lsh_sampler.add(embeddings)
    
    indices, weights = lsh_sampler.importance_sampling(
        target_function, 
        sample_size=200,
        strategy='proportional',  # Используем более быструю стратегию
        precompute_batch_size=500  # Батчи для предвычислений
    )
    
    lsh_time = time.time() - start_time
    print(f"LSH выборка: {len(indices)} точек за {lsh_time:.2f} сек")
    print(f"Средний вес: {np.mean(weights):.4f}")
    
    # Демонстрация PQ семплинга
    print("\n--- PQ Sampling ---")
    start_time = time.time()
    
    pq_sampler = PQWithSampling(
        input_dim=embeddings.shape[1],
        num_subspaces=8,
        num_centroids=256,
        cache_enabled=True,
        max_memory_points=20000
    )
    pq_sampler.train(embeddings)
    
    indices_pq, weights_pq = pq_sampler.importance_sampling(
        target_function,
        sample_size=200,
        strategy='proportional',  # Используем более быструю стратегию
        precompute_batch_size=500
    )
    
    pq_time = time.time() - start_time
    print(f"PQ выборка: {len(indices_pq)} точек за {pq_time:.2f} сек")
    print(f"Средний вес: {np.mean(weights_pq):.4f}")
    
    # Сравнение производительности
    print(f"\n--- СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ---")
    print(f"LSH: {lsh_time:.2f} сек")
    print(f"PQ:  {pq_time:.2f} сек")
    
    return lsh_sampler, pq_sampler

def create_data_blobs():
    print("Создание данных для теста encoded_blobs.parquet")
    X, y = make_blobs(
        n_samples=15000,
        n_features=2048,
        centers=64,
        cluster_std=0.15,
        center_box=(-640.0, 640.0),
        random_state=42
    )
    clustering_original = AgglomerativeClustering(
            n_clusters=64,
            metric='euclidean',
            linkage='ward'
        )
    predy = clustering_original.fit_predict(X)
    adjusted_rand_score(y, predy)
    transformer = KernelPCA(n_components=320, kernel='rbf')
    X = transformer.fit_transform(X)
    df = pd.DataFrame(X)
    df['label'] = y
    df.columns = [f'emb_{i}' for i in range(320)] + ['label']
    df.to_parquet('encoded_blobs.parquet', index=False)


def main():
    """Основная демонстрационная функция"""
    print("ДЕМОНСТРАЦИЯ МЕТОДОВ СЖАТИЯ ДАННЫХ И СЕМПЛИНГА")

    # Создание данных для теста encoded_blobs.parquet
    create_data_blobs()
    
    # Демонстрация всех функций
    demo_all_features()
    
    # Демонстрация PQ
    pq_results = demo_pq()
    
    # Демонстрация LSH  
    lsh_results = demo_lsh()
    
    # Демонстрация семплинга
    sampling_results = demo_sampling()
    
    # Сводка результатов
    print("\n" + "="*50)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("="*50)
    print(f"PQ:  ARI исходные={pq_results[0]:.4f}, ARI сжатые={pq_results[1]:.4f}, Сжатие={pq_results[2]:.1f}x")
    print(f"LSH: ARI исходные={lsh_results[0]:.4f}, ARI сжатые={lsh_results[1]:.4f}, Сжатие={lsh_results[2]:.1f}x")


if __name__ == "__main__":
    main()