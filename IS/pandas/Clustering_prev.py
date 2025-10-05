import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import DBSCAN as SklearnDBSCAN, KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import time
import sys
import os

# Добавляем пути к нашим классам
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MonsterDBSCAN import DBSCAN as MonsterDBSCAN
from MonsterKmeans import GradientKMeans

def generate_datasets():
    """Генерация синтетических датасетов для тестирования"""
    datasets = {}
    
    # Blobs dataset (простой случай)
    X_blobs, y_blobs = make_blobs(n_samples=1000, centers=3, cluster_std=0.8, random_state=42)
    datasets['blobs'] = (X_blobs, y_blobs)
    
    # Moons dataset (нелинейный случай)
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.1, random_state=42)
    datasets['moons'] = (X_moons, y_moons)
    
    # Circles dataset (концентрические круги)
    X_circles, y_circles = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
    datasets['circles'] = (X_circles, y_circles)
    
    # Анизотропные данные
    X_aniso, y_aniso = make_blobs(n_samples=1000, centers=3, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    datasets['aniso'] = (X_aniso, y_aniso)
    
    return datasets

def evaluate_clustering(y_true, y_pred, X_data=None):
    """Оценка качества кластеризации"""
    metrics = {}
    
    # ARI (Adjusted Rand Index)
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        metrics['ari'] = adjusted_rand_score(y_true, y_pred)
    else:
        metrics['ari'] = 0.0
    
    # Homogeneity
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        metrics['homogeneity'] = homogeneity_score(y_true, y_pred)
    else:
        metrics['homogeneity'] = 0.0
    
    # Silhouette Score (только если есть более 1 кластера)
    if X_data is not None and len(np.unique(y_pred)) > 1:
        try:
            metrics['silhouette'] = silhouette_score(X_data, y_pred)
        except:
            metrics['silhouette'] = -1.0
    else:
        metrics['silhouette'] = -1.0
    
    # Количество найденных кластеров (исключая шум для DBSCAN)
    n_clusters = len(np.unique(y_pred))
    if 0 in y_pred:  # Если есть шум (метка 0)
        n_clusters -= 1
    metrics['n_clusters'] = n_clusters
    
    return metrics

def compare_dbscan(datasets):
    """Сравнение DBSCAN реализаций"""
    results = {}
    
    for name, (X, y_true) in datasets.items():
        print(f"\n=== Сравнение DBSCAN на датасете {name} ===")
        
        # Нормализация данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_torch = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Параметры DBSCAN
        eps = 0.3
        min_samples = 5
        
        # Sklearn DBSCAN
        start_time = time.time()
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
        y_sklearn = sklearn_dbscan.fit_predict(X_scaled)
        sklearn_time = time.time() - start_time
        
        sklearn_metrics = evaluate_clustering(y_true, y_sklearn, X_scaled)
        sklearn_metrics['time'] = sklearn_time
        
        # Monster DBSCAN
        start_time = time.time()
        monster_dbscan = MonsterDBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        y_monster = monster_dbscan.fit_predict(X_torch).cpu().numpy()
        monster_time = time.time() - start_time
        
        monster_metrics = evaluate_clustering(y_true, y_monster, X_scaled)
        monster_metrics['time'] = monster_time
        
        results[name] = {
            'sklearn': sklearn_metrics,
            'monster': monster_metrics,
            'data': (X, y_true, y_sklearn, y_monster)
        }
        
        # Вывод результатов
        print(f"Sklearn DBSCAN - ARI: {sklearn_metrics['ari']:.3f}, "
              f"Homogeneity: {sklearn_metrics['homogeneity']:.3f}, "
              f"Clusters: {sklearn_metrics['n_clusters']}, Time: {sklearn_time:.3f}s")
        
        print(f"Monster DBSCAN - ARI: {monster_metrics['ari']:.3f}, "
              f"Homogeneity: {monster_metrics['homogeneity']:.3f}, "
              f"Clusters: {monster_metrics['n_clusters']}, Time: {monster_time:.3f}s")
    
    return results

def compare_kmeans(datasets):
    """Сравнение K-means реализаций"""
    results = {}
    
    for name, (X, y_true) in datasets.items():
        print(f"\n=== Сравнение K-means на датасете {name} ===")
        
        # Нормализация данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_torch = torch.tensor(X_scaled, dtype=torch.float32)
        
        n_clusters = len(np.unique(y_true))
        
        # Sklearn K-means
        start_time = time.time()
        sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_sklearn = sklearn_kmeans.fit_predict(X_scaled)
        sklearn_time = time.time() - start_time
        
        sklearn_metrics = evaluate_clustering(y_true, y_sklearn, X_scaled)
        sklearn_metrics['time'] = sklearn_time
        sklearn_metrics['inertia'] = sklearn_kmeans.inertia_
        
        # Monster K-means
        start_time = time.time()
        monster_kmeans = GradientKMeans(
            n_clusters=n_clusters, 
            n_features=X_scaled.shape[1],
            distance_metric='euclidean',
            loss_function='standard',
            max_iters=100,
            verbose=False
        )
        y_monster = monster_kmeans.fit_predict(X_torch).cpu().numpy()
        monster_time = time.time() - start_time
        
        monster_metrics = evaluate_clustering(y_true, y_monster, X_scaled)
        monster_metrics['time'] = monster_time
        monster_metrics['inertia'] = monster_kmeans.score(X_torch)
        
        results[name] = {
            'sklearn': sklearn_metrics,
            'monster': monster_metrics,
            'data': (X, y_true, y_sklearn, y_monster)
        }
        
        # Вывод результатов
        print(f"Sklearn K-means - ARI: {sklearn_metrics['ari']:.3f}, "
              f"Homogeneity: {sklearn_metrics['homogeneity']:.3f}, "
              f"Inertia: {sklearn_metrics['inertia']:.2f}, Time: {sklearn_time:.3f}s")
        
        print(f"Monster K-means - ARI: {monster_metrics['ari']:.3f}, "
              f"Homogeneity: {monster_metrics['homogeneity']:.3f}, "
              f"Inertia: {monster_metrics['inertia']:.2f}, Time: {monster_time:.3f}s")
    
    return results

def plot_comparison(results_dbscan, results_kmeans, datasets):
    """Визуализация результатов сравнения"""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    dataset_names = list(datasets.keys())
    
    for i, name in enumerate(dataset_names):
        X, y_true = datasets[name]
        
        # Исходные данные
        axes[i, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=10)
        axes[i, 0].set_title(f'{name}\n(Истинные метки)')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # DBSCAN результаты
        if name in results_dbscan:
            _, _, y_sklearn_dbscan, y_monster_dbscan = results_dbscan[name]['data']
            
            axes[i, 1].scatter(X[:, 0], X[:, 1], c=y_sklearn_dbscan, cmap='viridis', s=10)
            axes[i, 1].set_title(f'Sklearn DBSCAN\n(ARI: {results_dbscan[name]["sklearn"]["ari"]:.3f})')
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            
            axes[i, 2].scatter(X[:, 0], X[:, 1], c=y_monster_dbscan, cmap='viridis', s=10)
            axes[i, 2].set_title(f'Monster DBSCAN\n(ARI: {results_dbscan[name]["monster"]["ari"]:.3f})')
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
        
        # K-means результаты
        if name in results_kmeans:
            _, _, y_sklearn_kmeans, y_monster_kmeans = results_kmeans[name]['data']
            
            axes[i, 3].scatter(X[:, 0], X[:, 1], c=y_sklearn_kmeans, cmap='viridis', s=10)
            axes[i, 3].set_title(f'Sklearn K-means\n(ARI: {results_kmeans[name]["sklearn"]["ari"]:.3f})')
            axes[i, 3].set_xticks([])
            axes[i, 3].set_yticks([])
            
            axes[i, 4].scatter(X[:, 0], X[:, 1], c=y_monster_kmeans, cmap='viridis', s=10)
            axes[i, 4].set_title(f'Monster K-means\n(ARI: {results_kmeans[name]["monster"]["ari"]:.3f})')
            axes[i, 4].set_xticks([])
            axes[i, 4].set_yticks([])
    
    # Убираем лишние subplots
    for i in range(len(dataset_names), 4):
        for j in range(5):
            axes[i, j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def print_summary_table(results_dbscan, results_kmeans):
    """Вывод сводной таблицы результатов"""
    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)
    
    # DBSCAN результаты
    print("\nDBSCAN Сравнение:")
    print("Dataset           | ARI (Sklearn) | ARI (Monster) | Time (Sklearn) | Time (Monster)")
    print("-" * 80)
    
    for name in results_dbscan.keys():
        sklearn = results_dbscan[name]['sklearn']
        monster = results_dbscan[name]['monster']
        print(f"{name:15} | {sklearn['ari']:12.3f} | {monster['ari']:12.3f} | "
              f"{sklearn['time']:13.3f} | {monster['time']:12.3f}")
    
    # K-means результаты
    print("\nK-means Сравнение:")
    print("Dataset           | ARI (Sklearn) | ARI (Monster) | Inertia (Sklearn) | Inertia (Monster)")
    print("-" * 80)
    
    for name in results_kmeans.keys():
        sklearn = results_kmeans[name]['sklearn']
        monster = results_kmeans[name]['monster']
        print(f"{name:15} | {sklearn['ari']:12.3f} | {monster['ari']:12.3f} | "
              f"{sklearn['inertia']:15.2f} | {monster['inertia']:14.2f}")

def test_incremental_learning():
    """Тестирование инкрементного обучения"""
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ ИНКРЕМЕНТАЛЬНОГО ОБУЧЕНИЯ")
    print("="*80)
    
    # Генерируем большой датасет
    X_large, y_large = make_blobs(n_samples=5000, centers=4, cluster_std=1.2, random_state=42)
    X_torch = torch.tensor(X_large, dtype=torch.float32)
    
    # Стандартное обучение
    start_time = time.time()
    kmeans_standard = GradientKMeans(n_clusters=4, n_features=2, max_iters=50, verbose=False)
    y_standard = kmeans_standard.fit_predict(X_torch)
    standard_time = time.time() - start_time
    
    # Инкрементное обучение
    start_time = time.time()
    kmeans_incremental = GradientKMeans(n_clusters=4, n_features=2, max_iters=10, verbose=False)
    
    batch_size = 1000
    for i in range(0, len(X_torch), batch_size):
        batch = X_torch[i:i+batch_size]
        kmeans_incremental.fit_incremental(batch, n_epochs=5)
    
    y_incremental = kmeans_incremental.predict(X_torch)
    incremental_time = time.time() - start_time
    
    # Оценка качества
    ari_standard = adjusted_rand_score(y_large, y_standard.cpu().numpy())
    ari_incremental = adjusted_rand_score(y_large, y_incremental.cpu().numpy())
    
    print(f"Стандартное обучение - ARI: {ari_standard:.3f}, Время: {standard_time:.3f}s")
    print(f"Инкрементное обучение - ARI: {ari_incremental:.3f}, Время: {incremental_time:.3f}s")
    print(f"Ускорение: {standard_time/incremental_time:.2f}x")

def main():
    """Основная функция сравнения"""
    print("Генерация синтетических датасетов...")
    datasets = generate_datasets()
    
    print("Сравнение DBSCAN...")
    results_dbscan = compare_dbscan(datasets)
    
    print("\nСравнение K-means...")
    results_kmeans = compare_kmeans(datasets)
    
    # Визуализация
    print("\nСоздание визуализации...")
    plot_comparison(results_dbscan, results_kmeans, datasets)
    
    # Сводная таблица
    print_summary_table(results_dbscan, results_kmeans)
    
    # Тестирование инкрементного обучения
    test_incremental_learning()

if __name__ == "__main__":
    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    main()