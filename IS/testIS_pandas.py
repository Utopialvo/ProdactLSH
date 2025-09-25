import numpy as np
from sklearn.datasets import make_blobs, make_regression, make_classification
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import adjusted_rand_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class SamplingComparison:
    """Сравнение качества моделей при использовании семплинга LSH и PQ"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def generate_clustering_data(self, n_samples=30000, n_features=100, n_clusters=3):
        """Генерация данных для кластеризации"""
        print("Генерация данных для кластеризации...")
        X, y_true = make_blobs(
            n_samples=n_samples, 
            n_features=n_features, 
            centers=n_clusters,
            random_state=self.random_state,
            cluster_std=25.0
        )
        X = self.scaler.fit_transform(X)
        return torch.tensor(X, dtype=torch.float32), y_true
    
    def generate_regression_data(self, n_samples=10000, n_features=100):
        """Генерация данных для регрессии"""
        print("Генерация данных для регрессии...")
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.15,
            random_state=self.random_state
        )
        X = self.scaler.fit_transform(X)
        # Преобразуем в бинарную задачу для упрощения
        y_binary = ((y > (y.mean() + 2*y.std())).any() or (y < (y.mean() - 2*y.std())).any()).astype(int)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), y_binary
    
    def generate_classification_data(self, n_samples=10000, n_features=100, n_classes=3):
        """Генерация данных для классификации"""
        print("Генерация данных для классификации...")
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=self.random_state
        )
        X = self.scaler.fit_transform(X)
        return torch.tensor(X, dtype=torch.float32), y
    
    def evaluate_clustering(self, X, y_true, sample_size=1000):
        """Сравнение кластеризации с семплингом и без"""
        print("\n=== СРАВНЕНИЕ КЛАСТЕРИЗАЦИИ ===")
        
        # 1. KMeans на полных данных
        print("1. Обучение KMeans на полных данных...")
        kmeans_full = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
        y_pred_full = kmeans_full.fit_predict(X.numpy())
        ari_full = adjusted_rand_score(y_true, y_pred_full)
        print(f"   ARI (полные данные): {ari_full:.4f}")
        
        results = {'full': ari_full}
        
        # 2. Семплинг LSH и KMeans
        print("2. Семплинг LSH + KMeans...")
        def cluster_target_function(x):
            # Функция для оценки важности - расстояние до ближайшего центроида
            with torch.no_grad():
                distances = torch.cdist(x, torch.tensor(kmeans_full.cluster_centers_, dtype=torch.float32))
                min_distances = distances.min(dim=1)[0]
            return min_distances
        
        # LSH семплинг
        lsh_sampler = LSHWithSampling(
            input_dim=X.shape[1],
            num_tables=10,
            hash_size=12,
            distance_type='cosine'
        )
        lsh_sampler.add(X)
        
        lsh_indices, lsh_weights = lsh_sampler.importance_sampling(
            cluster_target_function,
            sample_size=sample_size,
            strategy='variance_based'
        )
        
        if len(lsh_indices) > 0:
            X_lsh_sample = X[lsh_indices].numpy()
            kmeans_lsh = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
            kmeans_lsh.fit(X_lsh_sample)
            y_pred_lsh = kmeans_lsh.predict(X.numpy())
            ari_lsh = adjusted_rand_score(y_true, y_pred_lsh)
            print(f"   ARI (LSH семплинг): {ari_lsh:.4f}")
            results['lsh'] = ari_lsh
        else:
            print("   LSH семплинг не удался")
            results['lsh'] = 0.0
        
        # 3. Семплинг PQ и KMeans
        print("3. Семплинг PQ + KMeans...")
        pq_sampler = PQWithSampling(
            input_dim=X.shape[1],
            num_subspaces=10,
            num_centroids=256
        )
        pq_sampler.train(X)
        
        pq_indices, pq_weights = pq_sampler.importance_sampling(
            cluster_target_function,
            sample_size=sample_size,
            strategy='residual_variance'
        )
        
        if len(pq_indices) > 0:
            X_pq_sample = X[pq_indices].numpy()
            kmeans_pq = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
            kmeans_pq.fit(X_pq_sample)
            y_pred_pq = kmeans_pq.predict(X.numpy())
            ari_pq = adjusted_rand_score(y_true, y_pred_pq)
            print(f"   ARI (PQ семплинг): {ari_pq:.4f}")
            results['pq'] = ari_pq
        else:
            print("   PQ семплинг не удался")
            results['pq'] = 0.0
        
        # 4. Случайный семплинг (базлайн)
        print("4. Случайный семплинг + KMeans...")
        random_indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_random_sample = X[random_indices].numpy()
        kmeans_random = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
        kmeans_random.fit(X_random_sample)
        y_pred_random = kmeans_random.predict(X.numpy())
        ari_random = adjusted_rand_score(y_true, y_pred_random)
        print(f"   ARI (случайный семплинг): {ari_random:.4f}")
        results['random'] = ari_random
        
        return results
    
    def evaluate_regression(self, X, y, y_binary, sample_size=1000):
        """Сравнение регрессии с семплингом и без"""
        print("\n=== СРАВНЕНИЕ РЕГРЕССИИ ===")
        
        # 1. Линейная регрессия на полных данных
        print("1. Обучение LinearRegression на полных данных...")
        reg_full = LinearRegression()
        reg_full.fit(X.numpy(), y.numpy())
        y_pred_full = reg_full.predict(X.numpy())
        mse_full = mean_squared_error(y.numpy(), y_pred_full)
        print(f"   MSE (полные данные): {mse_full:.4f}")
        
        results = {'full': mse_full}
        
        # 2. Семплинг LSH и регрессия
        print("2. Семплинг LSH + LinearRegression...")
        def reg_target_function(x):
            # Функция для оценки важности - абсолютное значение целевой переменной
            with torch.no_grad():
                # Используем бинарные метки для важности
                importance = torch.abs(torch.randn(x.shape[0]))  # Заглушка
            return importance
        
        lsh_sampler = LSHWithSampling(
            input_dim=X.shape[1],
            num_tables=10,
            hash_size=12,
            distance_type='cosine'
        )
        lsh_sampler.add(X)
        
        lsh_indices, lsh_weights = lsh_sampler.importance_sampling(
            reg_target_function,
            sample_size=sample_size,
            strategy='variance_based'
        )
        
        if len(lsh_indices) > 0:
            X_lsh_sample = X[lsh_indices].numpy()
            y_lsh_sample = y[lsh_indices].numpy()
            
            reg_lsh = LinearRegression()
            reg_lsh.fit(X_lsh_sample, y_lsh_sample)
            y_pred_lsh = reg_lsh.predict(X.numpy())
            mse_lsh = mean_squared_error(y.numpy(), y_pred_lsh)
            print(f"   MSE (LSH семплинг): {mse_lsh:.4f}")
            results['lsh'] = mse_lsh
        else:
            print("   LSH семплинг не удался")
            results['lsh'] = float('inf')
        
        # 3. Семплинг PQ и регрессия
        print("3. Семплинг PQ + LinearRegression...")
        pq_sampler = PQWithSampling(
            input_dim=X.shape[1],
            num_subspaces=10,
            num_centroids=256
        )
        pq_sampler.train(X)
        
        pq_indices, pq_weights = pq_sampler.importance_sampling(
            reg_target_function,
            sample_size=sample_size,
            strategy='residual_variance'
        )
        
        if len(pq_indices) > 0:
            X_pq_sample = X[pq_indices].numpy()
            y_pq_sample = y[pq_indices].numpy()
            
            reg_pq = LinearRegression()
            reg_pq.fit(X_pq_sample, y_pq_sample)
            y_pred_pq = reg_pq.predict(X.numpy())
            mse_pq = mean_squared_error(y.numpy(), y_pred_pq)
            print(f"   MSE (PQ семплинг): {mse_pq:.4f}")
            results['pq'] = mse_pq
        else:
            print("   PQ семплинг не удался")
            results['pq'] = float('inf')
        
        # 4. Случайный семплинг
        print("4. Случайный семплинг + LinearRegression...")
        random_indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_random_sample = X[random_indices].numpy()
        y_random_sample = y[random_indices].numpy()
        
        reg_random = LinearRegression()
        reg_random.fit(X_random_sample, y_random_sample)
        y_pred_random = reg_random.predict(X.numpy())
        mse_random = mean_squared_error(y.numpy(), y_pred_random)
        print(f"   MSE (случайный семплинг): {mse_random:.4f}")
        results['random'] = mse_random
        
        return results
    
    def evaluate_classification(self, X, y, sample_size=1000):
        """Сравнение классификации с семплингом и без"""
        print("\n=== СРАВНЕНИЕ КЛАССИФИКАЦИИ ===")
        
        # 1. Логистическая регрессия на полных данных
        print("1. Обучение LogisticRegression на полных данных...")
        clf_full = LogisticRegression(random_state=self.random_state, max_iter=1000)
        clf_full.fit(X.numpy(), y)
        y_pred_full = clf_full.predict(X.numpy())
        acc_full = accuracy_score(y, y_pred_full)
        print(f"   Accuracy (полные данные): {acc_full:.4f}")
        
        results = {'full': acc_full}
        
        # 2. Семплинг LSH и классификация
        print("2. Семплинг LSH + LogisticRegression...")
        def clf_target_function(x):
            # Функция для оценки важности - энтропия предсказаний
            with torch.no_grad():
                # Используем случайные важности как заглушку
                importance = torch.abs(torch.randn(x.shape[0]))
            return importance
        
        lsh_sampler = LSHWithSampling(
            input_dim=X.shape[1],
            num_tables=10,
            hash_size=12,
            distance_type='cosine'
        )
        lsh_sampler.add(X)
        
        lsh_indices, lsh_weights = lsh_sampler.importance_sampling(
            clf_target_function,
            sample_size=sample_size,
            strategy='variance_based'
        )
        
        if len(lsh_indices) > 0:
            X_lsh_sample = X[lsh_indices].numpy()
            y_lsh_sample = y[lsh_indices]
            
            clf_lsh = LogisticRegression(random_state=self.random_state, max_iter=1000)
            clf_lsh.fit(X_lsh_sample, y_lsh_sample)
            y_pred_lsh = clf_lsh.predict(X.numpy())
            acc_lsh = accuracy_score(y, y_pred_lsh)
            print(f"   Accuracy (LSH семплинг): {acc_lsh:.4f}")
            results['lsh'] = acc_lsh
        else:
            print("   LSH семплинг не удался")
            results['lsh'] = 0.0
        
        # 3. Семплинг PQ и классификация
        print("3. Семплинг PQ + LogisticRegression...")
        pq_sampler = PQWithSampling(
            input_dim=X.shape[1],
            num_subspaces=10,
            num_centroids=256
        )
        pq_sampler.train(X)
        
        pq_indices, pq_weights = pq_sampler.importance_sampling(
            clf_target_function,
            sample_size=sample_size,
            strategy='residual_variance'
        )
        
        if len(pq_indices) > 0:
            X_pq_sample = X[pq_indices].numpy()
            y_pq_sample = y[pq_indices]
            
            clf_pq = LogisticRegression(random_state=self.random_state, max_iter=1000)
            clf_pq.fit(X_pq_sample, y_pq_sample)
            y_pred_pq = clf_pq.predict(X.numpy())
            acc_pq = accuracy_score(y, y_pred_pq)
            print(f"   Accuracy (PQ семплинг): {acc_pq:.4f}")
            results['pq'] = acc_pq
        else:
            print("   PQ семплинг не удался")
            results['pq'] = 0.0
        
        # 4. Случайный семплинг
        print("4. Случайный семплинг + LogisticRegression...")
        random_indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_random_sample = X[random_indices].numpy()
        y_random_sample = y[random_indices]
        
        clf_random = LogisticRegression(random_state=self.random_state, max_iter=1000)
        clf_random.fit(X_random_sample, y_random_sample)
        y_pred_random = clf_random.predict(X.numpy())
        acc_random = accuracy_score(y, y_pred_random)
        print(f"   Accuracy (случайный семплинг): {acc_random:.4f}")
        results['random'] = acc_random
        
        return results
    
    def run_comprehensive_comparison(self, n_samples=5000, sample_size=1000):
        """Полное сравнение всех методов на всех задачах"""
        print("=" * 60)
        print("КОМПЛЕКСНОЕ СРАВНЕНИЕ МЕТОДОВ СЕМПЛИНГА")
        print("=" * 60)
        
        all_results = {}
        
        # 1. Кластеризация
        X_cluster, y_cluster = self.generate_clustering_data(n_samples=n_samples)
        cluster_results = self.evaluate_clustering(X_cluster, y_cluster, sample_size)
        all_results['clustering'] = cluster_results
        
        # 2. Регрессия
        X_reg, y_reg, y_reg_binary = self.generate_regression_data(n_samples=n_samples)
        reg_results = self.evaluate_regression(X_reg, y_reg, y_reg_binary, sample_size)
        all_results['regression'] = reg_results
        
        # 3. Классификация
        X_clf, y_clf = self.generate_classification_data(n_samples=n_samples)
        clf_results = self.evaluate_classification(X_clf, y_clf, sample_size)
        all_results['classification'] = clf_results
        
        # Визуализация результатов
        self._plot_results(all_results)
        
        return all_results
    
    def _plot_results(self, results):
        """Визуализация результатов сравнения"""
        tasks = list(results.keys())
        methods = ['full', 'lsh', 'pq', 'random']
        colors = ['blue', 'green', 'red', 'orange']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, task in enumerate(tasks):
            values = [results[task][method] for method in methods]
            
            # Для регрессии меньшие значения лучше, для других - большие
            if task == 'regression':
                values = [-v for v in values]  # Инвертируем для визуализации
                ylabel = 'Negative MSE (больше → лучше)'
            else:
                ylabel = 'Score (больше → лучше)'
            
            bars = axes[i].bar(methods, values, color=colors, alpha=0.7)
            axes[i].set_title(f'{task.capitalize()}')
            axes[i].set_ylabel(ylabel)
            axes[i].set_xlabel('Метод')
            
            # Добавляем значения на столбцы
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                            f'{abs(results[task][methods[bars.index(bar)]]):.3f}',
                            ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('sampling_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Таблица сравнения
        print("\n" + "=" * 60)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 60)
        
        for task in tasks:
            print(f"\n{task.upper()}:")
            for method in methods:
                score = results[task][method]
                if task == 'regression':
                    print(f"  {method}: MSE = {score:.4f}")
                else:
                    print(f"  {method}: Score = {score:.4f}")

# Запуск комплексного сравнения
if __name__ == "__main__":
    comparator = SamplingComparison(random_state=42)
    results = comparator.run_comprehensive_comparison(n_samples=5000, sample_size=1000)
