import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch
from IS_PQ import PQWithSampling, SamplingEvaluator  # Предполагается, что у вас есть этот файл


# Параметры генерации данных
n_samples = 30000
n_features = 256
n_clusters = 4
random_state = 42


print("Генерация данных с помощью make_blobs...")
# Генерация изотропных Gaussian blobs:cite[1]
X, y_true = make_blobs(
    n_samples=n_samples,
    n_features=n_features,  # 256 измерений:cite[1]
    centers=n_clusters,
    cluster_std=[2.0, 4.0, 1.0, 3.0],  # Разные стандартные отклонения для реалистичности
    random_state=random_state,
    shuffle=True
)


print("Стандартизация данных...")
# Стандартизация для устойчивости кластеризации
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Преобразуем в тензор PyTorch
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)


# =============================================================================
# ВАЖНОСТНОЕ СЕМПЛИРОВАНИЕ С ИСПОЛЬЗОВАНИЕМ PQ
# =============================================================================


print("Инициализация и обучение PQ модели...")
pq_sampler = PQWithSampling(
    input_dim=n_features,
    num_subspaces=8,  # Разделяем 256D на 8 подпространств по 32D
    num_centroids=64,  # 64 центроида на подпространство
    device='auto'
)


# Обучаем PQ на всех данных
pq_sampler.train(X_tensor, max_iter=50, batch_size=512)


# Определяем целевую функцию для семплирования (норма вектора)
def target_function_norm(x):
    return torch.norm(x, dim=1)


print("Выполнение importance sampling...")
# Выполняем семплирование 300 элементов:cite[2]:cite[3]
sample_size = 2000
sampled_indices, sampling_weights = pq_sampler.importance_sampling(
    target_function=target_function_norm,
    sample_size=sample_size,
    strategy='residual_variance',  # Стратегия, основанная на дисперсии остатков
    temperature=2.8  # Контроль дисперсии выборки
)


print(f"Размер семпла: {len(sampled_indices)}")
print(f"Диапазон весов: min={min(sampling_weights):.4f}, max={max(sampling_weights):.4f}")

# Получаем статистику семплинга для диагностики
sampling_stats = pq_sampler.get_sampling_stats()
print(f"Использовано подпространств: {sampling_stats['num_subspaces']}")
print(f"Всего кластеров: {sampling_stats['total_clusters']}")


# Создаем семплированные данные
X_sampled = X_scaled[sampled_indices]
y_sampled = y_true[sampled_indices]
sampling_weights_array = np.array(sampling_weights)


# =============================================================================
# КЛАСТЕРИЗАЦИЯ И СРАВНЕНИЕ
# =============================================================================


print("\n" + "=" * 60)
print("СРАВНЕНИЕ КЛАСТЕРИЗАЦИИ НА ПОЛНЫХ И СЕМПЛИРОВАННЫХ ДАННЫХ")
print("=" * 60)


# Метод 1: KMeans на полных данных
print("Обучение KMeans на полных данных...")
kmeans_full = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
kmeans_full.fit(X_scaled)
labels_full = kmeans_full.labels_


# Метод 2: KMeans на семплированных данных с весами
print("Обучение KMeans на семплированных данных с весами...")
kmeans_sampled = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)


# Нормализуем веса для KMeans (важно: KMeans ожидает веса, пропорциональные важности точек)
sample_weights_normalized = sampling_weights_array / sampling_weights_array.sum() * len(sampling_weights_array)
kmeans_sampled.fit(X_sampled, sample_weight=sample_weights_normalized)


# Предсказание на всех данных обеими моделями
labels_full_predict = kmeans_full.predict(X_scaled)
labels_sampled_predict = kmeans_sampled.predict(X_scaled)


# =============================================================================
# ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ
# =============================================================================


def evaluate_clustering(true_labels, pred_labels, X_data, method_name):
    """Вычисление метрик качества кластеризации"""
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    silhouette = silhouette_score(X_data, pred_labels)
    
    print(f"\n{method_name}:")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    
    return {'ARI': ari, 'NMI': nmi, 'Silhouette': silhouette}


# Оценка всех методов
metrics_full = evaluate_clustering(y_true, labels_full_predict, X_scaled, "KMeans на полных данных")
metrics_sampled = evaluate_clustering(y_true, labels_sampled_predict, X_scaled, "KMeans на семплированных данных")


# =============================================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================


print("\nВизуализация результатов...")


# Понижаем размерность для визуализации с помощью PCA
from sklearn.decomposition import PCA


pca = PCA(n_components=2, random_state=random_state)
X_pca = pca.fit_transform(X_scaled)


plt.figure(figsize=(15, 5))


# 1. Исходные истинные кластеры
plt.subplot(1, 3, 1)
for cluster_idx in range(n_clusters):
    cluster_mask = (y_true == cluster_idx)
    plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
               alpha=0.6, label=f'Кластер {cluster_idx}')
plt.title('Истинные кластеры (PCA проекция)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()


# 2. KMeans на полных данных
plt.subplot(1, 3, 2)
for cluster_idx in range(n_clusters):
    cluster_mask = (labels_full_predict == cluster_idx)
    plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
               alpha=0.6, label=f'Кластер {cluster_idx}')
plt.title('KMeans на полных данных (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()


# 3. KMeans на семплированных данных + семплы
plt.subplot(1, 3, 3)
for cluster_idx in range(n_clusters):
    cluster_mask = (labels_sampled_predict == cluster_idx)
    plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
               alpha=0.3, label=f'Кластер {cluster_idx}')


# Показываем семплированные точки
plt.scatter(X_pca[sampled_indices, 0], X_pca[sampled_indices, 1],
           c='black', marker='x', s=50, alpha=0.8, label='Семплированные точки')


plt.title('KMeans на семплированных данных\n(черные X - семпл)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()


plt.tight_layout()
plt.show()


# =============================================================================
# СРАВНЕНИЕ СТАТИСТИК
# =============================================================================


print("\n" + "=" * 60)
print("СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ")
print("=" * 60)


# Сравнение статистик по кластерам
original_counts = np.bincount(y_true)
sampled_counts = np.bincount(y_sampled, minlength=n_clusters)


print("Распределение по кластерам:")
print(f"{'Кластер':<10} {'Исходные':<12} {'Семпл':<10} {'Отн. частота':<15}")
for i in range(n_clusters):
    orig_count = original_counts[i]
    samp_count = sampled_counts[i] if i < len(sampled_counts) else 0
    orig_freq = orig_count / n_samples
    samp_freq = samp_count / sample_size
    print(f"{i:<10} {orig_count:<12} {samp_count:<10} {samp_freq/orig_freq:.3f}")


# Сравнение центроидов
print(f"\nСравнение центроидов:")
centroid_diff = np.linalg.norm(kmeans_full.cluster_centers_ - kmeans_sampled.cluster_centers_, axis=1)
print(f"Среднее расстояние между центроидами: {centroid_diff.mean():.6f}")
print(f"Максимальное расстояние: {centroid_diff.max():.6f}")


# =============================================================================
# АНАЛИЗ ЭФФЕКТИВНОСТИ
# =============================================================================


import time


print("\n" + "=" * 60)
print("АНАЛИЗ ЭФФЕКТИВНОСТИ")
print("=" * 60)


# Замер времени обучения
start_time = time.time()
kmeans_test_full = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
kmeans_test_full.fit(X_scaled)
full_time = time.time() - start_time


start_time = time.time()
kmeans_test_sampled = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
kmeans_test_sampled.fit(X_sampled, sample_weight=sample_weights_normalized)
sampled_time = time.time() - start_time


print(f"Время обучения на полных данных: {full_time:.4f} сек")
print(f"Время обучения на семплированных данных: {sampled_time:.4f} сек")
print(f"Ускорение: {full_time/sampled_time:.2f}x")
print(f"Эффективность использования данных: {sample_size}/{n_samples} ({sample_size/n_samples*100:.1f}%)")

# Оценка эффективного размера выборки
effective_size = SamplingEvaluator.effective_sample_size(sampling_weights)
print(f"Эффективный размер выборки: {effective_size:.2f} ({effective_size/sample_size*100:.1f}% от номинального)")


# =============================================================================
# ИТОГОВЫЙ АНАЛИЗ
# =============================================================================


print("\n" + "=" * 60)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 60)


print("1. КАЧЕСТВО КЛАСТЕРИЗАЦИИ:")
print(f"   • Полные данные: ARI = {metrics_full['ARI']:.4f}")
print(f"   • Семпл с весами: ARI = {metrics_sampled['ARI']:.4f}")
print(f"   • Разница: {abs(metrics_full['ARI'] - metrics_sampled['ARI']):.4f}")


print("\n2. ЭФФЕКТИВНОСТЬ:")
print(f"   • Ускорение обучения: {full_time/sampled_time:.2f}x")
print(f"   • Использовано данных: {sample_size/n_samples*100:.1f}%")
print(f"   • Эффективный размер выборки: {effective_size:.2f} ({effective_size/sample_size*100:.1f}%)")


print("\n3. РЕКОМЕНДАЦИИ:")
if abs(metrics_full['ARI'] - metrics_sampled['ARI']) < 0.05:
    print("   ✓ Качество кластеризации сохраняется при значительном ускорении")
    print("   ✓ Метод пригоден для больших datasets")
    if abs(metrics_full['ARI'] - metrics_sampled['ARI']) < 0.02:
        print("   ✓ Отличный результат - рассмотрите уменьшение размера семпла")
else:
    print("   ⚠ Качество снизилось - рассмотрите:")
    print("     • Увеличение размера семпла до 500-600")
    print("     • Настройку стратегии семплирования")
    print("     • Увеличение количества центроидов в PQ")


# Дополнительная визуализация: распределение весов
plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)
plt.hist(sampling_weights, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Распределение весов семплирования')
plt.xlabel('Вес')
plt.ylabel('Частота')


plt.subplot(1, 2, 2)
# Сравнение норм исходных и семплированных данных
original_norms = np.linalg.norm(X_scaled, axis=1)
sampled_norms = np.linalg.norm(X_sampled, axis=1)


plt.hist(original_norms, bins=30, alpha=0.5, label='Все данные', color='gray')
plt.hist(sampled_norms, bins=30, alpha=0.7, label='Семпл', color='blue')
plt.xlabel('Норма вектора')
plt.ylabel('Частота')
plt.title('Сравнение норм векторов')
plt.legend()


plt.tight_layout()
plt.show()


# Новая секция: оценка матожидания целевой функции
print("\n" + "=" * 60)
print("ОЦЕНКА МАТОЖИДАНИЯ ЦЕЛЕВОЙ ФУНКЦИИ")
print("=" * 60)

# Оценка матожидания нормы на семплированных данных
samples_tensor = torch.tensor(X_sampled, dtype=torch.float32)
expectation_estimate = SamplingEvaluator.estimate_expectation(
    samples_tensor, sampling_weights, target_function_norm
)

# Истинное матожидание (на всех данных)
true_expectation = target_function_norm(X_tensor).mean().item()

print(f"Истинное матожидание нормы: {true_expectation:.4f}")
print(f"Оценка через importance sampling: {expectation_estimate:.4f}")
print(f"Относительная ошибка: {abs(true_expectation - expectation_estimate)/true_expectation*100:.2f}%")

# Оценка дисперсии
variance_estimate = SamplingEvaluator.estimate_variance(
    samples_tensor, sampling_weights, target_function_norm
)
true_variance = target_function_norm(X_tensor).var().item()

print(f"\nИстинная дисперсия нормы: {true_variance:.4f}")
print(f"Оценка дисперсии через IS: {variance_estimate:.4f}")
print(f"Относительная ошибка дисперсии: {abs(true_variance - variance_estimate)/true_variance*100:.2f}%")