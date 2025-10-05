from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit, rand, monotonically_increasing_id
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
import logging
import pyspark.sql.functions as F
import time

from DistributedKMeans import DistributedKMeans
from PQ import ProductQuantization, PQEvaluator
from PQSampling import PQDataSampler, PQImportanceSampling

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_pq_pipeline_optimized():
    """Оптимизированная демонстрация пайплайна PQ."""
    spark = SparkSession.builder.appName("PQDemo") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewedJoin.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()
    
    # Генерация тестовых данных
    from sklearn.datasets import make_blobs
    import pandas as pd
    
    logger.info("Generating test data...")
    X, _ = make_blobs(n_samples=3000, n_features=50, centers=10, random_state=42)
    
    # Конвертируем в Spark DataFrame
    pdf = pd.DataFrame({'features': [Vectors.dense(x) for x in X]})
    df = spark.createDataFrame(pdf)
    df = df.withColumn('index', F.monotonically_increasing_id())
    
    # Кэшируем данные для многократного использования
    df.cache()
    df_count = df.count()
    logger.info(f"DataFrame cached with {df_count} rows")
    
    # Инициализация и обучение PQ
    logger.info("Training PQ model...")
    start_time = time.time()
    pq = ProductQuantization(M=8, K=256, use_opq=False, use_ivf=False)  # Упрощаем для скорости
    pq.fit(df)
    training_time = time.time() - start_time
    logger.info(f"PQ training completed in {training_time:.2f} seconds")
    
    # Кодирование данных
    logger.info("Encoding data...")
    start_time = time.time()
    encoded_df = pq.encode(df)
    encoded_df.cache()
    encoded_count = encoded_df.count()
    encoding_time = time.time() - start_time
    logger.info(f"Encoding completed in {encoding_time:.2f} seconds for {encoded_count} rows")
    
    print("Encoded data schema:")
    encoded_df.printSchema()
    
    # Быстрая оценка качества на небольшой выборке
    logger.info("Evaluating quality...")
    evaluator = PQEvaluator(pq)
    
    # Используем небольшую выборку для быстрой оценки
    sample_size = 100
    sample_df = encoded_df.limit(sample_size)
    
    # Декодируем выборку
    pq_codes = [row['pq_code'] for row in sample_df.select('pq_code').collect()]
    if pq_codes:
        reconstructed_vectors = pq.decode(pq_codes)
        
        # Создаем DataFrame с восстановленными векторами
        reconstructed_pdf = pd.DataFrame({
            'features': [Vectors.dense(x) for x in reconstructed_vectors]
        })
        reconstructed_df = spark.createDataFrame(reconstructed_pdf)
        
        # Вычисляем MSE
        mse = evaluator.compute_quantization_error(
            sample_df.select('features'), 
            reconstructed_df, 
            'features'
        )
        print(f"Sample Quantization MSE: {mse:.6f}")
        
        # Быстрая оценка сохранения расстояний
        distance_metrics = evaluator.evaluate_distance_preservation_fast(
            sample_df.select('features'), 
            reconstructed_df,
            'features', 
            sample_size=50
        )
        print(f"Distance correlation: {distance_metrics['distance_correlation']:.4f}")
    
    compression_ratio = evaluator.compute_compression_ratio(50)
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Быстрый семплинг
    logger.info("Sampling data...")
    sampler = PQDataSampler(pq)
    
    # Равномерный семплинг
    start_time = time.time()
    uniform_sample = sampler.uniform_sampling(encoded_df, 30)
    uniform_count = uniform_sample.count()
    uniform_time = time.time() - start_time
    print(f"Uniform sample count: {uniform_count} (took {uniform_time:.2f}s)")
    
    # Быстрый стратифицированный семплинг
    start_time = time.time()
    stratified_sample = sampler.stratified_sampling_optimized(encoded_df, 30, 'balanced')
    stratified_count = stratified_sample.count()
    stratified_time = time.time() - start_time
    print(f"Stratified sample count: {stratified_count} (took {stratified_time:.2f}s)")
    
    # Быстрая генерация синтетических данных
    logger.info("Generating synthetic data...")
    start_time = time.time()
    synthetic_data = sampler.generate_synthetic_data_fast('random_combination', 100, 0.05)
    synthetic_count = synthetic_data.count()
    synthetic_time = time.time() - start_time
    print(f"Synthetic data count: {synthetic_count} (took {synthetic_time:.2f}s)")
    
    # Быстрый Importance Sampling на небольшой выборке
    logger.info("Running importance sampling...")
    
    def example_target_function(x):
        return np.sum(x ** 2)  # Квадрат нормы вектора
    
    importance_sampler = PQImportanceSampling(pq)
    
    # Используем небольшую выборку для скорости
    importance_sample_size = min(200, encoded_count)
    start_time = time.time()
    expectation, sample_df = importance_sampler.importance_sampling_optimized(
        encoded_df.limit(importance_sample_size),
        example_target_function, 
        50
    )
    importance_time = time.time() - start_time
    print(f"Importance sampling expectation estimate: {expectation:.4f} (took {importance_time:.2f}s)")
    
    # Очистка кэша
    df.unpersist()
    encoded_df.unpersist()
    
    logger.info("Demo completed successfully!")
    spark.stop()

if __name__ == "__main__":
    demo_pq_pipeline_optimized()