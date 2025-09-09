import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from fast_rolsh_sampler import FastRoLSHsampler
from product_quantizer import BaseQuantizer, ProductQuantizer, AdvancedProductQuantizer, PQConfig

logger = logging.getLogger(__name__)

@dataclass
class UnifiedConfig:
    """Конфигурация для унифицированной системы квантования"""
    # Параметры LSH
    m: int = 100
    k: int = 10
    L: int = 5
    w: float = 1.0
    distance_metric: str = 'euclidean'
    initial_radius: Optional[float] = None
    radius_expansion: float = 2.0
    sampling_ratio: float = 0.1
    
    # Параметры продуктного квантования
    pq_num_subspaces: int = 8
    pq_num_clusters: int = 256
    pq_use_diffusion: bool = False
    pq_use_optimized_kmeans: bool = True
    pq_batch_size: int = 1000
    
    # Параметры гибридного режима
    hybrid_mode: str = 'two_stage'  # 'two_stage', 'pq_only', 'lsh_only'
    hybrid_candidate_multiplier: int = 10
    hybrid_use_compressed: bool = True
    
    # Общие параметры
    optimization_interval: int = 10
    random_state: int = 42

class UnifiedQuantizationEngine:
    """
    Унифицированная система квантования, объединяющая FastRoLSH и ProductQuantizer.
    Поддерживает три режима работы:
    1. LSH-only: только поиск с использованием LSH
    2. PQ-only: только продуктное квантование
    3. Hybrid: двухэтапный поиск (LSH для кандидатов + PQ для точного расстояния)
    """
    
    def __init__(self, d: int, config: Optional[UnifiedConfig] = None, 
                 dataset_id: Optional[int] = None, dataset_name: Optional[str] = None, 
                 db_pool: Optional[Any] = None):
        
        self.d = d
        self.config = config or UnifiedConfig()
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.db_pool = db_pool
        
        # Инициализация компонентов
        self.lsh_sampler = FastRoLSHsampler(
            d=d,
            m=self.config.m,
            k=self.config.k,
            L=self.config.L,
            w=self.config.w,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            db_pool=db_pool,
            distance_metric=self.config.distance_metric,
            initial_radius=self.config.initial_radius,
            radius_expansion=self.config.radius_expansion,
            sampling_ratio=self.config.sampling_ratio,
            optimization_interval=self.config.optimization_interval
        )
        
        pq_config = PQConfig(
            num_subspaces=self.config.pq_num_subspaces,
            num_clusters=self.config.pq_num_clusters,
            use_diffusion=self.config.pq_use_diffusion,
            use_optimized_kmeans=self.config.pq_use_optimized_kmeans,
            batch_size=self.config.pq_batch_size,
            distance_metric=self.config.distance_metric,
            random_state=self.config.random_state
        )
        
        if self.config.pq_use_diffusion:
            self.pq_quantizer = AdvancedProductQuantizer(config=pq_config)
        else:
            self.pq_quantizer = ProductQuantizer(config=pq_config)
        
        self.is_trained = False
        self.total_points = 0
        
    async def update(self, batch_data: torch.Tensor, batch_id: Optional[str] = None) -> bool:
        """
        Обновление системы новыми данными
        """
        try:
            # Обновляем LSH семплер
            lsh_success = await self.lsh_sampler.update(batch_data, batch_id)
            
            if not lsh_success:
                logger.error(f"Ошибка обновления LSH семплера для батча {batch_id}")
                return False
            
            # Если в гибридном режиме, обучаем PQ на накопленных данных
            if self.config.hybrid_mode != 'lsh_only' and not self.is_trained:
                # Используем reservoir sample из LSH для обучения PQ
                if hasattr(self.lsh_sampler, 'reservoir_sample') and self.lsh_sampler.reservoir_sample is not None:
                    reservoir_data = self.lsh_sampler.reservoir_sample.cpu().numpy()
                    if len(reservoir_data) >= self.config.pq_num_clusters * 2:
                        self.pq_quantizer.train(reservoir_data)
                        self.is_trained = True
                        logger.info("PQ квантователь обучен на reservoir sample")
            
            self.total_points += batch_data.shape[0]
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления UnifiedQuantizationEngine: {e}")
            return False
    
    async def query(self, queries: torch.Tensor, k: int = 10) -> List[List[int]]:
        """
        Поиск ближайших соседей в зависимости от выбранного режима
        """
        if self.config.hybrid_mode == 'lsh_only':
            return await self.lsh_sampler.batched_query(queries, k)
        
        elif self.config.hybrid_mode == 'pq_only':
            if not self.is_trained:
                raise ValueError("PQ квантователь не обучен")
            
            # Преобразуем запросы в numpy
            queries_np = queries.cpu().numpy()
            
            # Кодируем запросы
            query_codes = self.pq_quantizer.encode(queries_np)
            
            # Для каждого запроса ищем ближайшие точки
            results = []
            for i in range(queries_np.shape[0]):
                # Вычисляем расстояния до всех точек (в реальной системе нужно использовать эффективный поиск)
                # Здесь упрощенная реализация для демонстрации
                all_codes = self.pq_quantizer.encode(self._get_all_data())
                distances = self._pq_distance(query_codes[i], all_codes)
                
                # Выбираем k ближайших
                nearest_indices = np.argsort(distances)[:k]
                results.append(nearest_indices.tolist())
            
            return results
        
        else:  # hybrid_mode == 'two_stage'
            # Первый этап: LSH для поиска кандидатов
            candidate_multiplier = self.config.hybrid_candidate_multiplier
            candidates = await self.lsh_sampler.batched_query(queries, k * candidate_multiplier)
            
            if not self.is_trained:
                # Если PQ не обучен, возвращаем результаты LSH
                return [[c[:k] for c in candidates]]
            
            # Второй этап: точное ранжирование кандидатов с помощью PQ
            queries_np = queries.cpu().numpy()
            results = []
            
            for i, candidate_indices in enumerate(candidates):
                if not candidate_indices:
                    results.append([])
                    continue
                
                # Получаем данные кандидатов
                candidate_data = self._get_data_by_indices(candidate_indices)
                
                # Кодируем запрос и кандидатов
                query_code = self.pq_quantizer.encode(queries_np[i:i+1])[0]
                candidate_codes = self.pq_quantizer.encode(candidate_data)
                
                # Вычисляем расстояния
                distances = self._pq_distance(query_code, candidate_codes)
                
                # Выбираем k ближайших
                nearest_candidate_indices = np.argsort(distances)[:k]
                results.append([candidate_indices[j] for j in nearest_candidate_indices])
            
            return results
    
    def _pq_distance(self, query_code: np.ndarray, candidate_codes: np.ndarray) -> np.ndarray:
        """
        Вычисление расстояний между PQ кодами
        """
        # Упрощенная реализация - в реальной системе нужно использовать
        # эффективное вычисление расстояний через предрасчитанные таблицы
        query_reconstructed = self.pq_quantizer.decode(query_code.reshape(1, -1))
        candidates_reconstructed = self.pq_quantizer.decode(candidate_codes)
        
        if self.config.distance_metric == 'euclidean':
            return np.linalg.norm(query_reconstructed - candidates_reconstructed, axis=1)
        elif self.config.distance_metric == 'cosine':
            query_norm = query_reconstructed / np.linalg.norm(query_reconstructed)
            candidates_norm = candidates_reconstructed / np.linalg.norm(candidates_reconstructed, axis=1, keepdims=True)
            return 1 - np.dot(candidates_norm, query_norm.T).flatten()
        else:
            raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")
    
    def _get_all_data(self) -> np.ndarray:
        """
        Получение всех данных (заглушка - в реальной системе нужно получать из хранилища)
        """
        # В реальной системе это должно получать данные из базы или другого хранилища
        return np.random.randn(self.total_points, self.d)
    
    def _get_data_by_indices(self, indices: List[int]) -> np.ndarray:
        """
        Получение данных по индексам (заглушка)
        """
        # В реальной системе это должно получать данные из базы или другого хранилища
        return np.random.randn(len(indices), self.d)
    
    async def sample(self, strategy: str = 'proportional', size: int = 1000) -> Tuple[List[int], List[float]]:
        """
        Семплирование данных с использованием LSH
        """
        return await self.lsh_sampler.sample(strategy, size)
    
    async def train_pq(self, data: Optional[np.ndarray] = None) -> bool:
        """
        Обучение PQ квантователя на предоставленных данных или накопленных данных
        """
        try:
            if data is None:
                # Используем reservoir sample из LSH
                if hasattr(self.lsh_sampler, 'reservoir_sample') and self.lsh_sampler.reservoir_sample is not None:
                    data = self.lsh_sampler.reservoir_sample.cpu().numpy()
                else:
                    raise ValueError("Нет данных для обучения PQ")
            
            self.pq_quantizer.train(data)
            self.is_trained = True
            logger.info("PQ квантователь успешно обучен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения PQ квантователя: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики системы
        """
        lsh_stats = await self.lsh_sampler.get_stats()
        
        stats = {
            'unified_config': self.config.__dict__,
            'lsh_stats': lsh_stats,
            'pq_trained': self.is_trained,
            'total_points': self.total_points,
            'pq_memory_usage': self.pq_quantizer.get_memory_usage() if self.is_trained else {}
        }
        
        return stats
    
    def save_state(self, filepath: str) -> bool:
        """
        Сохранение состояния системы
        """
        try:
            import joblib
            
            state = {
                'config': self.config,
                'lsh_state': self.lsh_sampler.save_state(filepath + '.lsh'),
                'pq_state': self.pq_quantizer.save(filepath + '.pq') if self.is_trained else None,
                'is_trained': self.is_trained,
                'total_points': self.total_points
            }
            
            joblib.dump(state, filepath)
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Загрузка состояния системы
        """
        try:
            import joblib
            
            state = joblib.load(filepath)
            self.config = state['config']
            self.lsh_sampler.load_state(state['lsh_state'])
            
            if state['pq_state']:
                self.pq_quantizer.load(state['pq_state'])
            
            self.is_trained = state['is_trained']
            self.total_points = state['total_points']
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")
            return False

# Утилиты для преобразования между форматами данных
def torch_to_numpy(tensor_data: torch.Tensor) -> np.ndarray:
    """Преобразование torch tensor в numpy array"""
    return tensor_data.cpu().numpy()

def numpy_to_torch(np_data: np.ndarray) -> torch.Tensor:
    """Преобразование numpy array в torch tensor"""
    return torch.from_numpy(np_data).float()