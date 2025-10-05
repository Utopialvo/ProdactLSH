# intelligent_caching.py
"""
Модуль интеллектуального кэширования для эффективного управления памятью при работе с большими наборами тензорных данных.

Реализует двухуровневую систему хранения:
1. Оперативная память (RAM) - для часто используемых данных
2. Дисковое хранилище - для редко используемых данных

Основные компоненты:
- BatchTensorDiskCache: батчевое хранение тензоров на диске
- MemoryEfficientTensorStorage: прозрачное управление памятью с автоматическим перемещением данных
"""

import torch
import os
import time
from collections import defaultdict
from typing import List, Optional
import warnings


class BatchTensorDiskCache:
    """
    Кэш для батчевого хранения тензоров на диске.
    
    Оптимизирует хранение больших наборов данных путем группировки тензоров в батчи,
    что значительно уменьшает количество файлов и улучшает производительность при
    работе с большими датасетами, не помещающимися в оперативной памяти.
    
    Attributes:
        cache_dir (str): Директория для хранения кэшированных файлов
        max_memory_size (int): Максимальный размер памяти для кэширования в байтах
        batch_size (int): Количество тензоров в одном батче
        current_memory_usage (int): Текущее использование памяти в байтах
        access_times (dict): Время последнего доступа к каждому батчу для LRU-эвристики
        batch_counter (int): Счетчик для генерации уникальных идентификаторов батчей
        batch_to_indices (defaultdict): Маппинг батч_id -> список индексов тензоров
        index_to_batch (dict): Маппинг индекс_тензора -> (батч_id, позиция_в_батче)
    """
    
    def __init__(self, cache_dir: str = "./tensor_cache", max_memory_size: int = 1024 * 1024 * 1024 * 1024,
                 batch_size: int = 2048):
        """
        Инициализация кэша.
        
        Args:
            cache_dir: Директория для хранения кэшированных файлов
            max_memory_size: Максимальный размер памяти для кэширования в байтах
            batch_size: Количество тензоров в одном батче
        """
        self.cache_dir = cache_dir
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.current_memory_usage = 0
        self._ensure_cache_dir()
        self.access_times = {}
        self.batch_counter = 0
        self.current_batch = {}
        self.batch_to_indices = defaultdict(list)
        self.index_to_batch = {}
        
    def _ensure_cache_dir(self):
        """Создает директорию для кэша если она не существует."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_batch_filename(self, batch_id: int) -> str:
        """
        Генерирует имя файла для батча.
        
        Args:
            batch_id: Уникальный идентификатор батча
            
        Returns:
            Полный путь к файлу батча
        """
        return os.path.join(self.cache_dir, f"batch_{batch_id:06d}.pt")
    
    def store_tensors(self, tensors: List[torch.Tensor], indices: List[int]) -> None:
        """
        Сохраняет батч тензоров на диск.
        
        Args:
            tensors: Список тензоров для сохранения
            indices: Список индексов соответствующих тензоров
        """
        if not tensors:
            return
            
        # Создаем новый батч с уникальным ID
        batch_id = self.batch_counter
        self.batch_counter += 1
        
        # Подготавливаем данные для сохранения
        batch_data = {
            'tensors': torch.stack(tensors),
            'indices': indices
        }
        
        # Сохраняем батч на диск
        batch_filename = self._get_batch_filename(batch_id)
        torch.save(batch_data, batch_filename)
        
        # Обновляем маппинги для быстрого доступа
        for idx, tensor_idx in enumerate(indices):
            self.batch_to_indices[batch_id].append(tensor_idx)
            self.index_to_batch[tensor_idx] = (batch_id, idx)
        
        # Обновляем время доступа для LRU
        self.access_times[batch_id] = time.time()
    
    def load_tensor(self, index: int, device: str = None) -> torch.Tensor:
        """
        Загружает тензор по индексу.
        
        Args:
            index: Индекс тензора для загрузки
            device: Устройство для загрузки тензора ('cuda' или 'cpu')
            
        Returns:
            Загруженный тензор
            
        Raises:
            ValueError: Если индекс не найден в кэше
            FileNotFoundError: Если файл батча не найден
        """
        if index not in self.index_to_batch:
            raise ValueError(f"Tensor index {index} not found in cache")
            
        # Получаем информацию о расположении тензора
        batch_id, tensor_idx = self.index_to_batch[index]
        batch_filename = self._get_batch_filename(batch_id)
        
        if not os.path.exists(batch_filename):
            raise FileNotFoundError(f"Batch file {batch_filename} not found")
        
        # Загружаем весь батч и извлекаем нужный тензор
        batch_data = torch.load(batch_filename)
        tensor = batch_data['tensors'][tensor_idx]
        
        # Обновляем время доступа
        self.access_times[batch_id] = time.time()
        
        # Перемещаем на нужное устройство если указано
        if device:
            tensor = tensor.to(device)
            
        return tensor
    
    def cleanup_old_batches(self, keep_latest: int = 10):
        """
        Удаляет старые батчи, оставляя только последние keep_latest (LRU эвристика).
        
        Args:
            keep_latest: Количество последних батчей для сохранения
        """
        if len(self.access_times) <= keep_latest:
            return
            
        # Сортируем батчи по времени доступа
        sorted_batches = sorted(self.access_times.items(), key=lambda x: x[1])
        
        # Удаляем самые старые батчи
        for batch_id, _ in sorted_batches[:-keep_latest]:
            self.remove_batch(batch_id)
    
    def remove_batch(self, batch_id: int):
        """Удаляет батч из кэша."""
        batch_filename = self._get_batch_filename(batch_id)
        
        # Удаляем файл с диска
        if os.path.exists(batch_filename):
            os.remove(batch_filename)
        
        # Удаляем маппинги из памяти
        if batch_id in self.batch_to_indices:
            for index in self.batch_to_indices[batch_id]:
                if index in self.index_to_batch:
                    del self.index_to_batch[index]
            del self.batch_to_indices[batch_id]
        
        if batch_id in self.access_times:
            del self.access_times[batch_id]


class MemoryEfficientTensorStorage:
    """
    Эффективное хранение тензоров с автоматическим кэшированием на диск.
    
    Прозрачно управляет хранением тензоров, автоматически перемещая редко используемые
    данные на диск для экономии оперативной памяти. Предоставляет единый интерфейс
    доступа независимо от физического расположения данных.
    
    Attributes:
        cache_enabled (bool): Включено ли кэширование на диск
        max_memory_points (int): Максимальное количество тензоров в оперативной памяти
        cache (BatchTensorDiskCache): Объект кэша для дискового хранения
        memory_tensors (list): Тензоры в оперативной памяти
        memory_indices (list): Индексы тензоров в памяти
        next_index (int): Следующий доступный индекс для нового тензора
    """
    
    def __init__(self, cache_enabled: bool = True, max_memory_points: int = 100000, 
                 cache_dir: str = "./tensor_cache"):
        """
        Инициализация хранилища.
        
        Args:
            cache_enabled: Включено ли кэширование на диск
            max_memory_points: Максимальное количество тензоров в оперативной памяти
            cache_dir: Директория для хранения кэшированных файлов
        """
        self.cache_enabled = cache_enabled
        self.max_memory_points = max_memory_points
        self.cache = BatchTensorDiskCache(cache_dir=cache_dir)
        
        self.memory_tensors = []  # Тензоры в оперативной памяти
        self.memory_indices = []  # Индексы тензоров в памяти
        self.next_index = 0  # Счетчик для генерации уникальных индексов
        
    def add_tensors(self, tensors: List[torch.Tensor]) -> List[int]:
        """
        Добавляет тензоры в хранилище и возвращает их индексы.
        
        Автоматически решает, хранить ли тензоры в оперативной памяти или на диске
        на основе текущей загрузки памяти и настроек хранилища.
        
        Args:
            tensors: Список тензоров для добавления
            
        Returns:
            Список индексов добавленных тензоров
        """
        indices = []
        tensors_to_cache = []
        indices_to_cache = []
        
        for tensor in tensors:
            # Генерируем уникальный индекс для тензора
            idx = self.next_index
            self.next_index += 1
            
            # Решаем, где хранить тензор на основе текущей загрузки памяти
            if (self.cache_enabled and 
                len(self.memory_tensors) >= self.max_memory_points):
                # Сохраняем в кэш (на диск)
                tensors_to_cache.append(tensor.cpu())
                indices_to_cache.append(idx)
            else:
                # Сохраняем в память (RAM)
                self.memory_tensors.append(tensor.cpu())
                self.memory_indices.append(idx)
            
            indices.append(idx)
        
        # Сохраняем батч в кэш если есть тензоры для кэширования
        if tensors_to_cache:
            self.cache.store_tensors(tensors_to_cache, indices_to_cache)
        
        return indices
    
    def get_tensor(self, index: int, device: str = None) -> torch.Tensor:
        """
        Получает тензор по индексу.
        
        Прозрачно загружает тензор из оперативной памяти или с диска.
        
        Args:
            index: Индекс тензора для загрузки
            device: Устройство для загрузки тензора
            
        Returns:
            Загруженный тензор
            
        Raises:
            ValueError: Если индекс не найден
        """
        # Ищем в оперативной памяти сначала (быстрее)
        if index in self.memory_indices:
            pos = self.memory_indices.index(index)
            tensor = self.memory_tensors[pos]
            if device:
                tensor = tensor.to(device)
            return tensor
        
        # Если не найдено в памяти, ищем в кэше на диске
        if self.cache_enabled:
            return self.cache.load_tensor(index, device)
        
        raise ValueError(f"Tensor index {index} not found")
    
    def cleanup(self):
        """Очистка старых данных из кэша для освобождения места."""
        if self.cache_enabled:
            self.cache.cleanup_old_batches()