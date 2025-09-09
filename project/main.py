from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import asyncpg
import torch
import numpy as np
import joblib
import hashlib
from datetime import datetime
import logging
from contextlib import asynccontextmanager
import json
import os

from fast_rolsh_sampler import FastRoLSHsampler
from product_quantizer import BaseQuantizer, ProductQuantizer, AdvancedProductQuantizer, PQConfig
from unified_quantization import UnifiedQuantizationEngine, UnifiedConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request validation
class DatasetCreate(BaseModel):
    """Request model for creating a new dataset"""
    name: str  # Name of the dataset
    dimension: int  # Dimensionality of the data vectors
    m: Optional[int] = 100  # Number of hash functions per table
    k: Optional[int] = 10  # Number of hash bits per function
    L: Optional[int] = 5  # Number of hash tables
    w: Optional[float] = 1.0  # Bucket width for Euclidean distance
    distance_metric: Optional[str] = "euclidean"  # Distance metric to use
    initial_radius: Optional[float] = None  # Initial search radius for roLSH
    radius_expansion: Optional[float] = 2.0  # Radius expansion factor for roLSH
    sampling_ratio: Optional[float] = 0.1  # Feature sampling ratio for FastLSH
    quantization_method: Optional[str] = "lsh"  # Quantization method: lsh, pq, hybrid
    pq_num_subspaces: Optional[int] = 8  # Number of subspaces for PQ
    pq_num_clusters: Optional[int] = 256  # Number of clusters per subspace for PQ
    pq_use_diffusion: Optional[bool] = False  # Use diffusion maps for PQ

    @validator('distance_metric')
    def validate_distance_metric(cls, v):
        """Validate that distance metric is either 'euclidean' or 'cosine'"""
        if v not in ["euclidean", "cosine"]:
            raise ValueError("Distance metric must be 'euclidean' or 'cosine'")
        return v
    
    @validator('quantization_method')
    def validate_quantization_method(cls, v):
        """Validate quantization method"""
        if v not in ["lsh", "pq", "hybrid"]:
            raise ValueError("Quantization method must be 'lsh', 'pq', or 'hybrid'")
        return v

class BatchData(BaseModel):
    """Request model for processing a batch of data"""
    dataset_name: str  # Name of the dataset to add data to
    batch_id: Optional[str] = None  # Optional custom batch ID
    data: List[List[float]]  # List of data vectors

class QueryRequest(BaseModel):
    """Request model for querying nearest neighbors"""
    dataset_name: str  # Name of the dataset to query
    queries: List[List[float]]  # List of query vectors
    k: int = 10  # Number of neighbors to return

class SampleRequest(BaseModel):
    """Request model for sampling data"""
    dataset_name: str  # Name of the dataset to sample from
    strategy: str = "proportional"  # Sampling strategy
    size: int = 1000  # Number of samples to return

class ModelSaveRequest(BaseModel):
    """Request model for saving model state"""
    dataset_name: str  # Name of the dataset
    filepath: str = "model_state.joblib"  # Path to save model state

class QuantizationTrainRequest(BaseModel):
    """Request model for training quantization"""
    dataset_name: str  # Name of the dataset
    method: str = "pq"  # Quantization method to train: pq or apq
    use_diffusion: Optional[bool] = False  # Use diffusion maps for APQ

# Global variables for model and database connection management
models = {}  # Dictionary to store model instances by dataset name
db_pool = None  # Database connection pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    global db_pool
    
    # Startup: Create database connection pool
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/fast_rolsh_db")
    db_pool = await asyncpg.create_pool(db_url)
    
    # Load existing datasets and their models from database
    await load_existing_datasets()
    
    # Yield control to the application
    yield
    
    # Shutdown: Close database connection pool
    if db_pool:
        await db_pool.close()

# Create FastAPI application with lifespan management
app = FastAPI(
    title="FastRoLSH API", 
    description="API for batch processing with FastRoLSH and Product Quantization", 
    lifespan=lifespan
)

async def load_existing_datasets():
    """Load existing datasets from database and initialize their models"""
    try:
        async with db_pool.acquire() as conn:
            # Query all datasets from database
            datasets = await conn.fetch("SELECT id, name, dimension, m, k, L, w, distance_metric, initial_radius, radius_expansion, sampling_ratio, quantization_method, pq_num_subspaces, pq_num_clusters, pq_use_diffusion FROM datasets")
            
            # Initialize model for each dataset
            for dataset in datasets:
                if dataset['quantization_method'] == 'lsh':
                    models[dataset['name']] = FastRoLSHsampler(
                        d=dataset['dimension'],
                        m=dataset['m'],
                        k=dataset['k'],
                        L=dataset['L'],
                        w=dataset['w'],
                        dataset_id=dataset['id'],
                        dataset_name=dataset['name'],
                        db_pool=db_pool,
                        distance_metric=dataset['distance_metric'],
                        initial_radius=dataset['initial_radius'],
                        radius_expansion=dataset['radius_expansion'],
                        sampling_ratio=dataset['sampling_ratio']
                    )
                else:
                    # Create unified quantization engine
                    unified_config = UnifiedConfig(
                        m=dataset['m'],
                        k=dataset['k'],
                        L=dataset['L'],
                        w=dataset['w'],
                        distance_metric=dataset['distance_metric'],
                        initial_radius=dataset['initial_radius'],
                        radius_expansion=dataset['radius_expansion'],
                        sampling_ratio=dataset['sampling_ratio'],
                        pq_num_subspaces=dataset['pq_num_subspaces'],
                        pq_num_clusters=dataset['pq_num_clusters'],
                        pq_use_diffusion=dataset['pq_use_diffusion'],
                        hybrid_mode='hybrid' if dataset['quantization_method'] == 'hybrid' else 'pq_only'
                    )
                    
                    models[dataset['name']] = UnifiedQuantizationEngine(
                        d=dataset['dimension'],
                        config=unified_config,
                        dataset_id=dataset['id'],
                        dataset_name=dataset['name'],
                        db_pool=db_pool
                    )
                
                # Load model state from database
                if isinstance(models[dataset['name']], FastRoLSHsampler):
                    await models[dataset['name']].load_state_from_db()
                # For unified engine, we need to load from file if exists
                
        logger.info(f"Loaded {len(datasets)} datasets from database")
    except Exception as e:
        logger.error(f"Error loading existing datasets: {e}")

async def get_db_connection():
    """Dependency for getting database connection from pool"""
    async with db_pool.acquire() as connection:
        yield connection

@app.get("/")
async def root():
    """Root endpoint returning basic API information"""
    return {"message": "FastRoLSH API Server with Product Quantization"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring API and database status"""
    try:
        # Test database connection
        async with db_pool.acquire() as conn:
            await conn.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@app.post("/datasets/")
async def create_dataset(dataset: DatasetCreate, conn=Depends(get_db_connection)):
    """Create a new dataset with specified parameters"""
    try:
        # Check if dataset with this name already exists
        existing = await conn.fetchrow(
            "SELECT id FROM datasets WHERE name = $1", dataset.name
        )
        
        if existing:
            raise HTTPException(status_code=400, detail="Dataset with this name already exists")
        
        # Insert new dataset into database
        dataset_id = await conn.fetchval(
            """INSERT INTO datasets (name, dimension, m, k, L, w, distance_metric, initial_radius, 
            radius_expansion, sampling_ratio, quantization_method, pq_num_subspaces, pq_num_clusters, pq_use_diffusion) 
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) RETURNING id""",
            dataset.name, dataset.dimension, dataset.m, dataset.k, dataset.L, dataset.w, 
            dataset.distance_metric, dataset.initial_radius, dataset.radius_expansion, 
            dataset.sampling_ratio, dataset.quantization_method, dataset.pq_num_subspaces,
            dataset.pq_num_clusters, dataset.pq_use_diffusion
        )
        
        # Initialize appropriate model based on quantization method
        if dataset.quantization_method == 'lsh':
            models[dataset.name] = FastRoLSHsampler(
                d=dataset.dimension,
                m=dataset.m,
                k=dataset.k,
                L=dataset.L,
                w=dataset.w,
                dataset_id=dataset_id,
                dataset_name=dataset.name,
                db_pool=db_pool,
                distance_metric=dataset.distance_metric,
                initial_radius=dataset.initial_radius,
                radius_expansion=dataset.radius_expansion,
                sampling_ratio=dataset.sampling_ratio
            )
        else:
            # Create unified quantization engine
            unified_config = UnifiedConfig(
                m=dataset.m,
                k=dataset.k,
                L=dataset.L,
                w=dataset.w,
                distance_metric=dataset.distance_metric,
                initial_radius=dataset.initial_radius,
                radius_expansion=dataset.radius_expansion,
                sampling_ratio=dataset.sampling_ratio,
                pq_num_subspaces=dataset.pq_num_subspaces,
                pq_num_clusters=dataset.pq_num_clusters,
                pq_use_diffusion=dataset.pq_use_diffusion,
                hybrid_mode='hybrid' if dataset.quantization_method == 'hybrid' else 'pq_only'
            )
            
            models[dataset.name] = UnifiedQuantizationEngine(
                d=dataset.dimension,
                config=unified_config,
                dataset_id=dataset_id,
                dataset_name=dataset.name,
                db_pool=db_pool
            )
        
        return {"dataset_id": dataset_id, "message": "Dataset created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batches/")
async def process_batch(batch: BatchData, background_tasks: BackgroundTasks):
    """Process a batch of data vectors asynchronously"""
    try:
        # Get model for the specified dataset
        model = models.get(batch.dataset_name)
        if model is None:
            raise HTTPException(status_code=404, detail="Dataset not found or model not initialized")
        
        # Convert data to tensor
        tensor_data = torch.tensor(batch.data, dtype=torch.float32)
        
        # Generate batch ID if not provided
        if batch.batch_id is None:
            data_hash = hashlib.sha256(tensor_data.numpy().tobytes()).hexdigest()
            batch_id = f"batch_{data_hash[:16]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            batch_id = batch.batch_id
        
        # Add background task for processing the batch
        background_tasks.add_task(process_batch_background, model, tensor_data, batch_id, batch.dataset_name)
        
        return {"message": "Batch processing started", "batch_id": batch_id}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch_background(model, tensor_data: torch.Tensor, batch_id: str, dataset_name: str):
    """Background task for processing a batch of data"""
    try:
        # Update model with new batch data
        if isinstance(model, FastRoLSHsampler):
            success = await model.update(tensor_data, batch_id)
        else:  # UnifiedQuantizationEngine
            success = await model.update(tensor_data, batch_id)
        
        if success:
            # Save batch metadata to database
            async with db_pool.acquire() as conn:
                data_hash = hashlib.sha256(tensor_data.numpy().tobytes()).hexdigest()
                await conn.execute(
                    """INSERT INTO batches (dataset_id, batch_id, size, features_hash) 
                       VALUES ($1, $2, $3, $4)""",
                    model.dataset_id if hasattr(model, 'dataset_id') else None, 
                    batch_id, len(tensor_data), data_hash
                )
            
            logger.info(f"Batch {batch_id} processed successfully for dataset {dataset_name}")
        else:
            logger.error(f"Failed to process batch {batch_id} for dataset {dataset_name}")
            
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {str(e)}")

@app.post("/query/")
async def query_neighbors(request: QueryRequest):
    """Query for nearest neighbors of given vectors"""
    try:
        # Get model for the specified dataset
        model = models.get(request.dataset_name)
        if model is None:
            raise HTTPException(status_code=404, detail="Dataset not found or model not initialized")
        
        # Convert queries to tensor
        queries_tensor = torch.tensor(request.queries, dtype=torch.float32)
        
        # Execute query based on model type
        if isinstance(model, FastRoLSHsampler):
            results = await model.batched_query(queries_tensor, request.k)
        else:  # UnifiedQuantizationEngine
            results = await model.query(queries_tensor, request.k)
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sample/")
async def sample_data(request: SampleRequest):
    """Sample data points using LSH-based sampling strategies"""
    try:
        # Get model for the specified dataset
        model = models.get(request.dataset_name)
        if model is None:
            raise HTTPException(status_code=404, detail="Dataset not found or model not initialized")
        
        # Execute sampling
        if isinstance(model, FastRoLSHsampler):
            indices, weights = await model.sample(
                strategy=request.strategy,
                size=request.size
            )
        else:  # UnifiedQuantizationEngine
            indices, weights = await model.sample(
                strategy=request.strategy,
                size=request.size
            )
        
        return {"indices": indices, "weights": weights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quantization/train/")
async def train_quantization(request: QuantizationTrainRequest):
    """Train quantization model on existing data"""
    try:
        # Get model for the specified dataset
        model = models.get(request.dataset_name)
        if model is None:
            raise HTTPException(status_code=404, detail="Dataset not found or model not initialized")
        
        # Check if model supports quantization training
        if not hasattr(model, 'train_pq'):
            raise HTTPException(status_code=400, detail="Model does not support quantization training")
        
        # Train quantization
        success = await model.train_pq()
        
        if success:
            return {"message": "Quantization model trained successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to train quantization model")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets/")
async def list_datasets(conn=Depends(get_db_connection)):
    """List all datasets in the system"""
    try:
        # Query all datasets from database
        datasets = await conn.fetch("SELECT id, name, dimension, m, k, L, w, distance_metric, initial_radius, radius_expansion, sampling_ratio, quantization_method, pq_num_subspaces, pq_num_clusters, pq_use_diffusion, created_at FROM datasets ORDER BY created_at DESC")
        return [dict(dataset) for dataset in datasets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/state/{dataset_name}")
async def get_model_state(dataset_name: str):
    """Get current state and statistics of a model"""
    try:
        # Get model for the specified dataset
        model = models.get(dataset_name)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get model statistics
        if isinstance(model, FastRoLSHsampler):
            stats = await model.get_stats()
        else:  # UnifiedQuantizationEngine
            stats = await model.get_stats()
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/save/")
async def save_model(request: ModelSaveRequest):
    """Save model state to a file"""
    try:
        # Get model for the specified dataset
        model = models.get(request.dataset_name)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(request.filepath), exist_ok=True)
        
        # Save model state
        if isinstance(model, FastRoLSHsampler):
            success = model.save_state(request.filepath)
        else:  # UnifiedQuantizationEngine
            success = model.save_state(request.filepath)
        
        if success:
            return {"message": "Model saved successfully", "filepath": request.filepath}
        else:
            raise HTTPException(status_code=500, detail="Failed to save model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/load/")
async def load_model(request: ModelSaveRequest, conn=Depends(get_db_connection)):
    """Load model state from a file"""
    try:
        # Get dataset information from database
        dataset = await conn.fetchrow(
            "SELECT id, name, dimension, m, k, L, w, distance_metric, initial_radius, radius_expansion, sampling_ratio, quantization_method, pq_num_subspaces, pq_num_clusters, pq_use_diffusion FROM datasets WHERE name = $1", 
            request.dataset_name
        )
        
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Create or get model instance
        if request.dataset_name not in models:
            if dataset['quantization_method'] == 'lsh':
                models[request.dataset_name] = FastRoLSHsampler(
                    d=dataset['dimension'],
                    m=dataset['m'],
                    k=dataset['k'],
                    L=dataset['L'],
                    w=dataset['w'],
                    dataset_id=dataset['id'],
                    dataset_name=request.dataset_name,
                    db_pool=db_pool,
                    distance_metric=dataset['distance_metric'],
                    initial_radius=dataset['initial_radius'],
                    radius_expansion=dataset['radius_expansion'],
                    sampling_ratio=dataset['sampling_ratio']
                )
            else:
                # Create unified quantization engine
                unified_config = UnifiedConfig(
                    m=dataset['m'],
                    k=dataset['k'],
                    L=dataset['L'],
                    w=dataset['w'],
                    distance_metric=dataset['distance_metric'],
                    initial_radius=dataset['initial_radius'],
                    radius_expansion=dataset['radius_expansion'],
                    sampling_ratio=dataset['sampling_ratio'],
                    pq_num_subspaces=dataset['pq_num_subspaces'],
                    pq_num_clusters=dataset['pq_num_clusters'],
                    pq_use_diffusion=dataset['pq_use_diffusion'],
                    hybrid_mode='hybrid' if dataset['quantization_method'] == 'hybrid' else 'pq_only'
                )
                
                models[request.dataset_name] = UnifiedQuantizationEngine(
                    d=dataset['dimension'],
                    config=unified_config,
                    dataset_id=dataset['id'],
                    dataset_name=request.dataset_name,
                    db_pool=db_pool
                )
        
        # Load model state from file
        if isinstance(models[request.dataset_name], FastRoLSHsampler):
            success = models[request.dataset_name].load_state(request.filepath)
        else:  # UnifiedQuantizationEngine
            success = models[request.dataset_name].load_state(request.filepath)
        
        if success:
            return {"message": "Model loaded successfully", "filepath": request.filepath}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets/{dataset_name}/info")
async def get_dataset_info(dataset_name: str, conn=Depends(get_db_connection)):
    """Get detailed information about a dataset"""
    try:
        # Get dataset information from database
        dataset = await conn.fetchrow(
            "SELECT id, name, dimension, m, k, L, w, distance_metric, initial_radius, radius_expansion, sampling_ratio, quantization_method, pq_num_subspaces, pq_num_clusters, pq_use_diffusion, created_at FROM datasets WHERE name = $1",
            dataset_name
        )
        
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get batch information for the dataset
        batches = await conn.fetch(
            "SELECT batch_id, size, processed_at FROM batches WHERE dataset_id = $1 ORDER BY processed_at",
            dataset['id']
        )
        
        # Get model statistics if available
        model_stats = {}
        if dataset_name in models:
            if isinstance(models[dataset_name], FastRoLSHsampler):
                model_stats = await models[dataset_name].get_stats()
            else:  # UnifiedQuantizationEngine
                model_stats = await models[dataset_name].get_stats()
        
        return {
            "dataset": dict(dataset),
            "batches": [dict(batch) for batch in batches],
            "model_stats": model_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets/{dataset_name}/batches/{batch_id}")
async def get_batch_info(dataset_name: str, batch_id: str, conn=Depends(get_db_connection)):
    """Get information about a specific batch"""
    try:
        # Get batch information from database
        batch = await conn.fetchrow(
            """SELECT b.batch_id, b.size, b.processed_at, b.features_hash, d.name as dataset_name
               FROM batches b JOIN datasets d ON b.dataset_id = d.id
               WHERE d.name = $1 AND b.batch_id = $2""",
            dataset_name, batch_id
        )
        
        if batch is None:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return dict(batch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/optimize/{dataset_name}")
async def optimize_model_parameters(
    dataset_name: str, 
    sample_size: int = 1000,
    conn=Depends(get_db_connection)
):
    """Optimize model parameters based on data characteristics"""
    try:
        # Get model for the specified dataset
        model = models.get(dataset_name)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Optimize parameters based on model type
        if isinstance(model, FastRoLSHsampler):
            result = await model.optimize_parameters(sample_size)
            
            # Update parameters in database
            await conn.execute(
                "UPDATE datasets SET w = $1, initial_radius = $2, updated_at = NOW() WHERE name = $3",
                result.get('w', model.w),
                result.get('initial_radius', model.initial_radius),
                dataset_name
            )
        else:
            # For unified engine, we need to implement optimization
            result = {"message": "Parameter optimization not yet implemented for unified engine"}
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)