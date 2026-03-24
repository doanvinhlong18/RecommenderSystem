"""
Device Configuration Module for GPU/CPU Detection and Management.

This module provides centralized device management for the recommendation system,
enabling automatic GPU detection with graceful CPU fallback.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global device setting
_DEVICE: Optional[str] = None
_FORCE_CPU: bool = False


def _detect_device() -> str:
    """Detect available device (CUDA GPU or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA GPU detected: {device_name}")
            logger.info(f"GPU Memory: {total_memory:.2f} GB")
            return "cuda"
        else:
            logger.info("No CUDA GPU detected, using CPU")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch not installed, using CPU")
        return "cpu"
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}, using CPU")
        return "cpu"


def init_device(force_cpu: bool = False) -> str:
    """
    Initialize and return the device to use for training.

    Args:
        force_cpu: If True, force CPU usage even if GPU is available

    Returns:
        Device string ("cuda" or "cpu")
    """
    global _DEVICE, _FORCE_CPU

    _FORCE_CPU = force_cpu

    if force_cpu:
        _DEVICE = "cpu"
        logger.info("Force CPU mode enabled")
    else:
        _DEVICE = _detect_device()

    logger.info(f"{'='*50}")
    logger.info(f"Using device: {_DEVICE.upper()}")
    logger.info(f"{'='*50}")

    return _DEVICE


def get_device() -> str:
    """
    Get the current device setting.

    Returns:
        Device string ("cuda" or "cpu")
    """
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = _detect_device()
    return _DEVICE


def is_gpu_available() -> bool:
    """Check if GPU is available and not forced to CPU."""
    return get_device() == "cuda" and not _FORCE_CPU


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory usage information.

    Returns:
        Dictionary with memory info or empty dict if not on GPU
    """
    if not is_gpu_available():
        return {}

    try:
        import torch
        return {
            "allocated": torch.cuda.memory_allocated(0) / (1024**3),
            "reserved": torch.cuda.memory_reserved(0) / (1024**3),
            "total": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    except Exception:
        return {}


def log_gpu_memory(prefix: str = ""):
    """Log current GPU memory usage."""
    if not is_gpu_available():
        return

    info = get_gpu_memory_info()
    if info:
        msg = f"{prefix}GPU Memory - Allocated: {info['allocated']:.2f}GB, Reserved: {info['reserved']:.2f}GB, Total: {info['total']:.2f}GB"
        logger.info(msg)


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if is_gpu_available():
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")


# FAISS GPU utilities
def get_faiss_gpu_resources():
    """
    Get FAISS GPU resources if available.

    Returns:
        Tuple of (resources, gpu_available) or (None, False)
    """
    if not is_gpu_available():
        return None, False

    try:
        import faiss
        # Check if faiss-gpu is installed
        if hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            logger.info("FAISS GPU resources initialized")
            return res, True
        else:
            logger.info("FAISS GPU not available, using CPU FAISS")
            return None, False
    except Exception as e:
        logger.warning(f"FAISS GPU initialization failed: {e}")
        return None, False


def faiss_index_to_gpu(index, gpu_resources=None):
    """
    Move FAISS index to GPU if possible.

    Args:
        index: FAISS CPU index
        gpu_resources: Optional pre-created GPU resources

    Returns:
        GPU index if possible, otherwise original CPU index
    """
    if not is_gpu_available():
        return index

    try:
        import faiss
        if gpu_resources is None:
            gpu_resources, available = get_faiss_gpu_resources()
            if not available:
                return index

        gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
        logger.info("FAISS index moved to GPU")
        return gpu_index
    except Exception as e:
        logger.warning(f"Failed to move FAISS index to GPU: {e}")
        return index


# Implicit ALS GPU utilities
def get_implicit_als_class():
    """
    Get the appropriate ALS class (GPU or CPU).

    Tests if GPU ALS actually works before returning it.

    Returns:
        Tuple of (ALS class, is_gpu_enabled)
    """
    if is_gpu_available():
        try:
            from implicit.gpu.als import AlternatingLeastSquares as GPUAlS
            # Test if GPU ALS can actually be instantiated
            # This will raise ValueError if CUDA extension is not built
            test_model = GPUAlS(factors=10, iterations=1)
            del test_model
            logger.info("Using Implicit ALS GPU implementation")
            return GPUAlS, True
        except ImportError:
            logger.info("Implicit GPU module not available, using CPU ALS")
        except ValueError as e:
            # "No CUDA extension has been built, can't train on GPU."
            logger.warning(f"Implicit GPU ALS not available: {e}")
            logger.info("Falling back to CPU ALS")
        except Exception as e:
            logger.warning(f"Implicit GPU ALS error: {e}, using CPU ALS")

    try:
        from implicit.cpu.als import AlternatingLeastSquares
        logger.info("Using Implicit ALS CPU implementation")
        return AlternatingLeastSquares, False
    except ImportError:
        from implicit.als import AlternatingLeastSquares
        logger.info("Using Implicit ALS (legacy import)")
        return AlternatingLeastSquares, False


# Batch size recommendations
def get_optimal_batch_size(task: str = "sbert") -> int:
    """
    Get optimal batch size based on device and task.

    Args:
        task: Type of task ("sbert", "matrix", "general")

    Returns:
        Recommended batch size
    """
    is_gpu = is_gpu_available()

    batch_sizes = {
        "sbert": 128 if is_gpu else 32,
        "matrix": 1024 if is_gpu else 256,
        "general": 512 if is_gpu else 128
    }

    return batch_sizes.get(task, batch_sizes["general"])
