"""
GPU Setup Verification Script.

Run this script to check if GPU acceleration is properly configured.

Usage:
    python check_gpu.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


def check_pytorch():
    """Check PyTorch and CUDA availability."""
    print("=" * 60)
    print("PyTorch Configuration")
    print("=" * 60)

    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"[OK] CUDA version: {torch.version.cuda}")
            print(f"[OK] GPU device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"[OK] GPU Memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"[OK] CUDA Capability: {props.major}.{props.minor}")

            # Test tensor creation on GPU
            x = torch.randn(1000, 1000, device="cuda")
            print(f"[OK] GPU tensor test: PASSED (created {x.shape} tensor)")
            del x
            torch.cuda.empty_cache()
        else:
            print("[FAIL] No CUDA GPU detected")
            print("  - Check NVIDIA drivers are installed")
            print("  - Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")

    except ImportError:
        print("[FAIL] PyTorch not installed")
        print("  - Install with: pip install torch")
    except Exception as e:
        print(f"[FAIL] Error: {e}")
    print()


def check_faiss():
    """Check FAISS configuration."""
    print("=" * 60)
    print("FAISS Configuration")
    print("=" * 60)

    try:
        import faiss
        print(f"✓ FAISS installed")
        print(f"✓ FAISS version info: {faiss.__version__ if hasattr(faiss, '__version__') else 'N/A'}")

        # Check GPU support
        if hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                print("✓ FAISS GPU support: AVAILABLE")

                # Test GPU index
                import numpy as np
                d = 128
                index = faiss.IndexFlatIP(d)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

                vectors = np.random.random((1000, d)).astype('float32')
                gpu_index.add(vectors)
                print(f"✓ FAISS GPU test: PASSED (indexed 1000 vectors)")
            except Exception as e:
                print(f"✗ FAISS GPU test failed: {e}")
        else:
            print("✗ FAISS GPU support: NOT AVAILABLE (using faiss-cpu)")
            print("  - For GPU support: pip install faiss-gpu")

    except ImportError:
        print("✗ FAISS not installed")
        print("  - Install with: pip install faiss-cpu")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def check_implicit():
    """Check implicit library configuration."""
    print("=" * 60)
    print("Implicit ALS Configuration")
    print("=" * 60)

    try:
        import implicit
        print(f"✓ Implicit installed: {implicit.__version__}")

        # Check GPU support
        try:
            from implicit.gpu.als import AlternatingLeastSquares as GPUAlS
            print("✓ Implicit GPU support: AVAILABLE")
        except ImportError:
            print("✗ Implicit GPU support: NOT AVAILABLE")
            print("  - For GPU support: pip install implicit[gpu]")

        # Try CPU version
        try:
            from implicit.cpu.als import AlternatingLeastSquares as CPUAlS
            print("✓ Implicit CPU support: AVAILABLE")
        except ImportError:
            from implicit.als import AlternatingLeastSquares as CPUAlS
            print("✓ Implicit CPU support: AVAILABLE (legacy import)")

    except ImportError:
        print("✗ Implicit not installed")
        print("  - Install with: pip install implicit")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def check_sentence_transformers():
    """Check Sentence Transformers configuration."""
    print("=" * 60)
    print("Sentence Transformers Configuration")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        print("✓ Sentence Transformers installed")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ Will use device: {device}")

        # Quick test
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        embedding = model.encode(["Test sentence"], show_progress_bar=False)
        print(f"✓ SBERT test: PASSED (embedding shape: {embedding.shape})")

        if device == "cuda":
            print("✓ SBERT running on GPU")
        else:
            print("✗ SBERT running on CPU")

    except ImportError:
        print("✗ Sentence Transformers not installed")
        print("  - Install with: pip install sentence-transformers")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def check_device_config():
    """Check device_config module."""
    print("=" * 60)
    print("Device Config Module")
    print("=" * 60)

    try:
        from device_config import (
            get_device, is_gpu_available, init_device,
            get_optimal_batch_size, get_implicit_als_class
        )

        device = init_device()
        print(f"✓ Device initialized: {device}")
        print(f"✓ GPU available: {is_gpu_available()}")
        print(f"✓ SBERT batch size: {get_optimal_batch_size('sbert')}")
        print(f"✓ Matrix batch size: {get_optimal_batch_size('matrix')}")

        ALSClass, is_gpu = get_implicit_als_class()
        print(f"✓ Implicit ALS class: {'GPU' if is_gpu else 'CPU'}")

    except ImportError as e:
        print(f"✗ device_config import error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("GPU SETUP VERIFICATION")
    print("=" * 60 + "\n")

    check_pytorch()
    check_faiss()
    check_implicit()
    check_sentence_transformers()
    check_device_config()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            print("🚀 GPU acceleration is READY!")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print("\n   Run training with:")
            print("   python train.py")
        else:
            print("⚠️  Running in CPU mode")
            print("   GPU acceleration not available")
            print("\n   To enable GPU:")
            print("   1. Install NVIDIA CUDA Toolkit")
            print("   2. pip install torch --index-url https://download.pytorch.org/whl/cu118")
    except:
        print("⚠️  Running in CPU mode")

    print()


if __name__ == "__main__":
    main()
