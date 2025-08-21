import platform, shutil, subprocess, json
from datetime import datetime
from typing import Dict, Any, Optional
import psutil

def _try(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None

def nvidia_smi() -> Optional[str]:
    return _try(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])

def collect_system_info() -> Dict[str, Any]:
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_free_gb": round(shutil.disk_usage("/").free / (1024**3), 2),
    }
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as e:
        info.update({"torch": f"not available ({e})"})
    info["nvidia_smi"] = nvidia_smi()
    return info
