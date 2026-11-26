"""Lightweight GPU telemetry helpers."""
import subprocess


def query_nvidia_smi():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=3)
    except Exception:
        return []
    rows = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append({
            "id": int(parts[0]), "name": parts[1], "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]), "utilization_percent": float(parts[4]),
        })
    return rows
