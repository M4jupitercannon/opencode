#!/usr/bin/env python3
"""Select the N least-utilized GPUs. Supports AMD (rocm-smi) and NVIDIA (nvidia-smi)."""
import subprocess
import sys
import json


def get_amd_gpus():
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        gpus = {}
        for key, info in data.items():
            if not key.startswith("card"):
                continue
            idx = int(key.replace("card", ""))
            use_str = str(info.get("GPU use (%)", "0"))
            use = float(use_str.replace("%", "").strip())
            gpus[idx] = [idx, use, 0.0]

        try:
            mem_result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=10,
            )
            if mem_result.returncode == 0:
                mem_data = json.loads(mem_result.stdout)
                for key, info in mem_data.items():
                    if not key.startswith("card"):
                        continue
                    idx = int(key.replace("card", ""))
                    if idx in gpus:
                        used_bytes = float(
                            str(info.get("VRAM Total Used Memory (B)", "0"))
                        )
                        gpus[idx][2] = used_bytes / (1024 ** 2)
        except Exception:
            pass

        return [tuple(v) for v in gpus.values()]
    except Exception:
        return None


def get_nvidia_gpus():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx, util, mem = int(parts[0]), float(parts[1]), float(parts[2])
                gpus.append((idx, util, mem))
        return gpus
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: select_gpus.py <num_gpus>", file=sys.stderr)
        sys.exit(1)

    n = int(sys.argv[1])

    gpus = get_amd_gpus()
    vendor = "amd"
    if gpus is None:
        gpus = get_nvidia_gpus()
        vendor = "nvidia"

    if gpus is None or len(gpus) == 0:
        print(",".join(str(i) for i in range(n)))
        sys.exit(0)

    # Sort by utilization ascending, then memory used ascending
    gpus.sort(key=lambda x: (x[1], x[2]))

    selected = [str(g[0]) for g in gpus[:n]]
    print(",".join(selected))

    print(f"GPU vendor: {vendor}", file=sys.stderr)
    print(f"Total GPUs available: {len(gpus)}", file=sys.stderr)
    print(f"Selected {n} most free: {','.join(selected)}", file=sys.stderr)
    for g in gpus:
        marker = " <-- selected" if str(g[0]) in selected else ""
        print(f"  GPU {g[0]}: util={g[1]}%, mem_used={g[2]}MB{marker}", file=sys.stderr)


if __name__ == "__main__":
    main()
