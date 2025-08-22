from typing import Dict, Any, Optional
from time import perf_counter
import math, statistics, os
try:
    import psutil
except Exception:
    psutil = None

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader
import torch, torch.nn as nn
from .models import make_resnet18


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = False):
    model.eval()
    correct = 0
    total = 0
    if amp and device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = torch.amp.autocast("cpu", enabled=False)
    with autocast_ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _shm_usage():
    if psutil is None:
        return None
    try:
        u = psutil.disk_usage("/dev/shm")
        return u.total, u.free
    except Exception:
        return None


def _should_disable_workers(workers: int) -> bool:
    if workers <= 0:
        return False
    u = _shm_usage()
    if not u:
        return False
    total, free = u
    return (total and total < 256 * 1024**2) or (free and free < 128 * 1024**2)


def _silence_worker_stdout(_wid: int):
    import sys
    sys.stdout = open(os.devnull, "w")


def _rainbow_markup(text: str) -> str:
    colors = ["red", "dark_orange3", "yellow1", "chartreuse1",
              "turquoise2", "dodger_blue2", "medium_purple"]
    out = []
    for i, ch in enumerate(text):
        out.append(f"[{colors[i % len(colors)]}]{ch}[/]")
    return "".join(out)


def print_score_box(score_seconds: float, console: Console):
    # Радужный бокс «ВАШ СЧЕТ: <число> s» (меньше — лучше)
    bar_colors = ["red", "dark_orange3", "yellow1", "chartreuse1",
                  "turquoise2", "dodger_blue2", "medium_purple"]
    bar = "".join(f"[{c}]▀[/]" for c in bar_colors for _ in range(8))
    title = _rainbow_markup("ВАШ СЧЕТ:")
    value = _rainbow_markup(f"{score_seconds:.3f} s")
    body = f"{bar}\n[b]{title}[/b]\n{bar}\n[bold]{value}[/bold]\n{bar}"
    console.print(Panel.fit(
        body,
        border_style="bright_magenta",
        title=_rainbow_markup("Score = mean(epoch) + var(epoch)"),
        subtitle=_rainbow_markup("Меньше — лучше"),
    ))


def train_bench(
    train_ds, val_ds, num_classes: int,
    epochs: int = 10, batch_size: int = 128, workers: int = 4, lr: float = 1e-3,
    device_str: str = "auto", amp: bool = True, img_size: int = 224,
    console: Console | None = None,
) -> Dict[str, Any]:
    console = console or Console()
    device = _resolve_device(device_str)

    model = make_resnet18(num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    pin_memory = (device.type == "cuda")

    eff_workers = workers
    if _should_disable_workers(workers):
        console.print("[yellow]Низкий лимит /dev/shm — переключаюсь на num_workers=0[/yellow]")
        eff_workers = 0

    loader_kwargs = dict(batch_size=batch_size,
                         num_workers=eff_workers,
                         pin_memory=pin_memory,
                         persistent_workers=(eff_workers > 0))
    if eff_workers > 0:
        loader_kwargs.update(prefetch_factor=2,
                             worker_init_fn=_silence_worker_stdout)

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    total_start = perf_counter()
    best_acc = 0.0
    epoch_times: list[float] = []
    total_samples = 0

    progress_columns = [
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = perf_counter()
        with Progress(*progress_columns, transient=True, console=console) as prog:
            task = prog.add_task(f"Epoch {epoch}/{epochs}", total=len(train_loader))
            for x, y in train_loader:
                total_samples += x.size(0)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                if amp and device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    opt.step()
                prog.update(task, advance=1)

        epoch_time = perf_counter() - t0
        epoch_times.append(epoch_time)
        acc = evaluate(model, val_loader, device, amp=amp)
        best_acc = max(best_acc, acc)
        console.print(f"[green]✓[/green] Epoch {epoch} finished: "
                      f"time={epoch_time:.2f}s, val_acc={acc*100:.2f}%")

    total_time = perf_counter() - total_start
    throughput = total_samples / total_time if total_time > 0 else 0.0

    # --- Новый скор: mean(epoch_time) + var(epoch_time) ---
    mean_epoch = sum(epoch_times) / max(1, len(epoch_times))
    var_epoch = statistics.pvariance(epoch_times) if len(epoch_times) > 1 else 0.0
    score_time_var = mean_epoch + var_epoch  # секунды + секунды^2 (меньше — лучше)

    print_score_box(score_time_var, console)

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "lr": lr,
        "device": str(device),
        "amp": bool(amp and device.type == "cuda"),
        "img_size": img_size,
        "total_time_s": round(total_time, 3),
        "avg_epoch_time_s": round(mean_epoch, 3),
        "epoch_time_var_s2": round(var_epoch, 6),
        "best_val_acc": round(best_acc, 4),
        "train_samples": total_samples,
        "throughput_img_s": round(throughput, 2),
        "score_time_var_s": round(score_time_var, 3),
    }


def print_results_table(sysinfo: Dict[str, Any], results: Dict[str, Any],
                        console: Console | None = None):
    console = console or Console()
    t = Table(title="PyResBench — Training Benchmark", show_lines=False)
    t.add_column("Metric", style="cyan", no_wrap=True)
    t.add_column("Value", style="white")
    rows = [
        ("OS", sysinfo.get("os", "")),
        ("Python", sysinfo.get("python", "")),
        ("Torch", sysinfo.get("torch", "")),
        ("CUDA avail", str(sysinfo.get("cuda_available", ""))),
        ("CUDA", str(sysinfo.get("cuda_version", ""))),
        ("cuDNN", str(sysinfo.get("cudnn", ""))),
        ("GPU", str(sysinfo.get("gpu_name", ""))),
        ("CPU", f'{sysinfo.get("cpu","")} ({sysinfo.get("cpu_count_logical","")} threads)'),
        ("RAM (GB)", str(sysinfo.get("ram_total_gb",""))),
        ("Disk free (GB)", str(sysinfo.get("disk_free_gb",""))),
        ("----", "----"),
        ("Device", results["device"]),
        ("AMP", str(results["amp"])),
        ("Epochs", str(results["epochs"])),
        ("Batch", str(results["batch_size"])),
        ("Workers", str(results["workers"])),
        ("LR", str(results["lr"])),
        ("Img size", str(results["img_size"])),
        ("Total time (s)", f'[bold]{results["total_time_s"]}[/bold]'),
        ("Avg epoch (s)", f'{results["avg_epoch_time_s"]}'),
        ("Epoch time var (s²)", f'{results["epoch_time_var_s2"]}'),
        ("Throughput (img/s)", f'[bold green]{results["throughput_img_s"]}[/bold green]'),
        ("Best Val Acc", f'[bold magenta]{results["best_val_acc"]*100:.2f}%[/bold magenta]'),
        ("Score = mean + var (s)", f'[bold]{results.get("score_time_var_s", 0):.3f}[/bold]'),
    ]
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)
