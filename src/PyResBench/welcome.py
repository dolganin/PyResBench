from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
from rich.console import Console
from pathlib import Path
import json

from .utils import load_ascii_art, get_version_fallback, get_git_short_hash, random_tip, HOME_DIR

def _load_last_run():
    p = HOME_DIR / "last_run.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def render_welcome(console: Console, sysinfo: dict, args: object, data_dir: str):
    art = load_ascii_art()
    version = get_version_fallback()
    git = get_git_short_hash()
    ver_line = f"[bold]PyResBench[/bold] v{version}" + (f" ([dim]{git}[/dim])" if git else "")

    # Левая колонка: ASCII + версия
    left = Panel.fit(f"[cyan]{art}[/cyan]\n{ver_line}", border_style="cyan", title="PyResBench")

    # Правая верх: конфиг запуска
    cfg = Table.grid(padding=(0,1))
    cfg.add_row("[bold]Dataset[/bold]", str(args.dataset))
    cfg.add_row("[bold]Epochs[/bold]", str(args.epochs))
    cfg.add_row("[bold]Batch[/bold]", str(args.batch_size))
    cfg.add_row("[bold]Workers[/bold]", str(args.workers))
    cfg.add_row("[bold]Device[/bold]", args.device)
    cfg.add_row("[bold]AMP[/bold]", str(not args.no_amp))
    cfg.add_row("[bold]Data dir[/bold]", str(Path(data_dir).resolve()))
    cfg_panel = Panel(cfg, title="Run config", border_style="magenta")

    # Правая средняя: система
    sys = Table.grid(padding=(0,1))
    sys.add_row("[bold]OS[/bold]", str(sysinfo.get("os", "")))
    sys.add_row("[bold]Python[/bold]", str(sysinfo.get("python", "")))
    sys.add_row("[bold]Torch[/bold]", str(sysinfo.get("torch", "")))
    sys.add_row("[bold]CUDA avail[/bold]", str(sysinfo.get("cuda_available", "")))
    sys.add_row("[bold]CUDA[/bold]", str(sysinfo.get("cuda_version", "")))
    sys.add_row("[bold]GPU[/bold]", str(sysinfo.get("gpu_name", "")))
    sys_panel = Panel(sys, title="Environment", border_style="green")

    # Правая нижняя: прошлый прогон
    last = _load_last_run()
    if last:
        r = last.get("results", {})
        last_tbl = Table.grid(padding=(0,1))
        last_tbl.add_row("[bold]Prev dataset[/bold]", str(r.get("dataset","")))
        last_tbl.add_row("[bold]Throughput[/bold]", f'{r.get("throughput_img_s","")} img/s')
        last_tbl.add_row("[bold]Total time[/bold]", f'{r.get("total_time_s","")} s')
        last_tbl.add_row("[bold]Best val acc[/bold]", f'{float(r.get("best_val_acc",0))*100:.2f}%')
        last_panel = Panel(last_tbl, title="Last run", border_style="yellow")
    else:
        last_panel = Panel("[dim]Нет прошлых прогонов[/dim]", title="Last run", border_style="yellow")

    tip_panel = Panel(random_tip(), title="Hint", border_style="blue")

    right = Columns([cfg_panel, sys_panel, last_panel, tip_panel], expand=True)
    console.print(Columns([left, right], equal=True))
