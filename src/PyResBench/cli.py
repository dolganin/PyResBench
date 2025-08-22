import argparse
from pathlib import Path
from rich.console import Console
from .system_info import collect_system_info
from .datasets import get_dataset
from .train import train_bench, print_results_table
from .utils import dump_json, append_history_csv, HOME_DIR
from .welcome import render_welcome  # <-- NEW

def main():
    p = argparse.ArgumentParser(description="Benchmark your system by training a small ResNet.")
    p.add_argument("--dataset", default="pets", choices=["pets","cifar10","stl10","synthetic"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", default="auto", help="'auto'|'cuda'|'cpu'")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--json-out", default=None)
    p.add_argument("--csv-out", default=None)
    args = p.parse_args()

    console = Console()

    # Быстрая системная инфа для welcome
    sysinfo = collect_system_info()
    render_welcome(console, sysinfo, args, args.data_dir)

    # Пока готовим датасет — показываем спиннер
    with console.status("[bold]Подготовка датасета…[/bold]"):
        train_ds, val_ds, num_classes = get_dataset(
            args.dataset, args.data_dir, img_size=args.img_size, console=console  # <— добавили console
        )

    # Без статуса поверх прогресса — просто разделитель
    console.rule("[bold]Тренировка[/bold]")
    results = train_bench(
        train_ds, val_ds, num_classes=num_classes,
        epochs=args.epochs, batch_size=args.batch_size,
        workers=args.workers, lr=args.lr,
        device_str=args.device, amp=not args.no_amp, img_size=args.img_size,
        console=console,  # одна консоль на весь ран
    )
    results["dataset"] = args.dataset

    print_results_table(sysinfo, results, console=console)

    last = {"system": sysinfo, "results": results}
    dump_json(last, HOME_DIR / "last_run.json")

    if args.json_out:
        dump_json(last, Path(args.json_out))
        console.print(f"[green]Saved JSON to[/green] {args.json_out}")

    row = {
        "dataset": args.dataset,
        "epochs": results["epochs"],
        "batch": results["batch_size"],
        "workers": results["workers"],
        "device": results["device"],
        "amp": results["amp"],
        "total_time_s": results["total_time_s"],
        "avg_epoch_time_s": results["avg_epoch_time_s"],
        "throughput_img_s": results["throughput_img_s"],
        "best_val_acc": results["best_val_acc"],
    }
    append_history_csv(row)
    if args.csv_out:
        from csv import DictWriter
        path = Path(args.csv_out)
        exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = DictWriter(f, fieldnames=row.keys())
            if not exists: w.writeheader()
            w.writerow(row)
        console.print(f"[green]Appended CSV to[/green] {args.csv_out}")
