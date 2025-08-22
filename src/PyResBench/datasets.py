from typing import Tuple, Optional
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import utils as tvu
import torch
import os, contextlib, urllib.request, requests
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TransferSpeedColumn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _common_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, test_tf

class OxfordPetsBinary(datasets.OxfordIIITPet):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        path = self._images[index]
        name = path.name.lower()
        is_cat = ("cat" in name)
        y = 0 if is_cat else 1
        return img, y

# ---------- Rich download patches ----------

def _progress_columns():
    return [
        TextColumn("[bold blue]Скачивание[/bold blue] {task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

@contextlib.contextmanager
def _rich_torchvision_download(console: Optional[Console]):
    """
    Патчим и tvu.download_url, и tvu.download_file_from_google_drive, чтобы рисовать Rich‑прогресс.
    Если console=None — просто глушим stdout/stderr torchvision (старое поведение без прогресса).
    """
    from torchvision.datasets import utils as tvu
    if console is None:
        # Тихий режим: глушим проценты torchvision
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            yield
        return

    # --- НИЧЕГО НЕ ГЛУШИМ, когда console есть! Иначе Rich не видно. ---
    orig_download_url = tvu.download_url
    orig_download_gdrv = tvu.download_file_from_google_drive

    def _rich_download_url(url: str, root: str, filename: Optional[str] = None,
                           md5: Optional[str] = None, max_redirect_hops: int = 3) -> None:
        os.makedirs(root, exist_ok=True)
        if filename is None:
            filename = os.path.basename(url) or "file"
        fpath = os.path.join(root, filename)

        if os.path.isfile(fpath) and (md5 is None or tvu.check_md5(fpath, md5)):
            return

        columns = [
            TextColumn("[bold blue]Скачивание[/bold blue] {task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        with Progress(*columns, console=console, transient=True) as prog:
            task = prog.add_task(filename, total=None)

            def _hook(blocknum, blocksize, totalsize):
                if totalsize > 0:
                    prog.update(task, total=totalsize)
                prog.update(task, advance=blocksize)

            urllib.request.urlretrieve(url, fpath, reporthook=_hook)

        if md5 is not None and not tvu.check_md5(fpath, md5):
            try: os.remove(fpath)
            except Exception: pass
            raise RuntimeError(f"MD5 mismatch for {filename}")

    def _rich_download_gdrive(file_id: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None):
        import requests
        os.makedirs(root, exist_ok=True)
        if filename is None:
            filename = f"gdrive_{file_id}"
        fpath = os.path.join(root, filename)

        if os.path.isfile(fpath) and (md5 is None or tvu.check_md5(fpath, md5)):
            return

        session = requests.Session()
        URL = "https://docs.google.com/uc?export=download"
        response = session.get(URL, params={"id": file_id}, stream=True)
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                response = session.get(URL, params={"id": file_id, "confirm": v}, stream=True)
                break

        total = int(response.headers.get("Content-Length", "0")) or None
        columns = [
            TextColumn("[bold blue]Скачивание[/bold blue] {task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        with Progress(*columns, console=console, transient=True) as prog:
            task = prog.add_task(filename, total=total)
            with open(fpath, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    prog.update(task, advance=len(chunk))

        if md5 is not None and not tvu.check_md5(fpath, md5):
            try: os.remove(fpath)
            except Exception: pass
            raise RuntimeError(f"MD5 mismatch for {filename}")

    # Применяем патчи
    tvu.download_url = _rich_download_url
    tvu.download_file_from_google_drive = _rich_download_gdrive
    try:
        # БЕЗ redirect_stdout/redirect_stderr!
        yield
    finally:
        tvu.download_url = orig_download_url
        tvu.download_file_from_google_drive = orig_download_gdrv


# ----------------------------------------------------

def get_dataset(name: str, data_dir: str = "./data", img_size=224, seed=42,
                console: Optional[Console] = None) -> Tuple[Dataset, Dataset, int]:
    name = name.lower()
    train_tf, test_tf = _common_transforms(img_size)
    g = torch.Generator().manual_seed(seed)

    with _rich_torchvision_download(console):
        if name in ("pets", "catsdogs", "cats_dogs", "oxfordpets"):
            full = OxfordPetsBinary(root=data_dir, split="trainval", target_types="category",
                                    download=True, transform=train_tf)
            val_len = max(500, int(0.1 * len(full)))
            train_len = len(full) - val_len
            train_ds, val_ds = random_split(full, [train_len, val_len], generator=g)
            val_ds.dataset.transform = test_tf
            num_classes = 2

        elif name == "cifar10":
            train_ds = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_tf)
            val_ds   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
            num_classes = 10

        elif name == "stl10":
            train_ds = datasets.STL10(root=data_dir, split="train", download=True, transform=train_tf)
            val_ds   = datasets.STL10(root=data_dir, split="test",  download=True, transform=test_tf)
            num_classes = 10

        elif name == "synthetic":
            class Synth(Dataset):
                def __init__(self, n, c=3, h=224, w=224, k=10):
                    self.x = torch.randn(n, c, h, w)
                    self.y = torch.randint(0, k, (n,))
                    self.k = k
                def __len__(self): return len(self.y)
                def __getitem__(self, i): return self.x[i], int(self.y[i])
            train_ds, val_ds = Synth(5000), Synth(1000)
            num_classes = 10
        else:
            raise ValueError(f"Unknown dataset: {name}")

    return train_ds, val_ds, num_classes
