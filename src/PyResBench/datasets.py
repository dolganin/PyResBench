from typing import Tuple, Optional
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import torch
import os, contextlib, hashlib, urllib.request
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TransferSpeedColumn
from torchvision.datasets import utils as tvu  # для монкипатча

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

# ---------- Красивое скачивание через Rich ----------

@contextlib.contextmanager
def _rich_torchvision_download(console: Optional[Console]):
    """
    Патчим torchvision.datasets.utils.download_url, чтобы рисовать Rich-прогресс.
    Возвращаем всё назад по выходу из контекста.
    """
    if console is None:
        yield
        return

    orig_download_url = tvu.download_url

    def _rich_download_url(url: str, root: str, filename: Optional[str] = None,
                           md5: Optional[str] = None, max_redirect_hops: int = 3) -> None:
        os.makedirs(root, exist_ok=True)
        if filename is None:
            filename = os.path.basename(url)
            if not filename:
                filename = "file"
        fpath = os.path.join(root, filename)

        # если уже есть и md5 совпадает — просто выходим
        if os.path.isfile(fpath) and (md5 is None or tvu.check_md5(fpath, md5)):
            return

        # Качаем с прогрессом
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

            # urlretrieve сам дергает hook; он же пишет файл
            tmp_path, headers = urllib.request.urlretrieve(url, fpath, reporthook=_hook)  # noqa: F841

        # Проверка MD5 (если нужен)
        if md5 is not None and not tvu.check_md5(fpath, md5):
            try:
                os.remove(fpath)
            except Exception:
                pass
            raise RuntimeError(f"MD5 mismatch for {filename}")

    # Патчим
    tvu.download_url = _rich_download_url
    try:
        yield
    finally:
        # Возвращаем исходную функцию
        tvu.download_url = orig_download_url

# ----------------------------------------------------

def get_dataset(name: str, data_dir: str = "./data", img_size=224, seed=42,
                console: Optional[Console] = None) -> Tuple[Dataset, Dataset, int]:
    """
    ВАЖНО: теперь можно передать console для красивого прогресса скачивания.
    """
    name = name.lower()
    train_tf, test_tf = _common_transforms(img_size)
    g = torch.Generator().manual_seed(seed)

    # перехватываем скачивание внутри этого блока
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

    # Небольшой спиннер на распаковке (если torchvision её делает внутри) у нас уже не нужен,
    # так как мы перехватили именно download_url; распаковка обычно тихая.

    return train_ds, val_ds, num_classes
