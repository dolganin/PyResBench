from typing import Tuple
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import torch
import contextlib, os

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

@contextlib.contextmanager
def _silence_download():
    # Глушим весь спам из внутреннего downloader'а torchvision
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        yield

class OxfordPetsBinary(datasets.OxfordIIITPet):
    # 37 пород; сведем к «cat» vs «dog»
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        path = self._images[index]
        name = path.name.lower()
        is_cat = ("cat" in name)
        y = 0 if is_cat else 1
        return img, y

def get_dataset(name: str, data_dir: str = "./data", img_size=224, seed=42) -> Tuple[Dataset, Dataset, int]:
    name = name.lower()
    train_tf, test_tf = _common_transforms(img_size)
    g = torch.Generator().manual_seed(seed)

    if name in ("pets", "catsdogs", "cats_dogs", "oxfordpets"):
        with _silence_download():
            full = OxfordPetsBinary(root=data_dir, split="trainval", target_types="category",
                                    download=True, transform=train_tf)
        val_len = max(500, int(0.1 * len(full)))
        train_len = len(full) - val_len
        train_ds, val_ds = random_split(full, [train_len, val_len], generator=g)
        val_ds.dataset.transform = test_tf
        num_classes = 2

    elif name == "cifar10":
        with _silence_download():
            train_ds = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_tf)
            val_ds   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
        num_classes = 10

    elif name == "stl10":
        with _silence_download():
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
