from typing import Tuple
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch

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
    # 37 пород; сведем к «cat» vs «dog»
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # в torchvision target: 0..36 породы, self._labels[index][1] — вид? (в старых версиях)
        # Универсально: определим по имени файла (pet_image_label.txt использует 'Cat'/'Dog')
        # В актуальном torchvision у датасета атрибут 'classes' — породы. Для бинаризации используем путь:
        path = self._images[index]
        name = path.name.lower()
        is_cat = ("cat" in name)  # имена включают породу; у кошек встречается 'cat_'
        y = 0 if is_cat else 1
        return img, y

def get_dataset(name: str, data_dir: str = "./data", img_size=224, seed=42) -> Tuple[Dataset, Dataset, int]:
    name = name.lower()
    train_tf, test_tf = _common_transforms(img_size)
    g = torch.Generator().manual_seed(seed)

    if name in ("pets", "catsdogs", "cats_dogs", "oxfordpets"):
        full = OxfordPetsBinary(root=data_dir, split="trainval", target_types="category", download=True, transform=train_tf)
        # валидацию отделим из того же сплита
        val_len = max(500, int(0.1 * len(full)))
        train_len = len(full) - val_len
        train_ds, val_ds = random_split(full, [train_len, val_len], generator=g)
        # для val заменим трансформ
        val_ds.dataset.transform = test_tf
        num_classes = 2

    elif name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
        num_classes = 10

    elif name == "stl10":
        train_ds = datasets.STL10(root=data_dir, split="train", download=True, transform=train_tf)
        val_ds   = datasets.STL10(root=data_dir, split="test",  download=True, transform=test_tf)
        num_classes = 10

    elif name == "synthetic":
        # Быстрый синтетический набор
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
