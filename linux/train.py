import copy
import datetime
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import f1_score as F1
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tiny_cnn import tiny_cnn_s

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Цвет заливки для паддинга и поворотов. Один и тот же везде, иначе
# train и val получают разные артефакты по краям кадра.
FILL = (0, 0, 0)


def get_folders(path):
    return sorted(
        item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))
    )


class ResizePad:
    """Ресайз по длинной стороне с сохранением пропорций + паддинг до квадрата.

    В отличие от Resize((H, W)) не искажает соотношение сторон объекта,
    которое для вытянутых/круглых форм является признаком класса.
    """

    def __init__(self, size: int, fill=FILL):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new('RGB', (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas


class LabeledDataset(Dataset):
    def __init__(self, root, transform=None, return_path=False):
        self.root = root
        self.transform = transform
        self.return_path = return_path

        self.classes = get_folders(root)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Храним полные пути сразу — не собираем их заново в __getitem__.
        self.samples = []
        for class_name in self.classes:
            class_root = os.path.join(root, class_name)
            for image_name in sorted(os.listdir(class_root)):
                self.samples.append(
                    (os.path.join(class_root, image_name), self.class_to_idx[class_name])
                )

        if not self.samples:
            raise RuntimeError(f'не найдено изображений в {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'не читается: {path}')
        # Модель обучается на RGB -> на устройстве тоже кормить RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.int64)
        if self.return_path:
            return image, label, path
        return image, label


def return_transforms(size: int):
    """ВНИМАНИЕ: только ToTensor, без Normalize.

    Это напрямую определяет конфиг RKNN:
        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]]
    Если добавите Normalize — обязательно поправьте convert_rknn.py.
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        ResizePad(size),                       # сначала приводим к общему масштабу,
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=FILL),
        transforms.RandomRotation(180, fill=FILL),  # затем аугментируем — артефакты
        transforms.ToTensor(),                      # одинакового размера
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        ResizePad(size),
        transforms.ToTensor(),
    ])
    return train_transform, val_transform


def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, y_preds, y_trues = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            losses.append(loss_fn(out, y).item())
            y_trues.extend(y.cpu().numpy())
            y_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
    return float(np.mean(losses)), F1(y_trues, y_preds, average='weighted')


def main():
    size = 32
    train_data_path = 'linux/data/data_for_nn/train'
    val_data_path = 'linux/data/data_for_nn/test'
    model_path = 'linux/models'
    epochs = 20
    batch_size = 8

    train_transform, val_transform = return_transforms(size)
    train_dataset = LabeledDataset(train_data_path, transform=train_transform)
    val_dataset = LabeledDataset(val_data_path, transform=val_transform)
    print(f'классы: {train_dataset.class_to_idx}')
    print(f'train: {len(train_dataset)}, val: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = tiny_cnn_s(num_classes=len(train_dataset.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_loss = float('inf')
    best_model = copy.deepcopy(model)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        sched.step()

        val_loss, f1_sc = evaluate(model, val_loader, loss_fn, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            marker = ' *'
        else:
            marker = ''

        print(f'Epoch {epoch + 1}/{epochs}  train {np.mean(train_loss):.4f}  '
              f'val {val_loss:.4f}  F1 {f1_sc:.4f}{marker}')

    # --- экспорт: .pth и .onnx — ОДНА И ТА ЖЕ лучшая модель ---
    best_model.eval().cpu()

    stamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs('onnx', exist_ok=True)

    torch_path = os.path.join(model_path, f'model_{stamp}.pth')
    onnx_path = os.path.join('onnx', f'model_{stamp}.onnx')
    rknn_path = os.path.join('output', f'model_{stamp}.rknn')

    torch.save(best_model.state_dict(), torch_path)

    dummy_input = torch.randn(1, 3, size, size)
    export_kwargs = dict(
        input_names=['input'], output_names=['output'],
        opset_version=12, do_constant_folding=True, dynamic_axes=None,
    )
    try:
        # В torch >= 2.9 по умолчанию работает dynamo-экспортёр; rknn-toolkit2
        # заметно лучше переваривает граф старого TorchScript-экспортёра.
        torch.onnx.export(best_model, dummy_input, onnx_path,
                          dynamo=False, **export_kwargs)
    except TypeError:
        # старые версии torch параметра dynamo не знают
        torch.onnx.export(best_model, dummy_input, onnx_path, **export_kwargs)
    print('ONNX сохранён:', onnx_path)

    # Всё, что нужно конвертеру, включая геометрию входа и нормализацию.
    models_info = {
        'torch': torch_path,
        'onnx': onnx_path,
        'rknn': rknn_path,
        'input_size': [1, 3, size, size],
        'mean_values': [[0, 0, 0]],
        'std_values': [[255, 255, 255]],
        'classes': train_dataset.classes,
        'best_val_loss': best_loss,
    }
    with open('models.yaml', 'w') as f:
        yaml.dump(models_info, f, allow_unicode=True)
    print('models.yaml обновлён')


if __name__ == '__main__':
    main()