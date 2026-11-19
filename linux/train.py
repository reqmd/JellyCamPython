import argparse
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
    """Вписывает RGB/PIL изображение в (width, height) без искажения пропорций.

    width=height сохраняет прежний препроцессинг 32x32. При width!=height
    масштаб выбирается по ОБЕИМ сторонам: вертикальные зёрна не обрезаются.
    """

    def __init__(self, width: int, height: int | None = None, fill=FILL):
        self.width = int(width)
        self.height = int(height if height is not None else width)
        if self.width <= 0 or self.height <= 0:
            raise ValueError('ширина и высота входа должны быть положительными')
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError('пустое изображение')
        scale = min(self.width / w, self.height / h)
        new_w = min(self.width, max(1, round(w * scale)))
        new_h = min(self.height, max(1, round(h * scale)))
        # cv2.INTER_LINEAR: тот же алгоритм интерполяции, что в sorter.py,
        # test.py и convert.py; PIL.BILINEAR при уменьшении даёт другие пиксели.
        rgb = np.asarray(img.convert('RGB'))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.height, self.width, 3), self.fill, dtype=np.uint8)
        x0, y0 = (self.width - new_w) // 2, (self.height - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return Image.fromarray(canvas)


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


def return_transforms(width: int, height: int | None = None):
    """У train/val и RKNN одинаковые размеры, fill и нормализация.

    Для 32x32 старый порядок аугментаций оставлен без изменений.
    Для прямоугольника поворот выполняется ДО вписывания с expand=True:
    иначе при повороте на 90 градусов часть зерна обрезается.
    """
    height = int(height if height is not None else width)
    if width == height:
        train_ops = [
            transforms.ToPILImage(),
            ResizePad(width, height),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=FILL),
            transforms.RandomRotation(180, fill=FILL),
            transforms.ToTensor(),
        ]
    else:
        train_ops = [
            transforms.ToPILImage(),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3, fill=FILL),
            transforms.RandomRotation(180, expand=True, fill=FILL),
            ResizePad(width, height),
            transforms.ToTensor(),
        ]
    train_transform = transforms.Compose(train_ops)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        ResizePad(width, height),
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
    parser = argparse.ArgumentParser(description='Обучение / экспорт TinyCNN для RKNN')
    parser.add_argument('--width', type=int, default=32, help='ширина, по умолчанию 32')
    parser.add_argument('--height', type=int, default=32, help='высота, по умолчанию 32')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--weights', default=None,
                        help='существующий .pth: инициализация обучения или экспорт')
    parser.add_argument('--export-only', action='store_true',
                        help='только экспорт весов из --weights, без обучения')
    parser.add_argument('--config-out', default=None,
                        help='путь YAML; по умолчанию models.yaml для 32x32, '
                             'models_ШИРИНАxВЫСОТА.yaml для прямоугольника')
    args = parser.parse_args()
    width, height = args.width, args.height
    if width <= 0 or height <= 0 or args.epochs < 1:
        parser.error('width, height и epochs должны быть положительными')
    if args.export_only and not args.weights:
        parser.error('--export-only требует --weights /path/to/model.pth')

    train_data_path = 'linux/data/data_for_nn/train'
    val_data_path = 'linux/data/data_for_nn/test'
    model_path = 'linux/models'
    batch_size = 8
    config_out = args.config_out or (
        'models.yaml' if width == height == 32 else f'models_{width}x{height}.yaml'
    )

    # Список классов обязателен также при экспорте готовых весов.
    train_transform, val_transform = return_transforms(width, height)
    train_dataset = LabeledDataset(train_data_path, transform=train_transform)
    print(f'классы: {train_dataset.class_to_idx}')
    device = 'cuda' if torch.cuda.is_available() and not args.export_only else 'cpu'
    model = tiny_cnn_s(num_classes=len(train_dataset.classes)).to(device)
    if args.weights:
        # Веса должны относиться к той же архитектуре и тем же классам.
        state = torch.load(args.weights, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f'загружены веса: {args.weights}')

    best_loss = None
    if args.export_only:
        best_model = model
        print(f'Экспорт без обучения: вход {width}x{height}')
    else:
        val_dataset = LabeledDataset(val_data_path, transform=val_transform)
        if val_dataset.classes != train_dataset.classes:
            raise RuntimeError('классы в train и test не совпадают')
        print(f'train: {len(train_dataset)}, val: {len(val_dataset)}')
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, pin_memory=True)
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        best_loss = float('inf')
        best_model = copy.deepcopy(model)
        for epoch in range(args.epochs):
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
            print(f'Epoch {epoch + 1}/{args.epochs}  train {np.mean(train_loss):.4f}  '
                  f'val {val_loss:.4f}  F1 {f1_sc:.4f}{marker}')

    best_model.eval().cpu()
    stamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    suffix = '' if width == height == 32 else f'_{width}x{height}'
    os.makedirs(model_path, exist_ok=True)
    os.makedirs('onnx', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    torch_path = os.path.join(model_path, f'model_{stamp}{suffix}.pth')
    onnx_path = os.path.join('onnx', f'model_{stamp}{suffix}.onnx')
    rknn_path = os.path.join('output', f'model_{stamp}{suffix}.rknn')
    torch.save(best_model.state_dict(), torch_path)

    dummy_input = torch.randn(1, 3, height, width)
    export_kwargs = dict(input_names=['input'], output_names=['output'],
                         opset_version=12, do_constant_folding=True,
                         dynamic_axes=None)
    try:
        torch.onnx.export(best_model, dummy_input, onnx_path,
                          dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(best_model, dummy_input, onnx_path, **export_kwargs)
    print('ONNX сохранён:', onnx_path)

    models_info = {
        'torch': torch_path,
        'onnx': onnx_path,
        'rknn': rknn_path,
        'input_size': [1, 3, height, width],
        'mean_values': [[0, 0, 0]],
        'std_values': [[255, 255, 255]],
        'classes': train_dataset.classes,
        'best_val_loss': best_loss,
    }
    with open(config_out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(models_info, f, allow_unicode=True, sort_keys=False)
    print(f'{config_out} обновлён; вход: {width}x{height}')


if __name__ == '__main__':
    main()