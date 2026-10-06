import torch
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import f1_score as F1
import datetime
import copy
import yaml

from model import MobileNet

def get_folders(path):
    return [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]

class LabeledDataset(Dataset):
    def __init__(self, root, transform = None, test=False, hsv_with_rgb = False):
        self.root = root
        self.hsv_with_rgb = hsv_with_rgb
        self.classes = get_folders(root)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.transfrom = transform
        self.images = []
        self.labels = []
        self.root_images = []
        self.image_name = ''
        for class_name in self.classes:
            self.class_root = os.path.join(self.root, class_name)
            for image_name in os.listdir(self.class_root):
                self.images.append(image_name)
                self.labels.append(self.class_to_idx[class_name])
                if test == True:
                    self.root_images.append(os.path.join(self.class_root, image_name))
                
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = cv2.imread(os.path.join(self.root, self.classes[label], self.images[idx]), flags=cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_name = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        if self.transfrom is not None:
            image = self.transfrom(image)  

        if self.hsv_with_rgb:
            rgb_numpy = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            hsv = cv2.cvtColor(rgb_numpy, cv2.COLOR_RGB2HSV)
            hsv = torch.FloatTensor(hsv).permute(2, 0, 1) / 255.0
            combined = torch.cat([image, hsv], dim=0)
            return combined, label
        else:
            return image, label

def return_transforms(resolutions):
    H, W = resolutions
    train_transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.RandomRotation(180),
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((H, W)),
        transforms.ToTensor()
    ])
    return train_transform, val_transform

train_data_path = 'linux/data/data_for_nn/train'
val_data_path = 'linux/data/data_for_nn/test'
train_transform, val_transform = return_transforms(resolutions=[224 ,224])

train_dataset = LabeledDataset(train_data_path, transform=train_transform)
val_dataset = LabeledDataset(val_data_path, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)

if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'
model = MobileNet(alpha = 0.25, num_classes=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(params = model.parameters(), lr = 0.001, weight_decay=0.0001)
epochs = 20
date = datetime.datetime.now()
model_path = 'linux/models'
formatted_string_ru = date.strftime("%d-%m-%Y_%H-%M-%S")
model_name = f"model_{formatted_string_ru}.pth"

min_loss = 100

for epoch in range(epochs):
    train_loss = []
    model.train()
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        optim.zero_grad()
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()
        train_loss.append(loss.cpu().detach().numpy())
    print(f'Epochs: {epoch+1}/{epochs}.....Loss: {np.mean(train_loss)}')

    test_loss = []
    y_preds = []
    y_trues = []
    

    model.eval()
    for X, y in val_loader:
        X = X.to(device)
        y = y.to(device)
        y_trues.extend(y.cpu().numpy())
        y_pred = model(X)
        val_loss = loss_fn(y_pred, y)
        test_loss.append(val_loss.cpu().detach().numpy())
        y_pred = torch.argmax(y_pred, dim=1)
        y_preds.extend(y_pred.cpu().numpy())
    f1_sc = F1(y_trues, y_preds, average='weighted')
    if min_loss > np.mean(test_loss):
        min_loss = np.mean(test_loss)
        model_to_save = copy.deepcopy(model)
    print(f'F1: {f1_sc:.6f}, Test Loss {np.mean(test_loss)}\n')
torch_path = os.path.join(model_path, model_name)
torch.save(model.state_dict(), torch_path)

if not os.path.exists('onnx'):
    os.mkdir('onnx')

if not os.path.exists('rknn'):
    os.mkdir('rknn')

model_to_save.eval()
model_to_save.cpu()

dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = os.path.join('onnx', f"model_{formatted_string_ru}.onnx")
rknn_path = os.path.join('rknn', f"model_{formatted_string_ru}.rknn")
models_info = {'torch':torch_path, 'onnx':onnx_path, 'rknn':rknn_path}
torch.onnx.export(model_to_save, dummy_input, onnx_path, input_names=['input'], output_names=['output'], opset_version=12, dynamic_axes=None)
print("ONNX сохранён:", onnx_path)
with open ('models.yaml', 'w') as file:
    yaml.dump(models_info, file, allow_unicode = True)
