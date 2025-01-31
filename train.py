import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

# Гиперпараметры
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001
DATASET_PATH = "dataset/russian_sign_language"  # Путь к распакованному датасету
MODEL_SAVE_PATH = "models/sign_model.pth"

# Проверяем доступность GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Преобразование изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Загрузка датасета
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Загрузка предобученной модели MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(1280, len(dataset.classes))  # Меняем последний слой под наш датасет
model = model.to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Обучение модели
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Эпоха [{epoch+1}/{EPOCHS}], Потери: {running_loss/len(dataloader):.4f}")

# Сохранение обученной модели
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Модель сохранена в {MODEL_SAVE_PATH}")
