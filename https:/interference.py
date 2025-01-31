import cv2
import mediapipe as mp
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# Пути к модели и меткам классов
MODEL_PATH = "models/sign_model.pth"
LABELS = ["А", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я"] 

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Загрузка обученной модели
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(1280, len(LABELS))  # Меняем выходной слой
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Преобразование изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Запуск веб-камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Конвертируем изображение в RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Преобразуем координаты в тензор
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            input_tensor = torch.tensor(landmarks).unsqueeze(0)

            # Предсказание жеста
            prediction = model(input_tensor)
            predicted_label = torch.argmax(prediction, dim=1).item()

            # Вывод результата
            cv2.putText(frame, f'Жест: {LABELS[predicted_label]}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Распознавание жестов', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
