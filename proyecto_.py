# -*- coding: utf-8 -*-


!pip install tensorflow

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras import backend as K


# Clear the TensorFlow session to avoid cache issues
K.clear_session()


def preprocess_image(image, size=(256, 256)):

    image = cv2.resize(image, size)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    return cv2.cvtColor(blurred, cv2.COLOR_HSV2RGB)

def load_images_from_folder(folder, size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = preprocess_image(img, size)
            images.append(img)
    return np.array(images, dtype=np.uint8)  # Convertir a uint8 para compatibilidad con OpenCV

endo_micorrizas = load_images_from_folder('/content/drive/MyDrive/Endomicorrizas')
ecto_micorrizas = load_images_from_folder('/content/drive/MyDrive/Ectomicorrizas')

labels_endo = np.zeros(len(endo_micorrizas))  # Etiqueta 0 para endo-micorrizas
labels_ecto = np.ones(len(ecto_micorrizas))   # Etiqueta 1 para ecto-micorrizas

# Concatenar las imágenes y las etiquetas
X = np.concatenate([endo_micorrizas, ecto_micorrizas], axis=0)
y = np.concatenate([labels_endo, labels_ecto], axis=0)

# Normalizar las imágenes entre 0 y 1 (manteniendo uint8 para OpenCV)
X = (X / 255.0).astype(np.float32)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=50,
                    callbacks=[early_stopping])


# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%")

# Predicciones en el conjunto de prueba
y_pred = np.argmax(model.predict(X_test), axis=1)

# Generar reporte de clasificación y matriz de confusión
print("Clasificación de cada clase:")
print(classification_report(y_test, y_pred, target_names=["Endo-micorrizas", "Ecto-micorrizas"]))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('micorriza_detector.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo convertido a TensorFlow Lite y guardado como 'micorriza_detector.tflite'.")

model.save('micorriza_detector.h5')
print("Modelo guardado como 'micorriza_detector.h5'.")

import matplotlib.pyplot as plt
#Grafico de perdida y precisión
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()


plt.tight_layout()
plt.show()