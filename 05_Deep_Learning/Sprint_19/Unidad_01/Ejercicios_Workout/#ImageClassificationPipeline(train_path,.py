#ImageClassificationPipeline(train_path, test_path, image_size=(64, 64))

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.io import imread
from sklearn.utils import shuffle
import keras
import cv2

class ImageClassificationPipeline:
    def __init__(self, train_path, test_path, image_size=(64, 64)):
        self.train_path = train_path
        self.test_path = test_path
        self.image_size = image_size
        self.model = None
        self.history = None
        self.map = None

    def read_data(self, directorio):
        X = []
        y = []
        for folder in os.listdir(directorio):
            if os.path.isdir(os.path.join(directorio, folder)):
                for file in os.listdir(os.path.join(directorio, folder)):
                    image = imread(os.path.join(directorio, folder, file))
                    image = cv2.resize(image, self.image_size)  # Redimensionamos las imágenes
                    X.append(image)
                    y.append(folder)
        return np.array(X), np.array(y)

    def show_images_batch(self, paisajes, names=[], n_cols=5, size_scale=2):
        n_rows = ((len(paisajes) - 1) // n_cols + 1)
        plt.figure(figsize=(n_cols * size_scale, n_rows * 1.1 * size_scale))
        for index, paisaje in enumerate(paisajes):
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(paisaje, cmap="Greys")
            plt.axis("off")
            if len(names):
                plt.title(names[index])
        plt.show()

    def build_model(self):
        capas = [keras.layers.Conv2D(filters=30, kernel_size=(3, 3), input_shape=(self.image_size[0], self.image_size[1], 3), padding="same", activation='relu'),
                 keras.layers.MaxPooling2D(pool_size=(2, 2)),
                 keras.layers.Conv2D(filters=15, kernel_size=(3, 3), padding="same", activation='relu'),
                 keras.layers.MaxPooling2D(pool_size=(2, 2)),
                 keras.layers.Flatten(),
                 keras.layers.Dense(units=150, activation='relu'),
                 keras.layers.Dense(units=100, activation='relu'),
                 keras.layers.Dense(6, activation='softmax')]

        self.model = keras.Sequential(capas)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train_model(self, X_train, y_train):
        self.history = self.model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, 
                                      callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    def evaluate_model(self, X_test, y_test):
        self.model.evaluate(X_test, y_test)
        y_pred = [np.argmax(prediction) for prediction in self.model.predict(X_test)]

        inverse_map = {valor: clave for clave, valor in self.map.items()}
        y_test_labels = [inverse_map[y] for y in y_test]
        y_pred_labels = [inverse_map[y] for y in y_pred]

        print(classification_report(y_test_labels, y_pred_labels))
        ConfusionMatrixDisplay.from_predictions(y_test_labels, y_pred_labels, normalize="true")
        plt.show()

        confianza = [prediction.max() for prediction in self.model.predict(X_test)]
        pred_df = pd.DataFrame({"True": y_test_labels, "Predicted": y_pred_labels, "Confianza": confianza})
        error = pred_df["True"] != pred_df["Predicted"]
        errores = pred_df[error].sort_values("Confianza", ascending=False)

        for true_label, predicted_label in errores.iloc[0:10, 0:2].values:
            print(true_label, predicted_label)

        self.show_images_batch(X_test[errores.iloc[0:10].index], 
                               names=[f"T:{val1}, P:{val2}, C:{round(val3, 2)}%" for val1, val2, val3 in errores.iloc[0:10].values], 
                               size_scale=4)

    def run(self):
        # Leer los datos
        X_train, y_train = self.read_data(self.train_path)
        X_test, y_test = self.read_data(self.test_path)

        # Mostrar algunas imágenes
        indices = np.random.randint(0, len(X_train), 20)
        self.show_images_batch(X_train[indices], names=y_train[indices], n_cols=5)

        # Normalizar las imágenes
        X_train = X_train / 255
        X_test = X_test / 255

        # Construir el modelo
        self.build_model()

        # Convertir etiquetas a valores numéricos
        targets = pd.Series(y_train)
        self.map = {tipo: indice for indice, tipo in enumerate(targets.unique())}
        y_train_num = np.array([self.map[y] for y in y_train])
        y_test_num = np.array([self.map[y] for y in y_test])

        # Mezclar los datos
        X_train, y_train_num = shuffle(X_train, y_train_num)

        # Entrenar el modelo
        self.train_model(X_train, y_train_num)

        # Representar resultados
        history_df = pd.DataFrame(self.history.history)
        plt.plot(history_df['accuracy'], label='Accuracy')
        plt.plot(history_df['val_accuracy'], label='Val_accuracy')
        plt.legend()
        plt.show()

        plt.plot(history_df['loss'], label='Loss')
        plt.plot(history_df['val_loss'], label='Val_loss')
        plt.legend()
        plt.show()

        # Evaluar el modelo
        self.evaluate_model(X_test, y_test_num)

# Uso del pipeline
train_path = 'ruta_a_tus_datos_de_entrenamiento'
test_path = 'ruta_a_tus_datos_de_prueba'
pipeline = ImageClassificationPipeline(train_path, test_path, image_size=(64, 64))
pipeline.run()
