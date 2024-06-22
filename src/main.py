import os
import tensorflow as tf
from keras._tf_keras.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNet
import coremltools as ct

def main():
    # Загрузка предобученной модели MobileNet
    model = MobileNet(weights='imagenet')

    # Конвертация модели в формат Core ML
    coreml_model = ct.convert(model, source='tensorflow')

    # Определение пути для сохранения модели
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    output_path = os.path.join(output_dir, 'MobileNet.mlmodel')

    # Сохранение модели
    coreml_model.save(output_path)

if __name__ == "__main__":
    main()
