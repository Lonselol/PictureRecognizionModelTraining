import tensorflow as tf
from tensorflow.keras.applications import MobileNet
import coremltools as ct
import os

def main():
  # Загрузка предобученной модели MobileNet
  model = MobileNet(weights='imagenet')

  output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
  output_path = os.path.join(output_dir, 'MobileNet.mlmodel')

  # Конвертация модели в формат Core ML
  coreml_model = ct.convert(model)
  coreml_model.save(output_path)

if __name__ == "__main.py__":
  main()
