import os
from tensorflow.keras.api.applications.mobilenet import MobileNet
import coremltools as ct

def main():

    # Загрузка предобученной модели MobileNet
    model = MobileNet(weights='imagenet')

    # Конвертация модели в формат Core ML
    coreml_model = ct.convert(model, source='TensorFlow')

    # Определение пути для сохранения модели
    #output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    #output_path = os.path.join(output_dir, 'MobileNet.mlmodelM')

    # Сохранение модели
    coreml_model.save("MobileNet.mlmodel")

if __name__ == "__main__":
    main()
