import torchvision
import torch
import coremltools as ct 
import os

def main():
    # Load a pre-trained version of MobileNetV2
    torch_model = torchvision.models.mobilenet_v2(pretrained=True)
    # Set the model in evaluation mode.
    torch_model.eval()

    # Trace the model with random data.
    example_input = torch.rand(1, 3, 224, 224) 
    traced_model = torch.jit.trace(torch_model, example_input)
    out = traced_model(example_input)

    # Using image_input in the inputs parameter:
    # Convert to Core ML using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)]
     )

    # Определение пути для сохранения модели
    output_path = 'MobileNet.mlmodel'
    #output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    #output_path = os.path.join(output_dir, 'MobileNet.mlmodel')

    # Сохранение модели
    model.save(output_path)

if __name__ == "__main__":
    main()