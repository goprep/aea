import torch
import torchvision
from torchvision.models import resnet18

# Load the pre-trained ResNet18 model
model = resnet18(pretrained=True)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(model)
