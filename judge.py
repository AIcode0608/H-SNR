import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load the pretrained model
model = models.resnet50(pretrained=True)

# Preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example input image
input_image = Image.open("1.jpg")
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# Set the model to evaluation mode
model.eval()

# Perform inference using the model
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class index
_, predicted_idx = torch.max(output, 1)

# Load class labels
with open("synset_words.txt") as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]

# Print the predicted result
print("Prediction: ", labels[predicted_idx.item()])