import torch
from torchvision import transforms, models
from PIL import Image
import sys
import os

def inference(image_path, model_path='model.pth'):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Initialize model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    else:
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        return

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

    # Define attribute names
    attribute_names = ["Attr1", "Attr2", "Attr3", "Attr4"]
    
    # Thresholding (typical is 0.5)
    present_attributes = [name for name, prob in zip(attribute_names, probs) if prob > 0.5]
    
    print(f"Attributes present in {image_path}:")
    if present_attributes:
        print(", ".join(present_attributes))
    else:
        print("None")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
    else:
        inference(sys.argv[1])
