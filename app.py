from flask import Flask, request, render_template
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F  # For softmax

app = Flask(__name__)

# CIFAR-100 class names list
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (ResNet-18) and modify the final layer for CIFAR-100
model = models.resnet18(weights=None)  # Initialize a fresh ResNet18 model
model.fc = torch.nn.Linear(model.fc.in_features, 100)  # Adjust final layer for CIFAR-100
model.load_state_dict(torch.load('D:/JENE 161/SEM III/AI/mini project/cifar100_model.pth', map_location=device))  # Load weights
model = model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the uploaded image
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')  # Ensure RGB format

        # Preprocess the image
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension

        # Run the model and get prediction
        with torch.no_grad():
            outputs = model(img)
            
            # Apply softmax to get probabilities (confidence scores)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the predicted class and its probability
            confidence, predicted = torch.max(probabilities, 1)
            class_idx = predicted.item()
            class_name = CIFAR100_CLASSES[class_idx]
            confidence_score = confidence.item() * 100  # Convert to percentage

        # Render result with prediction and confidence
        return render_template('index.html', prediction=class_name, confidence=round(confidence_score, 2))

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction="Error", confidence=0.0)

if __name__ == "__main__":
    app.run(debug=True)
.0
