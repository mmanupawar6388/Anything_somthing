from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from src.model import SimpleCNN

app = Flask(__name__)

# CIFAR-10 class names
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN()
model_path = './checkpoints/cifar_net.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path}")
else:
    print(f"Warning: Model not found at {model_path}. Please train the model first.")

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return predictions"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        # Read and preprocess the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top prediction
        top_prob, top_class = torch.max(probabilities, 0)
        
        # Prepare all predictions
        all_predictions = {
            CLASSES[i]: float(probabilities[i]) 
            for i in range(len(CLASSES))
        }
        
        # Sort by confidence
        sorted_predictions = dict(sorted(all_predictions.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True))
        
        return jsonify({
            'success': True,
            'prediction': CLASSES[top_class],
            'confidence': float(top_prob),
            'all_predictions': sorted_predictions
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
