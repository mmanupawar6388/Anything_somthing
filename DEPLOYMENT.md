# Flask Web Application - Deployment Guide

## 🚀 Quick Start

The CIFAR-10 CNN model is now deployed as a web application! Users can upload images and get real-time predictions.

### Starting the Server

```bash
cd C:\Users\manup\.gemini\antigravity\scratch\cifar_cnn
python app.py
```

The server will start on:
- **Local**: http://127.0.0.1:5000
- **Network**: http://10.227.42.151:5000 (accessible from other devices on your network)

### Accessing the Application

1. Open your web browser
2. Navigate to `http://localhost:5000`
3. You'll see the CIFAR-10 Image Classifier interface

## 🎨 Features

### Modern Web Interface
- **Drag & Drop Upload**: Simply drag an image onto the upload area
- **Click to Browse**: Click the upload area to select a file
- **Image Preview**: See your uploaded image before prediction
- **Real-time Results**: Get instant predictions with confidence scores
- **Visual Feedback**: Beautiful gradient design with smooth animations

### Prediction Display
- **Top Prediction**: Large display of the most likely class with confidence
- **All Probabilities**: Bar chart showing confidence for all 10 classes
- **Sorted Results**: Classes ordered by confidence (highest first)

### Supported Classes
✈️ Plane | 🚗 Car | 🐦 Bird | 🐱 Cat | 🦌 Deer  
🐕 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚚 Truck

## 📂 Project Structure

```
cifar_cnn/
├── app.py                  # Flask application
├── templates/
│   └── index.html         # Web interface
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── main.js        # Client-side logic
├── src/                   # Model code
├── checkpoints/
│   └── cifar_net.pth      # Trained model
└── requirements.txt
```

## 🔧 Technical Details

### Backend (Flask)
- **Framework**: Flask
- **Model**: PyTorch CNN (74.64% accuracy)
- **Image Processing**: PIL (Pillow)
- **Endpoint**: `/predict` (POST)

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern gradients, animations, responsive design
- **JavaScript**: Vanilla JS (no frameworks)
- **AJAX**: Fetch API for async predictions

### API Endpoint

**POST /predict**

Request:
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

Response:
```json
{
  "success": true,
  "prediction": "car",
  "confidence": 0.909,
  "all_predictions": {
    "car": 0.909,
    "truck": 0.045,
    "plane": 0.023,
    ...
  }
}
```

## 🎯 Usage Examples

### Example 1: Upload a Car Image
1. Upload an image of a car
2. See prediction: "car" with ~90% confidence
3. View all class probabilities in the bar chart

### Example 2: Upload a Cat Image
1. Upload an image of a cat
2. See prediction: "cat" with ~62% confidence
3. Note: Animal classes have lower accuracy than vehicles

### Example 3: Test with Different Images
- Try images of different sizes (they'll be resized to 32x32)
- Try different formats (JPG, PNG, etc.)
- Try images from the internet or your own photos

## 🌐 Deployment Options

### Local Development (Current)
```bash
python app.py
```
- Runs on http://localhost:5000
- Debug mode enabled
- Auto-reload on code changes

### Production Deployment

#### Option 1: Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Option 2: Waitress (Windows)
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

#### Option 3: Docker
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t cifar-classifier .
docker run -p 5000:5000 cifar-classifier
```

### Cloud Deployment

#### Heroku
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create cifar-classifier
git push heroku main
```

#### AWS/Azure/GCP
- Use their respective container services
- Upload the Docker image
- Configure port 5000
- Set environment variables if needed

## 🔒 Security Considerations

### For Production:
1. **Disable Debug Mode**: Set `debug=False` in `app.run()`
2. **File Size Limits**: Add max file size validation
3. **File Type Validation**: Strictly validate image formats
4. **Rate Limiting**: Add Flask-Limiter to prevent abuse
5. **HTTPS**: Use SSL certificates (Let's Encrypt)
6. **CORS**: Configure CORS if needed for API access

### Example Security Enhancements:
```python
from flask import Flask
from flask_limiter import Limiter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

limiter = Limiter(
    app,
    default_limits=["100 per hour"]
)
```

## 🐛 Troubleshooting

### Server won't start
- Check if port 5000 is already in use
- Try a different port: `app.run(port=8000)`

### Model not found error
- Ensure `checkpoints/cifar_net.pth` exists
- Train the model first: `python main.py --mode train --epochs 10`

### Predictions are wrong
- Model accuracy is 74.64%, not perfect
- Works best with clear, centered images
- CIFAR-10 images are 32x32, so quality may vary

### Slow predictions
- CPU inference takes 1-2 seconds
- For faster inference, use GPU if available
- Or use model quantization/optimization

## 📊 Performance

- **Model Load Time**: ~1 second
- **Prediction Time**: ~0.5-1 second (CPU)
- **Image Upload**: Depends on file size and network
- **Total Response Time**: ~1-2 seconds

## 🎓 Next Steps

1. **Improve Accuracy**: Train for more epochs or use a better architecture
2. **Add Features**: 
   - Batch prediction (multiple images)
   - Prediction history
   - Confidence threshold settings
3. **Optimize**: 
   - Model quantization
   - ONNX export for faster inference
   - Caching frequently predicted images
4. **Deploy**: Host on a cloud platform for public access

## 📝 Notes

- The model was trained on CIFAR-10 (32x32 images)
- Best results with images similar to CIFAR-10 style
- Vehicle classes (car, truck, plane, ship) have highest accuracy
- Animal classes (cat, dog, bird) have moderate accuracy

---

**Server is running!** Open http://localhost:5000 in your browser to start classifying images! 🎉
