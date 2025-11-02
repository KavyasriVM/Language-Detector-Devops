from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import re
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
label_encoder = None
device = None

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer, label_encoder, device
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        logger.info("✅ Tokenizer loaded")
        
        # Load model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=22)
        
        # Load trained weights if available
        model_path = 'models/modelname.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("✅ Trained model weights loaded")
        else:
            logger.warning("⚠️ No trained model found, using base model")
        
        model.to(device)
        model.eval()
        
        # Create label encoder (you'll need to save this during training)
        # For now, using the languages from your dataset
        languages = ['Arabic', 'Chinese', 'Dutch', 'English', 'Estonian', 'French', 
                    'Hindi', 'Indonesian', 'Japanese', 'Korean', 'Latin', 'Persian', 
                    'Portugese', 'Pushto', 'Romanian', 'Russian', 'Spanish', 'Swedish', 
                    'Tamil', 'Thai', 'Turkish', 'Urdu']
        label_encoder = LabelEncoder()
        label_encoder.fit(languages)
        
        logger.info("✅ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise e

def preprocess_text(text):
    """Preprocess text for prediction"""
    # Clean text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    
    # Pad sequence
    padded = tokens + [0] * (512 - len(tokens))
    attention_mask = [1] * len(tokens) + [0] * (512 - len(tokens))
    
    return torch.tensor([padded]).to(device), torch.tensor([attention_mask]).to(device)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown'
    })

@app.route('/predict', methods=['POST'])
def predict_language():
    """Predict language from text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Preprocess text
        input_ids, attention_mask = preprocess_text(text)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1)
            confidence = torch.max(predictions, dim=-1)[0].item()
        
        # Get language name
        predicted_language = label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
        
        return jsonify({
            'text': text,
            'predicted_language': predicted_language,
            'confidence': float(round(float(confidence) * 100, 2)),
            'all_predictions': {
                lang: float(round(float(prob) * 100, 2))
                for lang, prob in zip(label_encoder.classes_, predictions[0].cpu().numpy())
            }
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    return jsonify({
        'supported_languages': label_encoder.classes_.tolist() if label_encoder else []
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'Language Detection API',
        'version': '1.0.0',
        'endpoints': {
            'POST /predict': 'Predict language from text',
            'GET /languages': 'Get supported languages',
            'GET /health': 'Health check'
        },
        'example_usage': {
            'url': '/predict',
            'method': 'POST',
            'body': {'text': 'Hello, how are you?'}
        }
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Language Detection API...")
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)




