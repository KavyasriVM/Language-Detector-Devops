#!/bin/bash

echo "🚀 Starting Language Detection Model Container..."

# Check if dataset exists
if [ ! -f "dataset.csv" ]; then
    echo "❌ Error: dataset.csv not found!"
    echo "Please make sure dataset.csv is in the container"
    exit 1
fi

echo "✅ Dataset found: dataset.csv"

# Check if model file exists
if [ ! -f "models/modelname.pth" ]; then
    echo "❌ Error: Pre-trained model not found!"
    echo "Please mount the models directory with the required model file."
    exit 1
else
    echo "✅ Pre-trained model found: models/modelname.pth"
fi

# Start the Flask API
echo "🌐 Starting Flask API server..."
python app.py --host=0.0.0.0 --port=5000






