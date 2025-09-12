#!/bin/bash

echo "🚀 Setting up AI Meal Plan Generator..."

# Create necessary directories
mkdir -p models/trained_models
mkdir -p logs

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Train the AI model
echo "🧠 Training AI model..."
python train_model.py

# Make scripts executable
chmod +x test_api.py
chmod +x train_model.py

echo "✅ Setup completed!"
echo ""
echo "Next steps:"
echo "1. Start the server: python api/meal_plan_server.py"
echo "2. Test the API: python test_api.py"
echo "3. Check the documentation: README.md"
