from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
from datetime import datetime, timedelta
import traceback
import joblib
import numpy as np

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.meal_plan_ai import MealPlanAI

app = Flask(__name__)
CORS(app)

# Initialize AI model
ai_model = MealPlanAI()

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set custom JSON encoder
app.json_encoder = NumpyEncoder

# Load trained models with corrected logic
def load_models():
    model_path = './models/trained_models'
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist")
        return False
    
    try:
        # Load main models
        for target in ai_model.target_columns:
            model_file = f'{model_path}/{target}_model.pkl'
            scaler_file = f'{model_path}/{target}_scaler.pkl'
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                ai_model.models[target] = joblib.load(model_file)
                ai_model.scalers[target] = joblib.load(scaler_file)
        
        # Load encoders
        encoders_file = f'{model_path}/encoders.pkl'
        if os.path.exists(encoders_file):
            ai_model.encoders = joblib.load(encoders_file)
        
        # Load goal models
        goal_models_file = f'{model_path}/goal_models.pkl'
        if os.path.exists(goal_models_file):
            ai_model.goal_models = joblib.load(goal_models_file)
        
        # Load goal-specific scalers
        for goal in ai_model.goal_models.keys():
            for target in ai_model.target_columns:
                scaler_file = f'{model_path}/{goal}_{target}_scaler.pkl'
                if os.path.exists(scaler_file):
                    ai_model.scalers[f'{goal}_{target}'] = joblib.load(scaler_file)
        
        # Load meal distribution rules
        distribution_file = f'{model_path}/meal_distribution_rules.pkl'
        if os.path.exists(distribution_file):
            ai_model.meal_distribution_rules = joblib.load(distribution_file)
        
        print(f"Models loaded successfully from {model_path}")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Convert numpy types to Python types
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Load models
model_loaded = load_models()

if not model_loaded:
    print("Warning: Could not load trained models. Please train the models first.")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/generate-meal-plan', methods=['POST'])
def generate_meal_plan():
    """Generate personalized meal plan"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'height_cm', 'weight_kg', 'body_fat_percentage']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400
        
        # Extract user profile
        user_profile = {
            'age': int(data['age']),
            'gender': data['gender'],
            'height_cm': float(data['height_cm']),
            'weight_kg': float(data['weight_kg']),
            'body_fat_percentage': float(data['body_fat_percentage']),
            'sport_league': data.get('sport_league', 'General Fitness'),
            'activity_level': data.get('activity_level', 'Moderate'),
            'training_experience_years': data.get('training_experience_years', 5),
            'adherence_rate': data.get('adherence_rate', 0.8),
            'competition_level': data.get('competition_level', 'Recreational'),
            'user_scenario': data.get('user_scenario', 'fitness_enthusiast')
        }
        
        # Extract goal and time range
        goal = data.get('goal', 'general_wellness')
        time_range_days = int(data.get('time_range_days', 30))
        
        # Extract exercise schedule if provided
        exercise_schedule = data.get('exercise_schedule', [])
        
        # Generate meal plan
        meal_plan = ai_model.generate_meal_plan(
            user_profile=user_profile,
            goal=goal,
            time_range_days=time_range_days,
            exercise_schedule=exercise_schedule
        )
        
        # Convert numpy types to Python types
        meal_plan_converted = convert_numpy_types(meal_plan)
        
        return jsonify({
            'status': 'success',
            'meal_plan': meal_plan_converted
        })
        
    except Exception as e:
        print(f"Error generating meal plan: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/generate-adaptive-meal-plan', methods=['POST'])
def generate_adaptive_meal_plan():
    """Generate adaptive meal plan based on logged meals"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'height_cm', 'weight_kg', 'body_fat_percentage', 'user_id']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400
        
        # Extract user profile
        user_profile = {
            'user_id': data['user_id'],
            'age': int(data['age']),
            'gender': data['gender'],
            'height_cm': float(data['height_cm']),
            'weight_kg': float(data['weight_kg']),
            'body_fat_percentage': float(data['body_fat_percentage']),
            'sport_league': data.get('sport_league', 'General Fitness'),
            'activity_level': data.get('activity_level', 'Moderate'),
            'training_experience_years': data.get('training_experience_years', 5),
            'adherence_rate': data.get('adherence_rate', 0.8),
            'competition_level': data.get('competition_level', 'Recreational'),
            'user_scenario': data.get('user_scenario', 'fitness_enthusiast')
        }
        
        # Extract goal and time range
        goal = data.get('goal', 'general_wellness')
        time_range_days = int(data.get('time_range_days', 30))
        
        # Extract exercise schedule if provided
        exercise_schedule = data.get('exercise_schedule', [])
        
        # Extract meal logging data if provided
        meal_logging_data = data.get('meal_logging_data', [])
        
        # Extract current time if provided
        current_time = data.get('current_time')
        if current_time:
            try:
                # Convert ISO format to datetime, handling timezone info
                current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                # Convert to naive datetime to avoid comparison issues
                current_time = current_time.replace(tzinfo=None)
            except ValueError:
                current_time = datetime.now()
        else:
            current_time = datetime.now()
        
        # Generate adaptive meal plan
        adaptive_meal_plan = ai_model.generate_adaptive_meal_plan(
            user_profile=user_profile,
            goal=goal,
            time_range_days=time_range_days,
            exercise_schedule=exercise_schedule,
            meal_logging_data=meal_logging_data,
            current_time=current_time
        )
        
        # Convert numpy types to Python types
        adaptive_meal_plan_converted = convert_numpy_types(adaptive_meal_plan)
        
        return jsonify({
            'status': 'success',
            'adaptive_meal_plan': adaptive_meal_plan_converted,
            'current_time': current_time.isoformat()
        })
        
    except Exception as e:
        print(f"Error generating adaptive meal plan: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/log-meal', methods=['POST'])
def log_meal():
    """Log a meal consumed by the user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['user_id', 'meal_type', 'calories', 'carbs_g', 'protein_g', 'fat_g']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400
        
        # Extract meal data
        meal_data = {
            'meal_type': data['meal_type'],
            'consumption_time': data.get('consumption_time', datetime.now().isoformat()),
            'calories': float(data['calories']),
            'carbs_g': float(data['carbs_g']),
            'protein_g': float(data['protein_g']),
            'fat_g': float(data['fat_g']),
            'consumed': data.get('consumed', '100%')
        }
        
        # Log the meal
        logged_meals = ai_model.log_meal(data['user_id'], meal_data)
        
        # Get updated nutrition status
        nutrition_status = ai_model.get_user_nutrition_status(data['user_id'])
        
        return jsonify({
            'status': 'success',
            'logged_meal': meal_data,
            'nutrition_status': nutrition_status,
            'logged_meals_count': len(logged_meals),
            'message': 'Meal logged successfully'
        })
        
    except Exception as e:
        print(f"Error logging meal: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/nutrition-status/<user_id>', methods=['GET'])
def get_nutrition_status(user_id):
    """Get nutrition status for a user"""
    try:
        nutrition_status = ai_model.get_user_nutrition_status(user_id)
        
        if not nutrition_status:
            return jsonify({
                'status': 'error',
                'message': 'No meal plan found for user'
            }), 404
        
        return jsonify({
            'status': 'success',
            'nutrition_status': nutrition_status,
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"Error getting nutrition status: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/reset-user/<user_id>', methods=['POST'])
def reset_user(user_id):
    """Reset user session data"""
    try:
        success = ai_model.reset_user_session(user_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'User session for {user_id} reset successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'No session found for user {user_id}'
            }), 404
        
    except Exception as e:
        print(f"Error resetting user session: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/predict-nutrition', methods=['POST'])
def predict_nutrition():
    """Predict nutrition needs only"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'height_cm', 'weight_kg', 'body_fat_percentage']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400
        
        # Extract user profile
        user_profile = {
            'age': int(data['age']),
            'gender': data['gender'],
            'height_cm': float(data['height_cm']),
            'weight_kg': float(data['weight_kg']),
            'body_fat_percentage': float(data['body_fat_percentage']),
            'sport_league': data.get('sport_league', 'General Fitness'),
            'activity_level': data.get('activity_level', 'Moderate'),
            'training_experience_years': data.get('training_experience_years', 5),
            'adherence_rate': data.get('adherence_rate', 0.8),
            'competition_level': data.get('competition_level', 'Recreational'),
            'user_scenario': data.get('user_scenario', 'fitness_enthusiast')
        }
        
        # Extract goal and time range
        goal = data.get('goal', 'general_wellness')
        time_range_days = int(data.get('time_range_days', 30))
        
        # Predict nutrition needs
        nutrition_needs = ai_model.predict_nutrition_needs(
            user_profile=user_profile,
            goal=goal,
            time_range_days=time_range_days
        )
        
        # Convert numpy types to Python types
        nutrition_converted = convert_numpy_types(nutrition_needs)
        
        return jsonify({
            'status': 'success',
            'nutrition_needs': nutrition_converted,
            'user_profile': user_profile,
            'goal': goal,
            'time_range_days': time_range_days
        })
        
    except Exception as e:
        print(f"Error predicting nutrition: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/goals', methods=['GET'])
def get_available_goals():
    """Get available goals"""
    goals = [
        {
            'id': 'general_wellness',
            'name': 'General Wellness',
            'description': 'Maintain overall health and fitness'
        },
        {
            'id': 'weight_loss',
            'name': 'Weight Loss',
            'description': 'Lose weight and reduce body fat'
        },
        {
            'id': 'weight_gain',
            'name': 'Weight Gain',
            'description': 'Gain weight and build mass'
        },
        {
            'id': 'muscle_building',
            'name': 'Muscle Building',
            'description': 'Build muscle mass and strength'
        },
        {
            'id': 'performance_optimization',
            'name': 'Performance Optimization',
            'description': 'Optimize athletic performance'
        },
        {
            'id': 'endurance_improvement',
            'name': 'Endurance Improvement',
            'description': 'Improve cardiovascular endurance'
        },
        {
            'id': 'strength_improvement',
            'name': 'Strength Improvement',
            'description': 'Increase muscular strength'
        }
    ]
    
    return jsonify({
        'status': 'success',
        'goals': goals
    })

@app.route('/api/sports', methods=['GET'])
def get_available_sports():
    """Get available sports"""
    sports = [
        'Cricket', 'NFL', 'NBA', 'Soccer', 'Tennis', 'Boxing', 'Swimming', 'Cycling',
        'Basketball', 'Football', 'Baseball', 'Volleyball', 'Golf', 'Martial Arts',
        'Rugby', 'Lacrosse', 'Field Hockey', 'Softball', 'Gymnastics', 'Diving',
        'Triathlon', 'Squash', 'Wrestling', 'General Fitness', 'Recreational Athlete'
    ]
    
    return jsonify({
        'status': 'success',
        'sports': sports
    })

@app.route('/api/activity-levels', methods=['GET'])
def get_activity_levels():
    """Get available activity levels"""
    activity_levels = [
        {
            'id': 'Low',
            'name': 'Low Activity',
            'description': 'Minimal exercise, mostly sedentary'
        },
        {
            'id': 'Moderate',
            'name': 'Moderate Activity',
            'description': 'Regular exercise 3-4 times per week'
        },
        {
            'id': 'High',
            'name': 'High Activity',
            'description': 'Intensive exercise 5-6 times per week'
        },
        {
            'id': 'Very High',
            'name': 'Very High Activity',
            'description': 'Professional athlete level training'
        }
    ]
    
    return jsonify({
        'status': 'success',
        'activity_levels': activity_levels
    })

@app.route('/api/example-request', methods=['GET'])
def get_example_request():
    """Get example request format"""
    example = {
        'age': 28,
        'gender': 'M',
        'height_cm': 175,
        'weight_kg': 70,
        'body_fat_percentage': 15,
        'user_id': 'athlete_123',  # New: Required for adaptive meal planning
        'sport_league': 'Cricket',
        'activity_level': 'High',
        'training_experience_years': 8,
        'adherence_rate': 0.85,
        'competition_level': 'Professional',
        'user_scenario': 'professional',
        'goal': 'performance_optimization',
        'time_range_days': 30,
        'exercise_schedule': [
            {
                'type': 'Skills Training',
                'duration_minutes': 90,
                'intensity': 'High',
                'time': '9:00 AM'
            },
            {
                'type': 'Fitness Training',
                'duration_minutes': 60,
                'intensity': 'Moderate',
                'time': '3:00 PM'
            }
        ],
        'meal_logging_data': [  # New: Optional meal logging data
            {
                'meal_type': 'Breakfast',
                'consumption_time': '9:30 AM',
                'calories': 450,
                'carbs_g': 55,
                'protein_g': 25,
                'fat_g': 18,
                'consumed': '90%'  # Percentage consumed (optional)
            }
        ]
    }
    
    return jsonify({
        'status': 'success',
        'example_request': example
    })

if __name__ == '__main__':
    print("Starting AI Meal Plan Server...")
    print("Server will run on http://localhost:5001")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /api/generate-meal-plan - Generate meal plan")
    print("  POST /api/generate-adaptive-meal-plan - Generate adaptive meal plan")
    print("  POST /api/log-meal - Log meal consumption")
    print("  GET  /api/nutrition-status/<user_id> - Get user nutrition status")
    print("  POST /api/reset-user/<user_id> - Reset user session")
    print("  POST /api/predict-nutrition - Predict nutrition needs")
    print("  GET  /api/goals - Get available goals")
    print("  GET  /api/sports - Get available sports")
    print("  GET  /api/activity-levels - Get activity levels")
    print("  GET  /api/example-request - Get example request format")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
