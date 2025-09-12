#!/usr/bin/env python3
"""
Test script for adaptive meal planning functionality
"""

import requests
import json
from datetime import datetime, timedelta
import time

# API endpoint
API_URL = "http://localhost:5001"

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check")
    
    try:
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        
        print(f"Status: {data['status']}")
        print(f"Model loaded: {data['model_loaded']}")
        print(f"Timestamp: {data['timestamp']}")
        
        return data['model_loaded']
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_generate_meal_plan():
    """Test basic meal plan generation"""
    print_section("Testing Basic Meal Plan Generation")
    
    test_data = {
        "user_id": "test_user_123",
        "age": 28,
        "gender": "M",
        "height_cm": 175,
        "weight_kg": 70,
        "body_fat_percentage": 15,
        "sport_league": "Cricket",
        "activity_level": "High",
        "goal": "performance_optimization",
        "time_range_days": 30,
        "exercise_schedule": [
            {
                "type": "Skills Training",
                "duration_minutes": 90,
                "intensity": "High",
                "time": "9:00 AM"
            }
        ]
    }
    
    try:
        response = requests.post(f"{API_URL}/api/generate-meal-plan", json=test_data)
        data = response.json()
        
        if data['status'] == 'success':
            print("✅ Meal plan generated successfully")
            meal_plan = data['meal_plan']
            print(f"Total calories: {meal_plan['total_nutrition']['total_daily_calories']:.1f}")
            print(f"Total carbs: {meal_plan['total_nutrition']['total_daily_carbs_g']:.1f}g")
            print(f"Total protein: {meal_plan['total_nutrition']['total_daily_protein_g']:.1f}g")
            print(f"Total fat: {meal_plan['total_nutrition']['total_daily_fat_g']:.1f}g")
            print(f"Number of meals: {len(meal_plan['meal_plan'])}")
            
            # Print meal details
            print("\nMeal Details:")
            for i, meal in enumerate(meal_plan['meal_plan'], 1):
                print(f"  Meal {i}: {meal['meal_type']} at {meal['time']}")
                print(f"    Calories: {meal['calories']:.1f}, Carbs: {meal['carbs_g']:.1f}g, "
                      f"Protein: {meal['protein_g']:.1f}g, Fat: {meal['fat_g']:.1f}g")
            
            return meal_plan
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_log_meal(user_id, meal_type, calories, carbs_g, protein_g, fat_g, consumed="100%"):
    """Test meal logging"""
    print_section(f"Testing Meal Logging: {meal_type}")
    
    meal_data = {
        "user_id": user_id,
        "meal_type": meal_type,
        "consumption_time": datetime.now().isoformat(),
        "calories": calories,
        "carbs_g": carbs_g,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "consumed": consumed
    }
    
    try:
        response = requests.post(f"{API_URL}/api/log-meal", json=meal_data)
        data = response.json()
        
        if data['status'] == 'success':
            print(f"✅ Meal '{meal_type}' logged successfully")
            
            if 'nutrition_status' in data and data['nutrition_status']:
                nutrition_status = data['nutrition_status']
                print("\nNutrition Status:")
                print(f"  Consumed calories: {nutrition_status['consumed']['calories']:.1f}")
                print(f"  Remaining calories: {nutrition_status['remaining']['calories']:.1f}")
                print(f"  Progress: {nutrition_status['progress']['calories']:.1f}%")
            
            return data
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_nutrition_status(user_id):
    """Test getting nutrition status"""
    print_section(f"Testing Nutrition Status for User: {user_id}")
    
    try:
        response = requests.get(f"{API_URL}/api/nutrition-status/{user_id}")
        
        if response.status_code == 200:
            data = response.json()
            nutrition_status = data['nutrition_status']
            
            print("✅ Nutrition status retrieved successfully")
            print("\nNutrition Status:")
            print(f"  Consumed calories: {nutrition_status['consumed']['calories']:.1f}")
            print(f"  Remaining calories: {nutrition_status['remaining']['calories']:.1f}")
            print(f"  Progress: {nutrition_status['progress']['calories']:.1f}%")
            
            return nutrition_status
        elif response.status_code == 404:
            print("❌ No meal plan found for user")
            return None
        else:
            data = response.json()
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_adaptive_meal_plan(user_id, meal_logging_data=None):
    """Test adaptive meal plan generation"""
    print_section("Testing Adaptive Meal Plan Generation")
    
    test_data = {
        "user_id": user_id,
        "age": 28,
        "gender": "M",
        "height_cm": 175,
        "weight_kg": 70,
        "body_fat_percentage": 15,
        "sport_league": "Cricket",
        "activity_level": "High",
        "goal": "performance_optimization",
        "time_range_days": 30,
        "exercise_schedule": [
            {
                "type": "Skills Training",
                "duration_minutes": 90,
                "intensity": "High",
                "time": "9:00 AM"
            }
        ],
        "meal_logging_data": meal_logging_data or []
    }
    
    try:
        response = requests.post(f"{API_URL}/api/generate-adaptive-meal-plan", json=test_data)
        data = response.json()
        
        if data['status'] == 'success':
            print("✅ Adaptive meal plan generated successfully")
            adaptive_plan = data['adaptive_meal_plan']
            
            # Check if adaptation was applied
            if 'adaptation_status' in adaptive_plan:
                print(f"Adaptation status: {adaptive_plan['adaptation_status']}")
                
                if adaptive_plan['adaptation_status'] == 'adapted':
                    print("\nAdaptation Details:")
                    details = adaptive_plan['adaptation_details']
                    print(f"  Missed meals: {details['missed_meals']}")
                    print(f"  Partial meals: {details['partial_meals']}")
                    print(f"  Adaptation reason: {details['adaptation_reason']}")
                    
                    # Print deficit
                    deficit = details['deficit']
                    print("\nNutrition Deficit:")
                    print(f"  Calories: {deficit['calories']:.1f}")
                    print(f"  Carbs: {deficit['carbs_g']:.1f}g")
                    print(f"  Protein: {deficit['protein_g']:.1f}g")
                    print(f"  Fat: {deficit['fat_g']:.1f}g")
            
            # Print adapted meal plan
            print("\nAdapted Meal Plan:")
            for i, meal in enumerate(adaptive_plan['meal_plan'], 1):
                status = meal.get('status', 'normal')
                status_marker = "🔄" if status == 'adapted' else "✓" if status == 'normal' else "❌"
                
                print(f"  {status_marker} Meal {i}: {meal['meal_type']} at {meal['time']}")
                print(f"    Calories: {meal['calories']:.1f}, Carbs: {meal['carbs_g']:.1f}g, "
                      f"Protein: {meal['protein_g']:.1f}g, Fat: {meal['fat_g']:.1f}g")
                
                if 'adaptation_note' in meal:
                    print(f"    Note: {meal['adaptation_note']}")
            
            return adaptive_plan
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_reset_user(user_id):
    """Test resetting user session"""
    print_section(f"Testing Reset User: {user_id}")
    
    try:
        response = requests.post(f"{API_URL}/api/reset-user/{user_id}")
        data = response.json()
        
        if data['status'] == 'success':
            print(f"✅ User session reset successfully: {data['message']}")
            return True
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_adaptive_meal_plan_scenario():
    """Run a complete adaptive meal plan scenario"""
    print_section("RUNNING ADAPTIVE MEAL PLAN SCENARIO")
    
    # Check if API is healthy
    if not test_health():
        print("❌ API is not healthy. Make sure the server is running.")
        return
    
    # Generate a unique user ID for this test
    user_id = f"test_user_{int(time.time())}"
    print(f"Using test user ID: {user_id}")
    
    # Step 1: Generate initial meal plan
    print("\n📋 Step 1: Generate initial meal plan")
    meal_plan = test_generate_meal_plan()
    if not meal_plan:
        print("❌ Failed to generate initial meal plan. Aborting test.")
        return
    
    # Extract meals from the plan
    meals = meal_plan['meal_plan']
    
    # Use the first meal in the plan instead of specifically looking for 'Breakfast'
    first_meal = meals[0] if meals else None
    lunch = next((m for m in meals if 'Lunch' in m['meal_type'] or 'Post-Exercise Meal' in m['meal_type']), None)
    
    if not first_meal or not lunch:
        print("❌ Meal plan doesn't contain expected meals. Aborting test.")
        return
    
    # Step 2: Log first meal as partially consumed (70%)
    print(f"\n🍳 Step 2: Log {first_meal['meal_type']} as partially consumed (70%)")
    consumed_percentage = 0.7
    logged_meal = test_log_meal(
        user_id=user_id,
        meal_type=first_meal['meal_type'],
        calories=first_meal['calories'] * consumed_percentage,
        carbs_g=first_meal['carbs_g'] * consumed_percentage,
        protein_g=first_meal['protein_g'] * consumed_percentage,
        fat_g=first_meal['fat_g'] * consumed_percentage,
        consumed="70%"
    )
    
    # Step 3: Check nutrition status
    print("\n📊 Step 3: Check nutrition status")
    nutrition_status = test_nutrition_status(user_id)
    
    # Step 4: Generate adaptive meal plan
    print("\n🔄 Step 4: Generate adaptive meal plan")
    # Create meal logging data with the partial breakfast
    meal_logging_data = [{
        'meal_type': first_meal['meal_type'],
        'consumption_time': datetime.now().isoformat(),
        'calories': first_meal['calories'] * consumed_percentage,
        'carbs_g': first_meal['carbs_g'] * consumed_percentage,
        'protein_g': first_meal['protein_g'] * consumed_percentage,
        'fat_g': first_meal['fat_g'] * consumed_percentage,
        'consumed': "70%"
    }]
    
    # Set current time to after breakfast but before lunch
    current_time = (datetime.now().replace(hour=11, minute=30)).isoformat()
    
    # Generate adaptive plan
    adaptive_plan = test_adaptive_meal_plan(user_id, meal_logging_data)
    
    # Step 5: Reset user session
    print("\n🔄 Step 5: Reset user session")
    test_reset_user(user_id)
    
    print("\n✅ Adaptive meal plan scenario completed successfully!")

if __name__ == "__main__":
    run_adaptive_meal_plan_scenario() 