#!/usr/bin/env python3
"""
Test script for the AI Meal Plan API
Tests all endpoints and functionality
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5001"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data['status']}")
            print(f"  Model loaded: {data['model_loaded']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_goals():
    """Test goals endpoint"""
    print("\nTesting goals endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/goals")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Goals retrieved: {len(data['goals'])} goals available")
            for goal in data['goals'][:3]:  # Show first 3
                print(f"  - {goal['name']}: {goal['description']}")
            return True
        else:
            print(f"✗ Goals request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Goals request error: {e}")
        return False

def test_sports():
    """Test sports endpoint"""
    print("\nTesting sports endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/sports")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Sports retrieved: {len(data['sports'])} sports available")
            print(f"  Sample sports: {', '.join(data['sports'][:5])}")
            return True
        else:
            print(f"✗ Sports request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Sports request error: {e}")
        return False

def test_activity_levels():
    """Test activity levels endpoint"""
    print("\nTesting activity levels endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/activity-levels")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Activity levels retrieved: {len(data['activity_levels'])} levels available")
            for level in data['activity_levels']:
                print(f"  - {level['name']}: {level['description']}")
            return True
        else:
            print(f"✗ Activity levels request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Activity levels request error: {e}")
        return False

def test_example_request():
    """Test example request endpoint"""
    print("\nTesting example request endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/example-request")
        if response.status_code == 200:
            data = response.json()
            print("✓ Example request retrieved")
            example = data['example_request']
            print(f"  - Age: {example['age']}")
            print(f"  - Gender: {example['gender']}")
            print(f"  - Weight: {example['weight_kg']}kg")
            print(f"  - Goal: {example['goal']}")
            return True
        else:
            print(f"✗ Example request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Example request error: {e}")
        return False

def test_nutrition_prediction():
    """Test nutrition prediction endpoint"""
    print("\nTesting nutrition prediction endpoint...")
    
    test_data = {
        "age": 28,
        "gender": "M",
        "height_cm": 175,
        "weight_kg": 70,
        "body_fat_percentage": 15,
        "sport_league": "Cricket",
        "activity_level": "High",
        "training_experience_years": 8,
        "adherence_rate": 0.85,
        "competition_level": "Professional",
        "user_scenario": "professional",
        "goal": "performance_optimization",
        "time_range_days": 30
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/predict-nutrition", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print("✓ Nutrition prediction successful")
            nutrition = data['nutrition_needs']
            print(f"  - Total Calories: {nutrition['total_daily_calories']:.1f}")
            print(f"  - Carbohydrates: {nutrition['total_daily_carbs_g']:.1f}g")
            print(f"  - Protein: {nutrition['total_daily_protein_g']:.1f}g")
            print(f"  - Fat: {nutrition['total_daily_fat_g']:.1f}g")
            return True
        else:
            print(f"✗ Nutrition prediction failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Nutrition prediction error: {e}")
        return False

def test_meal_plan_generation():
    """Test meal plan generation endpoint"""
    print("\nTesting meal plan generation endpoint...")
    
    test_data = {
        "age": 28,
        "gender": "M",
        "height_cm": 175,
        "weight_kg": 70,
        "body_fat_percentage": 15,
        "sport_league": "Cricket",
        "activity_level": "High",
        "training_experience_years": 8,
        "adherence_rate": 0.85,
        "competition_level": "Professional",
        "user_scenario": "professional",
        "goal": "performance_optimization",
        "time_range_days": 30,
        "exercise_schedule": [
            {
                "type": "Skills Training",
                "duration_minutes": 90,
                "intensity": "High",
                "time": "9:00 AM"
            },
            {
                "type": "Fitness Training",
                "duration_minutes": 60,
                "intensity": "Moderate",
                "time": "3:00 PM"
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/generate-meal-plan", json=test_data)
        if response.status_code == 200:
            data = response.json()
            print("✓ Meal plan generation successful")
            meal_plan = data['meal_plan']
            print(f"  - Goal: {meal_plan['goal']}")
            print(f"  - Time Range: {meal_plan['time_range_days']} days")
            print(f"  - Exercise Count: {meal_plan['exercise_count']}")
            print(f"  - Total Calories: {meal_plan['total_nutrition']['total_daily_calories']:.1f}")
            print(f"  - Number of Meals: {len(meal_plan['meal_plan'])}")
            
            print("\n  Meal Schedule:")
            for i, meal in enumerate(meal_plan['meal_plan'][:3]):  # Show first 3 meals
                print(f"    {i+1}. {meal['meal_type']} at {meal['time']}")
                print(f"       Calories: {meal['calories']:.1f}, Carbs: {meal['carbs_g']:.1f}g, Protein: {meal['protein_g']:.1f}g")
            
            return True
        else:
            print(f"✗ Meal plan generation failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Meal plan generation error: {e}")
        return False

def test_meal_plan_update():
    """Test meal plan update endpoint"""
    print("\nTesting meal plan update endpoint...")
    
    # First generate a meal plan
    test_data = {
        "age": 28,
        "gender": "M",
        "height_cm": 175,
        "weight_kg": 70,
        "body_fat_percentage": 15,
        "sport_league": "Cricket",
        "activity_level": "High",
        "goal": "performance_optimization",
        "time_range_days": 30
    }
    
    try:
        # Generate initial meal plan
        response = requests.post(f"{BASE_URL}/api/generate-meal-plan", json=test_data)
        if response.status_code != 200:
            print(f"✗ Initial meal plan generation failed: {response.status_code}")
            return False
        
        initial_meal_plan = response.json()['meal_plan']
        
        # Update meal plan
        update_data = {
            "current_meal_plan": initial_meal_plan,
            "updates": {
                "weight_kg": 75,  # Weight gain
                "activity_level": "Very High",  # Increased activity
                "goal": "muscle_building"  # Changed goal
            }
        }
        
        response = requests.post(f"{BASE_URL}/api/update-meal-plan", json=update_data)
        if response.status_code == 200:
            data = response.json()
            print("✓ Meal plan update successful")
            updated_plan = data['updated_meal_plan']
            print(f"  - Changes applied: {data['changes_applied']}")
            print(f"  - New total calories: {updated_plan['total_nutrition']['total_daily_calories']:.1f}")
            print(f"  - New goal: {updated_plan['goal']}")
            return True
        else:
            print(f"✗ Meal plan update failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Meal plan update error: {e}")
        return False

def main():
    print("=" * 60)
    print("AI MEAL PLAN API TESTING")
    print("=" * 60)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    # Run all tests
    tests = [
        test_health,
        test_goals,
        test_sports,
        test_activity_levels,
        test_example_request,
        test_nutrition_prediction,
        test_meal_plan_generation,
        test_meal_plan_update
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
