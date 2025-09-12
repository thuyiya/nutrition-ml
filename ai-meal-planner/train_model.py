#!/usr/bin/env python3
"""
AI Meal Plan Model Training Script
Trains the AI model using the collected dataset
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from meal_plan_ai import MealPlanAI
from adaptive_meal_plan import AdaptiveMealPlanner

def main():
    print("=" * 60)
    print("AI MEAL PLAN MODEL TRAINING")
    print("=" * 60)
    
    # Initialize AI model
    print("Initializing AI model...")
    ai_model = MealPlanAI()
    
    # Load data
    print("\nLoading training data...")
    data_path = '../dataset/data-collection'
    
    try:
        athletes_df, meal_plans_df, training_df, sports_df = ai_model.load_data(data_path)
        
        print(f"✓ Loaded {len(athletes_df)} athlete profiles")
        print(f"✓ Loaded {len(meal_plans_df)} meal plan entries")
        print(f"✓ Loaded {len(training_df)} training sessions")
        print(f"✓ Loaded {len(sports_df)} sports configurations")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    # Prepare features
    print("\nPreparing features for training...")
    try:
        training_data = ai_model.prepare_features(athletes_df, meal_plans_df, training_df, sports_df)
        print(f"✓ Prepared {len(training_data)} training samples")
        print(f"✓ Generated {len(ai_model.feature_columns)} features")
        
    except Exception as e:
        print(f"✗ Error preparing features: {e}")
        return False
    
    # Train models
    print("\nTraining AI models...")
    try:
        ai_model.train_models(training_data)
        print("✓ All models trained successfully")
        
    except Exception as e:
        print(f"✗ Error training models: {e}")
        return False
    
    # Generate adaptive meal planning data
    print("\nGenerating adaptive meal planning data...")
    try:
        # Initialize adaptive planner
        adaptive_planner = AdaptiveMealPlanner()
        
        # Create adaptive meal planning examples
        print("✓ Adaptive meal planner initialized")
        print("✓ Adaptive meal planning functionality ready")
        
    except Exception as e:
        print(f"✗ Error initializing adaptive meal planner: {e}")
        # Continue with the process even if adaptive planner fails
    
    # Save models
    print("\nSaving trained models...")
    try:
        model_path = './models/trained_models'
        ai_model.save_models(model_path)
        print(f"✓ Models saved to {model_path}")
        
    except Exception as e:
        print(f"✗ Error saving models: {e}")
        return False
    
    # Test the model
    print("\nTesting the trained model...")
    try:
        # Create a test user profile
        test_profile = {
            'age': 28,
            'gender': 'M',
            'height_cm': 175,
            'weight_kg': 70,
            'body_fat_percentage': 15,
            'sport_league': 'Cricket',
            'activity_level': 'High',
            'training_experience_years': 8,
            'adherence_rate': 0.85,
            'competition_level': 'Professional',
            'user_scenario': 'professional'
        }
        
        # Test nutrition prediction
        nutrition = ai_model.predict_nutrition_needs(test_profile, 'performance_optimization', 30)
        print("✓ Test nutrition prediction successful")
        print(f"  - Calories: {nutrition['total_daily_calories']:.1f}")
        print(f"  - Carbs: {nutrition['total_daily_carbs_g']:.1f}g")
        print(f"  - Protein: {nutrition['total_daily_protein_g']:.1f}g")
        print(f"  - Fat: {nutrition['total_daily_fat_g']:.1f}g")
        
        # Test meal plan generation
        meal_plan = ai_model.generate_meal_plan(test_profile, 'performance_optimization', 30)
        print("✓ Test meal plan generation successful")
        print(f"  - Generated {len(meal_plan['meal_plan'])} meals")
        
        # Test adaptive meal plan generation
        print("\nTesting adaptive meal plan generation...")
        
        # Create test meal logging data (partially consumed breakfast)
        breakfast = next((m for m in meal_plan['meal_plan'] if m['meal_type'] == 'Breakfast'), None)
        if breakfast:
            # Create meal logging data with 70% consumed breakfast
            consumed_percentage = 0.7
            meal_logging_data = [{
                'meal_type': breakfast['meal_type'],
                'consumption_time': datetime.now().isoformat(),
                'calories': breakfast['calories'] * consumed_percentage,
                'carbs_g': breakfast['carbs_g'] * consumed_percentage,
                'protein_g': breakfast['protein_g'] * consumed_percentage,
                'fat_g': breakfast['fat_g'] * consumed_percentage,
                'consumed': "70%"
            }]
            
            # Generate adaptive meal plan
            adaptive_meal_plan = ai_model.generate_adaptive_meal_plan(
                user_profile=test_profile,
                goal='performance_optimization',
                time_range_days=30,
                meal_logging_data=meal_logging_data
            )
            
            print("✓ Test adaptive meal plan generation successful")
            
            if 'adaptation_status' in adaptive_meal_plan:
                print(f"  - Adaptation status: {adaptive_meal_plan['adaptation_status']}")
                
                if adaptive_meal_plan['adaptation_status'] == 'adapted':
                    print("  - Adaptation applied successfully")
                    print(f"  - Adaptation reason: {adaptive_meal_plan['adaptation_details']['adaptation_reason']}")
        else:
            print("✗ Could not test adaptive meal plan (no breakfast in meal plan)")
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("1. Start the API server: python api/meal_plan_server.py")
    print("2. Test the API endpoints")
    print("3. Test adaptive meal planning: python test_adaptive_meal_plan.py")
    print("4. Integrate with your application")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
