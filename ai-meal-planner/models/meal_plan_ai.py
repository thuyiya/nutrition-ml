import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MealPlanAI:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_columns = ['total_daily_calories', 'total_daily_carbs_g', 'total_daily_protein_g', 'total_daily_fat_g']
        self.meal_distribution_model = None
        self.goal_models = {}
        
    def load_data(self, data_path):
        """Load and prepare training data"""
        print("Loading athlete profiles...")
        athletes_df = pd.read_csv(f'{data_path}/athlete_profiles_ml.csv')
        
        print("Loading meal plans...")
        meal_plans_df = pd.read_csv(f'{data_path}/assigned_meal_plans.csv')
        
        print("Loading training data...")
        training_df = pd.read_csv(f'{data_path}/training_data_ml.csv')
        
        print("Loading sports data...")
        sports_df = pd.read_csv(f'{data_path}/sports.csv')
        
        return athletes_df, meal_plans_df, training_df, sports_df
    
    def prepare_features(self, athletes_df, meal_plans_df, training_df, sports_df):
        """Prepare features for AI model training"""
        print("Preparing features...")
        
        # Create daily meal plan summaries
        daily_plans = meal_plans_df.groupby(['athlete_id', 'date']).agg({
            'total_daily_calories': 'first',
            'total_daily_carbs_g': 'first',
            'total_daily_protein_g': 'first',
            'total_daily_fat_g': 'first',
            'exercise_count': 'first',
            'sport_league': 'first',
            'user_scenario': 'first',
            'competition_level': 'first',
            'activity_level': 'first',
            'adherence_rate': 'first'
        }).reset_index()
        
        # Merge with athlete profiles
        training_data = daily_plans.merge(
            athletes_df[['athlete_id', 'age', 'gender', 'height_cm', 'weight_kg', 
                        'body_fat_percentage', 'training_experience_years', 'goals']], 
            on='athlete_id', how='left'
        )
        
        # Process goals column
        training_data['goals_processed'] = training_data['goals'].apply(
            lambda x: self._process_goals(x) if pd.notna(x) else 'general_wellness'
        )
        
        # Add exercise intensity features
        exercise_features = training_df.groupby(['athlete_id', 'date']).agg({
            'duration_minutes': ['sum', 'mean', 'count'],
            'intensity_level': lambda x: self._encode_intensity(x.mode().iloc[0] if len(x.mode()) > 0 else 'Moderate'),
            'calories_burned': 'sum'
        }).reset_index()
        
        exercise_features.columns = ['athlete_id', 'date', 'total_duration', 'avg_duration', 
                                   'session_count', 'intensity_level', 'total_calories_burned']
        
        training_data = training_data.merge(exercise_features, on=['athlete_id', 'date'], how='left')
        training_data = training_data.fillna(0)
        
        # Add sport-specific features
        sport_features = sports_df[['name', 'METs', 'carbWeightRatio', 'proteinWeightRatio']].rename(
            columns={'name': 'sport_league'}
        )
        training_data = training_data.merge(sport_features, on='sport_league', how='left')
        training_data = training_data.fillna({'METs': 5.0, 'carbWeightRatio': 6.0, 'proteinWeightRatio': 1.5})
        
        # Feature engineering
        training_data['bmi'] = training_data['weight_kg'] / ((training_data['height_cm'] / 100) ** 2)
        training_data['lean_body_mass'] = training_data['weight_kg'] * (1 - training_data['body_fat_percentage'] / 100)
        training_data['exercise_intensity_score'] = training_data['intensity_level'] * training_data['total_duration']
        training_data['age_group'] = pd.cut(training_data['age'], bins=[0, 25, 35, 45, 100], labels=['young', 'adult', 'middle', 'senior'])
        
        # Encode categorical variables
        categorical_columns = ['gender', 'sport_league', 'user_scenario', 'competition_level', 
                             'activity_level', 'goals_processed', 'age_group']
        
        for col in categorical_columns:
            if col in training_data.columns:
                le = LabelEncoder()
                training_data[f'{col}_encoded'] = le.fit_transform(training_data[col].astype(str))
                self.encoders[col] = le
        
        # Select features for training
        self.feature_columns = [
            'age', 'height_cm', 'weight_kg', 'body_fat_percentage', 'bmi', 'lean_body_mass',
            'training_experience_years', 'adherence_rate', 'exercise_count', 'total_duration',
            'avg_duration', 'session_count', 'intensity_level', 'total_calories_burned',
            'METs', 'carbWeightRatio', 'proteinWeightRatio', 'exercise_intensity_score',
            'gender_encoded', 'sport_league_encoded', 'user_scenario_encoded',
            'competition_level_encoded', 'activity_level_encoded', 'goals_processed_encoded',
            'age_group_encoded'
        ]
        
        return training_data
    
    def _process_goals(self, goals_str):
        """Process goals string to extract primary goal"""
        if pd.isna(goals_str):
            return 'general_wellness'
        
        goals_str = str(goals_str).lower()
        
        if 'weight_loss' in goals_str or 'lose' in goals_str:
            return 'weight_loss'
        elif 'weight_gain' in goals_str or 'gain' in goals_str:
            return 'weight_gain'
        elif 'muscle_gain' in goals_str or 'muscle_building' in goals_str:
            return 'muscle_building'
        elif 'performance' in goals_str:
            return 'performance_optimization'
        elif 'endurance' in goals_str:
            return 'endurance_improvement'
        elif 'strength' in goals_str:
            return 'strength_improvement'
        else:
            return 'general_wellness'
    
    def _encode_intensity(self, intensity):
        """Encode exercise intensity to numeric"""
        intensity_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
        return intensity_map.get(intensity, 2)
    
    def train_models(self, training_data):
        """Train multiple AI models for different aspects"""
        print("Training AI models...")
        
        X = training_data[self.feature_columns]
        
        # Train models for each macro target
        for target in self.target_columns:
            print(f"Training model for {target}...")
            y = training_data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            }
            
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                
                print(f"  {name}: R² = {score:.4f}")
            
            self.models[target] = best_model
            self.scalers[target] = scaler
            print(f"Best model for {target}: R² = {best_score:.4f}")
        
        # Train goal-specific models
        self._train_goal_models(training_data)
        
        # Train meal distribution model
        self._train_meal_distribution_model(training_data)
    
    def _train_goal_models(self, training_data):
        """Train models for different goals"""
        print("Training goal-specific models...")
        
        goals = training_data['goals_processed'].unique()
        
        for goal in goals:
            goal_data = training_data[training_data['goals_processed'] == goal]
            
            if len(goal_data) < 50:  # Skip if not enough data
                continue
            
            X = goal_data[self.feature_columns]
            
            goal_models = {}
            for target in self.target_columns:
                y = goal_data[target]
                
                # Use XGBoost for goal-specific models
                model = xgb.XGBRegressor(n_estimators=50, random_state=42)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                
                goal_models[target] = model
                self.scalers[f'{goal}_{target}'] = scaler
            
            self.goal_models[goal] = goal_models
            print(f"Trained models for goal: {goal}")
    
    def _train_meal_distribution_model(self, meal_plans_df):
        """Train model for meal distribution patterns"""
        print("Training meal distribution model...")
        
        # Analyze meal distribution patterns
        meal_distribution = meal_plans_df.groupby(['exercise_count', 'meal_type']).agg({
            'meal_percentage': 'mean',
            'meal_calories': 'mean',
            'meal_carbs_g': 'mean',
            'meal_protein_g': 'mean',
            'meal_fat_g': 'mean'
        }).reset_index()
        
        # Create distribution rules
        self.meal_distribution_rules = {}
        
        for exercise_count in meal_distribution['exercise_count'].unique():
            exercise_data = meal_distribution[meal_distribution['exercise_count'] == exercise_count]
            
            distribution = {}
            for _, meal in exercise_data.iterrows():
                distribution[meal['meal_type']] = {
                    'percentage': meal['meal_percentage'],
                    'calories': meal['meal_calories'],
                    'carbs': meal['meal_carbs_g'],
                    'protein': meal['meal_protein_g'],
                    'fat': meal['meal_fat_g']
                }
            
            self.meal_distribution_rules[exercise_count] = distribution
        
        print("Meal distribution model trained successfully")
    
    def predict_nutrition_needs(self, user_profile, goal='general_wellness', time_range_days=30):
        """Predict nutrition needs for a user"""
        
        # Prepare user features
        user_features = self._prepare_user_features(user_profile, goal)
        
        # Predict using goal-specific model if available
        if goal in self.goal_models:
            predictions = {}
            for target in self.target_columns:
                model = self.goal_models[goal][target]
                scaler = self.scalers[f'{goal}_{target}']
                
                user_scaled = scaler.transform([user_features])
                prediction = model.predict(user_scaled)[0]
                predictions[target] = max(0, prediction)  # Ensure non-negative
        else:
            # Use general models
            predictions = {}
            for target in self.target_columns:
                model = self.models[target]
                scaler = self.scalers[target]
                
                user_scaled = scaler.transform([user_features])
                prediction = model.predict(user_scaled)[0]
                predictions[target] = max(0, prediction)
        
        # Adjust for time range and goal
        predictions = self._adjust_for_goal_and_time(predictions, user_profile, goal, time_range_days)
        
        return predictions
    
    def _prepare_user_features(self, user_profile, goal):
        """Prepare user features for prediction"""
        features = []
        
        # Basic features
        features.extend([
            user_profile['age'],
            user_profile['height_cm'],
            user_profile['weight_kg'],
            user_profile['body_fat_percentage'],
            user_profile['weight_kg'] / ((user_profile['height_cm'] / 100) ** 2),  # BMI
            user_profile['weight_kg'] * (1 - user_profile['body_fat_percentage'] / 100),  # Lean body mass
            user_profile.get('training_experience_years', 5),
            user_profile.get('adherence_rate', 0.8),
            user_profile.get('exercise_count', 1),
            user_profile.get('total_duration', 60),
            user_profile.get('avg_duration', 60),
            user_profile.get('session_count', 1),
            user_profile.get('intensity_level', 2),
            user_profile.get('total_calories_burned', 300),
            user_profile.get('METs', 5.0),
            user_profile.get('carbWeightRatio', 6.0),
            user_profile.get('proteinWeightRatio', 1.5),
            user_profile.get('exercise_intensity_score', 120)
        ])
        
        # Categorical features
        categorical_features = ['gender', 'sport_league', 'user_scenario', 'competition_level', 'activity_level', 'goals_processed', 'age_group']
        
        for col in categorical_features:
            if col in self.encoders:
                value = user_profile.get(col, 'unknown')
                if col == 'goals_processed':
                    value = goal
                elif col == 'age_group':
                    age = user_profile['age']
                    if age <= 25:
                        value = 'young'
                    elif age <= 35:
                        value = 'adult'
                    elif age <= 45:
                        value = 'middle'
                    else:
                        value = 'senior'
                
                try:
                    encoded_value = self.encoders[col].transform([value])[0]
                except ValueError:
                    # Handle unseen categories
                    encoded_value = 0
                features.append(encoded_value)
            else:
                features.append(0)
        
        return features
    
    def _adjust_for_goal_and_time(self, predictions, user_profile, goal, time_range_days):
        """Adjust predictions based on goal and time range"""
        
        # Goal-based adjustments
        if goal == 'weight_loss':
            predictions['total_daily_calories'] *= 0.85  # 15% calorie deficit
            predictions['total_daily_carbs_g'] *= 0.8   # Reduce carbs
            predictions['total_daily_protein_g'] *= 1.1  # Increase protein
        elif goal == 'weight_gain':
            predictions['total_daily_calories'] *= 1.15  # 15% calorie surplus
            predictions['total_daily_carbs_g'] *= 1.2   # Increase carbs
            predictions['total_daily_protein_g'] *= 1.1  # Increase protein
        elif goal == 'muscle_building':
            predictions['total_daily_calories'] *= 1.1   # Slight surplus
            predictions['total_daily_carbs_g'] *= 1.1   # Increase carbs
            predictions['total_daily_protein_g'] *= 1.3  # Significant protein increase
        elif goal == 'performance_optimization':
            predictions['total_daily_calories'] *= 1.05  # Slight surplus
            predictions['total_daily_carbs_g'] *= 1.15   # Increase carbs for energy
            predictions['total_daily_protein_g'] *= 1.1  # Increase protein
        
        # Time range adjustments
        if time_range_days <= 7:
            # Short-term: more aggressive changes
            if goal == 'weight_loss':
                predictions['total_daily_calories'] *= 0.9
            elif goal == 'weight_gain':
                predictions['total_daily_calories'] *= 1.2
        elif time_range_days <= 30:
            # Medium-term: moderate changes
            pass  # Use base adjustments
        else:
            # Long-term: conservative changes
            if goal == 'weight_loss':
                predictions['total_daily_calories'] *= 0.95
            elif goal == 'weight_gain':
                predictions['total_daily_calories'] *= 1.05
        
        return predictions
    
    def generate_meal_plan(self, user_profile, goal='general_wellness', time_range_days=30, exercise_schedule=None):
        """Generate complete meal plan"""
        
        # Predict nutrition needs
        nutrition_needs = self.predict_nutrition_needs(user_profile, goal, time_range_days)
        
        # Determine exercise count
        exercise_count = len(exercise_schedule) if exercise_schedule else user_profile.get('exercise_count', 0)
        
        # Generate meal schedule
        meal_schedule = self._generate_meal_schedule(exercise_count, exercise_schedule)
        
        # Distribute macros across meals
        meal_plan = self._distribute_macros_to_meals(nutrition_needs, meal_schedule, exercise_count)
        
        return {
            'user_profile': user_profile,
            'goal': goal,
            'time_range_days': time_range_days,
            'total_nutrition': nutrition_needs,
            'exercise_count': exercise_count,
            'meal_plan': meal_plan,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_meal_schedule(self, exercise_count, exercise_schedule=None):
        """Generate meal schedule based on exercise count"""
        
        if exercise_count == 0:
            # Rest day schedule
            return [
                {'meal_type': 'Breakfast', 'time': '9:00 AM', 'meal_key': 'breakfast'},
                {'meal_type': 'Morning Snack', 'time': '11:00 AM', 'meal_key': 'morning_snack'},
                {'meal_type': 'Lunch', 'time': '1:00 PM', 'meal_key': 'lunch'},
                {'meal_type': 'Afternoon Snack', 'time': '4:00 PM', 'meal_key': 'afternoon_snack'},
                {'meal_type': 'Dinner', 'time': '7:00 PM', 'meal_key': 'dinner'}
            ]
        elif exercise_count == 1:
            # Single exercise day
            return [
                {'meal_type': 'Pre-Exercise Snack', 'time': '8:00 AM', 'meal_key': 'pre_snack'},
                {'meal_type': 'Post-Exercise Meal', 'time': '11:00 AM', 'meal_key': 'post_meal'},
                {'meal_type': 'Lunch', 'time': '3:00 PM', 'meal_key': 'meal_2'},
                {'meal_type': 'Recovery Snack', 'time': '6:00 PM', 'meal_key': 'recovery_snack'},
                {'meal_type': 'Dinner', 'time': '8:00 PM', 'meal_key': 'dinner'}
            ]
        else:
            # Multiple exercise day
            return [
                {'meal_type': 'Pre-Exercise Snack 1', 'time': '7:00 AM', 'meal_key': 'pre_snack_1'},
                {'meal_type': 'Post-Exercise Meal 1', 'time': '10:00 AM', 'meal_key': 'post_meal_1'},
                {'meal_type': 'Pre-Exercise Snack 2', 'time': '2:00 PM', 'meal_key': 'pre_snack_2'},
                {'meal_type': 'Post-Exercise Meal 2', 'time': '5:00 PM', 'meal_key': 'post_meal_2'},
                {'meal_type': 'Evening Meal', 'time': '8:00 PM', 'meal_key': 'evening_meal'},
                {'meal_type': 'Recovery Snack', 'time': '10:00 PM', 'meal_key': 'recovery_snack'}
            ]
    
    def _distribute_macros_to_meals(self, nutrition_needs, meal_schedule, exercise_count):
        """Distribute macros across meals using trained patterns"""
        
        # Get distribution rules
        if exercise_count in self.meal_distribution_rules:
            distribution_rules = self.meal_distribution_rules[exercise_count]
        else:
            # Default distribution
            distribution_rules = self._get_default_distribution(exercise_count)
        
        meal_plan = []
        
        for meal_info in meal_schedule:
            meal_type = meal_info['meal_type']
            meal_key = meal_info['meal_key']
            
            if meal_type in distribution_rules:
                distribution = distribution_rules[meal_type]
            else:
                # Use default percentages
                distribution = self._get_default_meal_distribution(meal_key, exercise_count)
            
            meal_data = {
                'meal_type': meal_type,
                'time': meal_info['time'],
                'calories': round(nutrition_needs['total_daily_calories'] * distribution['percentage'] / 100, 1),
                'carbs_g': round(nutrition_needs['total_daily_carbs_g'] * distribution['percentage'] / 100, 1),
                'protein_g': round(nutrition_needs['total_daily_protein_g'] * distribution['percentage'] / 100, 1),
                'fat_g': round(nutrition_needs['total_daily_fat_g'] * distribution['percentage'] / 100, 1),
                'percentage': distribution['percentage']
            }
            
            meal_plan.append(meal_data)
        
        return meal_plan
    
    def _get_default_distribution(self, exercise_count):
        """Get default meal distribution rules"""
        if exercise_count == 0:
            return {
                'Breakfast': {'percentage': 25.0},
                'Morning Snack': {'percentage': 10.0},
                'Lunch': {'percentage': 30.0},
                'Afternoon Snack': {'percentage': 10.0},
                'Dinner': {'percentage': 25.0}
            }
        elif exercise_count == 1:
            return {
                'Pre-Exercise Snack': {'percentage': 10.0},
                'Post-Exercise Meal': {'percentage': 30.0},
                'Lunch': {'percentage': 30.0},
                'Recovery Snack': {'percentage': 10.0},
                'Dinner': {'percentage': 20.0}
            }
        else:
            return {
                'Pre-Exercise Snack 1': {'percentage': 8.0},
                'Post-Exercise Meal 1': {'percentage': 25.0},
                'Pre-Exercise Snack 2': {'percentage': 8.0},
                'Post-Exercise Meal 2': {'percentage': 25.0},
                'Evening Meal': {'percentage': 24.0},
                'Recovery Snack': {'percentage': 10.0}
            }
    
    def _get_default_meal_distribution(self, meal_key, exercise_count):
        """Get default distribution for a specific meal"""
        default_percentages = {
            'breakfast': 25.0,
            'morning_snack': 10.0,
            'lunch': 30.0,
            'afternoon_snack': 10.0,
            'dinner': 25.0,
            'pre_snack': 10.0,
            'post_meal': 30.0,
            'meal_2': 30.0,
            'recovery_snack': 10.0,
            'pre_snack_1': 8.0,
            'post_meal_1': 25.0,
            'pre_snack_2': 8.0,
            'post_meal_2': 25.0,
            'evening_meal': 24.0
        }
        
        percentage = default_percentages.get(meal_key, 10.0)
        return {'percentage': percentage}
    
    def save_models(self, model_path):
        """Save trained models"""
        import os
        os.makedirs(model_path, exist_ok=True)
        
        # Save main models
        for target, model in self.models.items():
            joblib.dump(model, f'{model_path}/{target}_model.pkl')
        
        # Save scalers
        for target, scaler in self.scalers.items():
            joblib.dump(scaler, f'{model_path}/{target}_scaler.pkl')
        
        # Save encoders
        joblib.dump(self.encoders, f'{model_path}/encoders.pkl')
        
        # Save goal models
        joblib.dump(self.goal_models, f'{model_path}/goal_models.pkl')
        
        # Save meal distribution rules
        joblib.dump(self.meal_distribution_rules, f'{model_path}/meal_distribution_rules.pkl')
        
        print(f"Models saved to {model_path}")
    
    def load_models(self, model_path):
        """Load trained models"""
        import os
        
        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist")
            return False
        
        try:
            # Load main models
            for target in self.target_columns:
                model_file = f'{model_path}/{target}_model.pkl'
                if os.path.exists(model_file):
                    self.models[target] = joblib.load(model_file)
            
            # Load scalers
            for target in self.target_columns:
                scaler_file = f'{model_path}/{target}_scaler.pkl'
                if os.path.exists(scaler_file):
                    self.scalers[target] = joblib.load(scaler_file)
            
            # Load encoders
            encoders_file = f'{model_path}/encoders.pkl'
            if os.path.exists(encoders_file):
                self.encoders = joblib.load(encoders_file)
            
            # Load goal models
            goal_models_file = f'{model_path}/goal_models.pkl'
            if os.path.exists(goal_models_file):
                self.goal_models = joblib.load(goal_models_file)
            
            # Load meal distribution rules
            distribution_file = f'{model_path}/meal_distribution_rules.pkl'
            if os.path.exists(distribution_file):
                self.meal_distribution_rules = joblib.load(distribution_file)
            
            print(f"Models loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

if __name__ == "__main__":
    # Initialize AI model
    ai_model = MealPlanAI()
    
    # Load and prepare data
    athletes_df, meal_plans_df, training_df, sports_df = ai_model.load_data('../dataset/data-collection')
    
    # Prepare features
    training_data = ai_model.prepare_features(athletes_df, meal_plans_df, training_df, sports_df)
    
    # Train models
    ai_model.train_models(training_data)
    
    # Save models
    ai_model.save_models('./trained_models')
    
    print("AI model training completed successfully!")
