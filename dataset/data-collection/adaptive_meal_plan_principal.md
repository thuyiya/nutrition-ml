## Core Principles

1. **Real-time Adaptation**: The system continuously monitors meal consumption and adjusts future meals based on what has already been consumed.

2. **Nutritional Balance**: Adjustments maintain the proper balance of macronutrients (proteins, carbohydrates, and fats).

3. **Meal Hierarchy**: Main meals receive proportionally larger adjustments than snacks.

4. **Safety Limits**: Adjustments are capped to prevent extreme changes to meal recommendations.

5. **User-Centric**: The system works invisibly, providing seamless nutrition guidance without disrupting the user experience.

## How It Works

### Detecting Missed Nutrition

The system identifies nutritional deficits or surpluses by:

1. Comparing the recommended nutrition values with actual consumed values
2. Checking if a meal's scheduled time has passed
3. Determining if the meal meets its nutritional targets

### Calculating Adjustments

When a nutritional gap is detected:

1. **Calculate the deficit/surplus**: 
   - Deficit = Recommended nutrition - Consumed nutrition
   - This is calculated separately for calories, proteins, and carbohydrates

2. **Distribute among remaining meals**:
   - The deficit is distributed among all remaining meals and snacks
   - Main meals receive twice the adjustment of snacks
   - Formula: Adjustment per unit = Deficit ÷ (Remaining meals × 2 + Remaining snacks)

3. **Apply safety limits**:
   - Maximum adjustment is capped at 5% of the original meal value
   - This prevents extreme changes that could make meals unhealthy

### Applying Adjustments

Adjustments are applied to:
- Future meals only (meals that haven't been consumed yet)
- All macronutrients (calories, proteins, carbohydrates)
- Regular meals and snacks (excluding specialized pre-workout snacks)

### Cleanup Process

The system maintains data integrity by:
- Removing adjustments from meals that have already passed
- Clearing zero-value adjustments
- Resetting at the end of each day

## Example Scenarios

### Scenario 1: Missed Breakfast

When a user skips breakfast entirely:

- **Nutritional deficit**: Full breakfast nutrition (e.g., 500 calories)
- **Distribution**: Deficit is spread across lunch, dinner, and snacks
- **Result**: Lunch and dinner get larger increases, snacks get smaller increases

### Scenario 2: Partially Consumed Lunch

When a user only eats half their lunch:

- **Nutritional deficit**: Half of lunch nutrition (e.g., 300 calories)
- **Distribution**: Deficit is spread across remaining afternoon snack and dinner
- **Result**: Dinner gets a larger increase, afternoon snack gets a smaller increase

### Scenario 3: Over-consumed Meal

When a user eats more than recommended:

- **Nutritional surplus**: Extra consumed nutrition (e.g., 200 extra calories)
- **Distribution**: Surplus is subtracted from remaining meals
- **Result**: Future meals get reduced recommendations to compensate

## Implementation Considerations

### When Adjustments Apply

Adjustments are applied when:
- A meal's scheduled time has passed
- The meal doesn't meet its nutritional recommendations
- There are remaining meals/snacks in the day
- Custom nutrition plans don't override the system

### When Adjustments Don't Apply

Adjustments are not applied when:
- It's the last meal of the day (nothing left to adjust)
- All nutritional recommendations have been perfectly met
- A custom nutrition plan overrides the system
- No remaining meals/snacks are available

### Special Cases

- **Pre-workout snacks**: These are excluded from adjustments to maintain exercise performance
- **Custom nutrition plans**: These can override the adaptation system when needed
- **Last meal of day**: No adjustments are made as there are no future meals to adjust

## Benefits

1. **Flexible Nutrition**: Users can miss meals without compromising daily nutrition targets
2. **Balanced Approach**: Maintains proper macronutrient ratios throughout adjustments
3. **Seamless Experience**: Works automatically without requiring user intervention
4. **Personalized**: Adapts to each user's unique eating patterns and schedule
5. **Safety-First**: Prevents extreme meal recommendations through capping mechanisms

## Technical Requirements

For successful implementation, the system needs:

1. Real-time tracking of meal consumption
2. Accurate timestamps for meal scheduling
3. User timezone information for proper timing
4. Meal type classification (meal vs. snack)
5. Nutritional data for all meals (calories, proteins, carbohydrates)

## Adjustment Algorithm

```
FUNCTION mealAdjustmentAlgorithm(meals, currentTime, timezone):
    FOR each meal in meals:
        IF meal.startDate <= currentTime AND meal is not last meal:
            IF meal does not meet recommendations:
                deficit = calculateDeficit(meal)
                remainingMeals = countRemainingMeals(meals, currentIndex)
                
                IF remainingMeals > 0:
                    adjustmentPerMeal = deficit / (remainingMeals * 2 + remainingSnacks)
                    
                    FOR each future meal:
                        IF future meal is not pre-workout snack:
                            adjustment = adjustmentPerMeal * (2 if meal, 1 if snack)
                            
                            IF adjustment > 5% of meal value:
                                adjustment = 5% of meal value
                            
                            futureMeal.adjustments = adjustment
    
    cleanupOldAdjustments(meals, currentTime)
    RETURN meals
```

This adaptation system ensures users receive optimal nutrition guidance even when their eating patterns deviate from recommendations, supporting long-term nutritional success while maintaining flexibility for real-life situations.