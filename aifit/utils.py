# trainer/utils.py

def generate_diet_plan(diet_type):
    # Example meal plans for each diet type
    diet_plans = {
        'vegetarian': {
            'breakfast': 'Oatmeal with fruits and nuts',
            'lunch': 'Quinoa salad with chickpeas and avocado',
            'dinner': 'Stir-fried tofu with vegetables and brown rice',
            'snacks': 'Greek yogurt with honey, carrots, and hummus'
        },
        'vegan': {
            'breakfast': 'Smoothie with spinach, banana, and almond butter',
            'lunch': 'Lentil soup with whole-grain bread',
            'dinner': 'Vegan burrito with black beans, avocado, and salsa',
            'snacks': 'Fruit salad, nuts, and seeds'
        },
        'keto': {
            'breakfast': 'Scrambled eggs with spinach and avocado',
            'lunch': 'Grilled chicken salad with olive oil dressing',
            'dinner': 'Baked salmon with steamed broccoli',
            'snacks': 'Almonds, cheese slices, and cucumber'
        },
        'mediterranean': {
            'breakfast': 'Greek yogurt with berries and honey',
            'lunch': 'Grilled chicken with tabbouleh and hummus',
            'dinner': 'Baked fish with roasted vegetables and olive oil',
            'snacks': 'Olives, nuts, and fresh fruit'
        },
        'paleo': {
            'breakfast': 'Egg muffins with spinach and bacon',
            'lunch': 'Turkey lettuce wraps with avocado and tomato',
            'dinner': 'Grilled steak with sweet potatoes and asparagus',
            'snacks': 'Mixed nuts, beef jerky, and apple slices'
        },
        'custom': {
            'breakfast': 'Choose your preferred breakfast based on your needs',
            'lunch': 'Choose your lunch with lean proteins and vegetables',
            'dinner': 'Choose a balanced meal with healthy fats, proteins, and carbs',
            'snacks': 'Choose nutrient-dense snacks like nuts, seeds, and fruits'
        }
    }

    # Return the selected diet plan
    return diet_plans.get(diet_type, diet_plans['custom'])
