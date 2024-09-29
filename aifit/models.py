from django.db import models
from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    DIET_CHOICES = [
        ('vegetarian', 'Vegetarian'),
        ('vegan', 'Vegan'),
        ('keto', 'Keto'),
        ('mediterranean', 'Mediterranean'),
        ('paleo', 'Paleo'),
        ('custom', 'Custom'),
    ]

    name = models.CharField(max_length=255)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    activity_level = models.CharField(max_length=50)
    sleep_hours = models.DecimalField(max_digits=4, decimal_places=1)
    diet_type = models.CharField(max_length=15, choices=DIET_CHOICES, default='custom')
    mental_state = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

class Recommendation(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    generated_text = models.TextField()  # The AI-generated recommendation
    created_at = models.DateTimeField(auto_now_add=True)

class ActivityLog(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    activity_type = models.CharField(max_length=100)  # e.g., Yoga, Running
    duration_minutes = models.IntegerField()
    mood_before = models.CharField(max_length=100)
    mood_after = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)


class ChatMessage(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User: {self.user_message}, Bot: {self.bot_response}"

class User(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    password = models.CharField(max_length=255)
    # Additional fields...

class Workout(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField()
    exercise_name = models.CharField(max_length=100)
    sets = models.IntegerField()
    reps = models.IntegerField()
    weight = models.FloatField()


class Meal(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField()
    meal_type = models.CharField(max_length=50)  # Breakfast, Lunch, Dinner, Snack
    calories = models.FloatField()
    protein = models.FloatField()
    fats = models.FloatField()
    carbs = models.FloatField()

class Progress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField()
    weight = models.FloatField()
    body_fat_percentage = models.FloatField()
    measurements = models.JSONField()  # Store measurements as JSON


# models.py



class Goal(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    completed = models.BooleanField(default=False)

    def __str__(self):
        return self.title
