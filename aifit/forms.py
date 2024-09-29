from django import forms
from .models import UserProfile, ActivityLog

from django import forms

class UserProfileForm(forms.Form):
    name = forms.CharField(label='Name', max_length=100)
    age = forms.IntegerField(label='Age')
    gender = forms.ChoiceField(label='Gender', choices=[('male', 'Male'), ('female', 'Female')])
    activity_level = forms.ChoiceField(label='Activity Level', choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High')
    ])
    sleep_hours = forms.DecimalField(label='Sleep Hours')
    diet_type = forms.ChoiceField(label='Diet Type', choices=[
        ('vegan', 'Vegan'),
        ('vegetarian', 'Vegetarian'),
        ('non-vegetarian', 'Non-Vegetarian'),
    ])
    mental_state = forms.ChoiceField(label='Mental State', choices=[
        ('good', 'Good'),
        ('average', 'Average'),
        ('bad', 'Bad'),
    ])

    def __init__(self, *args, **kwargs):
        super(UserProfileForm, self).__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})  # Add Bootstrap class to all fields


class ActivityLogForm(forms.ModelForm):
    class Meta:
        model = ActivityLog
        fields = ['activity_type', 'duration_minutes', 'mood_before', 'mood_after']


from .models import Workout, Meal, Progress, User  # Assuming User is in models

class ProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password']  # Include other fields as necessary

class WorkoutForm(forms.ModelForm):
    class Meta:
        model = Workout
        fields = ['exercise_name', 'sets', 'reps', 'weight']

class MealForm(forms.ModelForm):
    class Meta:
        model = Meal
        fields = ['meal_type', 'calories', 'protein', 'fats', 'carbs']

class ProgressForm(forms.ModelForm):
    class Meta:
        model = Progress
        fields = ['weight', 'body_fat_percentage', 'measurements']  # Include other fields as necessary
