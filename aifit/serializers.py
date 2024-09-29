from rest_framework import serializers
from .models import UserProfile, Recommendation, ActivityLog,Workout,Meal,Progress

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = '__all__'

class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = '__all__'

class ActivityLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ActivityLog
        fields = '__all__'






from rest_framework import serializers
from .models import Workout, Meal, Progress, Goal


class WorkoutSerializer(serializers.ModelSerializer):
    class Meta:
        model = Workout
        fields = ['date', 'weight']  # Add other fields as necessary

    def create(self, validated_data):
        user = validated_data.pop('user')  # Remove user from validated data
        workout = Workout.objects.create(user=user, **validated_data)
        return workout


class MealSerializer(serializers.ModelSerializer):
    class Meta:
        model = Meal
        fields = ['date', 'calories']  # Add other fields as necessary

    def create(self, validated_data):
        user = validated_data.pop('user')
        meal = Meal.objects.create(user=user, **validated_data)
        return meal


class ProgressSerializer(serializers.ModelSerializer):
    class Meta:
        model = Progress
        fields = ['date', 'weight']  # Add other fields as necessary

    def create(self, validated_data):
        user = validated_data.pop('user')
        progress = Progress.objects.create(user=user, **validated_data)
        return progress


class GoalSerializer(serializers.ModelSerializer):
    class Meta:
        model = Goal
        fields = ['title', 'completed']  # Add other fields as necessary

