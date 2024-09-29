from rest_framework import generics

from .gemini_ai_service import generation_config
from .serializers import UserProfileSerializer, RecommendationSerializer, ActivityLogSerializer
import openai  # For GPT-based content generation (example)
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.views import View
from django.contrib.auth import logout
from .models import UserProfile, Recommendation, ActivityLog
from .forms import UserProfileForm, ActivityLogForm
from .models import UserProfile
from .forms import UserProfileForm
from .utils import generate_diet_plan
openai.api_key = 'your-openai-api-key'
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm


import cv2
import mediapipe as mp
import numpy as np
from django.http import StreamingHttpResponse
from django.shortcuts import render
class UserProfileListCreate(generics.ListCreateAPIView):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer


class ActivityLogListCreate(generics.ListCreateAPIView):
    queryset = ActivityLog.objects.all()
    serializer_class = ActivityLogSerializer


class RecommendationListCreate(generics.ListCreateAPIView):
    queryset = Recommendation.objects.all()
    serializer_class = RecommendationSerializer

    def perform_create(self, serializer):
        user = serializer.validated_data['user']

        # Example AI recommendation based on user data
        prompt = f"Create a mindfulness recommendation for a {user.age}-year-old {user.activity_level} individual, who feels {user.mental_state}."

        # Use an AI API like GPT to generate a recommendation
        ai_response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        generated_text = ai_response['choices'][0]['text']

        # Save the recommendation
        serializer.save(generated_text=generated_text)


# trainer/views.py



class UserProfileView(View):
    def get(self, request):
        form = UserProfileForm()
        return render(request, 'user_profile_form.html', {'form': form})

    def post(self, request):
        form = UserProfileForm(request.POST)
        if form.is_valid():
            # Save the user profile
            profile = form.save()

            # Generate a diet plan based on the diet type selected
            diet_plan = generate_diet_plan(profile.diet_type)

            # Pass the diet plan to the success page
            return render(request, 'diet_plan.html', {'diet_plan': diet_plan, 'profile': profile})

        return render(request, 'user_profile_form.html', {'form': form})


class ActivityLogView(View):
    def get(self, request):
        form = ActivityLogForm()
        return render(request, 'activity_log_form.html', {'form': form})

    def post(self, request):
        form = ActivityLogForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('activity-log-success')
        return render(request, 'activity_log_form.html', {'form': form})

class RecommendationView(View):
    def get(self, request, user_id):
        user = UserProfile.objects.get(id=user_id)
        recommendations = Recommendation.objects.filter(user=user)
        return render(request, 'recommendation_list.html', {'recommendations': recommendations})

def success(request):
    return render(request,'success.html')

def diet_plan(request):
    return render(request,'diet_plan.html')

def index(request):
    return render(request,'index.html')


# your_app/views.py



# aifit/views.py



def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')  # Use .get() method
        password = request.POST.get('password')  # Use .get() method

        # Check if both fields are filled
        if username and password:
            # Authenticate the user
            user = authenticate(request, username=username, password=password)

            if user is not None:
                # Login the user
                login(request, user)
                messages.success(request, 'You are now logged in!')
                return redirect('index')  # Redirect to home or dashboard page
            else:
                messages.error(request, 'Invalid username or password')
                return redirect('login')
        else:
            messages.error(request, 'Please fill in both fields.')
            return redirect('login')

    # If GET request, display the login form
    return render(request, 'login.html')



# your_app/views.py


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            return redirect('login')  # After registration, redirect to login page
    else:
        form = UserCreationForm()

    return render(request, 'register.html', {'form': form})

def custom_logout(request):
    # Log the user out
    logout(request)
    # Set a success message
    messages.success(request, 'You have successfully logged out.')
    # Redirect to the login page or a different page
    return redirect('login')  # Ensure 'login' is a valid URL name

# Other view functions can be defined here

from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
import mediapipe as mp
import numpy as np

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


# Function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# Function to generate frames
def exercise_tracker(exercise_choice):
    cap = cv2.VideoCapture(0)  # Start capturing from the webcam

    # Initialize counters and variables
    squat_counter, dumbbell_counter, weight_lifting_counter, pullup_counter = 0, 0, 0, 0
    stage = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get required landmarks
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Exercise tracking logic based on selected exercise
            if exercise_choice == "squats":
                angle_knee = calculate_angle(left_hip, left_knee, left_ankle)
                if angle_knee > 160:
                    stage = "up"
                if angle_knee < 90 and stage == "up":
                    stage = "down"
                    squat_counter += 1
                cv2.putText(image, f'Squats: {squat_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            elif exercise_choice == "dumbbell":
                angle_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)
                if angle_elbow > 160:
                    stage = "down"
                if angle_elbow < 30 and stage == "down":
                    stage = "up"
                    dumbbell_counter += 1
                cv2.putText(image, f'Dumbbell Lifting: {dumbbell_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2)

        except Exception as e:
            print(f"Error: {e}")
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Encode the frame to send to the browser
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


# Video feed view for streaming video
def video_feed(request):
    exercise_choice = request.GET.get('exercise', 'squats')  # Default to 'squats'
    return StreamingHttpResponse(exercise_tracker(exercise_choice),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


# Render HTML page
def tracker_view(request):
    return render(request, 'tracker.html')


# Render HTML page
def tracker_view(request):
    return render(request, 'tracker.html')

def dashboard(request):
    return render(request, 'dashboard.html')
def home(request):
    return render(request, 'home.html')
def logtemp(request):
    return render(request, 'logtemp.html')

from django.shortcuts import redirect, render
from fitnesbee.settings import GENERATIVE_AI_KEY
from aifit.models import ChatMessage
import google.generativeai as genai

def send_message(request):
    if request.method == 'POST':
        genai.configure(api_key=GENERATIVE_AI_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            # safety_settings = Adjust safety settings
            # See https://ai.google.dev/gemini-api/docs/safety-settings
            system_instruction="""
        You are an expert personal trainer, specializing in fitness, exercise routines, and nutrition guidance. You should only respond to prompts related to physical fitness, exercise plans, and health advice. For other types of questions, kindly inform the user that you're here to help with fitness-related queries.

        Your goal is to provide personalized, actionable fitness advice, helping users achieve their physical health goals. You are trained in strength training, cardiovascular fitness, flexibility exercises, and general nutrition. You provide detailed and structured workout plans, nutrition tips, and injury prevention advice.

        Always approach conversations with motivation and encouragement, ensuring users feel confident and empowered to reach their fitness goals. Maintain an energetic and supportive tone, especially when users express doubts or lack motivation.

        Be direct and clear, providing specific recommendations on exercise form, duration, and intensity, while adapting to each user's fitness level and goals. Encourage users to challenge themselves within safe and healthy limits.

        You help users with the following:

        1. *Fitness Assessment*: Evaluate user fitness levels and recommend suitable exercise routines.
        2. *Workout Plans*: Provide structured workout routines (e.g., cardio, strength training, flexibility, or HIIT) tailored to users' needs.
        3. *Exercise Guidance*: Correct exercise form and provide modifications for different fitness levels.
        4. *Motivation*: Keep users motivated and accountable, offering words of encouragement and progress tracking strategies.
        5. *Nutrition Advice*: Offer general nutrition tips that support fitness goals, such as healthy meal ideas or post-workout snacks.
        6. *Injury Prevention*: Suggest warm-up, cool-down, and recovery practices to avoid injury and enhance performance.
        7. *Goal Setting*: Help users set realistic, measurable fitness goals, and provide strategies to achieve them.

        Use a motivational, energetic, and supportive style, making users feel excited about their fitness journey. Provide structured responses with clear action points, like reps, sets, and exercise suggestions, or meal recommendations for a balanced approach to fitness.
        """,
        )

        user_message = request.POST.get('user_message')
        bot_response = model.generate_content(user_message)

        ChatMessage.objects.create(user_message=user_message, bot_response=bot_response.text)

    return redirect('list_messages')
def send_message1(request):
    if request.method == 'POST':
        genai.configure(api_key=GENERATIVE_AI_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro",generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
         system_instruction="You are an expert mental therapist, specializing in emotional support and psychological guidance. You should only respond to emotional , mental and skill related prompts otherwise tell them youre here for mental support cannot implement that. Your goal is to help users navigate through their mental and emotional challenges with compassion, understanding, and expert advice. You are trained in cognitive behavioral therapy (CBT), mindfulness, and active listening techniques. You provide a safe, non-judgmental space for users to express their thoughts and feelings.\n\nAlways approach conversations with empathy, making users feel heard and understood.\nMaintain a calm and reassuring tone, especially when users are anxious or distressed, instilling calmness and hope.\nBe non-judgmental, encouraging open communication, ensuring users feel safe to express themselves.\nProvide supportive and encouraging feedback, offering gentle motivation and reinforcing users' strengths.\nUse a friendly and approachable style that makes users feel comfortable and at ease.\nHelp users with the following:\n\nActive Listening: Reflect on what users say and validate their emotions.\nEmotion Identification and Regulation: Help users understand their emotions and offer techniques to regulate them.\nCognitive Behavioral Therapy (CBT): Guide users through reframing negative thoughts and help them see situations from a healthier perspective.\nMindfulness Guidance: Provide simple mindfulness exercises like breathing techniques or guided meditation to reduce stress.\nEncouraging Self-Care: Remind users of the importance of self-care, promoting habits that support mental well-being.\nCrisis Support: Offer immediate emotional support during distress and recommend professional help when needed.",
)

        user_message = request.POST.get('user_message')
        bot_response = model.generate_content(user_message)

        ChatMessage.objects.create(user_message=user_message, bot_response=bot_response.text)

    return redirect('list_messages1')
def list_messages1(request):
    messages = ChatMessage.objects.all()
    return render(request, 'list_messages1.html', { 'messages': messages })
def list_messages(request):
    messages = ChatMessage.objects.all()
    return render(request, 'list_messages.html', { 'messages': messages })

from django.shortcuts import redirect

# View to clear the messages
def clear_messages(request):
    if request.method == 'POST':
        # Assuming you are storing the messages in the database, delete all messages
        ChatMessage.objects.all().delete()
        return redirect('send_message')  # Redirect to the chat home



@csrf_exempt  # Use with caution; ensure you handle CSRF in production
def generate_diet_plan(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        profile_data = data.get('profile_data', {})


        diet_plan = f"Based on your profile, here's a healthy diet plan for you!"

        # Return the diet plan as a JSON response
        return JsonResponse({'plan': diet_plan})

    return JsonResponse({'error': 'Invalid request'}, status=400)
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from .models import Workout, Meal, Progress, Goal
from .serializers import WorkoutSerializer, MealSerializer, ProgressSerializer, GoalSerializer
from rest_framework.permissions import IsAuthenticated

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_workouts(request):
    workouts = Workout.objects.filter(user=request.user)
    serializer = WorkoutSerializer(workouts, many=True)
    return Response(serializer.data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_workout(request):
    serializer = WorkoutSerializer(data=request.data)
    if serializer.is_valid():
        # Associate the request.user with the new Workout instance
        workout = serializer.save(user=request.user)
        return Response({'message': 'Workout created successfully.', 'workout': WorkoutSerializer(workout).data}, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_meals(request):
    meals = Meal.objects.filter(user=request.user)
    serializer = MealSerializer(meals, many=True)
    return Response(serializer.data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_meal(request):
    serializer = MealSerializer(data=request.data)
    if serializer.is_valid():
        # Associate the request.user with the new Meal instance
        meal = serializer.save(user=request.user)
        return Response({'message': 'Meal created successfully.', 'meal': MealSerializer(meal).data}, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_progress(request):
    progress = Progress.objects.filter(user=request.user)
    serializer = ProgressSerializer(progress, many=True)
    return Response(serializer.data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_progress(request):
    serializer = ProgressSerializer(data=request.data)
    if serializer.is_valid():
        # Associate the request.user with the new Progress instance
        progress = serializer.save(user=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_goals(request):
    goals = Goal.objects.filter(user=request.user)
    serializer = GoalSerializer(goals, many=True)
    return Response(serializer.data)

import json
import urllib.request
import torch
from PIL import Image
import cv2
from django.http import StreamingHttpResponse
from django.shortcuts import render
from torchvision import models, transforms

# Load ImageNet labels from the URL
LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
response = urllib.request.urlopen(LABELS_URL)
IMAGENET_LABELS = json.loads(response.read())

# Define the ResNet model and set it to evaluation mode
model = models.resnet50(pretrained=True)
model.eval()

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Nutrition information for some foods
FOOD_NUTRITION_INFO = {
    'banana': {'calories': 89, 'protein': 1.1, 'fat': 0.3, 'carbs': 23},
    'pizza': {'calories': 266, 'protein': 11, 'fat': 10, 'carbs': 33},
    'broccoli': {'calories': 55, 'protein': 3.7, 'fat': 0.6, 'carbs': 11},
    'carrot': {'calories': 41, 'protein': 0.9, 'fat': 0.2, 'carbs': 10},
    'apple': {'calories': 52, 'protein': 0.3, 'fat': 0.2, 'carbs': 14},
    'orange': {'calories': 47, 'protein': 0.9, 'fat': 0.1, 'carbs': 12},
    'sandwich': {'calories': 205, 'protein': 7.8, 'fat': 9.6, 'carbs': 22},
    'cake': {'calories': 257, 'protein': 3.6, 'fat': 11.5, 'carbs': 37},
    'hotdog': {'calories': 290, 'protein': 10, 'fat': 27, 'carbs': 2.5},
    'hamburger': {'calories': 295, 'protein': 17, 'fat': 15, 'carbs': 26},
    'spaghetti': {'calories': 158, 'protein': 5.8, 'fat': 0.9, 'carbs': 31},
    'donut': {'calories': 452, 'protein': 4.9, 'fat': 25, 'carbs': 51},
    'french loaf': {'calories': 274, 'protein': 7.9, 'fat': 1.8, 'carbs': 56},
    'burrito': {'calories': 206, 'protein': 6.3, 'fat': 6.8, 'carbs': 30},
    'bagel': {'calories': 245, 'protein': 9.8, 'fat': 1.2, 'carbs': 48},
    'pretzel': {'calories': 380, 'protein': 8, 'fat': 4, 'carbs': 80},
}

# Generate a description of the food item
def generate_food_description(food_item):
    descriptions = {
        'banana': 'Bananas are a popular tropical fruit, rich in potassium and vitamins.',
        'pizza': 'Pizza is a savory dish from Italy consisting of a flatbread topped with tomato sauce, cheese, and various toppings.',
        'broccoli': 'Broccoli is a green vegetable that is high in fiber and protein, and is often used in salads and stir-fries.',
        'orange': 'Orange is a citrus fruit which is rich in vitamin C',
    }
    return descriptions.get(food_item, "No description available for this food item.")

# Food identification function using PyTorch
def identify_food_pytorch(frame):
    img = Image.fromarray(frame)
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    _, predicted_idx = outputs.max(1)
    predicted_label = IMAGENET_LABELS[predicted_idx]

    if predicted_label in FOOD_NUTRITION_INFO:
        return predicted_label
    else:
        return None

# Video streaming function
def video_stream(request):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        food_label = identify_food_pytorch(frame_rgb)

        if food_label:
            nutrition_info = FOOD_NUTRITION_INFO[food_label]
            description = generate_food_description(food_label)

            text = f"{food_label.title()}: {nutrition_info['calories']} kcal, Prot: {nutrition_info['protein']}g, Fat: {nutrition_info['fat']}g, Carbs: {nutrition_info['carbs']}g"
            description_text = f"Description: {description}"

            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, description_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(frame, "Not Food", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Stream response
def stream(request):
    return StreamingHttpResponse(video_stream(request), content_type='multipart/x-mixed-replace; boundary=frame')

# Render the HTML page
def food_recognition(request):
    return render(request, 'food_recognition.html')

# meal_planner/views.py

from django.shortcuts import render
import random
from google.generativeai import ChatSession
import os

# Load .env file
GOOGLE_API_KEY = os.getenv('GENERATIVE_AI_KEY')

# Set up the chat session with Google Generative AI
session = ChatSession(GOOGLE_API_KEY)

# Food data for different meals
food_items = {
    "breakfast": {
        "protein": {"eggs": 78, "greek_yogurt": 130, "cottage_cheese": 206, "turkey_slices": 104, "smoked_salmon": 117},
        "whole_grains": {"whole_wheat_bread": 79, "oatmeal": 150, "quinoa": 222, "whole_grain_cereal": 120, "granola": 494},
        "fruits": {"berries": 50, "bananas": 96, "apples": 52, "oranges": 62, "grapefruit": 52, "melon_slices": 30},
        "vegetables": {"spinach": 7, "tomatoes": 18, "avocado": 160, "bell_peppers": 25, "mushrooms": 15},
        "healthy_fats": {"nut_butter": 94, "nuts": 163, "chia_seeds": 58, "flaxseeds": 55, "avocado_slices": 50},
        "dairy": {"milk": 103, "cheese": 113, "yogurt": 150, "dairy-free_alternatives": 80},
        "other": {"honey": 64, "maple_syrup": 52, "coffee": 2, "jam": 49, "peanut_butter": 188, "cocoa_powder": 12}
    },
    "lunch": {
        "protein": {"chicken_breast": 165, "tofu": 144, "salmon": 206, "lean_beef": 250, "black_beans": 114},
        "whole_grains": {"brown_rice": 215, "quinoa": 222, "whole_wheat_pasta": 174, "barley": 123},
        "vegetables": {"broccoli": 55, "carrots": 41, "kale": 33, "bell_peppers": 25, "zucchini": 21},
        "healthy_fats": {"olive_oil": 119, "avocado": 160, "nuts": 163},
        "other": {"hummus": 100, "salsa": 36}
    },
    "dinner": {
        "protein": {"grilled_chicken": 165, "fish": 206, "pork_chops": 283, "tofu": 144},
        "whole_grains": {"quinoa": 222, "brown_rice": 215, "barley": 123},
        "vegetables": {"asparagus": 20, "brussels_sprouts": 38, "spinach": 7},
        "healthy_fats": {"olive_oil": 119, "nuts": 163},
        "other": {"soup": 100, "salad": 50}
    }
}

# Helper functions
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        bmr = 9.99 * weight + 6.25 * height - 4.92 * age + 5
    else:
        bmr = 9.99 * weight + 6.25 * height - 4.92 * age - 161
    return round(bmr, 2)

def knapsack(target_calories, food_groups):
    items = []
    for group, foods in food_groups.items():
        for item, calories in foods.items():
            items.append((calories, item))

    n = len(items)
    dp = [[0 for _ in range(target_calories + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(target_calories + 1):
            value, _ = items[i - 1]

            if value > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - value] + value)

    selected_items = []
    j = target_calories
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            _, item = items[i - 1]
            selected_items.append(item)
            j -= items[i - 1][0]

    return selected_items, dp[n][target_calories]

def meal_plan_view(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        age = int(request.POST.get('age'))
        weight = int(request.POST.get('weight'))
        height = int(request.POST.get('height'))
        gender = request.POST.get('gender')

        # Calculate BMR
        bmr = calculate_bmr(weight, height, age, gender)

        # Convert BMR to integer before passing it to knapsack
        bmr = int(bmr)

        # Generate meal plan for breakfast, lunch, and dinner
        breakfast_items, breakfast_calories = knapsack(bmr // 3, food_items["breakfast"])
        lunch_items, lunch_calories = knapsack(bmr // 3, food_items["lunch"])
        dinner_items, dinner_calories = knapsack(bmr // 3, food_items["dinner"])

        context = {
            'name': name,
            'bmr': bmr,
            'breakfast_items': breakfast_items,
            'breakfast_calories': breakfast_calories,
            'lunch_items': lunch_items,
            'lunch_calories': lunch_calories,
            'dinner_items': dinner_items,
            'dinner_calories': dinner_calories,
        }
        return render(request, 'meal_plan.html', context)

    return render(request, 'form.html')

