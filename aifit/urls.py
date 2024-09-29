from django.urls import path
from .views import UserProfileListCreate, RecommendationListCreate, ActivityLogListCreate
from django.conf import settings
from django.conf.urls.static import static
from .views import UserProfileView, ActivityLogView, RecommendationView,success,diet_plan,index,user_login,register,custom_logout,tracker_view,video_feed,logtemp,send_message,list_messages,clear_messages,dashboard
from .views import get_goals,send_message1,list_messages1,home,food_recognition,stream
from django.contrib.auth import views as auth_views
from . import views
from .views import get_workouts, create_workout, get_meals, create_meal, get_progress, create_progress

urlpatterns = [
    path('users/', UserProfileListCreate.as_view(), name='user-list-create'),
    path('recommendations/', RecommendationListCreate.as_view(), name='recommendation-list-create'),
    path('activities/', ActivityLogListCreate.as_view(), name='activity-log-list-create'),
    path('create-profile/', UserProfileView.as_view(), name='create-profile'),
    path('log-activity/', ActivityLogView.as_view(), name='log-activity'),
    path('recommendations/<int:user_id>/', RecommendationView.as_view(), name='recommendations'),
    path('success/',success,name='success'),
    path('diet/',diet_plan,name='diet_plan'),
    path('',index,name='index'),
    path('home/',home,name='home'),
    path('dashboard/',dashboard,name='dashboard'),
    path('login/', user_login, name='login'),  # Login URL
    path('register/', register, name='register'),  # Registration URL
    path('logout/', custom_logout, name='logout'),
    path('tracker/', views.tracker_view, name='tracker_view'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('log/',logtemp,name='logtemp'),
    path('send/', send_message, name='send_message'),
    path('sends/',send_message1, name='send_message1'),
    path('list1/', list_messages1, name='list_messages1'),
    path('list/', list_messages, name='list_messages'),
    path('clear_messages/', views.clear_messages, name='clear_messages'),
    path('workouts/', get_workouts, name='get_workouts'),
    path('workouts/create/', create_workout, name='create_workout'),
    path('meals/', get_meals, name='get_meals'),
    path('meals/create/', create_meal, name='create_meal'),
    path('progress/', get_progress, name='get_progress'),
    path('progress/create/', create_progress, name='create_progress'),
    path('goals/', get_goals, name='get_goals'),
    path('food-recognition/', food_recognition, name='food_recognition'),
    path('stream/', stream, name='video_feed1'),
    path('mealp/', views.meal_plan_view, name='meal_plan'),

]
if settings.DEBUG:
    urlpatterns +=static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
    urlpatterns +=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)