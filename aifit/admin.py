from django.contrib import admin
from .models  import UserProfile,Recommendation,ActivityLog,ChatMessage,Workout,Meal,Progress
# Register your models here.
admin.site.register(UserProfile)
admin.site.register(Recommendation)
admin.site.register(ActivityLog)
admin.site.register(ChatMessage)
admin.site.register(Workout)
admin.site.register(Meal)
admin.site.register(Progress)
