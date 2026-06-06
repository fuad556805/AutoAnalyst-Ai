from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict, name='predict'),
    path('download-model/', views.download_model, name='download_model'),
]
