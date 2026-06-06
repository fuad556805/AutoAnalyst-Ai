from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('dataset/<int:pk>/load/', views.load_dataset, name='load_dataset'),
    path('dataset/<int:pk>/delete/', views.delete_dataset, name='delete_dataset'),
]
