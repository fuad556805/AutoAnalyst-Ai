from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload, name='upload'),
    path('preview/', views.preview, name='preview'),
    path('select-target/', views.select_target, name='select_target'),
]
