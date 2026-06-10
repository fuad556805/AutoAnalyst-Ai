from django.urls import path
from . import views

urlpatterns = [
    path('',        views.chat_view,    name='chat'),
    path('send/',   views.send_message, name='chat_send'),
    path('clear/',  views.clear_history,name='chat_clear'),
    path('pdf/',    views.chat_pdf,     name='chat_pdf'),
]
