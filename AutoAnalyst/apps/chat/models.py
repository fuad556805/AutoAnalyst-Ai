from django.db import models
from django.contrib.auth.models import User


class ChatMessage(models.Model):
    ROLE_CHOICES = [('user', 'User'), ('assistant', 'Assistant')]

    user       = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_messages')
    role       = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content    = models.TextField()
    chart_data = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f'{self.user.username}[{self.role}]: {self.content[:60]}'
