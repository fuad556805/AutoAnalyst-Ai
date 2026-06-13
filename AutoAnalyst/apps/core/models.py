from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class Profile(models.Model):
    user       = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    image      = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    bio        = models.CharField(max_length=160, blank=True)

    def __str__(self):
        return f'{self.user.username} Profile'

    def get_avatar_letter(self):
        return (self.user.first_name or self.user.username)[0].upper()


@receiver(post_save, sender=User)
def create_or_update_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    else:
        Profile.objects.get_or_create(user=instance)
