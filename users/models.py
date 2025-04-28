from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Users(models.Model):
    # Add any custom fields here
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=150, blank=True)
    phone_number = models.CharField(max_length=15, blank=True)
    address = models.TextField(blank=True)
    image = models.ImageField(upload_to='images/', null=True, blank=True)  

    def __str__(self):
        return self.username

