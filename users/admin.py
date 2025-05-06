from django.contrib import admin
from .models import Users, ProtectedUser

# Register your models here.
admin.site.register(Users)
admin.site.register(ProtectedUser)
