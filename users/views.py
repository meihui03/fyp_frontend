# from django.shortcuts import render

# # Create your views here.
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.models import User

# views.py
def index(request):
    return HttpResponse("Hello, world. You're at the user index.")

# login.html
def login_users(request):
    return render(request, "login.html")