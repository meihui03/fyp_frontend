# from django.shortcuts import render

# # Create your views here.
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.contrib.auth import login, authenticate, logout as auth_logout

from django.contrib import messages
from django.contrib.auth.models import User

from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Users

from .forms import (
    UserCreationForm,
)
# views.py
# def index(request):
#     return HttpResponse("Hello, world. You're at the user index.")

def logout(request):
    auth_logout(request)
    return redirect("login")

# login.html
def login_users(request):
    if request.user.is_authenticated:
        return redirect("homepage")

    if request.method == "POST":
        username = request.POST["username"].lower()
        password = request.POST["password"]

        try:
            user = User.objects.get(username=username)
        except:
            messages.error(request, "Username does not exist")

        # Authenticate the user with the provided username and password
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect(
                request.GET["next"] if "next" in request.GET else "homepage"
            )

        else:
            messages.error(request, "Username OR password is incorrect")

    return render(request, "login.html")

# signup.html
def signup_users(request):
    form = UserCreationForm()

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.username = user.username.lower()
            user.save()

            messages.success(request, "User account was created!")

            login(request, user)
            return redirect("homepage")

        else:
            messages.success(request, "An error has occurred during registration")

    context = {"form": form}
    return render(request, "signup.html", context)

# homepage.html
def homepage(request):
    return render(request, "homepage.html")

# about.html
def about(request):
    return render(request, "about.html")

# contact.html
def contact(request):
    return render(request, "contact.html")

# uploadimage.html
def upload_image_page(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get('image')

        if uploaded_file:
            # associate the uploaded file with the user
            user = request.user  
            # user.image = uploaded_file  
            # user.save()
            user_profile, created = Users.objects.get_or_create(user=user)
            user_profile.image = uploaded_file
            user_profile.save()
            
            return redirect('loading') 

        else:
            messages.error(request, "No file was uploaded.")

    return render(request, 'uploadimage.html')

def loading(request):
    return render(request, 'loading.html')

def result(request):
    user_profile = Users.objects.get(user=request.user)
    context = {
        'user_profile': user_profile  
    }
    return render(request, 'result.html', context)

def selection(request):
    if not request.user.is_authenticated or not request.user.is_superuser:
        return redirect('homepage') 

    return render(request, 'selection.html')