# from django.shortcuts import render
import os
# # Create your views here.
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.contrib.auth import login, authenticate, logout as auth_logout

from django.contrib import messages
from django.contrib.auth.models import User

from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Users, ProtectedUser
from django.conf import settings
from .forms import (
    UserCreationForm,
)
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

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

    vggface2_dir = os.path.join(settings.BASE_DIR, 'vggface2')
    user_folders = [f for f in os.listdir(vggface2_dir) if os.path.isdir(os.path.join(vggface2_dir, f))]
    users = []

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    for folder in sorted(user_folders):
        folder_path = os.path.join(vggface2_dir, folder)
        images = [img for img in os.listdir(folder_path) if img.lower().endswith(image_extensions)]
        if images:
            images.sort()
            image_url = f"/media/{folder}/{images[0]}"
            # Check if user is protected
            try:
                protected_user = ProtectedUser.objects.get(user_id=folder)
                is_protected = protected_user.is_protected
            except ProtectedUser.DoesNotExist:
                is_protected = False
                
            users.append({
                'id': folder, 
                'image': image_url,
                'is_protected': is_protected
            })

    return render(request, 'selection.html', {'users': users})

@csrf_exempt #bypass CSRF protection
# @login_required
def save_protected_status(request):
    # if not request.user.is_superuser:
    #     return redirect('homepage')

    if request.method == 'POST':
        protected_users = request.POST.getlist('protected_users')
        
        # Get all users from the database
        all_users = ProtectedUser.objects.all()
        
        # Update or create protected status for each user
        for user in all_users:
            if user.user_id in protected_users:
                user.is_protected = True
            else:
                user.is_protected = False
            user.save()
            
        # Create new entries for users that don't exist yet
        for user_id in protected_users:
            ProtectedUser.objects.get_or_create(
                user_id=user_id,
                defaults={'is_protected': True}
            )
            
        messages.success(request, 'Protected status updated successfully!')
        return redirect('uploadimage')
    
    return redirect('selection')

# @login_required
@csrf_exempt
def get_protected_users(request):
    # if not request.user.is_superuser:
    #     return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    # Get all protected users
    protected_users = ProtectedUser.objects.filter(is_protected=True)
    
    # Get all users to determine row numbers
    vggface2_dir = os.path.join(settings.BASE_DIR, 'vggface2')
    user_folders = sorted([f for f in os.listdir(vggface2_dir) if os.path.isdir(os.path.join(vggface2_dir, f))])
    
    # Convert to array of dictionaries with row numbers
    protected_data = []
    for index, user in enumerate(protected_users, 1):
        # Find the position in the sorted list
        folder_index = user_folders.index(user.user_id) 
        protected_data.append({
            'folder_index': folder_index ,
            'user_id': user.user_id,
            'is_protected': user.is_protected,
            'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': user.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return JsonResponse({
        'protected_users': protected_data,
        'total_protected': len(protected_data)
    })