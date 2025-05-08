# from django.shortcuts import render
import os
import psutil
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
import json
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import subprocess
import threading

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
    # user_profile = Users.objects.get(user=request.user)
    # context = {
    #     'user_profile': user_profile  
    # }
    # return render(request, 'result.html', context)
    user_profile = Users.objects.filter(user=request.user).first()
    if user_profile:
        image_url = user_profile.image.url if user_profile.image else None
    else:
        image_url = None
    context = {
        'image_url': image_url
    }
    return render(request, 'result.html', context)
    # return render(request, 'result.html')


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

@csrf_exempt
def save_protected_status(request):
    if not request.user.is_superuser:
        return redirect('homepage')

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
        
        # Run generate_triggers.py in a separate thread
        def run_generate_triggers():
            try:
                # Run generate_triggers.py and capture output
                result = subprocess.run(['python', 'generate_triggers.py'], 
                                     capture_output=True, 
                                     text=True, 
                                     check=True)
                
                # Extract trigger path from output
                for line in result.stdout.split('\n'):
                    if 'TRIGGER_PATH:' in line:
                        trigger_path = line.split('TRIGGER_PATH:')[1].strip()
                        # Store the trigger path
                        with open('last_trigger_path.txt', 'w') as f:
                            f.write(trigger_path)
                        print(f'Trigger path saved: {trigger_path}')
                        break
                
            except subprocess.CalledProcessError as e:
                print(f"Error running generate_triggers.py: {e}")
                print(f"Error output: {e.stderr}")
        
        thread = threading.Thread(target=run_generate_triggers)
        thread.start()
            
        return redirect('generating_trigger')
    
    return redirect('selection')

@csrf_exempt
def get_trigger_path(request):
    """API endpoint to get the last generated trigger path"""
    try:
        with open('last_trigger_path.txt', 'r') as f:
            trigger_path = f.read().strip()
            # Normalize path to use forward slashes
            trigger_path = trigger_path.replace('\\', '/')
            return JsonResponse({'trigger_path': trigger_path})
    except FileNotFoundError:
        return JsonResponse({'error': 'No trigger path found'}, status=404)

def generating_trigger(request):
    return render(request, 'generating_trigger.html')

@csrf_exempt
def check_process_status(request):
    # Check if generate_triggers.py is still running
    is_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower() and 'generate_triggers.py' in ' '.join(proc.info['cmdline']):
                is_running = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return JsonResponse({
        'complete': not is_running
    })

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
    
    
@csrf_exempt
def post_upload_images(request):
    """
    API endpoint to handle image uploads. Accepts POST requests with an image file.
    Returns a JSON response with the uploaded image URL and a placeholder for the matched image URL.
    """
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image = request.FILES['image']
            # Create a FileSystemStorage instance with the correct media root
            fs = FileSystemStorage()
            
            # Ensure the directory exists
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            # os.makedirs(settings.MEDIA_ROOT, "images", exist_ok=True)
            
            # Save the file
            filename = fs.save(image.name, image)
            file_url = fs.url(filename)  # This will give us the correct URL based on MEDIA_URL

            return JsonResponse({
                'success': True, 
                'file_url': file_url,
                'redirect_url': "/loading/"
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Error saving file: {str(e)}'
            }, status=500)

    return JsonResponse({'success': False, 'error': 'No image uploaded'}, status=400)

@csrf_exempt
def get_upload_image(request):
    """API endpoint to get the last uploaded image path"""
    try:
        # Get the most recent uploaded image from the media directory
        if os.path.exists(settings.MEDIA_ROOT):
            files = [f for f in os.listdir(settings.MEDIA_ROOT) if os.path.isfile(os.path.join(settings.MEDIA_ROOT, f))]
            if files:
                # Sort by modification time, newest first
                latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(settings.MEDIA_ROOT, x)))
                file_url = f'{settings.MEDIA_URL}{latest_file}'
                return JsonResponse({
                    'success': True,
                    'file_url': file_url
                })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'No uploaded image found'}, status=404)