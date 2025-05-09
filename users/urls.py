from django.urls import path
from . import views

urlpatterns = [
    # path("", views.index, name="index"),
    path("", views.homepage, name="homepage"),
    path("login/", views.login_users, name="login"),
    path("logout/", views.logout, name="logout"),
    path("signup/", views.signup_users, name="signup"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path('uploadimage/', views.upload_image_page, name='uploadimage'),
    path('loading/', views.loading, name='loading'),
    path('result/', views.result, name='result'), 
    path('selection/', views.selection, name='selection'),
    path('save-protected-status/', views.save_protected_status, name='save_protected_status'),
    path('get-protected-users/', views.get_protected_users, name='get_protected_users'),
    path('post-upload-images/', views.post_upload_images, name='post_upload_image'),
    path('generating-trigger/', views.generating_trigger, name='generating_trigger'),
    path('check-process-status/', views.check_process_status, name='check_process_status'),
    path('get-trigger-path/', views.get_trigger_path, name='get_trigger_path'),
    path('get-upload-image/', views.get_upload_image, name='get_upload_image'),
    path('run-evaluation/', views.run_evaluation, name='run_evaluation'),
    path('check-evaluation-status/', views.check_evaluation_status, name='check_evaluation_status'),
    path('get-evaluation-results/', views.get_evaluation_results, name='get_evaluation_results'),
]