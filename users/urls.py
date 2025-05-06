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
]