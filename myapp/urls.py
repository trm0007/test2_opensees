from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    
    # Projects
    path('projects/', views.project_list, name='project_list'),
    path('projects/create/', views.project_create, name='project_create'),
    path('projects/<int:pk>/', views.project_detail, name='project_detail'),
    path('projects/<int:pk>/update/', views.project_update, name='project_update'),
    path('projects/<int:pk>/delete/', views.project_delete, name='project_delete'),
    
    # Tasks
    path('projects/<int:project_pk>/tasks/', views.task_list, name='task_list'),
    path('projects/<int:project_pk>/tasks/create/', views.task_create, name='task_create'),
    path('projects/<int:project_pk>/tasks/<int:pk>/', views.task_detail, name='task_detail'),
    path('projects/<int:project_pk>/tasks/<int:pk>/update/', views.task_update, name='task_update'),
    path('projects/<int:project_pk>/tasks/<int:pk>/delete/', views.task_delete, name='task_delete'),
]