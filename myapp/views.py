from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import Project, Task
from .forms import ProjectForm, TaskForm

# Auth Views
def home(request):
    return render(request, 'myapp/pages/home.html')

def about(request):
    return render(request, 'myapp/pages/about.html')

def contact(request):
    return render(request, 'myapp/pages/contact.html')

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'myapp/auth/login.html', {'form': form})

def user_register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'myapp/auth/register.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('home')

# Project Views
@login_required
def project_list(request):
    projects = Project.objects.filter(user=request.user)
    return render(request, 'myapp/projects/list.html', {'projects': projects})

@login_required
def project_detail(request, pk):
    project = get_object_or_404(Project, pk=pk, user=request.user)
    return render(request, 'myapp/projects/detail.html', {'project': project})

@login_required
def project_create(request):
    if request.method == 'POST':
        form = ProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            project.user = request.user
            project.save()
            return redirect('project_list')
    else:
        form = ProjectForm()
    return render(request, 'myapp/projects/form.html', {'form': form})

@login_required
def project_update(request, pk):
    project = get_object_or_404(Project, pk=pk, user=request.user)
    if request.method == 'POST':
        form = ProjectForm(request.POST, instance=project)
        if form.is_valid():
            form.save()
            return redirect('project_list')
    else:
        form = ProjectForm(instance=project)
    return render(request, 'myapp/projects/form.html', {'form': form})

@login_required
def project_delete(request, pk):
    project = get_object_or_404(Project, pk=pk, user=request.user)
    print("project delete activated")
    if request.method == 'POST':
        print("project delete activated1")
        project.delete()
        return redirect('project_list')
    return render(request, 'myapp/projects/confirm_delete.html', {'project': project})

# Task Views
@login_required
def task_list(request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    tasks = project.tasks.all()
    return render(request, 'myapp/tasks/list.html', {'project': project, 'tasks': tasks})

@login_required
def task_detail(request, project_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=pk, project=project)
    return render(request, 'myapp/tasks/detail.html', {'project': project, 'task': task})

@login_required
def task_create(request, project_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    if request.method == 'POST':
        form = TaskForm(request.POST)
        if form.is_valid():
            task = form.save(commit=False)
            task.project = project
            task.save()
            return redirect('task_list', project_pk=project.pk)
    else:
        form = TaskForm()
    return render(request, 'myapp/tasks/form.html', {'form': form, 'project': project})

@login_required
def task_update(request, project_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=pk, project=project)
    if request.method == 'POST':
        form = TaskForm(request.POST, instance=task)
        if form.is_valid():
            form.save()
            return redirect('task_list', project_pk=project.pk)
    else:
        form = TaskForm(instance=task)
    return render(request, 'myapp/tasks/form.html', {'form': form, 'project': project})

@login_required
def task_delete(request, project_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=pk, project=project)
    if request.method == 'POST':
        task.delete()
        return redirect('task_list', project_pk=project.pk)
    return render(request, 'myapp/tasks/confirm_delete.html', {'project': project, 'task': task})