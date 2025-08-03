from django.contrib import admin
from .models import Project, Task

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'created_at', 'updated_at')
    search_fields = ('title', 'description', 'user__username')
    list_filter = ('created_at', 'updated_at', 'user')

@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ('title', 'project', 'completed', 'created_at', 'updated_at')
    search_fields = ('title', 'description', 'project__title')
    list_filter = ('completed', 'created_at', 'updated_at')
