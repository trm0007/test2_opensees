from django.contrib import admin
from .models import (
    OpenSeesModel, TimeSeries, Material, Section,
    Node, Element, Constraint, Diaphragm
)

@admin.register(OpenSeesModel)
class OpenSeesModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'ndm', 'ndf', 'project', 'task', 'created_at', 'updated_at')
    search_fields = ('name', 'project__title', 'task__title')
    list_filter = ('created_at', 'updated_at')

@admin.register(TimeSeries)
class TimeSeriesAdmin(admin.ModelAdmin):
    list_display = ('type', 'tag', 'project', 'task', 'created_at', 'updated_at')
    search_fields = ('tag',)
    list_filter = ('type', 'created_at', 'updated_at')

@admin.register(Material)
class MaterialAdmin(admin.ModelAdmin):
    list_display = ('type', 'tag', 'project', 'task', 'created_at', 'updated_at')
    list_filter = ('type', 'created_at', 'updated_at')

@admin.register(Section)
class SectionAdmin(admin.ModelAdmin):
    list_display = ('type', 'tag', 'project', 'task', 'created_at', 'updated_at')
    list_filter = ('type', 'created_at', 'updated_at')

@admin.register(Node)
class NodeAdmin(admin.ModelAdmin):
    list_display = ('tag', 'x', 'y', 'z', 'project', 'task', 'created_at', 'updated_at')
    search_fields = ('tag',)
    list_filter = ('created_at', 'updated_at')

@admin.register(Element)
class ElementAdmin(admin.ModelAdmin):
    list_display = ('type', 'tag', 'project', 'task', 'created_at', 'updated_at')
    list_filter = ('type', 'created_at', 'updated_at')

@admin.register(Constraint)
class ConstraintAdmin(admin.ModelAdmin):
    list_display = ('type', 'project', 'task', 'created_at', 'updated_at')
    list_filter = ('type', 'created_at', 'updated_at')

@admin.register(Diaphragm)
class DiaphragmAdmin(admin.ModelAdmin):
    list_display = ('perp_dirn', 'master_node', 'project', 'task', 'created_at', 'updated_at')
    list_filter = ('created_at', 'updated_at')
