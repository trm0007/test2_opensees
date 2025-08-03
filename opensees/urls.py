# opensees/urls.py
from django.urls import path
from . import views

app_name = 'opensees'

urlpatterns = [


    path('<int:project_pk>/<int:task_pk>/input/', views.analysis_input, name='input'),
    path('<int:project_pk>/<int:task_pk>/run/', views.run_analysis, name='run_analysis'),
        # Calculator URLs
    path('<int:project_pk>/<int:task_pk>/calculator1/', views.calculator1, name='calculator1'),
    path('<int:project_pk>/<int:task_pk>/calculator2/', views.calculator2, name='calculator2'),
    path('<int:project_pk>/<int:task_pk>/calculator3/', views.calculator3, name='calculator3'),
    path('project/<int:project_pk>/task/<int:task_pk>/convert-to-html/', 
         views.convert_to_html, name='convert_to_html'),
    path('project/<int:project_pk>/task/<int:task_pk>/download-excel/<str:filename>/', 
         views.download_excel, name='download_excel'),

    # New RSA-specific patterns
    path('project/<int:project_pk>/task/<int:task_pk>/convert_to_html_rsa/', 
         views.convert_to_html_rsa, name='convert_to_html_rsa'),
    path('project/<int:project_pk>/task/<int:task_pk>/download_excel_rsa/<str:filename>/', 
         views.download_excel_rsa, name='download_excel_rsa'),

     # New seismic analysis calculator
    path('<int:project_id>/<int:task_id>/calculator4/', views.calculator4, name='calculator4'),
    # New Wind Analysis URL
    path('<int:project_id>/tasks/<int:task_id>/calculator5/', views.calculator5, name='calculator5'),
]