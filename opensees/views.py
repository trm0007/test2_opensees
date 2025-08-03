import ast
import json
import os
import tempfile
import traceback
import importlib.util
from datetime import datetime
from subprocess import PIPE, Popen

import matplotlib
from sectionproperties.pre import *
from sectionproperties.pre.library import *
from sectionproperties.analysis import Section


matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import numpy as np

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404

from myapp.models import Project, Task
from .forms import *
from .utils import *

from opensees.functions import *
from opensees.wall_meshing import create_and_visualize_model, generate_building_model, modify_building_model
from opensees.calculate_section_properties import analyze_section, model_generation
from opensees.calculate_concretesection_properties import concrete_model_generation, analyze_concrete_section
from openseespywin.opensees import *
import openseespy.opensees as ops
import opsvis as opsv
from concreteproperties.material import Concrete, SteelBar
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
    SteelElasticPlastic
)

import os
import json
import pandas as pd
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse


def get_common_context(request, project_pk, task_pk):
    """Get common context for all views"""
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    return {
        'project': project,
        'task': task,
        'user': request.user
    }




@login_required
def analysis_input(request, project_pk, task_pk):
    """Display input form for OpenSees analysis"""
    from django.template.loader import render_to_string

    context = get_common_context(request, project_pk, task_pk)
    default_input = render_to_string("opensees/data.html")

    context['default_input'] = default_input
    return render(request, 'opensees/input.html', context)



# @login_required
# def run_analysis(request, project_pk, task_pk):
#     print(f"Starting run_analysis for project {project_pk}, task {task_pk}")
#     context = get_common_context(request, project_pk, task_pk)
    
#     if request.method == 'POST':
#         print("POST request received")
#         try:
#             opensees_script = request.POST.get('input_data', '').strip()
#             print(f"Received script content (length: {len(opensees_script)} chars)")
            
#             # Create temp directory with user-specific subfolder
#             user_dir = os.path.join(settings.MEDIA_ROOT, 'temp_analysis', str(request.user.pk))
#             print(f"Creating user directory at: {user_dir}")
#             os.makedirs(user_dir, exist_ok=True)
            
#             # Create unique filename
#             script_filename = f"model_definition.py"
#             script_path = os.path.normpath(os.path.join(user_dir, script_filename))
            
#             # Debugging output
#             print(f"Script path: {script_path}")
#             print(f"Directory exists: {os.path.exists(user_dir)}")
#             print(f"Is directory: {os.path.isdir(user_dir)}")
            
#             # Write with proper line endings and encoding
#             print(f"Writing script to file (size: {len(opensees_script)} bytes)")
#             with open(script_path, 'w') as f:
#                 f.write(opensees_script)
            
#             print(f"Attempting to import module from {script_path}")
#             spec = importlib.util.spec_from_file_location("model_definition", script_path)
#             model_module = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(model_module)
#             print("Module imported successfully")
            
#             # Helper function to safely get attributes
#             def get_model_attr(module, name, default=None):
#                 attr = getattr(module, name, default) if hasattr(module, name) else default
#                 print(f"Retrieved attribute '{name}': {attr if isinstance(attr, (int, float, str)) else type(attr)}")
#                 return attr
            
#             # Extract all model data with defaults
#             model_data = {
#                 'materials': get_model_attr(model_module, 'materials', []),
#                 'nd_materials': get_model_attr(model_module, 'nd_materials', []),
#                 'section_properties': get_model_attr(model_module, 'section_properties', []),
#                 'elastic_section': get_model_attr(model_module, 'elastic_section', []),
#                 'aggregator_section': get_model_attr(model_module, 'aggregator_section', []),
#                 'shell_section': get_model_attr(model_module, 'shell_section', []),
#                 'nodes': get_model_attr(model_module, 'nodes', []),
#                 'transformations': get_model_attr(model_module, 'transformations', []),
#                 'beam_integrations': get_model_attr(model_module, 'beam_integrations', []),
#                 'frame_elements': get_model_attr(model_module, 'frame_elements', []),
#                 'shell_elements': get_model_attr(model_module, 'shell_elements', []),
#                 'fixities': get_model_attr(model_module, 'fixities', []),
#                 'diaphragms': get_model_attr(model_module, 'diaphragms', []),
#                 'node_loads': get_model_attr(model_module, 'node_loads', []),
#                 'element_uniform_loads': get_model_attr(model_module, 'element_uniform_loads', []),
#                 'shell_pressure_loads': get_model_attr(model_module, 'shell_pressure_loads', []),
#                 'zero_length_elements': get_model_attr(model_module, 'zero_length_elements', []),
#                 'Tn': get_model_attr(model_module, 'Tn', []),
#                 'Sa': get_model_attr(model_module, 'Sa', []),
#                 'load_cases': get_model_attr(model_module, 'load_cases', []),
#                 'load_combinations': get_model_attr(model_module, 'load_combinations', []),
#                 'num_points': get_model_attr(model_module, 'num_points', 5),
#                 'analysis_type': get_model_attr(model_module, 'analysis_type', 'gravity_analysis')
#             }

#             analysis_type = model_data['analysis_type']
#             print(f"Analysis type: {analysis_type}")

#             print("Model data extracted, creating structural model...")
#             # Create the structural model
#             (node_loads, element_uniform_loads, shell_pressure_loads, 
#             section_properties, elastic_section, aggregator_section, 
#             beam_integrations, frame_elements) = create_structural_model(
#                 model_data['materials'],
#                 model_data['nd_materials'],
#                 model_data['section_properties'],
#                 model_data['elastic_section'],
#                 model_data['aggregator_section'],
#                 model_data['shell_section'],
#                 model_data['nodes'],
#                 model_data['transformations'],
#                 model_data['beam_integrations'],
#                 model_data['frame_elements'],
#                 model_data['shell_elements'],
#                 model_data['fixities'],
#                 model_data['diaphragms'],
#                 model_data['node_loads'],
#                 model_data['element_uniform_loads'],
#                 model_data['shell_pressure_loads'],
#                 model_data['zero_length_elements'],
#             )
            
#             print(f"Structural model created with: {len(frame_elements)} frame elements, {len(node_loads)} node loads")
            
#             # Get project and task titles for folder structure
#             project = get_object_or_404(Project, pk=project_pk)
#             task = get_object_or_404(Task, pk=task_pk)
            
#             # Prepare output directory in MEDIA_ROOT
#             user_folder = os.path.join(settings.MEDIA_ROOT, project.title, task.title, f"user_{request.user.username}")
#             print(f"Creating user folder at: {user_folder}")
#             os.makedirs(user_folder, exist_ok=True)
#             output_dir = os.path.join(user_folder, "post_processing")
#             print(f"Creating output directory at: {output_dir}")
#             os.makedirs(output_dir, exist_ok=True)
            
#             print("Model creation completed. Starting analysis...")
            
#             # Initialize analysis results
#             analysis_results = {}
#             image_urls = {}
#             all_filepaths = []
            
#             # Run analysis based on type
#             if analysis_type == "gravity_analysis":
#                 print("Starting gravity analysis...")
#                 try:
#                     gravity_results = gravity_analysis(
#                         node_loads, 
#                         element_uniform_loads, 
#                         shell_pressure_loads, 
#                         section_properties, 
#                         elastic_section, 
#                         aggregator_section, 
#                         beam_integrations, 
#                         frame_elements,
#                         model_data['load_cases'],
#                         model_data['load_combinations'],
#                         num_points=model_data['num_points'], 
#                         output_folder=output_dir
#                     )
#                     analysis_results['gravity'] = gravity_results
#                     print(f"Gravity analysis completed with {len(gravity_results)} results")
#                 except Exception as e:
#                     print(f"Error in gravity analysis: {str(e)}")
#                     import traceback
#                     traceback.print_exc()
#                     analysis_results['gravity'] = {'error': str(e)}
            
#             elif analysis_type == "response_spectrum_analysis":
#                 # Run response spectrum analysis if Tn and Sa are provided
#                 if model_data['Tn'] and model_data['Sa']:
#                     print(f"Found Tn ({len(model_data['Tn'])} points) and Sa ({len(model_data['Sa'])} points) for RSA")
#                     try:
#                         print("Starting response spectrum analysis...")
#                         rsa_results = response_spectrum_analysis(
#                             section_properties, 
#                             model_data['Tn'], 
#                             model_data['Sa'],
#                             direction=1, 
#                             num_modes=7, 
#                             output_folder=output_dir
#                         )
#                         analysis_results['rsa'] = rsa_results
#                         print(f"Response spectrum analysis completed with {len(rsa_results)} results")
#                     except Exception as e:
#                         print(f"Error in response spectrum analysis: {str(e)}")
#                         import traceback
#                         traceback.print_exc()
#                         analysis_results['rsa'] = {'error': str(e)}
#                 else:
#                     print("Skipping response spectrum analysis - no Tn/Sa data provided")
            
#             # Collect all generated files from the output directory
#             if os.path.exists(output_dir):
#                 print(f"Scanning output directory: {output_dir}")
#                 for root, dirs, files in os.walk(output_dir):
#                     print(f"Found {len(files)} files in {root}")
#                     for file in files:
#                         filepath = os.path.join(root, file)
#                         all_filepaths.append(filepath)
            
#             # Create image URLs for display
#             print(f"Processing {len(all_filepaths)} output files for image URLs")
#             for filepath in all_filepaths:
#                 if filepath and os.path.exists(filepath):
#                     # Check if it's an image file
#                     if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                         rel_path = os.path.relpath(filepath, settings.MEDIA_ROOT)
#                         name = os.path.splitext(os.path.basename(filepath))[0]
#                         image_url = os.path.join(settings.MEDIA_URL, rel_path).replace('\\', '/')
#                         image_urls[name] = image_url
#                         print(f"Added image URL: {name} -> {image_url}")
            
#             print(f"Analysis completed. Generated {len(all_filepaths)} files, {len(image_urls)} images")
            
#             # Prepare context for template
#             context['result'] = {
#                 'analysis_results': analysis_results,
#                 'images': image_urls,
#                 'output_files': [os.path.basename(f) for f in all_filepaths],
#                 'output_directory': output_dir,
#                 'success': True
#             }
            
#         except Exception as e:
#             print(f"Critical error in analysis: {str(e)}")
#             import traceback
#             traceback.print_exc()
            
#             context['result'] = {
#                 'success': False,
#                 'error': str(e),
#                 'traceback': traceback.format_exc()
#             }
    
#     else:
#         print("Non-POST request received")
#         context['result'] = {'success': False}
    
#     print("Returning response")
#     return render(request, 'opensees/input.html', context)




# Modified run_analysis function to handle RSA properly
@login_required
def run_analysis(request, project_pk, task_pk):
    print(f"Starting run_analysis for project {project_pk}, task {task_pk}")
    context = get_common_context(request, project_pk, task_pk)
    
    if request.method == 'POST':
        print("POST request received")
        try:
            opensees_script = request.POST.get('input_data', '').strip()
            print(f"Received script content (length: {len(opensees_script)} chars)")
            
            # Create temp directory with user-specific subfolder
            user_dir = os.path.join(settings.MEDIA_ROOT, 'temp_analysis', str(request.user.pk))
            print(f"Creating user directory at: {user_dir}")
            os.makedirs(user_dir, exist_ok=True)
            
            # Create unique filename
            script_filename = f"model_definition.py"
            script_path = os.path.normpath(os.path.join(user_dir, script_filename))
            
            # Write with proper line endings and encoding
            print(f"Writing script to file (size: {len(opensees_script)} bytes)")
            with open(script_path, 'w') as f:
                f.write(opensees_script)
            
            print(f"Attempting to import module from {script_path}")
            spec = importlib.util.spec_from_file_location("model_definition", script_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            print("Module imported successfully")
            
            # Helper function to safely get attributes
            def get_model_attr(module, name, default=None):
                attr = getattr(module, name, default) if hasattr(module, name) else default
                print(f"Retrieved attribute '{name}': {attr if isinstance(attr, (int, float, str)) else type(attr)}")
                return attr
            
            # Extract all model data with defaults
            model_data = {
                'materials': get_model_attr(model_module, 'materials', []),
                'nd_materials': get_model_attr(model_module, 'nd_materials', []),
                'section_properties': get_model_attr(model_module, 'section_properties', []),
                'elastic_section': get_model_attr(model_module, 'elastic_section', []),
                'aggregator_section': get_model_attr(model_module, 'aggregator_section', []),
                'shell_section': get_model_attr(model_module, 'shell_section', []),
                'nodes': get_model_attr(model_module, 'nodes', []),
                'transformations': get_model_attr(model_module, 'transformations', []),
                'beam_integrations': get_model_attr(model_module, 'beam_integrations', []),
                'frame_elements': get_model_attr(model_module, 'frame_elements', []),
                'shell_elements': get_model_attr(model_module, 'shell_elements', []),
                'fixities': get_model_attr(model_module, 'fixities', []),
                'diaphragms': get_model_attr(model_module, 'diaphragms', []),
                'node_loads': get_model_attr(model_module, 'node_loads', []),
                'element_uniform_loads': get_model_attr(model_module, 'element_uniform_loads', []),
                'shell_pressure_loads': get_model_attr(model_module, 'shell_pressure_loads', []),
                'zero_length_elements': get_model_attr(model_module, 'zero_length_elements', []),
                'Tn': get_model_attr(model_module, 'Tn', []),
                'Sa': get_model_attr(model_module, 'Sa', []),
                'load_cases': get_model_attr(model_module, 'load_cases', []),
                'load_combinations': get_model_attr(model_module, 'load_combinations', []),
                'num_points': get_model_attr(model_module, 'num_points', 5),
                'analysis_type': get_model_attr(model_module, 'analysis_type', 'gravity_analysis'),
                'num_modes': get_model_attr(model_module, 'num_modes', 7),
                'direction': get_model_attr(model_module, 'direction', 1)
            }

            analysis_type = model_data['analysis_type']
            print(f"Analysis type: {analysis_type}")

            print("Model data extracted, creating structural model...")
            # Create the structural model
            (node_loads, element_uniform_loads, shell_pressure_loads, 
            section_properties, elastic_section, aggregator_section, 
            beam_integrations, frame_elements) = create_structural_model(
                model_data['materials'],
                model_data['nd_materials'],
                model_data['section_properties'],
                model_data['elastic_section'],
                model_data['aggregator_section'],
                model_data['shell_section'],
                model_data['nodes'],
                model_data['transformations'],
                model_data['beam_integrations'],
                model_data['frame_elements'],
                model_data['shell_elements'],
                model_data['fixities'],
                model_data['diaphragms'],
                model_data['node_loads'],
                model_data['element_uniform_loads'],
                model_data['shell_pressure_loads'],
                model_data['zero_length_elements'],
            )
            
            print(f"Structural model created with: {len(frame_elements)} frame elements, {len(node_loads)} node loads")
            
            # Get project and task titles for folder structure
            project = get_object_or_404(Project, pk=project_pk)
            task = get_object_or_404(Task, pk=task_pk)
            
            # Prepare output directory in MEDIA_ROOT
            user_folder = os.path.join(settings.MEDIA_ROOT, project.title, task.title, f"user_{request.user.username}")
            print(f"Creating user folder at: {user_folder}")
            os.makedirs(user_folder, exist_ok=True)
            output_dir = os.path.join(user_folder, "post_processing")
            print(f"Creating output directory at: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            print("Model creation completed. Starting analysis...")
            
            # Initialize analysis results
            analysis_results = {}
            image_urls = {}
            all_filepaths = []
            
            # Run analysis based on type
            if analysis_type == "gravity_analysis":
                print("Starting gravity analysis...")
                try:
                    gravity_results = gravity_analysis(
                        node_loads, 
                        element_uniform_loads, 
                        shell_pressure_loads, 
                        section_properties, 
                        elastic_section, 
                        aggregator_section, 
                        beam_integrations, 
                        frame_elements,
                        model_data['load_cases'],
                        model_data['load_combinations'],
                        num_points=model_data['num_points'], 
                        output_folder=output_dir
                    )
                    analysis_results['gravity'] = gravity_results
                    print(f"Gravity analysis completed with {len(gravity_results)} results")
                except Exception as e:
                    print(f"Error in gravity analysis: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    analysis_results['gravity'] = {'error': str(e)}
            
            elif analysis_type == "response_spectrum_analysis":
                # Run response spectrum analysis if Tn and Sa are provided
                if model_data['Tn'] and model_data['Sa']:
                    print(f"Found Tn ({len(model_data['Tn'])} points) and Sa ({len(model_data['Sa'])} points) for RSA")
                    try:
                        print("Starting response spectrum analysis...")
                        rsa_results = response_spectrum_analysis(
                            section_properties, 
                            model_data['Tn'], 
                            model_data['Sa'],
                            direction=model_data['direction'], 
                            num_modes=model_data['num_modes'], 
                            output_folder=output_dir
                        )
                        analysis_results['rsa'] = rsa_results
                        print(f"Response spectrum analysis completed")
                    except Exception as e:
                        print(f"Error in response spectrum analysis: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        analysis_results['rsa'] = {'error': str(e)}
                else:
                    print("Skipping response spectrum analysis - no Tn/Sa data provided")
            
            # Collect all generated files from the output directory
            if os.path.exists(output_dir):
                print(f"Scanning output directory: {output_dir}")
                for root, dirs, files in os.walk(output_dir):
                    print(f"Found {len(files)} files in {root}")
                    for file in files:
                        filepath = os.path.join(root, file)
                        all_filepaths.append(filepath)
            
            # Create image URLs for display
            print(f"Processing {len(all_filepaths)} output files for image URLs")
            for filepath in all_filepaths:
                if filepath and os.path.exists(filepath):
                    # Check if it's an image file
                    if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        rel_path = os.path.relpath(filepath, settings.MEDIA_ROOT)
                        name = os.path.splitext(os.path.basename(filepath))[0]
                        image_url = os.path.join(settings.MEDIA_URL, rel_path).replace('\\', '/')
                        image_urls[name] = image_url
                        print(f"Added image URL: {name} -> {image_url}")
            
            print(f"Analysis completed. Generated {len(all_filepaths)} files, {len(image_urls)} images")
            
            # Prepare context for template
            context['result'] = {
                'analysis_results': analysis_results,
                'images': image_urls,
                'output_files': [os.path.basename(f) for f in all_filepaths],
                'output_directory': output_dir,
                'analysis_type': analysis_type,
                'success': True
            }
            
        except Exception as e:
            print(f"Critical error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            
            context['result'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    else:
        print("Non-POST request received")
        context['result'] = {'success': False}
    
    print("Returning response")
    return render(request, 'opensees/input.html', context)


# New functions for RSA HTML conversion and Excel download
@login_required
def convert_to_html_rsa(request, project_pk, task_pk):
    context = get_common_context(request, project_pk, task_pk)
    
    # Get project and task for correct folder structure
    project = get_object_or_404(Project, pk=project_pk)
    task = get_object_or_404(Task, pk=task_pk)
    
    user_folder = os.path.join(settings.MEDIA_ROOT, project.title, task.title, f"user_{request.user.username}")
    output_dir = os.path.join(user_folder, "post_processing")
    rsa_dir = os.path.join(output_dir, "response_spectrum_analysis")
    
    if not os.path.exists(rsa_dir):
        context['error'] = f"RSA output directory not found: {rsa_dir}"
        return render(request, 'opensees/convert_to_html_rsa.html', context)
    
    try:
        json_file_path = os.path.join(rsa_dir, "rsa_analysis_results.json")
        
        if not os.path.exists(json_file_path):
            context['error'] = "RSA analysis results not found. Please run the analysis first."
            return render(request, 'opensees/convert_to_html_rsa.html', context)
        
        print(f"Processing RSA results from: {json_file_path}")
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Initialize results dictionary
        rsa_results = {
            'html_tables': {},
            'excel_files': {}
        }
        all_excel_files = {}
        
        # Process Modal Properties
        if 'modal_properties' in data:
            modal_props = data['modal_properties']
            
            # Create modal properties DataFrame
            modal_data = []
            for i, (period, eigenvalue) in enumerate(zip(modal_props.get('periods', []), 
                                                       modal_props.get('eigenvalues', []))):
                modal_data.append({
                    'Mode': i + 1,
                    'Period_s': round(period, 4),
                    'Frequency_Hz': round(1/period if period > 0 else 0, 4),
                    'Eigenvalue': round(eigenvalue, 6),
                    'Omega_rad_s': round(math.sqrt(eigenvalue) if eigenvalue > 0 else 0, 4)
                })
            
            df_modal = pd.DataFrame(modal_data)
            rsa_results['html_tables']['modal_properties'] = df_modal.to_html(classes='table table-striped', index=False)
            excel_filename = 'rsa_modal_properties.xlsx'
            excel_path = os.path.join(rsa_dir, excel_filename)
            df_modal.to_excel(excel_path, index=False)
            rsa_results['excel_files']['modal_properties'] = excel_filename
            all_excel_files['modal_properties'] = excel_filename
            
            # Process participation factors and effective masses
            participation_factors = modal_props.get('modal_participation_factors', {})
            effective_masses = modal_props.get('effective_masses', {})
            
            participation_data = []
            for direction in ['MX', 'MY', 'RMZ']:
                if direction in participation_factors and direction in effective_masses:
                    for i, (pf, em) in enumerate(zip(participation_factors[direction], 
                                                   effective_masses[direction])):
                        participation_data.append({
                            'Mode': i + 1,
                            'Direction': direction,
                            'Participation_Factor': round(pf, 6),
                            'Effective_Mass': round(em, 6)
                        })
            
            if participation_data:
                df_participation = pd.DataFrame(participation_data)
                rsa_results['html_tables']['participation_factors'] = df_participation.to_html(classes='table table-striped', index=False)
                excel_filename = 'rsa_participation_factors.xlsx'
                excel_path = os.path.join(rsa_dir, excel_filename)
                df_participation.to_excel(excel_path, index=False)
                rsa_results['excel_files']['participation_factors'] = excel_filename
                all_excel_files['participation_factors'] = excel_filename
        
        # Process Floor Properties
        if 'floor_properties' in data:
            df_floors = pd.DataFrame(data['floor_properties'])
            rsa_results['html_tables']['floor_properties'] = df_floors.to_html(classes='table table-striped', index=False)
            excel_filename = 'rsa_floor_properties.xlsx'
            excel_path = os.path.join(rsa_dir, excel_filename)
            df_floors.to_excel(excel_path, index=False)
            rsa_results['excel_files']['floor_properties'] = excel_filename
            all_excel_files['floor_properties'] = excel_filename
        
        # Process CQC Displacements
        if 'nodal_responses' in data and 'cqc_displacements' in data['nodal_responses']:
            cqc_displacements = data['nodal_responses']['cqc_displacements']
            displacement_data = []
            
            for node, disps in cqc_displacements.items():
                displacement_data.append({
                    'Node': node,
                    'Ux': round(disps.get('Ux', 0), 6),
                    'Uy': round(disps.get('Uy', 0), 6),
                    'Uz': round(disps.get('Uz', 0), 6),
                    'Rx': round(disps.get('Rx', 0), 6),
                    'Ry': round(disps.get('Ry', 0), 6),
                    'Rz': round(disps.get('Rz', 0), 6)
                })
            
            df_displacements = pd.DataFrame(displacement_data)
            rsa_results['html_tables']['cqc_displacements'] = df_displacements.to_html(classes='table table-striped', index=False)
            excel_filename = 'rsa_cqc_displacements.xlsx'
            excel_path = os.path.join(rsa_dir, excel_filename)
            df_displacements.to_excel(excel_path, index=False)
            rsa_results['excel_files']['cqc_displacements'] = excel_filename
            all_excel_files['cqc_displacements'] = excel_filename
        
        # Process CQC Reactions
        if 'nodal_responses' in data and 'cqc_reactions' in data['nodal_responses']:
            cqc_reactions = data['nodal_responses']['cqc_reactions']
            reaction_data = []
            
            for node, reactions in cqc_reactions.items():
                reaction_data.append({
                    'Node': node,
                    'Fx': round(reactions.get('Fx', 0), 3),
                    'Fy': round(reactions.get('Fy', 0), 3),
                    'Fz': round(reactions.get('Fz', 0), 3),
                    'Mx': round(reactions.get('Mx', 0), 3),
                    'My': round(reactions.get('My', 0), 3),
                    'Mz': round(reactions.get('Mz', 0), 3)
                })
            
            df_reactions = pd.DataFrame(reaction_data)
            rsa_results['html_tables']['cqc_reactions'] = df_reactions.to_html(classes='table table-striped', index=False)
            excel_filename = 'rsa_cqc_reactions.xlsx'
            excel_path = os.path.join(rsa_dir, excel_filename)
            df_reactions.to_excel(excel_path, index=False)
            rsa_results['excel_files']['cqc_reactions'] = excel_filename
            all_excel_files['cqc_reactions'] = excel_filename
        
        # Process Story Drifts
        if 'story_drifts' in data:
            story_drift_data = []
            for story, drifts in data['story_drifts'].items():
                if isinstance(drifts, dict):
                    story_drift_data.append({
                        'Story': story,
                        'Drift_X': round(drifts.get('drift_x', 0), 6),
                        'Drift_Y': round(drifts.get('drift_y', 0), 6),
                        'Height': round(drifts.get('height', 0), 3)
                    })
            
            if story_drift_data:
                df_drifts = pd.DataFrame(story_drift_data)
                rsa_results['html_tables']['story_drifts'] = df_drifts.to_html(classes='table table-striped', index=False)
                excel_filename = 'rsa_story_drifts.xlsx'
                excel_path = os.path.join(rsa_dir, excel_filename)
                df_drifts.to_excel(excel_path, index=False)
                rsa_results['excel_files']['story_drifts'] = excel_filename
                all_excel_files['story_drifts'] = excel_filename
        
        # Process Shear Results
        if 'shear_results' in data:
            shear_data = []
            for story, shears in data['shear_results'].items():
                if isinstance(shears, dict):
                    shear_data.append({
                        'Story': story,
                        'Base_Shear_X': round(shears.get('base_shear_x', 0), 3),
                        'Base_Shear_Y': round(shears.get('base_shear_y', 0), 3),
                        'Story_Shear_X': round(shears.get('story_shear_x', 0), 3),
                        'Story_Shear_Y': round(shears.get('story_shear_y', 0), 3)
                    })
            
            if shear_data:
                df_shears = pd.DataFrame(shear_data)
                rsa_results['html_tables']['shear_results'] = df_shears.to_html(classes='table table-striped', index=False)
                excel_filename = 'rsa_shear_results.xlsx'
                excel_path = os.path.join(rsa_dir, excel_filename)
                df_shears.to_excel(excel_path, index=False)
                rsa_results['excel_files']['shear_results'] = excel_filename
                all_excel_files['shear_results'] = excel_filename
        
        # Process Member Forces
        if 'member_forces' in data:
            member_forces = data['member_forces']
            
            # Process CQC Forces
            if 'cqc_forces' in member_forces:
                cqc_force_data = []
                for element_id, forces in member_forces['cqc_forces'].items():
                    if isinstance(forces, list):
                        for i, force in enumerate(forces):
                            if isinstance(force, dict):
                                cqc_force_data.append({
                                    'Element': element_id,
                                    'Section': i + 1,
                                    'N': round(force.get('N', 0), 3),
                                    'Vy': round(force.get('Vy', 0), 3),
                                    'Vz': round(force.get('Vz', 0), 3),
                                    'T': round(force.get('T', 0), 3),
                                    'My': round(force.get('My', 0), 3),
                                    'Mz': round(force.get('Mz', 0), 3)
                                })
                
                if cqc_force_data:
                    df_cqc_forces = pd.DataFrame(cqc_force_data)
                    rsa_results['html_tables']['cqc_member_forces'] = df_cqc_forces.to_html(classes='table table-striped', index=False)
                    excel_filename = 'rsa_cqc_member_forces.xlsx'
                    excel_path = os.path.join(rsa_dir, excel_filename)
                    df_cqc_forces.to_excel(excel_path, index=False)
                    rsa_results['excel_files']['cqc_member_forces'] = excel_filename
                    all_excel_files['cqc_member_forces'] = excel_filename
            
            # Process Critical Forces
            if 'critical_forces' in member_forces:
                critical_force_data = []
                for element_id, forces in member_forces['critical_forces'].items():
                    if isinstance(forces, dict):
                        for force_type, value in forces.items():
                            critical_force_data.append({
                                'Element': element_id,
                                'Force_Type': force_type,
                                'Max_Value': round(value.get('max', 0), 3) if isinstance(value, dict) else round(value, 3),
                                'Min_Value': round(value.get('min', 0), 3) if isinstance(value, dict) else 0
                            })
                
                if critical_force_data:
                    df_critical_forces = pd.DataFrame(critical_force_data)
                    rsa_results['html_tables']['critical_member_forces'] = df_critical_forces.to_html(classes='table table-striped', index=False)
                    excel_filename = 'rsa_critical_member_forces.xlsx'
                    excel_path = os.path.join(rsa_dir, excel_filename)
                    df_critical_forces.to_excel(excel_path, index=False)
                    rsa_results['excel_files']['critical_member_forces'] = excel_filename
                    all_excel_files['critical_member_forces'] = excel_filename
        
        context.update({
            'rsa_results': rsa_results,
            'all_excel_files': all_excel_files,
            'success': True
        })
        
    except Exception as e:
        print(f"Error processing RSA results: {str(e)}")
        import traceback
        traceback.print_exc()
        context['error'] = f"Error processing RSA results: {str(e)}"
    
    return render(request, 'opensees/convert_to_html_rsa.html', context)


@login_required
def download_excel_rsa(request, project_pk, task_pk, filename):
    # Get project and task for correct folder structure
    project = get_object_or_404(Project, pk=project_pk)
    task = get_object_or_404(Task, pk=task_pk)
    
    user_folder = os.path.join(settings.MEDIA_ROOT, project.title, task.title, f"user_{request.user.username}")
    output_dir = os.path.join(user_folder, "post_processing")
    rsa_dir = os.path.join(output_dir, "response_spectrum_analysis")
    
    # Check if filename exists in RSA directory
    file_path = os.path.join(rsa_dir, filename)
    
    if not os.path.exists(file_path):
        raise Http404("File not found")
    
    with open(file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response


@login_required
def download_excel(request, project_pk, task_pk, filename):
    # Get project and task for correct folder structure
    project = get_object_or_404(Project, pk=project_pk)
    task = get_object_or_404(Task, pk=task_pk)
    
    user_folder = os.path.join(settings.MEDIA_ROOT, project.title, task.title, f"user_{request.user.username}")
    output_dir = os.path.join(user_folder, "post_processing")
    
    # Check if filename contains combo name (subfolder structure)
    file_path = None
    if os.path.exists(os.path.join(output_dir, filename)):
        file_path = os.path.join(output_dir, filename)
    else:
        # Search in combo subfolders
        for root, dirs, files in os.walk(output_dir):
            if filename in files:
                file_path = os.path.join(root, filename)
                break
    
    if not file_path or not os.path.exists(file_path):
        raise Http404("File not found")
    
    with open(file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

@login_required
def convert_to_html(request, project_pk, task_pk):
    context = get_common_context(request, project_pk, task_pk)
    
    # Get project and task for correct folder structure
    project = get_object_or_404(Project, pk=project_pk)
    task = get_object_or_404(Task, pk=task_pk)
    
    user_folder = os.path.join(settings.MEDIA_ROOT, project.title, task.title, f"user_{request.user.username}")
    output_dir = os.path.join(user_folder, "post_processing")
    
    if not os.path.exists(output_dir):
        context['error'] = f"Output directory not found: {output_dir}"
        return render(request, 'opensees/convert_to_html.html', context)
    
    try:
        # Find all load combination folders
        combo_folders = []
        for item in os.listdir(output_dir):
            combo_path = os.path.join(output_dir, item)
            if os.path.isdir(combo_path):
                combo_folders.append(item)
        
        if not combo_folders:
            context['error'] = "No load combination folders found"
            return render(request, 'opensees/convert_to_html.html', context)
        
        print(f"Found load combination folders: {combo_folders}")
        
        # Process each load combination
        combo_results = {}
        all_excel_files = {}
        
        for combo_name in combo_folders:
            combo_path = os.path.join(output_dir, combo_name)
            json_file_path = os.path.join(combo_path, "analysis_results.json")
            
            if not os.path.exists(json_file_path):
                print(f"Skipping {combo_name} - no analysis_results.json found")
                continue
            
            print(f"Processing combination: {combo_name}")
            
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Initialize combo results
            combo_results[combo_name] = {
                'html_tables': {},
                'excel_files': {}
            }
            
            # Process nodal results
            if 'nodal_results' in data:
                if 'reactions' in data['nodal_results']:
                    df_reactions = pd.DataFrame(data['nodal_results']['reactions']).T
                    combo_results[combo_name]['html_tables']['reactions'] = df_reactions.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_reactions.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_reactions.to_excel(excel_path)
                    combo_results[combo_name]['excel_files']['reactions'] = excel_filename
                    all_excel_files[f'{combo_name}_reactions'] = excel_filename
                
                if 'displacements' in data['nodal_results']:
                    df_displacements = pd.DataFrame(data['nodal_results']['displacements']).T
                    combo_results[combo_name]['html_tables']['displacements'] = df_displacements.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_displacements.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_displacements.to_excel(excel_path)
                    combo_results[combo_name]['excel_files']['displacements'] = excel_filename
                    all_excel_files[f'{combo_name}_displacements'] = excel_filename
            
            # Process element results - beam data
            if 'element_results' in data and 'beam' in data['element_results']:
                beam_data = data['element_results']['beam']
                
                # Beam forces
                if 'forces' in beam_data:
                    forces_data = []
                    for element_id, force_list in beam_data['forces'].items():
                        for force_point in force_list:
                            force_point['element_id'] = element_id
                            forces_data.append(force_point)
                    
                    df_forces = pd.DataFrame(forces_data)
                    combo_results[combo_name]['html_tables']['beam_forces'] = df_forces.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_beam_forces.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_forces.to_excel(excel_path, index=False)
                    combo_results[combo_name]['excel_files']['beam_forces'] = excel_filename
                    all_excel_files[f'{combo_name}_beam_forces'] = excel_filename
                
                # Beam stresses
                if 'stresses' in beam_data:
                    stresses_data = []
                    for element_id, stress_list in beam_data['stresses'].items():
                        for stress_point in stress_list:
                            stress_point['element_id'] = element_id
                            stresses_data.append(stress_point)
                    
                    df_stresses = pd.DataFrame(stresses_data)
                    combo_results[combo_name]['html_tables']['beam_stresses'] = df_stresses.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_beam_stresses.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_stresses.to_excel(excel_path, index=False)
                    combo_results[combo_name]['excel_files']['beam_stresses'] = excel_filename
                    all_excel_files[f'{combo_name}_beam_stresses'] = excel_filename
                
                # Beam strains
                if 'strains' in beam_data:
                    strains_data = []
                    for element_id, strain_list in beam_data['strains'].items():
                        for strain_point in strain_list:
                            strain_point['element_id'] = element_id
                            strains_data.append(strain_point)
                    
                    df_strains = pd.DataFrame(strains_data)
                    combo_results[combo_name]['html_tables']['beam_strains'] = df_strains.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_beam_strains.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_strains.to_excel(excel_path, index=False)
                    combo_results[combo_name]['excel_files']['beam_strains'] = excel_filename
                    all_excel_files[f'{combo_name}_beam_strains'] = excel_filename
                
                # Beam deflections
                if 'deflections' in beam_data:
                    deflections_data = []
                    for element_id, deflection_list in beam_data['deflections'].items():
                        for deflection_point in deflection_list:
                            deflection_point['element_id'] = element_id
                            deflections_data.append(deflection_point)
                    
                    df_deflections = pd.DataFrame(deflections_data)
                    combo_results[combo_name]['html_tables']['beam_deflections'] = df_deflections.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_beam_deflections.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_deflections.to_excel(excel_path, index=False)
                    combo_results[combo_name]['excel_files']['beam_deflections'] = excel_filename
                    all_excel_files[f'{combo_name}_beam_deflections'] = excel_filename
                
                # Beam relative deflections
                if 'relative_deflections' in beam_data:
                    rel_deflections_data = []
                    for element_id, rel_deflection_list in beam_data['relative_deflections'].items():
                        for rel_deflection_point in rel_deflection_list:
                            rel_deflection_point['element_id'] = element_id
                            rel_deflections_data.append(rel_deflection_point)
                    
                    df_rel_deflections = pd.DataFrame(rel_deflections_data)
                    combo_results[combo_name]['html_tables']['beam_relative_deflections'] = df_rel_deflections.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_beam_relative_deflections.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_rel_deflections.to_excel(excel_path, index=False)
                    combo_results[combo_name]['excel_files']['beam_relative_deflections'] = excel_filename
                    all_excel_files[f'{combo_name}_beam_relative_deflections'] = excel_filename
                
                # Beam slopes
                if 'slopes' in beam_data:
                    slopes_data = []
                    for element_id, slope_list in beam_data['slopes'].items():
                        for slope_point in slope_list:
                            slope_point['element_id'] = element_id
                            slopes_data.append(slope_point)
                    
                    df_slopes = pd.DataFrame(slopes_data)
                    combo_results[combo_name]['html_tables']['beam_slopes'] = df_slopes.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_beam_slopes.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_slopes.to_excel(excel_path, index=False)
                    combo_results[combo_name]['excel_files']['beam_slopes'] = excel_filename
                    all_excel_files[f'{combo_name}_beam_slopes'] = excel_filename
                
                # Beam max/min deflections
                if 'max_min_deflections' in beam_data:
                    df_max_min_deflections = pd.DataFrame(beam_data['max_min_deflections']).T
                    combo_results[combo_name]['html_tables']['beam_max_min_deflections'] = df_max_min_deflections.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_beam_max_min_deflections.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_max_min_deflections.to_excel(excel_path)
                    combo_results[combo_name]['excel_files']['beam_max_min_deflections'] = excel_filename
                    all_excel_files[f'{combo_name}_beam_max_min_deflections'] = excel_filename
                
                # Beam max/min slopes
                if 'max_min_slopes' in beam_data:
                    df_max_min_slopes = pd.DataFrame(beam_data['max_min_slopes']).T
                    combo_results[combo_name]['html_tables']['beam_max_min_slopes'] = df_max_min_slopes.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_beam_max_min_slopes.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_max_min_slopes.to_excel(excel_path)
                    combo_results[combo_name]['excel_files']['beam_max_min_slopes'] = excel_filename
                    all_excel_files[f'{combo_name}_beam_max_min_slopes'] = excel_filename
                
                # Beam properties
                if 'beam_properties' in beam_data:
                    properties_data = []
                    for element_id, properties in beam_data['beam_properties'].items():
                        flattened_props = {'element_id': element_id}
                        
                        # Flatten nested dictionaries
                        for key, value in properties.items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, dict):
                                        for sub_sub_key, sub_sub_value in sub_value.items():
                                            flattened_props[f"{key}_{sub_key}_{sub_sub_key}"] = sub_sub_value
                                    else:
                                        flattened_props[f"{key}_{sub_key}"] = sub_value
                            elif isinstance(value, list):
                                flattened_props[key] = str(value)
                            else:
                                flattened_props[key] = value
                        
                        properties_data.append(flattened_props)
                    
                    df_properties = pd.DataFrame(properties_data)
                    combo_results[combo_name]['html_tables']['beam_properties'] = df_properties.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_beam_properties.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_properties.to_excel(excel_path, index=False)
                    combo_results[combo_name]['excel_files']['beam_properties'] = excel_filename
                    all_excel_files[f'{combo_name}_beam_properties'] = excel_filename
            
            # Process shell element results
            if 'element_results' in data and 'shell' in data['element_results']:
                shell_data = data['element_results']['shell']
                
                if 'forces' in shell_data:
                    df_shell_forces = pd.DataFrame(shell_data['forces']).T
                    combo_results[combo_name]['html_tables']['shell_forces'] = df_shell_forces.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_shell_forces.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_shell_forces.to_excel(excel_path)
                    combo_results[combo_name]['excel_files']['shell_forces'] = excel_filename
                    all_excel_files[f'{combo_name}_shell_forces'] = excel_filename
                
                if 'stresses' in shell_data:
                    shell_stresses_data = []
                    for element_id, stress_list in shell_data['stresses'].items():
                        for stress_point in stress_list:
                            stress_point['element_id'] = element_id
                            shell_stresses_data.append(stress_point)
                    
                    df_shell_stresses = pd.DataFrame(shell_stresses_data)
                    combo_results[combo_name]['html_tables']['shell_stresses'] = df_shell_stresses.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_shell_stresses.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_shell_stresses.to_excel(excel_path, index=False)
                    combo_results[combo_name]['excel_files']['shell_stresses'] = excel_filename
                    all_excel_files[f'{combo_name}_shell_stresses'] = excel_filename
                
                if 'strains' in shell_data:
                    shell_strains_data = []
                    for element_id, strain_list in shell_data['strains'].items():
                        for strain_point in strain_list:
                            strain_point['element_id'] = element_id
                            shell_strains_data.append(strain_point)
                    
                    df_shell_strains = pd.DataFrame(shell_strains_data)
                    combo_results[combo_name]['html_tables']['shell_strains'] = df_shell_strains.to_html(classes='table table-striped')
                    excel_filename = f'{combo_name}_shell_strains.xlsx'
                    excel_path = os.path.join(combo_path, excel_filename)
                    df_shell_strains.to_excel(excel_path, index=False)
                    combo_results[combo_name]['excel_files']['shell_strains'] = excel_filename
                    all_excel_files[f'{combo_name}_shell_strains'] = excel_filename
        
        context.update({
            'combo_results': combo_results,
            'all_excel_files': all_excel_files,
            'combo_folders': combo_folders,
            'success': True
        })
        
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        import traceback
        traceback.print_exc()
        context['error'] = f"Error processing results: {str(e)}"
    
    return render(request, 'opensees/convert_to_html.html', context)


@login_required
def calculator1(request, project_pk, task_pk):
    context = get_common_context(request, project_pk, task_pk)
    
    if request.method == 'POST':
        # Get parameters from POST data
        numBayX = int(request.POST['numBayX'])
        numBayY = int(request.POST['numBayY'])
        numFloor = int(request.POST['numFloor'])
        num_x_div = int(request.POST['num_x_div'])
        num_y_div = int(request.POST['num_y_div'])
        
        # Parse list inputs
        bayWidthX = ast.literal_eval(request.POST['bayWidthX'])
        bayWidthY = ast.literal_eval(request.POST['bayWidthY'])
        storyHeights = ast.literal_eval(request.POST['storyHeights'])
        
        # Helper function to safely parse inputs
        def parse_input(input_str, default):
            if not input_str.strip():
                return default
            try:
                return ast.literal_eval(input_str)
            except (ValueError, SyntaxError):
                return default
        
        # Process node operations
        add_nodes = parse_input(request.POST.get('add_nodes', ''), {})
        delete_nodes = parse_input(request.POST.get('delete_nodes', ''), [])
        
        # Process beam operations
        add_beams = parse_input(request.POST.get('add_beams', ''), {})
        delete_beams = parse_input(request.POST.get('delete_beams', ''), [])
        
        # Process shell operations
        add_shells = parse_input(request.POST.get('add_shells', ''), {})
        delete_shells = parse_input(request.POST.get('delete_shells', ''), [])

        # Create output directory
        user_dir = os.path.join(settings.MEDIA_ROOT, str(request.user.username))
        os.makedirs(user_dir, exist_ok=True)
        filename = os.path.join(user_dir, "model")
        # Generate model
        building = generate_building_model(
            num_bays_x=numBayX,
            num_bays_y=numBayY,
            num_floors=numFloor,
            bay_width_x=bayWidthX,
            bay_width_y=bayWidthY,
            story_heights=storyHeights,
            num_x_div=num_x_div,
            num_y_div=num_y_div
        )
        # Generate model
        building, filepath = create_and_visualize_model(
            building=building,
            num_bays_x=numBayX,
            num_bays_y=numBayY,
            num_floors=numFloor,
            bay_width_x=bayWidthX,
            bay_width_y=bayWidthY,
            story_heights=storyHeights,
            num_x_div=num_x_div,
            num_y_div=num_y_div,
            output_dir=user_dir,
            filename=filename  # Pass the complete path with base filename
        )
        print("abc1", building["beams"])
        building = modify_building_model(building, add_nodes, delete_nodes, add_beams, delete_beams, add_shells, delete_shells)
        print("abc2", building["beams"])
                # Generate model
        building, filepath = create_and_visualize_model(
            building=building,
            num_bays_x=numBayX,
            num_bays_y=numBayY,
            num_floors=numFloor,
            bay_width_x=bayWidthX,
            bay_width_y=bayWidthY,
            story_heights=storyHeights,
            num_x_div=num_x_div,
            num_y_div=num_y_div,
            output_dir=user_dir,
            filename=filename  # Pass the complete path with base filename
        )

        # building, filepath = create_and_visualize_model(
        #     num_bays_x=numBayX,
        #     num_bays_y=numBayY,
        #     num_floors=numFloor,
        #     bay_width_x=bayWidthX,
        #     bay_width_y=bayWidthY,
        #     story_heights=storyHeights,
        #     num_x_div=num_x_div,
        #     num_y_div=num_y_div,
        #     output_dir=user_dir,
        #     filename=filename  # Pass the complete path with base filename
        # )
        # Convert filesystem path to URL-accessible path
        relative_path = os.path.relpath(filepath, settings.MEDIA_ROOT)
        image_url = os.path.join(settings.MEDIA_URL, relative_path)
        
        context['image_url'] = image_url
        context['building'] = building
        context['frame_nodes'] = building["frame_nodes"]  # Changed from "nodes" to "frame_nodes"
        context['beams'] = building["beams"]
        context['shells'] = building["shells"]
        

    return render(request, 'opensees/calculator1.html', context)


# Helper function to safely get float values from POST data
def get_float(request, name, default=0.0):
    try:
        return float(request.POST.get(name, default))
    except (ValueError, TypeError):
        return default




@login_required
def calculator2(request, project_pk, task_pk):
    print("=== calculator2 view called ===")
    print(f"project_pk: {project_pk}, task_pk: {task_pk}")
    
    context = get_common_context(request, project_pk, task_pk)
    print("Context created")

    if request.method == 'POST':
        print("POST request received")
        
        # Debug: print all POST data
        print("All POST data:")
        # for key, value in request.POST.items():
        #     print(f"{key}: {value}")
        
        analysis_data_str = request.POST['analysis_parameters']
        # print("Raw analysis_data_str:")
        # print(analysis_data_str)
        
        # Create namespace for executed code
        analysis_data = {}
        # exec(analysis_data_str, {}, analysis_data)
        exec(
            analysis_data_str,
            {
                "Concrete": Concrete,
                "SteelBar": SteelBar,
                "ConcreteLinear": ConcreteLinear,
                "RectangularStressBlock": RectangularStressBlock,
                "SteelElasticPlastic": SteelElasticPlastic,
                "np": np  # include numpy if used in the code string
            },
            analysis_data
        )

        print("Python code executed successfully")
        
        # 2. Extract the configuration from executed code
        materials = analysis_data['materials']
        # advanced_materials = analysis_data['advanced_materials']
        advanced_materials = analysis_data.get('advanced_materials', [])

        geometry_definitions = analysis_data['geometry_definitions']
        rebar_definitions = analysis_data['rebar_definitions']
        load_data = analysis_data['load_data']
        mesh_size = analysis_data['mesh_size']
        print("# 2. Extract the configuration from executed code")
        
        # 3. Prepare output directory
        user_folder = os.path.join(settings.MEDIA_ROOT, f"user_{request.user.id}")
        os.makedirs(user_folder, exist_ok=True)
        output_dir = os.path.join(user_folder, "analysis_output")
        os.makedirs(output_dir, exist_ok=True)
        print("model_generation started ")
        
        # 4. Run the analysis
        section = model_generation(
            materials=materials,
            advanced_materials=advanced_materials,
            geometry_definitions=geometry_definitions,
            rebar_definitions=rebar_definitions,
            mesh_size=mesh_size,
            plot_title="Generated Section"
        )
        print("model_generation successful ")

        # Analyze with loads
        section_props, max_stresses, filepaths = analyze_section(
            section=section,
            output_dir=output_dir,
            load_data=load_data
        )
        # print("section_props:")
        # print(section_props)
        print("filepaths13:")
        print(filepaths)
        # 5. Prepare results
        image_urls = {}
        for path in filepaths:
            if path and os.path.exists(path):
                rel_path = os.path.relpath(path, settings.MEDIA_ROOT)
                name = os.path.splitext(os.path.basename(path))[0]  # extract filename without extension
                image_urls[name] = os.path.join(settings.MEDIA_URL, rel_path)


        context['result'] = {
            'properties': section_props,
            'stresses': max_stresses,
            'images': image_urls,
            'success': True
        }

    else:
        context['result'] = {'success': False}

    return render(request, 'opensees/calculator2.html', context)


@login_required
def calculator3(request, project_pk, task_pk):
    print("=== calculator2 view called ===")
    print(f"project_pk: {project_pk}, task_pk: {task_pk}")
    
    context = get_common_context(request, project_pk, task_pk)
    print("Context created")

    if request.method == 'POST':
        print("POST request received")
        
        # Debug: print all POST data
        print("All POST data:")
        # for key, value in request.POST.items():
        #     print(f"{key}: {value}")
        
        analysis_data_str = request.POST['analysis_parameters']
        # print("Raw analysis_data_str:")
        # print(analysis_data_str)
        
        # Create namespace for executed code
        analysis_data = {}
        # exec(analysis_data_str, {}, analysis_data)
        exec(
            analysis_data_str,
            {
                "Concrete": Concrete,
                "SteelBar": SteelBar,
                "ConcreteLinear": ConcreteLinear,
                "RectangularStressBlock": RectangularStressBlock,
                "SteelElasticPlastic": SteelElasticPlastic,
                "np": np  # include numpy if used in the code string
            },
            analysis_data
        )

        print("Python code executed successfully")
        
        # 2. Extract the configuration from executed code
        # materials = analysis_data['materials']
        materials = analysis_data.get('materials', [])
        advanced_materials = analysis_data['advanced_materials']
        geometry_definitions = analysis_data['geometry_definitions']
        rebar_definitions = analysis_data['rebar_definitions']
        load_data = analysis_data['load_data']
        mesh_size = analysis_data['mesh_size']
        print("# 2. Extract the configuration from executed code")
        
        # 3. Prepare output directory
        user_folder = os.path.join(settings.MEDIA_ROOT, f"user_{request.user.id}")
        os.makedirs(user_folder, exist_ok=True)
        output_dir = os.path.join(user_folder, "analysis_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create concrete results directory in MEDIA_ROOT
        concrete_results_dir = os.path.join(user_folder, "concrete_results")
        os.makedirs(concrete_results_dir, exist_ok=True)
        
        print("model_generation started ")
        

        section, compound_geom = concrete_model_generation(
            materials=materials,
            advanced_materials=advanced_materials,
            geometry_definitions=geometry_definitions,
            rebar_definitions=rebar_definitions,
            mesh_size=mesh_size,
            plot_title="Generated Section"
        )
        print("Model created successfully")

        from concreteproperties import ConcreteSection
        from concreteproperties.design_codes import AS3600, ACI318

        # Create concrete section
        conc_sec = ConcreteSection(compound_geom)

        # Initialize design code (either pass the class or string, but be consistent)
        design_code = ACI318  # The class itself, not an instance

        # Perform analysis - FIXED: Use concrete_results_dir instead of string
        analysis_results, image_paths = analyze_concrete_section(
            concrete_section=conc_sec,
            compressive_strength=32,
            steel_yield_strength=500,
            n_design=1000e3,  # 1000 kN
            plot_results=True,
            file_directory=concrete_results_dir,  # Use the absolute path instead of "concrete_results"
            design_code="ACI318"  # Pass the design code class
        )

        print("image_paths10:")
        print(image_paths)
        
        # 5. Prepare results
        image_urls = {}
        for path in image_paths:
            if path and os.path.exists(path):
                rel_path = os.path.relpath(path, settings.MEDIA_ROOT)
                name = os.path.splitext(os.path.basename(path))[0]  # extract filename without extension
                # Use forward slashes for URLs
                url_path = rel_path.replace('\\', '/')
                image_urls[name] = settings.MEDIA_URL + url_path

        print("image_paths11:")
        print(image_paths)

        context['result'] = {
            'properties': analysis_results,
            # 'stresses': max_stresses,
            'images': image_urls,
            'success': True
        }

    else:
        context['result'] = {'success': False}
    return render(request, 'opensees/calculator3.html', context)


# Import your seismic analysis functions
# from seismic import (
#     calculate_N, calculate_base_shear_and_overturning_moment,
#     get_seismic_zone_coefficient_town, get_importance_factor,
#     get_occupancy_category, get_system_info, calculate_building_period,
#     get_soil_parameters, calculate_normalized_acceleration_spectrum,
#     classify_spt_value, get_site_classification
# )

from opensees.seismic import calculate_N, calculate_base_shear_and_overturning_moment, calculate_building_period, calculate_normalized_acceleration_spectrum, classify_spt_value, get_importance_factor, get_occupancy_category, get_seismic_zone_coefficient_town, get_soil_parameters, get_system_info


@login_required
def calculator4(request, project_id, task_id):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    task = get_object_or_404(Task, id=task_id, project=project)
    
    result = None
    
    if request.method == 'POST':
        try:
            # Get the seismic parameters from textarea
            seismic_script = request.POST.get('seismic_parameters', '').strip()
            print(f"Received seismic script content (length: {len(seismic_script)} chars)")
            
            # Create temp directory with user-specific subfolder
            import tempfile
            import importlib.util
            import os
            from django.conf import settings
            
            user_dir = os.path.join(settings.MEDIA_ROOT, 'temp_seismic', str(request.user.pk))
            print(f"Creating user directory at: {user_dir}")
            os.makedirs(user_dir, exist_ok=True)
            
            # Create unique filename
            script_filename = f"seismic_params.py"
            script_path = os.path.normpath(os.path.join(user_dir, script_filename))
            
            # Write with proper line endings and encoding
            print(f"Writing script to file (size: {len(seismic_script)} bytes)")
            with open(script_path, 'w') as f:
                f.write(seismic_script)
            
            print(f"Attempting to import module from {script_path}")
            spec = importlib.util.spec_from_file_location("seismic_params", script_path)
            params_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(params_module)
            print("Module imported successfully")
            
            # Helper function to safely get attributes
            def get_param_attr(module, name, default=None):
                attr = getattr(module, name, default) if hasattr(module, name) else default
                print(f"Retrieved parameter '{name}': {attr}")
                return attr
            
            # Extract parameters from the module
            num_stories = get_param_attr(params_module, 'num_stories', 10)
            floor_height = get_param_attr(params_module, 'floor_height', 3.0)
            floor_weight = get_param_attr(params_module, 'floor_weight', 5000.0)
            building_type = get_param_attr(params_module, 'building_type', 'Concrete moment-resisting frames')
            location = get_param_attr(params_module, 'location', 'Dhaka')
            nature_of_occupancy = get_param_attr(params_module, 'nature_of_occupancy', 'Elementary school or secondary school facilities with a capacity greater than 250')
            system_type = get_param_attr(params_module, 'system_type', 'C. MOMENT RESISTING FRAME SYSTEMS (no shear wall)')
            subtypes = get_param_attr(params_module, 'subtypes', '5. Intermediate reinforced concrete moment frames')
            
            # SPT data
            spt_depths = get_param_attr(params_module, 'spt_depths', [5]*20)
            spt_values = get_param_attr(params_module, 'spt_values', [5,10,5,14,14,20,22,24,26,24,20,30,35,35,34,24,24,10,20,25])
            
            # Perform seismic analysis calculations
            result = perform_seismic_analysis(
                num_stories, floor_height, floor_weight, building_type,
                location, nature_of_occupancy, system_type, subtypes, spt_depths, spt_values
            )
            
        except Exception as e:
            print(f"Error in seismic analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            result = {'error': str(e)}
    
    context = {
        'project': project,
        'task': task,
        'result': result,
    }
    
    return render(request, 'opensees/calculator4.html', context)

def perform_seismic_analysis(num_stories, floor_height, floor_weight, building_type, 
                           location, nature_of_occupancy, system_type, subtypes, d, N):
    """
    Perform complete seismic analysis calculations
    """
    try:
        # Building parameters
        floor_heights = [floor_height] * num_stories
        total_height = sum(floor_heights)
        floor_weights = [floor_weight] * num_stories
        total_weight = sum(floor_weights)
        heights = [sum(floor_heights[:i+1]) for i in range(num_stories)]
        
        # Calculate corrected SPT value
        corrected_SPT_value = calculate_N(d, N)
        soil_type = classify_spt_value(corrected_SPT_value)
        
        # Get seismic zone coefficient
        Z = get_seismic_zone_coefficient_town(location)
        
        # Get importance factor
        occupancy_category = get_occupancy_category(nature_of_occupancy)
        I = get_importance_factor(occupancy_category)
        
        # Get structural system parameters
        system_info = get_system_info(system_type, subtypes)
        R = system_info['R']
        omega = system_info['omega']
        Cd = system_info['Cd']
        
        # Calculate building period
        Ct, m = calculate_base_shear_and_overturning_moment(building_type)
        T = calculate_building_period(total_height, m, Ct, structural_type="concrete")
        
        # Get soil parameters and calculate spectrum
        soil_factor, TB, TC, TD = get_soil_parameters(soil_type)
        Cs = calculate_normalized_acceleration_spectrum(soil_factor, T, TB, TC, TD)
        
        # Calculate base shear
        Sa = (2/3) * Z * I * Cs / R
        V = Sa * total_weight
        
        # Calculate story forces
        k = 1.0 if T <= 0.5 else 2.0 if T >= 2.5 else 1.0 + (T - 0.5)/2.0
        
        Cvx = []
        for i in range(num_stories):
            numerator = floor_weights[i] * heights[i]**k
            denominator = sum(floor_weights[j] * heights[j]**k for j in range(num_stories))
            Cvx.append(numerator / denominator)
        
        Fx = [V * c for c in Cvx]
        
        # Calculate story shears
        Vx = []
        for i in range(num_stories):
            Vx.append(sum(Fx[i:num_stories]))
        
        # Prepare results
        story_results = []
        for i in reversed(range(num_stories)):
            story_num = num_stories - i
            story_results.append({
                'story': story_num,
                'height': heights[i],
                'weight': floor_weights[i],
                'force': Fx[i],
                'shear': Vx[i]
            })
        
        result = {
            'building_parameters': {
                'building_type': building_type,
                'num_stories': num_stories,
                'total_height': total_height,
                'total_weight': total_weight,
                'location': location,
                'occupancy_category': occupancy_category,
                'importance_factor': I
            },
            'soil_parameters': {
                'corrected_spt': corrected_SPT_value,
                'soil_type': soil_type,
                'soil_factor': soil_factor,
                'TB': TB,
                'TC': TC,
                'TD': TD
            },
            'structural_parameters': {
                'system_type': system_type,
                'subtypes': subtypes,
                'R': R,
                'omega': omega,
                'Cd': Cd,
                'Ct': Ct,
                'm': m
            },
            'analysis_results': {
                'building_period': T,
                'seismic_zone_coefficient': Z,
                'normalized_spectrum': Cs,
                'design_acceleration': Sa,
                'base_shear': V,
                'k_factor': k
            },
            'story_results': story_results
        }
        
        return result
        
    except Exception as e:
        return {'error': f'Analysis error: {str(e)}'}
    


from opensees.wind import basic_wind_speed, importance_factor, wind_directionality_factor,calculate_topographic_factor, terrain_exposure_constants,velocity_pressure_coefficient, Compute_Gust_factor,calculate_wall_pressure_coefficient, calculate_base_shear_and_overturning_moment,calculate_building_period, analyze_wind_direction, calculate_wind_loads,get_occupancy_category



@login_required
def calculator5(request, project_id, task_id):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    task = get_object_or_404(Task, id=task_id, project=project)
    
    result = None
    
    if request.method == 'POST':
        try:
            # Get the wind parameters from textarea
            wind_script = request.POST.get('wind_parameters', '').strip()
            print(f"Received wind script content (length: {len(wind_script)} chars)")
            
            # Create temp directory with user-specific subfolder
            import tempfile
            import importlib.util
            import os
            from django.conf import settings
            
            user_dir = os.path.join(settings.MEDIA_ROOT, 'temp_wind', str(request.user.pk))
            print(f"Creating user directory at: {user_dir}")
            os.makedirs(user_dir, exist_ok=True)
            
            # Create unique filename
            script_filename = f"wind_params.py"
            script_path = os.path.normpath(os.path.join(user_dir, script_filename))
            
            # Write with proper line endings and encoding
            print(f"Writing script to file (size: {len(wind_script)} bytes)")
            with open(script_path, 'w') as f:
                f.write(wind_script)
            
            print(f"Attempting to import module from {script_path}")
            spec = importlib.util.spec_from_file_location("wind_params", script_path)
            params_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(params_module)
            print("Module imported successfully")
            
            # Helper function to safely get attributes
            def get_param_attr(module, name, default=None):
                attr = getattr(module, name, default) if hasattr(module, name) else default
                print(f"Retrieved parameter '{name}': {attr}")
                return attr
            
            # Extract parameters from the module
            num_stories = get_param_attr(params_module, 'num_stories', 10)
            story_height = get_param_attr(params_module, 'story_height', 3.2)
            building_dimensions = get_param_attr(params_module, 'building_dimensions', {"X": 30, "Y": 20})
            wind_direction = get_param_attr(params_module, 'wind_direction', 'X')
            structural_type = get_param_attr(params_module, 'structural_type', 'Concrete moment-resisting frames')
            location = get_param_attr(params_module, 'location', 'Dhaka')
            exposure_category = get_param_attr(params_module, 'exposure_category', 'B')
            nature_of_occupancy = get_param_attr(params_module, 'nature_of_occupancy', 'Elementary school or secondary school facilities with a capacity greater than 250')
            structure_type = get_param_attr(params_module, 'structure_type', 'Main Wind Force Resisting System')
            topographic_params = get_param_attr(params_module, 'topographic_params', {"k1": 0, "k2": 0, "k3": 0})
            
            # Prepare wind input data
            wind_input_data = {
                "num_stories": num_stories,
                "story_height": story_height, 
                "building_dimensions": building_dimensions,
                "wind_direction": wind_direction,
                "structural_type": structural_type,
                "location": location,
                "exposure_category": exposure_category,
                "nature_of_occupancy": nature_of_occupancy,
                "structure_type": structure_type,
                "topographic_params": topographic_params
            }
            
            # Perform wind analysis calculations
            result = perform_wind_analysis(wind_input_data)
            
        except Exception as e:
            print(f"Error in wind analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            result = {'error': str(e)}
    
    context = {
        'project': project,
        'task': task,
        'result': result,
    }
    
    return render(request, 'opensees/calculator5.html', context)


def perform_wind_analysis(wind_input_data):
    """
    Perform complete wind analysis calculations
    """
    try:
        # Extract all input parameters
        num_stories = wind_input_data["num_stories"]
        story_height = wind_input_data["story_height"]
        building_dimensions = wind_input_data["building_dimensions"]
        wind_direction = wind_input_data["wind_direction"]
        structural_type = wind_input_data["structural_type"]
        location = wind_input_data["location"]
        exposure_category = wind_input_data["exposure_category"]
        nature_of_occupancy = wind_input_data["nature_of_occupancy"]
        structure_type = wind_input_data["structure_type"]
        topographic_params = wind_input_data.get("topographic_params", {"k1": 0, "k2": 0, "k3": 0})
        
        # Determine occupancy category
        occupancy_category = get_occupancy_category(nature_of_occupancy)
        
        total_height = num_stories * story_height
        
        # Get length and width based on wind direction
        building_length, building_width = analyze_wind_direction(building_dimensions, wind_direction)
        L_over_B = building_length / building_width

        def validate_numeric(value, name):
            """Ensure a value is numeric and not None"""
            if value is None:
                raise ValueError(f"{name} cannot be None")
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {type(value)}")
            return value

        # Get basic parameters with validation
        V = validate_numeric(basic_wind_speed(location), "Wind speed (V)")
        I = validate_numeric(importance_factor(occupancy_category, V), "Importance factor (I)")
        Kd = validate_numeric(wind_directionality_factor(structure_type), "Directionality factor (Kd)")
        Kzt = validate_numeric(calculate_topographic_factor(
            topographic_params["k1"], topographic_params["k2"], topographic_params["k3"]),
            "Topographic factor (Kzt)")
        
        exposure_constants = terrain_exposure_constants(exposure_category)
        if not isinstance(exposure_constants, dict):
            raise ValueError("Exposure constants should be a dictionary")
            
        # Validate exposure constants
        for const in ['zg', 'alpha', 'z_min', 'c', 'L', 'epsilon', 'a_var', 'b_var']:
            validate_numeric(exposure_constants.get(const), f"Exposure constant {const}")

        Ct, m = calculate_base_shear_and_overturning_moment(structural_type)
        T = validate_numeric(calculate_building_period(total_height, m, Ct), "Building period (T)")

        # Calculate maximum qz at building top for leeward pressures
        Kz_top = validate_numeric(
            velocity_pressure_coefficient(total_height, exposure_constants['zg'], 
                                       exposure_constants['alpha'], exposure_category),
            "Velocity pressure coefficient at top (Kz)"
        )
        qz_max = 0.000613 * Kz_top * Kzt * Kd * (V**2) * I

        # Initialize result lists
        wind_pressures_windward = []
        wind_pressures_leeward = []
        total_pressures = []
        wind_forces = []
        
        for i in range(1, num_stories + 1):
            z = i * story_height
            
            # Velocity pressure coefficient for current story
            Kz = validate_numeric(
                velocity_pressure_coefficient(z, exposure_constants['zg'], 
                                           exposure_constants['alpha'], exposure_category),
                "Velocity pressure coefficient (Kz)"
            )
            
            # Velocity pressure for windward (story-specific)
            qz = 0.000613 * Kz * Kzt * Kd * (V**2) * I
            
            # Pressure coefficients
            Cp_windward = validate_numeric(
                calculate_wall_pressure_coefficient("Windward Wall", L_over_B),
                "Windward Cp"
            )
            Cp_leeward = validate_numeric(
                calculate_wall_pressure_coefficient("Leeward Wall", L_over_B),
                "Leeward Cp"
            )
                
            # Gust factor
            Gf = validate_numeric(
                Compute_Gust_factor(T, z, exposure_constants['z_min'], exposure_constants['c'],
                                 exposure_constants['L'], exposure_constants['epsilon'],
                                 V, exposure_constants['a_var'], exposure_constants['b_var'],
                                 total_height, building_length, building_width),
                "Gust factor (Gf)"
            )
            
            # Pressures (using qz_max for leeward)
            p_windward = qz * Gf * Cp_windward
            p_leeward = qz_max * Gf * Cp_leeward  # Using maximum qz
            p_total = p_windward + abs(p_leeward)  # Total pressure
            
            wind_pressures_windward.append(p_windward)
            wind_pressures_leeward.append(p_leeward)
            total_pressures.append(p_total)
            
            # Total force (windward + leeward)
            story_area = story_height * building_width
            F_total = p_total * story_area
            wind_forces.append(F_total)

        # Calculate story shears (cumulative from top)
        story_shears = []
        cumulative_force = 0
        for i in range(num_stories):
            cumulative_force += wind_forces[num_stories - 1 - i]
            story_shears.append(cumulative_force)
        story_shears.reverse()  # Reverse to match story order

        # Prepare story results
        story_results = []
        for i in range(num_stories):
            story_num = i + 1
            story_results.append({
                'story': story_num,
                'height': story_num * story_height,
                'windward_pressure': wind_pressures_windward[i],
                'leeward_pressure': wind_pressures_leeward[i],
                'total_pressure': total_pressures[i],
                'force': wind_forces[i],
                'shear': story_shears[i]
            })

        # Calculate total wind force
        total_wind_force = sum(wind_forces)
        
        result = {
            'building_parameters': {
                'num_stories': num_stories,
                'story_height': story_height,
                'total_height': total_height,
                'building_length': building_length,
                'building_width': building_width,
                'L_over_B': L_over_B,
                'wind_direction': wind_direction,
                'location': location,
                'occupancy_category': occupancy_category,
                'structural_type': structural_type
            },
            'wind_parameters': {
                'basic_wind_speed': V,
                'importance_factor': I,
                'directionality_factor': Kd,
                'topographic_factor': Kzt,
                'exposure_category': exposure_category,
                'structure_type': structure_type
            },
            'exposure_constants': exposure_constants,
            'analysis_results': {
                'building_period': T,
                'max_velocity_pressure': qz_max,
                'total_wind_force': total_wind_force,
                'Ct': Ct,
                'm': m
            },
            'story_results': story_results
        }
        
        return result
        
    except Exception as e:
        return {'error': f'Wind analysis error: {str(e)}'}







