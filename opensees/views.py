import ast
from collections import defaultdict
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

from opensees.center_of_mass import calculate_center_of_mass, generate_report, process_loads
from opensees.punching_shear import run_punching_shear_analysis


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
# filter_beam_x_by_z, filter_beam_y_by_z, filter_column_members_by_z,
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

def create_frame_elements_with_transformations(nodes, frame_elements_using_angle):

    transformations = []
    node_dict = {node[0]: node[1:4] for node in nodes}
    
    for elem in frame_elements_using_angle:
        tag = elem[1]
        iNode = elem[2]
        jNode = elem[3]
        angle = elem[4]
        
        # Get node coordinates
        i_coords = np.array(node_dict[iNode])
        j_coords = np.array(node_dict[jNode])
        
        # Calculate local x axis (from i to j)
        vecx = j_coords - i_coords
        vecx = vecx / np.linalg.norm(vecx)
        
        # For vertical elements (z-direction)
        if abs(vecx[2]) > 0.99:  # Mostly vertical
            # Default y-axis is global y (0,1,0)
            vecy = np.array([0.0, 1.0, 0.0])
            vecz = np.cross(vecx, vecy)
            vecz = vecz / np.linalg.norm(vecz)
        else:
            # Default z-axis is global z (0,0,1)
            vecz = np.array([0.0, 0.0, 1.0])
            vecy = np.cross(vecz, vecx)
            vecy = vecy / np.linalg.norm(vecy)
        
        # Apply rotation about local x-axis
        if angle != 0:
            angle_rad = math.radians(angle)
            rot_matrix = np.array([
                [1, 0, 0],
                [0, math.cos(angle_rad), -math.sin(angle_rad)],
                [0, math.sin(angle_rad), math.cos(angle_rad)]
            ])
            vecy = rot_matrix @ vecy
            vecz = rot_matrix @ vecz
        
        # The transformation is defined by the local x-z plane
        transformations.append([
            "Linear", 
            tag, 
            float(vecz[0]), 
            float(vecz[1]), 
            float(vecz[2])
        ])
    
    frame_elements = []
    for elem in frame_elements_using_angle:
        frame_elements.append([
            elem[0],  # element type
            elem[1],  # element tag
            elem[2],  # iNode
            elem[3],  # jNode
            elem[1],  # transformation tag (same as element tag)
            elem[5]   # integration tag
        ])
    return transformations, frame_elements




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
                # 'elastic_section': get_model_attr(model_module, 'elastic_section', []),
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

            # Elastic section definition using the properties from the list
            elastic_section = ["Elastic", 
                                1, 
                                model_data['section_properties'][0][9], # E
                                model_data['section_properties'][0][2],  # A
                                model_data['section_properties'][0][4],  # Iz
                                model_data['section_properties'][0][3],  # Iy
                                model_data['section_properties'][0][10], # G
                                model_data['section_properties'][0][5]] # J
            # Get frame elements with transformations if they weren't provided
            if not model_data['transformations'] and model_data['frame_elements']:
                print("Creating transformations for frame elements")
                transformations, frame_elements_with_trans = create_frame_elements_with_transformations()
                model_data['transformations'] = transformations
                model_data['frame_elements'] = frame_elements_with_trans
            else:
                print("Using provided transformations and frame elements")
                frame_elements_with_trans = model_data['frame_elements']
            print("Model data extracted, creating structural model...")
            # Create the structural model
            (node_loads, element_uniform_loads, shell_pressure_loads, 
            section_properties, elastic_section, aggregator_section, 
            beam_integrations, frame_elements) = create_structural_model(
                model_data['materials'],
                model_data['nd_materials'],
                model_data['section_properties'],
                elastic_section,
                model_data['aggregator_section'],
                model_data['shell_section'],
                model_data['nodes'],
                model_data['transformations'],
                # transformations,
                model_data['beam_integrations'],
                model_data['frame_elements'],
                # frame_elements,
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
    print("hello")
    context = get_common_context(request, project_pk, task_pk)
    
    # Get project and task for correct folder structure
    project = get_object_or_404(Project, pk=project_pk)
    task = get_object_or_404(Task, pk=task_pk)
    
    user_folder = os.path.join(settings.MEDIA_ROOT, project.title, task.title, f"user_{request.user.username}")
    output_dir = os.path.join(user_folder, "post_processing")
    output_dir = os.path.join(output_dir, "gravity_analysis")
    
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





def convert_frame_nodes(frame_nodes_dict, building=None):
    """
    Convert frame nodes from dictionary format to list format.
    Input format: {1: array([x,y,z]), 2: array([x,y,z]), ...}
    Output format: [[id, x, y, z, mass], ...]
    Also includes nodes from shell elements if building is provided.
    """
    frame_nodes_list = []
    all_nodes = dict(frame_nodes_dict)  # Copy original frame nodes
    
    # If building is provided, add shell element nodes to the node list
    if building:
        shell_node_ids = set()
        
        # Collect all shell nodes from all floors
        for floor_data in building.get('shells', []):
            # Get nodes from floor_data's nodes dictionary
            shell_node_ids.update(floor_data.get('nodes', {}).keys())
            
            # Also get nodes from each element in the floor
            for elem_data in floor_data.get('elements', {}).values():
                shell_node_ids.update(elem_data.get('nodes', []))
        
        # Add shell nodes that aren't already in frame_nodes
        for node_id in shell_node_ids:
            if node_id not in all_nodes:
                # Try to find the node in the building's all_nodes or floor nodes
                if 'all_nodes' in building and node_id in building['all_nodes']:
                    all_nodes[node_id] = building['all_nodes'][node_id]
                else:
                    # Search through floor data for the node coordinates
                    for floor_data in building.get('shells', []):
                        if node_id in floor_data.get('nodes', {}):
                            all_nodes[node_id] = np.array(floor_data['nodes'][node_id])
                            break
                    else:
                        print(f"Warning: Shell node {node_id} not found in any node data")
                        continue
    
    # Determine base nodes (z=0) and assign None mass
    base_nodes = [node_id for node_id, coords in all_nodes.items() 
                 if isinstance(coords, (list, np.ndarray)) and abs(coords[2]) < 1e-6]
    
    for node_id, coords in all_nodes.items():
        # Convert numpy array to list if needed
        if hasattr(coords, 'tolist'):
            coords = coords.tolist()
        elif not isinstance(coords, list):
            coords = list(coords)
        
        # Assign mass - None for base nodes, default mass for others
        mass = None if node_id in base_nodes else [200, 200, 200, 0, 0, 0]
        
        frame_nodes_list.append([
            node_id,
            float(coords[0]),  # x
            float(coords[1]),  # y
            float(coords[2]),  # z
            mass
        ])
    
    return frame_nodes_list


def convert_building_format(building):
    """
    Convert entire building data to the desired format
    """
    # print(200*"x")
    # print(building)
    # print(200*"-")
    converted = {
        'nodes': convert_frame_nodes(building.get('all_nodes', {})),
        'frame_elements': [],
        'shell_elements': []
    }
    
    # Convert beams to frame elements
    for elem_id, (i_node, j_node) in building.get('beams', {}).items():
        converted['frame_elements'].append([
            "forceBeamColumn",
            int(elem_id),
            int(i_node),
            int(j_node),
            1,  # transformation tag
            1   # integration tag
        ])
    
    # Convert shells
    for floor in building.get('shells', []):
        for elem_id, elem_data in floor.get('elements', {}).items():
            converted['shell_elements'].append([
                "ShellMITC4",
                int(elem_id),
                *[int(n) for n in elem_data.get('nodes', [])],
                20  # section tag
            ])
    
    # Preserve other data
    converted.update({
        'frame_nodes': building.get('frame_nodes', {}),
        'beams': building.get('beams', {}),
        'shells': building.get('shells', []),
        'image_filepaths': building.get('image_filepaths', [])
    })
    
    return converted




def calculator1(request, project_pk, task_pk):
    context = get_common_context(request, project_pk, task_pk)
    
    if request.method == 'POST':
        print("calculator1 has been called")
        try:
            # Strict JSON parsing with no defaults
            model_data = json.loads(request.POST['model_data'])
            
            # Mandatory parameters - will raise KeyError if missing
            num_bays_x = int(model_data['numBayX'])
            num_bays_y = int(model_data['numBayY'])
            num_floors = int(model_data['numFloor'])
            num_x_div = int(model_data['num_x_div'])
            num_y_div = int(model_data['num_y_div'])
            
            # Mandatory list parameters
            bay_width_x = model_data['bayWidthX']
            bay_width_y = model_data['bayWidthY']
            story_heights = model_data['storyHeights']
            
            # Operation parameters (empty if not provided)
            add_nodes = model_data.get('add_nodes', {})
            delete_nodes = model_data.get('delete_nodes', [])
            add_beams = model_data.get('add_beams', {})
            delete_beams = model_data.get('delete_beams', [])
            add_shells = model_data.get('add_shells', {})
            delete_shells = model_data.get('delete_shells', [])
            print("calculator1 is running1")
            
            # Validate all parameters exist and have correct types
            if not all(isinstance(x, (list, tuple)) for x in [bay_width_x, bay_width_y, story_heights]):
                raise ValueError("bayWidthX, bayWidthY, and storyHeights must be arrays")
                
            if not all(isinstance(x, (int, float)) for lst in [bay_width_x, bay_width_y, story_heights] for x in lst):
                raise ValueError("All bay widths and story heights must be numbers")
            
            if not all(isinstance(x, int) for x in [num_bays_x, num_bays_y, num_floors, num_x_div, num_y_div]):
                raise ValueError("All numeric parameters must be integers")
            
            # Validate array lengths match the counts
            if len(bay_width_x) != num_bays_x:
                raise ValueError(f"Expected {num_bays_x} bay widths in X direction, got {len(bay_width_x)}")
            if len(bay_width_y) != num_bays_y:
                raise ValueError(f"Expected {num_bays_y} bay widths in Y direction, got {len(bay_width_y)}")
            if len(story_heights) != num_floors:
                raise ValueError(f"Expected {num_floors} story heights, got {len(story_heights)}")
            
            # Create output directory for mesh plots
            user_dir = os.path.join(settings.MEDIA_ROOT, str(request.user.username))
            os.makedirs(user_dir, exist_ok=True)
            filename_base = f"model"
            
            # Create subdirectory for mesh plots
            mesh_plots_dir = os.path.join(user_dir, "mesh_plots")
            os.makedirs(mesh_plots_dir, exist_ok=True)
            print("calculator1 is running generate_building_model")
            
            # Generate model with mesh plotting - all parameters are required
            building = generate_building_model(
                num_bays_x=num_bays_x,
                num_bays_y=num_bays_y,
                num_floors=num_floors,
                bay_width_x=bay_width_x,
                bay_width_y=bay_width_y,
                story_heights=story_heights,
                num_x_div=num_x_div,
                num_y_div=num_y_div,
                add_nodes=add_nodes,
                add_beams=add_beams,
                add_shells=add_shells,
                remove_nodes=delete_nodes,
                remove_beams=delete_beams,
                remove_shells=delete_shells,
                save_plots=True,
                output_dir=mesh_plots_dir,
                show_plots=False,
                dpi=300,
                figsize=(12, 8)
            )
            
            print("calculator1 generate_building_model successfully")
            
            # Generate Z points for filtering (floor levels)
            z_points = [0]  # Ground level
            for i in range(num_floors):
                z_points.append(sum(story_heights[:i+1]))
            
            # Convert nodes to the expected format for filtering functions
            nodes_for_filtering = {}
            all_nodes_dict = building.get('all_nodes', building.get('frame_nodes', {}))
            for node_id, coords in all_nodes_dict.items():
                if hasattr(coords, 'tolist'):
                    coords = coords.tolist()
                elif not isinstance(coords, list):
                    coords = list(coords)
                nodes_for_filtering[str(node_id)] = (coords[0], coords[1], coords[2])
            
            # Convert beams to the expected format for filtering functions
            # Use only beam IDs without name prefixes
            beams_for_filtering = {}
            for beam_id, nodes in building['beams'].items():
                beams_for_filtering[str(beam_id)] = (str(nodes[0]), str(nodes[1]))
            
            # Filter nodes and elements by Z level using coordinate-based filtering
            filtered_nodes_by_z = filter_nodes_by_z(nodes_for_filtering, z_points)
            filtered_columns_by_z = filter_column_beams_by_coordinates(beams_for_filtering, nodes_for_filtering, z_points)
            filtered_beams_x_by_z = filter_beam_x_by_coordinates(beams_for_filtering, nodes_for_filtering, z_points)
            filtered_beams_y_by_z = filter_beam_y_by_coordinates(beams_for_filtering, nodes_for_filtering, z_points)
            
            # Create a directory for filtered element data and plots
            filtered_dir = os.path.join(user_dir, "filtered_elements")
            os.makedirs(filtered_dir, exist_ok=True)
            
            # Save filtered data to JSON files
            save_filtered_data(filtered_nodes_by_z, filtered_dir, "filtered_nodes.json")
            save_filtered_data(filtered_columns_by_z, filtered_dir, "filtered_columns.json")
            save_filtered_data(filtered_beams_x_by_z, filtered_dir, "filtered_beams_x.json")
            save_filtered_data(filtered_beams_y_by_z, filtered_dir, "filtered_beams_y.json")
            
            # Generate plots for each level
            filtered_plots_dir = os.path.join(filtered_dir, "plots")
            os.makedirs(filtered_plots_dir, exist_ok=True)
            
            # Plot elements for each level with all member types separated
            filtered_plot_filepaths = plot_filtered_elements(
                filtered_nodes_by_z,
                filtered_columns_by_z,
                filtered_beams_x_by_z,
                filtered_beams_y_by_z,
                filtered_plots_dir,
                prefix="structural_level"
            )
            
            # Convert plot filepaths to URLs for HTML rendering
            filtered_plot_urls = []
            for filepath in filtered_plot_filepaths:
                relative_path = os.path.relpath(filepath, settings.MEDIA_ROOT)
                url = os.path.join(settings.MEDIA_URL, relative_path)
                filtered_plot_urls.append({
                    'url': url,
                    'z_level': os.path.basename(filepath).split('_')[-1].replace('.png', ''),
                    'filename': os.path.basename(filepath)
                })
            
            # [Rest of your existing code continues...]
            
            # Update context with filtered plot URLs
            context.update({
                # [Your existing context updates...]
                'filtered_plot_urls': filtered_plot_urls,
                'filtered_data_dir': os.path.join(settings.MEDIA_URL, os.path.relpath(filtered_dir, settings.MEDIA_ROOT))
            })

            # Filter shell elements by Z level
            filtered_shells_by_z = {}
            for floor_data in building['shells']:
                floor_level = floor_data['floor_level']
                z_level = z_points[floor_level] if floor_level < len(z_points) else z_points[-1]
                
                # Find the closest Z level
                closest_z = min(z_points, key=lambda z: abs(z - z_level))
                
                if closest_z not in filtered_shells_by_z:
                    filtered_shells_by_z[closest_z] = {}
                
                filtered_shells_by_z[closest_z][f'floor_{floor_level}'] = floor_data
            
            # Generate overall building visualization
            building, main_filepath = create_and_visualize_model(
                building=building,
                num_bays_x=num_bays_x,
                num_bays_y=num_bays_y,
                num_floors=num_floors,
                bay_width_x=bay_width_x,
                bay_width_y=bay_width_y,
                story_heights=story_heights,
                num_x_div=num_x_div,
                num_y_div=num_y_div,
                output_dir=user_dir,
                filename=filename_base
            )

            # Process main visualization image
            main_relative_path = os.path.relpath(main_filepath, settings.MEDIA_ROOT)
            main_image_url = os.path.join(settings.MEDIA_URL, main_relative_path)
            
            # Process mesh plot images
            mesh_image_urls = []
            mesh_image_info = []
            
            if 'image_filepaths' in building and building['image_filepaths']:
                for i, mesh_filepath in enumerate(building['image_filepaths']):
                    if mesh_filepath and os.path.exists(mesh_filepath):
                        # Convert absolute path to relative path for URL generation
                        mesh_relative_path = os.path.relpath(mesh_filepath, settings.MEDIA_ROOT)
                        mesh_image_url = os.path.join(settings.MEDIA_URL, mesh_relative_path)
                        mesh_image_urls.append(mesh_image_url)
                        
                        # Extract floor number from filename or use index
                        floor_number = i + 1  # Floors start from 1
                        mesh_image_info.append({
                            'url': mesh_image_url,
                            'floor': floor_number,
                            'filename': os.path.basename(mesh_filepath)
                        })
            
            # Convert building to the desired format with shell nodes included
            building_formatted = convert_building_format(building)
            
            # Prepare response with formatted data and filtered data by Z levels
            context.update({
                'success': True,
                'image_url': main_image_url,  # Main building visualization
                'mesh_image_urls': mesh_image_urls,  # List of mesh plot URLs
                'mesh_image_info': mesh_image_info,  # Detailed info for each mesh plot
                'building': building_formatted,
                'frame_nodes': building_formatted["nodes"],  # Properly formatted nodes including shell nodes
                'beams': building_formatted["frame_elements"],  # Properly formatted beam elements
                'shells': building_formatted["shell_elements"],  # Properly formatted shell elements
                'num_floors': num_floors,  # Pass number of floors for template rendering
                # Filtered data by Z levels
                'z_points': z_points,
                'filtered_nodes_by_z': filtered_nodes_by_z,
                'filtered_columns_by_z': filtered_columns_by_z,
                'filtered_beams_x_by_z': filtered_beams_x_by_z,
                'filtered_beams_y_by_z': filtered_beams_y_by_z,
                'filtered_shells_by_z': filtered_shells_by_z
            })

        except KeyError as e:
            context['error'] = f"Missing required parameter: {str(e)}"
        except json.JSONDecodeError as e:
            context['error'] = "Invalid JSON data format"
        except ValueError as e:
            context['error'] = f"Invalid parameter value: {str(e)}"
        except Exception as e:
            context['error'] = "An unexpected error occurred"
            logger.error(f"Error in calculator1: {str(e)}", exc_info=True)

    return render(request, 'opensees/calculator1.html', context)


def filter_nodes_by_z(nodes, z_points):
    """Filter and group nodes by Z level"""
    grouped_nodes = defaultdict(dict)
    
    for node_id, (x, y, z) in nodes.items():
        for z_level in z_points:
            if abs(z - z_level) < 1e-6:
                grouped_nodes[z_level][node_id] = (x, y, z)
                break
    
    return dict(grouped_nodes)


def filter_column_beams_by_coordinates(beams, nodes, z_points):
    """Filter and group column beams by starting Z level based on coordinates"""
    grouped_beams = defaultdict(dict)
    
    for beam_id, (n1, n2) in beams.items():
        n1_coords = nodes[n1]
        n2_coords = nodes[n2]
        
        # Check if this is a vertical beam (column) - different Z coordinates
        if abs(n1_coords[2] - n2_coords[2]) > 1e-6:
            # Determine starting Z level (lower Z coordinate)
            start_z = min(n1_coords[2], n2_coords[2])
            
            # Find which z_level this column starts at
            for z_level in z_points[:-1]:  # No columns start at the top level
                if abs(start_z - z_level) < 1e-6:
                    grouped_beams[z_level][beam_id] = (n1, n2)
                    break
    
    return dict(grouped_beams)


def filter_beam_x_by_coordinates(beams, nodes, z_points):
    """Filter and group X-direction beams by Z level based on coordinates"""
    grouped_beams = defaultdict(dict)
    
    for beam_id, (n1, n2) in beams.items():
        n1_coords = nodes[n1]
        n2_coords = nodes[n2]
        
        # Check if this is a horizontal beam in X direction
        # Same Z coordinate and more X difference than Y difference
        if (abs(n1_coords[2] - n2_coords[2]) < 1e-6 and 
            abs(n1_coords[0] - n2_coords[0]) > abs(n1_coords[1] - n2_coords[1])):
            
            beam_z = n1_coords[2]  # Both nodes have same Z
            
            # Find which z_level this beam belongs to
            for z_level in z_points:
                if abs(beam_z - z_level) < 1e-6:
                    grouped_beams[z_level][beam_id] = (n1, n2)
                    break
    
    return dict(grouped_beams)


def filter_beam_y_by_coordinates(beams, nodes, z_points):
    """Filter and group Y-direction beams by Z level based on coordinates"""
    grouped_beams = defaultdict(dict)
    
    for beam_id, (n1, n2) in beams.items():
        n1_coords = nodes[n1]
        n2_coords = nodes[n2]
        
        # Check if this is a horizontal beam in Y direction
        # Same Z coordinate and more Y difference than X difference
        if (abs(n1_coords[2] - n2_coords[2]) < 1e-6 and 
            abs(n1_coords[1] - n2_coords[1]) > abs(n1_coords[0] - n2_coords[0])):
            
            beam_z = n1_coords[2]  # Both nodes have same Z
            
            # Find which z_level this beam belongs to
            for z_level in z_points:
                if abs(beam_z - z_level) < 1e-6:
                    grouped_beams[z_level][beam_id] = (n1, n2)
                    break
    
    return dict(grouped_beams)


def save_filtered_data(filtered_data, output_folder, filename):
    """Save filtered data to JSON file in the specified output folder"""
    full_path = os.path.join(output_folder, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)


def plot_filtered_elements(nodes_by_z, columns_by_z, beams_x_by_z, beams_y_by_z, output_dir, prefix="level"):
    """Plot filtered elements for each Z level with labels and return filepaths"""
    filepaths = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all z_levels in order
    z_levels = sorted(nodes_by_z.keys())
    
    for i, z_level in enumerate(z_levels):
        nodes = nodes_by_z[z_level]
        
        # Create figure with larger size for better readability
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes with labels
        x_coords = [coord[0] for coord in nodes.values()]
        y_coords = [coord[1] for coord in nodes.values()]
        z_coords = [coord[2] for coord in nodes.values()]
        ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o', s=50, label='Nodes')
        
        # Add node labels
        for node_id, (x, y, z) in nodes.items():
            ax.text(x, y, z, f'{node_id}', color='red', fontsize=8)
        
        # Plot columns (vertical members) - they span between current and next level
        if z_level in columns_by_z and i < len(z_levels) - 1:
            next_z = z_levels[i + 1]
            next_nodes = nodes_by_z.get(next_z, {})
            
            for beam_id, (n1, n2) in columns_by_z[z_level].items():
                if n1 in nodes and n2 in next_nodes:
                    x = [nodes[n1][0], nodes[n1][0]]  # Same X position (vertical)
                    y = [nodes[n1][1], nodes[n1][1]]  # Same Y position (vertical)
                    z = [nodes[n1][2], next_nodes[n2][2]]
                    
                    # Plot column
                    ax.plot(x, y, z, 'g-', linewidth=3, 
                           label='Columns' if beam_id == list(columns_by_z[z_level].keys())[0] else "")
                    
                    # Add column label at midpoint
                    mid_z = (z[0] + z[1]) / 2
                    ax.text(x[0], y[0], mid_z, f'{beam_id}', 
                           color='green', fontsize=8, ha='center', va='center')
        
        # Plot X beams (horizontal members)
        if z_level in beams_x_by_z:
            for beam_id, (n1, n2) in beams_x_by_z[z_level].items():
                if n1 in nodes and n2 in nodes:
                    x = [nodes[n1][0], nodes[n2][0]]
                    y = [nodes[n1][1], nodes[n2][1]]
                    z = [nodes[n1][2], nodes[n2][2]]
                    
                    # Plot beam
                    ax.plot(x, y, z, 'b-', linewidth=2, 
                           label='X Beams' if beam_id == list(beams_x_by_z[z_level].keys())[0] else "")
                    
                    # Add beam label at midpoint
                    mid_x = (x[0] + x[1]) / 2
                    mid_y = (y[0] + y[1]) / 2
                    mid_z = (z[0] + z[1]) / 2
                    ax.text(mid_x, mid_y, mid_z, f'{beam_id}', 
                           color='blue', fontsize=8, ha='center', va='center')
        
        # Plot Y beams (horizontal members)
        if z_level in beams_y_by_z:
            for beam_id, (n1, n2) in beams_y_by_z[z_level].items():
                if n1 in nodes and n2 in nodes:
                    x = [nodes[n1][0], nodes[n2][0]]
                    y = [nodes[n1][1], nodes[n2][1]]
                    z = [nodes[n1][2], nodes[n2][2]]
                    
                    # Plot beam
                    ax.plot(x, y, z, 'm-', linewidth=2, 
                           label='Y Beams' if beam_id == list(beams_y_by_z[z_level].keys())[0] else "")
                    
                    # Add beam label at midpoint
                    mid_x = (x[0] + x[1]) / 2
                    mid_y = (y[0] + y[1]) / 2
                    mid_z = (z[0] + z[1]) / 2
                    ax.text(mid_x, mid_y, mid_z, f'{beam_id}', 
                           color='magenta', fontsize=8, ha='center', va='center')
        
        ax.set_title(f'Structural Elements at Z={z_level:.2f}', pad=20)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        
        # Create custom legend without duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Adjust view for better visibility
        ax.view_init(elev=30, azim=45)
        
        # Save plot with high quality
        filename = f"{prefix}_{z_level:.2f}.png"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        filepaths.append(filepath)
    
    return filepaths



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

@login_required
def calculator6(request, project_id, task_id):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    task = get_object_or_404(Task, id=task_id, project=project)
    
    result = None
    
    if request.method == 'POST':
        try:
            # Get the mass parameters from textarea
            mass_script = request.POST.get('mass_parameters', '').strip()
            print(f"Received mass script content (length: {len(mass_script)} chars)")
            
            # Create temp directory with user-specific subfolder
            import tempfile
            import importlib.util
            import os
            from django.conf import settings
            
            user_dir = os.path.join(settings.MEDIA_ROOT, 'temp_mass', str(request.user.pk))
            print(f"Creating user directory at: {user_dir}")
            os.makedirs(user_dir, exist_ok=True)
            
            # Create unique filename
            script_filename = f"mass_params.py"
            script_path = os.path.normpath(os.path.join(user_dir, script_filename))
            
            # Write with proper line endings and encoding
            print(f"Writing script to file (size: {len(mass_script)} bytes)")
            with open(script_path, 'w') as f:
                f.write(mass_script)
            
            print(f"Attempting to import module from {script_path}")
            spec = importlib.util.spec_from_file_location("mass_params", script_path)
            params_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(params_module)
            print("Module imported successfully")
            
            # Helper function to safely get attributes
            def get_param_attr(module, name, default=None):
                attr = getattr(module, name, default) if hasattr(module, name) else default
                print(f"Retrieved parameter '{name}': {attr}")
                return attr
            
            # Extract parameters from the module
            nodes = get_param_attr(params_module, 'nodes', [])
            frame_elements = get_param_attr(params_module, 'frame_elements', [])
            load_cases = get_param_attr(params_module, 'load_cases', [])
            load_combinations = get_param_attr(params_module, 'load_combinations', [])
            
            shell_elements = getattr(params_module, 'shell_elements', [])
            
            # Process loads and generate report
            results = process_loads(nodes, shell_elements, load_cases)
            report_data = generate_report(results)
            print("ssdfffsf:", report_data)
            
            # Perform center of mass calculations
            result = {
                'nodes': nodes,
                'frame_elements': frame_elements,
                'load_cases': load_cases,
                'load_combinations': load_combinations,
                'floor_com': calculate_center_of_mass(nodes, frame_elements, load_cases, load_combinations, load_cases),
                'report_data': report_data  # report_data is a list with modified load cases

            }
            
            # report_data = {"report_data": report_data}
        except Exception as e:
            print(f"Error in mass analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            result = {'error': str(e)}
    
    context = {
        'project': project,
        'task': task,
        'result': result,
        # 'report_data': report_data,
    }
    
    return render(request, 'opensees/calculator6.html', context)



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





@login_required
def punching_shear_analysis(request, project_pk, task_pk):
    context = get_common_context(request, project_pk, task_pk)
    
    # Get project and task for correct folder structure
    project = get_object_or_404(Project, pk=project_pk)
    task = get_object_or_404(Task, pk=task_pk)
    
    user_folder = os.path.join(settings.MEDIA_ROOT, project.title, task.title, f"user_{request.user.username}")
    punching_shear_dir = os.path.join(user_folder, "punching_shear")
    
    if request.method == 'POST':
        try:
            # Parse input data from textarea
            input_data = request.POST.get('input_data', '').strip()
            
            # Parse the input data (expecting JSON format)
            import json
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                # Try to parse as key-value pairs if not JSON
                data = {}
                lines = input_data.split('\n')
                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert to appropriate data type
                        try:
                            if value.lower() in ['true', 'false']:
                                data[key] = value.lower() == 'true'
                            elif value.lower() == 'auto':
                                data[key] = 'auto'
                            elif ',' in value and '[' in value:
                                # Handle arrays/lists
                                data[key] = json.loads(value)
                            else:
                                data[key] = float(value)
                        except (ValueError, json.JSONDecodeError):
                            data[key] = value
            
            # Extract form data with defaults
            col_width = data.get('col_width', 20)
            col_depth = data.get('col_depth', 20)
            slab_avg_depth = data.get('slab_avg_depth', 8)
            condition = data.get('condition', 'I')
            studrail_length = data.get('studrail_length', 12)
            
            # Forces
            Vz = data.get('Vz', -200)
            Mx = data.get('Mx', 500)
            My = data.get('My', 300)
            
            # Material properties
            f_prime_c = data.get('f_prime_c', 4000)
            fy = data.get('fy', 60000)
            reinforcement_type = data.get('reinforcement_type', 'both')
            
            # Analysis options
            consider_ecc = data.get('consider_ecc', True)
            gamma_vx = data.get('gamma_vx', 'auto')
            gamma_vy = data.get('gamma_vy', 'auto')
            
            # Parse openings
            openings = data.get('openings', [])
            
            # Create output folder
            output_folder = os.path.join(punching_shear_dir, "analysis_results")
            os.makedirs(output_folder, exist_ok=True)
            
            # Run analysis
            results = run_punching_shear_analysis(
                col_width=col_width,
                col_depth=col_depth,
                slab_avg_depth=slab_avg_depth,
                condition=condition,
                studrail_length=studrail_length,
                Vz=Vz, Mx=Mx, My=My,
                openings=openings,
                gamma_vx=gamma_vx,
                gamma_vy=gamma_vy,
                consider_ecc=consider_ecc,
                f_prime_c=f_prime_c,
                fy=fy,
                reinforcement_type=reinforcement_type,
                output_folder=output_folder
            )
            
            # Read analysis results
            results_json_path = os.path.join(output_folder, "evaluation.json")
            if os.path.exists(results_json_path):
                with open(results_json_path, 'r') as f:
                    evaluation_results = json.load(f)
            else:
                evaluation_results = None
            
            # Read CSV results for display
            results_csv_path = os.path.join(output_folder, "results.csv")
            if os.path.exists(results_csv_path):
                import pandas as pd
                results_df = pd.read_csv(results_csv_path)
                results_html = results_df.to_html(classes='table table-striped table-responsive', table_id='results-table')
            else:
                results_html = None
            
            # Read report
            report_path = os.path.join(output_folder, "punching_shear_report.txt")
            report_content = None
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
            
            # File paths for download and display
            files_available = {}
            image_files = {}
            potential_files = [
                ('2d_plot', 'punching_shear_2d.png'),
                ('3d_plot', 'punching_shear_3d.html'),
                ('preview', 'perimeter_preview.png'),
                ('results_csv', 'results.csv'),
                ('results_json', 'results.json'),
                ('evaluation', 'evaluation.json'),
                ('report', 'punching_shear_report.txt')
            ]
            
            for file_key, filename in potential_files:
                file_path = os.path.join(output_folder, filename)
                if os.path.exists(file_path):
                    files_available[file_key] = filename
                    # Create media URL for images
                    if filename.endswith('.png'):
                        relative_path = os.path.relpath(file_path, settings.MEDIA_ROOT)
                        image_files[file_key] = os.path.join(settings.MEDIA_URL, relative_path).replace('\\', '/')
                    elif filename.endswith('.html'):
                        # For HTML files, read content to embed
                        with open(file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        image_files[file_key] = html_content
            
            context.update({
                'analysis_completed': True,
                'evaluation_results': evaluation_results,
                'results_html': results_html,
                'report_content': report_content,
                'files_available': files_available,
                'image_files': image_files,
                'success_message': 'Punching shear analysis completed successfully!'
            })
            
        except Exception as e:
            context['error_message'] = f"Analysis failed: {str(e)}"
            print(f"Punching shear analysis error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return render(request, 'opensees/punching_shear_analysis.html', context)


@login_required
def download_punching_shear_file(request, project_pk, task_pk, filename):
    # Get project and task for correct folder structure
    project = get_object_or_404(Project, pk=project_pk)
    task = get_object_or_404(Task, pk=task_pk)
    
    user_folder = os.path.join(settings.MEDIA_ROOT, project.title, task.title, f"user_{request.user.username}")
    punching_shear_dir = os.path.join(user_folder, "punching_shear", "analysis_results")
    
    file_path = os.path.join(punching_shear_dir, filename)
    
    if not os.path.exists(file_path):
        raise Http404("File not found")
    
    # Determine content type based on file extension
    content_type = 'application/octet-stream'
    if filename.endswith('.png'):
        content_type = 'image/png'
    elif filename.endswith('.html'):
        content_type = 'text/html'
    elif filename.endswith('.csv'):
        content_type = 'text/csv'
    elif filename.endswith('.json'):
        content_type = 'application/json'
    elif filename.endswith('.txt'):
        content_type = 'text/plain'
    
    with open(file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type=content_type)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response


from django.shortcuts import render
import subprocess
import tempfile
import os

from django.shortcuts import render
import subprocess
import tempfile
import os

def code_editor(request, project_id, task_id):  # Changed parameter names
    # Get common context and project/task objects
    context = get_common_context(request, project_id, task_id)  # Update parameter names
    project = get_object_or_404(Project, pk=project_id)
    task = get_object_or_404(Task, pk=task_id)

    
    # Initialize code execution variables
    output = ""
    code = ""
    
    if request.method == 'POST':
        code = request.POST.get('code', '')
        
        # Create temporary file for execution
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp.write(code.encode('utf-8'))
            temp_path = temp.name
        
        try:
            # Execute the Python code
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            output = result.stdout
            if result.stderr:
                output += "\nErrors:\n" + result.stderr
        except subprocess.TimeoutExpired:
            output = "Error: Code execution timed out (max 10 seconds)"
        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            os.unlink(temp_path)
    
    context.update({
        'project': project,
        'task': task,
        'output': output,
        'code': code or '# Your Python code here\nprint("Hello from Project {} Task {}")'.format(project_id, task_id),
    })
    
    return render(request, 'opensees/calculator11.html', context)





