
read the following code. the gravity analysis is done well. now i want to perform response spectrum analysis. the python functions are correctly done. now the problem is to render the results and filepaths. so modify or create the html code.

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
    
# input.html:
{% extends "opensees/base.html" %}

{% block content %}
<div class="container">
    <h1 class="text-center mb-4">Structural Analysis Input</h1>
    
	<!-- Calculator Navigation -->
    {% include "opensees/calculator_nav.html" %}

    {% if error %}
        <div class="alert alert-danger">{{ error }}</div>
    {% endif %}
    
    <form method="post" action="{% url 'opensees:run_analysis' project.pk task.pk %}">
        {% csrf_token %}
        <div class="form-group">
            <label for="input_data">OpenSees Script</label>
            <textarea class="form-control" id="input_data" name="input_data" rows="30" 
                    style="font-family: monospace; white-space: pre;">

{% include "opensees/data.html" %}

</textarea>
        </div>
        <button type="submit" class="btn btn-primary">Run Analysis</button>
    </form>

    <!-- Convert to HTML Form -->
    <form method="post" action="{% url 'opensees:convert_to_html' project.pk task.pk %}" class="mt-4">
        {% csrf_token %}
        <button type="submit" class="btn btn-success">Convert to HTML</button>
    </form>
    





{% if result.success %}
    <!-- Analysis Results Section -->
    <div class="mt-4 p-3 bg-light rounded">
        <h4>Analysis Summary:</h4>
        <div class="row">
            <div class="col-md-6">
                <p><strong>Output Directory:</strong> {{ result.output_directory }}</p>
                <p><strong>Generated Files:</strong> {{ result.output_files|length }}</p>
                <p><strong>Generated Images:</strong> {{ result.images|length }}</p>
            </div>
            <div class="col-md-6">
                <h6>Generated Files:</h6>
                <ul class="list-unstyled small">
                    {% for file in result.output_files %}
                    <li>• {{ file }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
    </div>

    <!-- Gravity Analysis Results -->
    {% if result.analysis_results.gravity %}
    <div class="mt-4 p-3 bg-light rounded">
        <h4>Gravity Analysis Results:</h4>
        {% if result.analysis_results.gravity.error %}
            <div class="alert alert-danger">
                <strong>Error:</strong> {{ result.analysis_results.gravity.error }}
            </div>
        {% else %}
            <p class="text-success">✓ Gravity analysis completed successfully</p>
            <small class="text-muted">Results saved to JSON files in output directory</small>
            {{ result.analysis_results.gravity }}
        {% endif %}
    </div>
    {% endif %}

    <!-- Response Spectrum Analysis Results -->
    {% if result.analysis_results.rsa %}
    <div class="mt-4 p-3 bg-light rounded">
        <h4>Response Spectrum Analysis Results:</h4>
        {% if result.analysis_results.rsa.error %}
            <div class="alert alert-danger">
                <strong>Error:</strong> {{ result.analysis_results.rsa.error }}
            </div>
        {% else %}
            <p class="text-success">✓ Response spectrum analysis completed successfully</p>
            {% if result.analysis_results.rsa.modal_properties %}
            <div class="row">
                <div class="col-md-6">
                    <h6>Modal Properties:</h6>
                    <ul class="small">
                        {% for period in result.analysis_results.rsa.modal_properties.periods %}
                        <li>Mode {{ forloop.counter }}: T = {{ period|floatformat:4 }} s</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
            {% endif %}
        {% endif %}
    </div>
    {% endif %}

    <!-- Generated Images/Plots -->
    {% if result.images %}
    <div class="mt-4 p-3 bg-light rounded">
        <h4>Generated Plots and Visualizations:</h4>
        <div class="row">
            {% for name, url in result.images.items %}
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header">
                        <strong>{{ name}}</strong>
                    </div>
                    <div class="card-body">
                        <img src="{{ url }}" alt="{{ name }}" class="img-fluid" style="max-width: 100%; height: auto;">
                        <p class="mt-2 text-muted small">File: {{ name }}</p>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

{% elif result.success == False and result.error %}
    <!-- Error Display -->
    <div class="mt-4 p-3 bg-danger text-white rounded">
        <h4>Analysis Error:</h4>
        <p><strong>Error:</strong> {{ result.error }}</p>
        {% if result.traceback %}
        <details>
            <summary>Show detailed traceback</summary>
            <pre class="mt-2 text-white">{{ result.traceback }}</pre>
        </details>
        {% endif %}
    </div>

{% endif %}










<div class="mt-4">
        <h4>Example Structure</h4>
        <pre>model("basic","-ndm",3,"-ndf",6)
            {% include "opensees/data.html" %}

        </pre>
    </div>
</div>
{% endblock %}

convert_to_html.html

{% extends "opensees/base.html" %}

{% block content %}

<div class="container">
    <h1 class="text-center mb-4">Analysis Results - HTML Format</h1>
    
    {% if error %}
        <div class="alert alert-danger">{{ error }}</div>
    {% endif %}
    
    {% if success %}
        <!-- Excel Download Buttons for All Combinations -->
        <div class="mb-4">
            <h4>Download All Excel Files:</h4>
            <div class="row">
                {% for name, filename in all_excel_files.items %}
                    <div class="col-md-4 mb-2">
                        <a href="{% url 'opensees:download_excel' project.pk task.pk filename %}" 
                           class="btn btn-primary btn-sm w-100" download>
                            {{ name|title }}
                        </a>
                    </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Navigation tabs for load combinations -->
        <ul class="nav nav-tabs mb-4" id="comboTabs" role="tablist">
            {% for combo_name in combo_folders %}
                <li class="nav-item" role="presentation">
                    <button class="nav-link {% if forloop.first %}active{% endif %}" 
                            id="{{ combo_name }}-tab" 
                            data-bs-toggle="tab" 
                            data-bs-target="#{{ combo_name }}-content" 
                            type="button" 
                            role="tab" 
                            aria-controls="{{ combo_name }}-content" 
                            aria-selected="{% if forloop.first %}true{% else %}false{% endif %}">
                        {{ combo_name|title }}
                    </button>
                </li>
            {% endfor %}
        </ul>
        
        <!-- Tab content for each load combination -->
        <div class="tab-content" id="comboTabContent">
            {% for combo_name, combo_data in combo_results.items %}
                <div class="tab-pane fade {% if forloop.first %}show active{% endif %}" 
                     id="{{ combo_name }}-content" 
                     role="tabpanel" 
                     aria-labelledby="{{ combo_name }}-tab">
                    
                    <h3 class="mb-4">{{ combo_name|title }} Results</h3>
                    
                    <!-- Excel downloads for this combination -->
                    <div class="mb-3">
                        <h5>Download Excel Files for {{ combo_name|title }}:</h5>
                        {% for name, filename in combo_data.excel_files.items %}
                            <a href="{% url 'opensees:download_excel' project.pk task.pk filename %}" 
                               class="btn btn-outline-primary btn-sm me-2 mb-2" download>
                                {{ name|title }} Excel
                            </a>
                        {% endfor %}
                    </div>
                    
                    <!-- Accordion for different result types -->
                    <div class="accordion" id="accordion{{ combo_name }}">
                        
                        <!-- Nodal Results -->
                        {% if 'reactions' in combo_data.html_tables or 'displacements' in combo_data.html_tables %}
                        <div class="accordion-item">
                            <h2 class="accordion-header" id="nodal{{ combo_name }}">
                                <button class="accordion-button collapsed" type="button" 
                                        data-bs-toggle="collapse" 
                                        data-bs-target="#nodalResults{{ combo_name }}" 
                                        aria-expanded="false" 
                                        aria-controls="nodalResults{{ combo_name }}">
                                    Nodal Results
                                </button>
                            </h2>
                            <div id="nodalResults{{ combo_name }}" 
                                 class="accordion-collapse collapse" 
                                 aria-labelledby="nodal{{ combo_name }}" 
                                 data-bs-parent="#accordion{{ combo_name }}">
                                <div class="accordion-body">
                                    {% if 'reactions' in combo_data.html_tables %}
                                        <h6>Reactions</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.reactions|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'displacements' in combo_data.html_tables %}
                                        <h6>Displacements</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.displacements|safe }}
                                        </div>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                        {% endif %}
                        
                        <!-- Beam Results -->
                        {% if 'beam_forces' in combo_data.html_tables or 'beam_stresses' in combo_data.html_tables or 'beam_strains' in combo_data.html_tables or 'beam_deflections' in combo_data.html_tables %}
                        <div class="accordion-item">
                            <h2 class="accordion-header" id="beam{{ combo_name }}">
                                <button class="accordion-button collapsed" type="button" 
                                        data-bs-toggle="collapse" 
                                        data-bs-target="#beamResults{{ combo_name }}" 
                                        aria-expanded="false" 
                                        aria-controls="beamResults{{ combo_name }}">
                                    Beam Results
                                </button>
                            </h2>
                            <div id="beamResults{{ combo_name }}" 
                                 class="accordion-collapse collapse" 
                                 aria-labelledby="beam{{ combo_name }}" 
                                 data-bs-parent="#accordion{{ combo_name }}">
                                <div class="accordion-body">
                                    {% if 'beam_forces' in combo_data.html_tables %}
                                        <h6>Beam Forces</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.beam_forces|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'beam_stresses' in combo_data.html_tables %}
                                        <h6>Beam Stresses</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.beam_stresses|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'beam_strains' in combo_data.html_tables %}
                                        <h6>Beam Strains</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.beam_strains|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'beam_deflections' in combo_data.html_tables %}
                                        <h6>Beam Deflections</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.beam_deflections|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'beam_relative_deflections' in combo_data.html_tables %}
                                        <h6>Beam Relative Deflections</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.beam_relative_deflections|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'beam_slopes' in combo_data.html_tables %}
                                        <h6>Beam Slopes</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.beam_slopes|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'beam_max_min_deflections' in combo_data.html_tables %}
                                        <h6>Beam Max/Min Deflections</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.beam_max_min_deflections|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'beam_max_min_slopes' in combo_data.html_tables %}
                                        <h6>Beam Max/Min Slopes</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.beam_max_min_slopes|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'beam_properties' in combo_data.html_tables %}
                                        <h6>Beam Properties</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.beam_properties|safe }}
                                        </div>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                        {% endif %}
                        
                        <!-- Shell Results -->
                        {% if 'shell_forces' in combo_data.html_tables or 'shell_stresses' in combo_data.html_tables or 'shell_strains' in combo_data.html_tables %}
                        <div class="accordion-item">
                            <h2 class="accordion-header" id="shell{{ combo_name }}">
                                <button class="accordion-button collapsed" type="button" 
                                        data-bs-toggle="collapse" 
                                        data-bs-target="#shellResults{{ combo_name }}" 
                                        aria-expanded="false" 
                                        aria-controls="shellResults{{ combo_name }}">
                                    Shell Results
                                </button>
                            </h2>
                            <div id="shellResults{{ combo_name }}" 
                                 class="accordion-collapse collapse" 
                                 aria-labelledby="shell{{ combo_name }}" 
                                 data-bs-parent="#accordion{{ combo_name }}">
                                <div class="accordion-body">
                                    {% if 'shell_forces' in combo_data.html_tables %}
                                        <h6>Shell Forces</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.shell_forces|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'shell_stresses' in combo_data.html_tables %}
                                        <h6>Shell Stresses</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.shell_stresses|safe }}
                                        </div>
                                    {% endif %}
                                    
                                    {% if 'shell_strains' in combo_data.html_tables %}
                                        <h6>Shell Strains</h6>
                                        <div class="table-responsive mb-4">
                                            {{ combo_data.html_tables.shell_strains|safe }}
                                        </div>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                        {% endif %}
                    </div>
                </div>
            {% endfor %}
        </div>
    {% endif %}
    
    <div class="mt-4">
        <a href="{% url 'opensees:run_analysis' project.pk task.pk %}" class="btn btn-secondary">Back to Analysis</a>
    </div>
</div>


<!-- Bootstrap JS for tabs and accordion functionality -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
{% endblock %}


urls.py
    path('project/<int:project_pk>/task/<int:task_pk>/convert-to-html/', 
         views.convert_to_html, name='convert_to_html'),
    path('project/<int:project_pk>/task/<int:task_pk>/download-excel/<str:filename>/', 
         views.download_excel, name='download_excel'),

    # New RSA-specific patterns
    path('project/<int:project_pk>/task/<int:task_pk>/convert_to_html_rsa/', 
         views.convert_to_html_rsa, name='convert_to_html_rsa'),
    path('project/<int:project_pk>/task/<int:task_pk>/download_excel_rsa/<str:filename>/', 
         views.download_excel_rsa, name='download_excel_rsa'),
