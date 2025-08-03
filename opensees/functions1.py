import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
import openseespy.opensees as ops
import opsvis as opsv
from opsvis.model import get_Ew_data_from_ops_domain_3d
from opsvis.secforces import section_force_distribution_3d
import json
import math
from typing import Dict, List, Tuple

def get_element_shapes(section_properties, elastic_section,aggregator_section, beam_integrations, frame_elements ):
    """Directly extracts element shapes from provided data structures"""

    section_shapes = {}
    for sec in section_properties:
        tag = sec[0]
        sec_type = sec[1].lower()  # Case-insensitive
        
        if sec_type == 'rectangular':
            section_shapes[tag] = ['rect', [sec[6], sec[7]]]  # B, H
            
        elif sec_type == 'circular':
            section_shapes[tag] = ['circ', [sec[6]]]  # D
            
        elif sec_type == 'i' or sec_type == 'wideflange':
            section_shapes[tag] = ['I', [
                sec[6],  # B (flange width)
                sec[7],  # H (total depth)
                sec[8],  # tf (flange thickness)
                sec[9]   # tw (web thickness)
            ]]
            
        elif sec_type == 'l' or sec_type == 'angle':
            section_shapes[tag] = ['L', [
                sec[6],  # H (leg length)
                sec[7],  # B (leg length)
                sec[8]   # t (thickness)
            ]]
            
        elif sec_type == 't':
            section_shapes[tag] = ['T', [
                sec[6],  # B (flange width)
                sec[7],  # H (total depth)
                sec[8],  # tf (flange thickness)
                sec[9]   # tw (stem thickness)
            ]]
            
        elif sec_type == 'c' or sec_type == 'channel':
            section_shapes[tag] = ['C', [
                sec[6],  # B (flange width)
                sec[7],  # H (depth)
                sec[8]   # t (thickness)
            ]]
            
        elif sec_type == 'tube' or sec_type == 'pipe':
            section_shapes[tag] = ['tube', [
                sec[6],  # D (outer diameter)
                sec[7]   # t (wall thickness)
            ]]
            
        elif sec_type == 'box':
            section_shapes[tag] = ['box', [
                sec[6],  # B (outer width)
                sec[7],  # H (outer depth)
                sec[8]   # t (wall thickness)
            ]]
            
        else:
            print(f"Warning: Unsupported section type '{sec[1]}' (tag: {tag})")
            continue  # Skip unsupported sections

    # Create mapping: {integration_tag: section_tag}
    integration_to_section = {integ[1]: integ[2] for integ in beam_integrations}

    # Process elements
    ele_shapes = {}
    for elem in frame_elements:
        ele_tag = elem[1]  # Element ID
        integ_tag = elem[5]  # Integration tag
        
        if integ_tag in integration_to_section:
            sec_tag = integration_to_section[integ_tag]
            if sec_tag in section_shapes:
                ele_shapes[ele_tag] = section_shapes[sec_tag]
    
    return ele_shapes

def extract_beam_results(ele_tag, nep, section_properties):
    """
    Extract forces, stresses, and strains for a beam element using section properties list
    
    Args:
        ele_tag: Element tag
        nep: Number of evaluation points
        section_properties: List of section properties [tag, type, A, Iy, Iz, J, B, H, t]
        
    Returns:
        Dictionary containing beam results
        
    Raises:
        ValueError: If required section properties are missing
    """
    if section_properties is None:
        raise ValueError("Section properties list must be provided")
    # print("its working")
    # Get the section tag assigned to the element
    # section_tag = ops.eleResponse(ele_tag, 'section')[0]
    # print(f"Element {ele_tag} is using section_tag: {section_tag}")
    # Find the section in our properties list
    section = None
    for sec in section_properties:
        # if sec[0] == section_tag:
        section = sec
        break
    
    if section is None:
        raise ValueError(f"Section with tag {section} not found in section properties")
    
    # Extract properties from the list
    section_type = section[1]
    A = section[2]
    Iy = section[3]
    Iz = section[4]
    J = section[5]
    B = section[6]
    H = section[7]
    t = section[8]
    # print(f"Section type: {section_type}, A: {A}, Iy: {Iy}, Iz: {Iz}, J: {J}, B: {B}, H: {H}, t: {t}")
    # Validate dimensions
    if B is None or H is None or B <= 0 or H <= 0:
        raise ValueError(f"Invalid section dimensions for element {ele_tag}: B={B}, H={H}")
    
    # Get element geometry
    node_tags = ops.eleNodes(ele_tag)
    ecrd = np.array([ops.nodeCoord(node_tags[0]), ops.nodeCoord(node_tags[1])])
    L = np.linalg.norm(ecrd[1] - ecrd[0])
    
    # Get element forces
    forces = ops.eleResponse(ele_tag, 'localForces')
    
    # Process distributed loads
    Ew = get_Ew_data_from_ops_domain_3d()
    eload_data = Ew.get(ele_tag, [['-beamUniform', 0., 0., 0.]])
    
    # Get force distribution
    s, xl, _ = section_force_distribution_3d(ecrd, forces, nep, eload_data)
    
    # Organize forces
    force_results = []
    for i in range(len(xl)):
        force_results.append({
            'position': float(xl[i]),
            'N': float(s[i,0]),
            'Vy': float(s[i,1]),
            'Vz': float(s[i,2]),
            'T': float(s[i,3]),
            'My': float(s[i,4]),
            'Mz': float(s[i,5])
        })
    
    # Material properties (steel defaults)
    E = 2.1e11  # Pa (Young's modulus)
    G = 8.1e10  # Pa (Shear modulus)
    
    # Calculate stresses and strains at each point
    stress_results = []
    strain_results = []
    
    for i in range(len(xl)):
        # Current forces
        N = s[i,0]
        Vy = s[i,1]
        Vz = s[i,2]
        T = s[i,3]
        My = s[i,4]
        Mz = s[i,5]
        
        # Calculate stresses
        σ_axial = N/A
        
        # Bending stresses depend on section type
        if section_type == "rectangular":
            σ_bending_y = My * (H/2) / Iy
            σ_bending_z = Mz * (B/2) / Iz
        elif section_type == "circular":
            D = B  # For circular sections, B = D
            σ_bending_y = My * (D/2) / Iy
            σ_bending_z = Mz * (D/2) / Iz
        elif section_type == "L":
            # For L-section, use extreme fiber distances
            σ_bending_y = My * (H/2) / Iy
            σ_bending_z = Mz * (B/2) / Iz
        
        # Torsional stress
        if section_type == "rectangular":
            τ_torsion = T * (H/2) / J
        elif section_type == "circular":
            τ_torsion = T * (D/2) / J
        elif section_type == "L":
            τ_torsion = T * t / J  # Approximate for thin-walled
        
        # Shear stresses
        if section_type == "rectangular":
            Qy = A*H/8  # First moment of area
            Qz = A*B/8
            τ_shear_y = Vy*Qz/(Iz*B)
            τ_shear_z = Vz*Qy/(Iy*H)
        elif section_type == "circular":
            τ_shear_y = 4/3 * Vy/A  # Average shear stress for circular
            τ_shear_z = 4/3 * Vz/A
        elif section_type == "L":
            # Simplified shear stress for L-sections
            τ_shear_y = Vy/A
            τ_shear_z = Vz/A
        
        # Von Mises stress
        σ_vm = np.sqrt((σ_axial + σ_bending_y + σ_bending_z)**2 + 
                      3*(max(τ_shear_y, τ_shear_z, τ_torsion))**2)
        
        # Calculate strains
        ε_axial = σ_axial/E
        ε_bending_y = σ_bending_y/E
        ε_bending_z = σ_bending_z/E
        γ_shear_y = τ_shear_y/G
        γ_shear_z = τ_shear_z/G
        γ_torsion = τ_torsion/G
        
        # Store results
        stress_results.append({
            'position': float(xl[i]),
            'σ_axial': float(σ_axial),
            'σ_bending_y': float(σ_bending_y),
            'σ_bending_z': float(σ_bending_z),
            'τ_shear_y': float(τ_shear_y),
            'τ_shear_z': float(τ_shear_z),
            'τ_torsion': float(τ_torsion),
            'von_mises': float(σ_vm)
        })
        
        strain_results.append({
            'position': float(xl[i]),
            'ε_axial': float(ε_axial),
            'ε_bending_y': float(ε_bending_y),
            'ε_bending_z': float(ε_bending_z),
            'γ_shear_y': float(γ_shear_y),
            'γ_shear_z': float(γ_shear_z),
            'γ_torsion': float(γ_torsion)
        })
    
    return {
        'length': float(L),
        'section_type': section_type,
        'forces': force_results,
        'stresses': stress_results,
        'strains': strain_results
    }

# Function to calculate floor masses (COM and mass)
def calculate_floor_masses(node_coords):
    # Group nodes by floor (Z coordinate)
    floors = {}
    for node_tag, coords in node_coords.items():
        z = coords[2]
        if z not in floors:
            floors[z] = []
        floors[z].append(node_tag)
    
    # Sort floors by elevation
    sorted_floors = sorted(floors.keys())
    
    # Calculate mass properties for each floor
    floor_masses = {}
    for z in sorted_floors:
        total_mass_x = 0
        total_mass_y = 0
        total_mass = 0
        for node_tag in floors[z]:
            mass = ops.nodeMass(node_tag)
            if sum(mass) > 0:  # Only nodes with mass
                x, y, _ = node_coords[node_tag]
                total_mass_x += x * mass[0]  # Using x-mass (DOF 1)
                total_mass_y += y * mass[1]  # Using y-mass (DOF 2)
                total_mass += mass[0]  # Assuming mass is same in x and y
        
        if total_mass > 0:
            com_x = total_mass_x / total_mass
            com_y = total_mass_y / total_mass
            floor_masses[z] = (com_x, com_y, z, total_mass)
        else:
            # For floors with no mass, use geometric center
            x_coords = [node_coords[tag][0] for tag in floors[z]]
            y_coords = [node_coords[tag][1] for tag in floors[z]]
            com_x = sum(x_coords)/len(x_coords) if x_coords else 0
            com_y = sum(y_coords)/len(y_coords) if y_coords else 0
            floor_masses[z] = (com_x, com_y, z, 0)
    
    return floor_masses

# Function to calculate floor stiffness properties
def calculate_floor_stiffness(node_coords, modal_props, eigs, floor_masses):
    # Get modal properties
    parti_factor_MX = modal_props["partiFactorMX"]
    parti_factor_MY = modal_props["partiFactorMY"]
    parti_factor_RMZ = modal_props["partiFactorRMZ"]
    
    floor_stiffness = {}
    floor_stiffness_values = {}
    
    # Find dominant translational modes
    x_modes = [i for i in range(len(parti_factor_MX)) if abs(parti_factor_MX[i]) > 0.3]
    y_modes = [i for i in range(len(parti_factor_MY)) if abs(parti_factor_MY[i]) > 0.3]
    
    if not x_modes or not y_modes:
        print("Warning: Could not identify clear translational modes for stiffness calculation")
        return floor_stiffness, floor_stiffness_values
    
    # We'll use the first significant mode in each direction
    x_mode = x_modes[0]
    y_mode = y_modes[0]
    
    # Calculate global stiffness eccentricities
    ey = parti_factor_RMZ[x_mode]/parti_factor_MX[x_mode] if parti_factor_MX[x_mode] != 0 else 0
    ex = -parti_factor_RMZ[y_mode]/parti_factor_MY[y_mode] if parti_factor_MY[y_mode] != 0 else 0
    
    # Calculate approximate stiffness values using modal frequencies
    omega_x = math.sqrt(eigs[x_mode])
    omega_y = math.sqrt(eigs[y_mode])
    
    # Total mass (sum of all floor masses)
    total_mass = sum(mass for _, _, _, mass in floor_masses.values())
    
    # Calculate floor stiffness properties
    for z, (com_x, com_y, _, mass) in floor_masses.items():
        if total_mass > 0:
            # Calculate stiffness values
            Kx = mass * omega_x**2 if mass > 0 else 0
            Ky = mass * omega_y**2 if mass > 0 else 0
            Kr = max(Kx, Ky) * 100  # Simplified rotational stiffness
            
            # Center of stiffness for this floor
            cos_x = com_x + ex
            cos_y = com_y + ey
            
            floor_stiffness[z] = (cos_x, cos_y, z)
            floor_stiffness_values[z] = (Kx, Ky, Kr)
    
    return floor_stiffness, floor_stiffness_values




def extract_story_drifts(cqc_displacements, node_tags, story_heights):
    """
    Calculate story drifts from nodal displacements
    
    Args:
        cqc_displacements (dict): CQC combined displacements
        node_tags (list): List of node tags
        story_heights (dict): Dictionary mapping story level to height {story: height}
    
    Returns:
        dict: Story drifts for each direction
    """
    story_drifts = {'X': {}, 'Y': {}}
    
    # Group nodes by story level (assuming node tags or coordinates indicate story)
    # This is a simplified approach - you may need to modify based on your node numbering
    story_nodes = {}
    for node_tag in node_tags:
        # Get node coordinates to determine story level
        coords = ops.nodeCoord(node_tag)
        z_coord = coords[2] if len(coords) > 2 else 0.0
        
        # Find which story this node belongs to
        story = None
        for story_level, height in story_heights.items():
            if abs(z_coord - height) < 0.01:  # tolerance for floating point comparison
                story = story_level
                break
        
        if story is not None:
            if story not in story_nodes:
                story_nodes[story] = []
            story_nodes[story].append(node_tag)
    
    # Calculate drifts between stories
    sorted_stories = sorted(story_nodes.keys())
    
    for i in range(1, len(sorted_stories)):
        upper_story = sorted_stories[i]
        lower_story = sorted_stories[i-1]
        
        # Get maximum displacement for each story
        upper_disp_x = max([abs(cqc_displacements['Ux'][node]) for node in story_nodes[upper_story]])
        lower_disp_x = max([abs(cqc_displacements['Ux'][node]) for node in story_nodes[lower_story]])
        
        upper_disp_y = max([abs(cqc_displacements['Uy'][node]) for node in story_nodes[upper_story]])
        lower_disp_y = max([abs(cqc_displacements['Uy'][node]) for node in story_nodes[lower_story]])
        
        # Calculate drift
        story_height = story_heights[upper_story] - story_heights[lower_story]
        drift_x = abs(upper_disp_x - lower_disp_x) / story_height
        drift_y = abs(upper_disp_y - lower_disp_y) / story_height
        
        story_drifts['X'][f"Story_{upper_story}"] = drift_x
        story_drifts['Y'][f"Story_{upper_story}"] = drift_y
    
    return story_drifts


def calculate_base_and_story_shears(output_dir, modal_reactions, cqc_reactions, story_heights, eigs,
                                  json_filepath='RSA_Base_Story_Shears.json'):
    """
    Calculate base shear and story shear forces from nodal reactions (both mode-wise and CQC-combined)
    
    Args:
        output_dir (str): Output directory path
        modal_reactions (dict): Modal reactions from extract_and_combine_nodal_responses
        cqc_reactions (dict): CQC combined reactions from extract_and_combine_nodal_responses
        story_heights (dict): Dictionary mapping story level to height {story: height}
        eigs (list): List of eigenvalues from modal analysis
        json_filepath (str): Path to save JSON results
    
    Returns:
        dict: Dictionary containing base shear and story shear results (both mode-wise and CQC-combined)
    """
    # Initialize results dictionary
    shear_results = {
        'base_shear': {
            'CQC': {'Fx': 0.0, 'Fy': 0.0, 'Fz': 0.0},
            'modal': {}
        },
        'story_shears': {
            'CQC': {},
            'modal': {}
        },
        'story_overturning_moments': {
            'CQC': {},
            'modal': {}
        },
        'modes': [f"Mode {i+1}" for i in range(len(eigs))]
    }
    
    # Get all node tags with reactions
    node_tags = list(cqc_reactions.keys())

    # Group nodes by story level
    story_nodes = {}
    for node_tag in node_tags:
        # Get node coordinates to determine story level
        coords = ops.nodeCoord(node_tag)
        z_coord = coords[2] if len(coords) > 2 else 0.0
        
        # Find which story this node belongs to
        story = None
        for story_level, height in story_heights.items():
            if abs(z_coord - height) < 0.01:  # tolerance for floating point comparison
                story = story_level
                break
        
        if story is not None:
            if story not in story_nodes:
                story_nodes[story] = []
            story_nodes[story].append(node_tag)
    
    # =============================================
    # CQC-combined results
    # =============================================
    
    # Calculate base shear (sum of reactions at base nodes)
    base_nodes = story_nodes.get(1, [])  # Assuming story 1 is the base
    for node_tag in base_nodes:
        shear_results['base_shear']['CQC']['Fx'] += cqc_reactions[node_tag]['Fx']
        shear_results['base_shear']['CQC']['Fy'] += cqc_reactions[node_tag]['Fy']
        shear_results['base_shear']['CQC']['Fz'] += cqc_reactions[node_tag]['Fz']
    
    # Calculate story shears and moments (CQC-combined)
    for story, nodes in story_nodes.items():
        story_shear_x = 0.0
        story_shear_y = 0.0
        story_moment_x = 0.0
        story_moment_y = 0.0
        
        for node_tag in nodes:
            # Get node coordinates
            coords = ops.nodeCoord(node_tag)
            z = coords[2] if len(coords) > 2 else 0.0
            
            # Sum shear forces
            story_shear_x += cqc_reactions[node_tag]['Fx']
            story_shear_y += cqc_reactions[node_tag]['Fy']
            
            # Sum moments about base (z=0)
            story_moment_x += cqc_reactions[node_tag]['Fy'] * z
            story_moment_y += cqc_reactions[node_tag]['Fx'] * z
        
        shear_results['story_shears']['CQC'][story] = {
            'Fx': story_shear_x,
            'Fy': story_shear_y
        }
        
        shear_results['story_overturning_moments']['CQC'][story] = {
            'Mx': story_moment_x,
            'My': story_moment_y
        }
    
    # Calculate cumulative story shears (from top down)
    sorted_stories = sorted(story_nodes.keys(), reverse=True)
    cumulative_shear_x = 0.0
    cumulative_shear_y = 0.0
    
    for story in sorted_stories:
        cumulative_shear_x += shear_results['story_shears']['CQC'][story]['Fx']
        cumulative_shear_y += shear_results['story_shears']['CQC'][story]['Fy']
        
        shear_results['story_shears']['CQC'][story]['cumulative_Fx'] = cumulative_shear_x
        shear_results['story_shears']['CQC'][story]['cumulative_Fy'] = cumulative_shear_y
    
    # =============================================
    # Mode-wise results
    # =============================================
    
    # Initialize modal results
    for mode in range(1, len(eigs)+1):
        shear_results['base_shear']['modal'][mode] = {'Fx': 0.0, 'Fy': 0.0, 'Fz': 0.0}
        shear_results['story_shears']['modal'][mode] = {}
        shear_results['story_overturning_moments']['modal'][mode] = {}
    
    # Calculate mode-wise base shear
    for mode in modal_reactions:
        shear_results['base_shear']['modal'][mode] = {'Fx': 0.0, 'Fy': 0.0, 'Fz': 0.0}
        for node_tag in base_nodes:
            if node_tag in modal_reactions[mode]:
                # Modal reactions are stored as [Fx, Fy, Fz, Mx, My, Mz]
                Fx = modal_reactions[mode][node_tag][0]
                Fy = modal_reactions[mode][node_tag][1]
                Fz = modal_reactions[mode][node_tag][2]
                shear_results['base_shear']['modal'][mode]['Fx'] += Fx
                shear_results['base_shear']['modal'][mode]['Fy'] += Fy
                shear_results['base_shear']['modal'][mode]['Fz'] += Fz
    
    # Calculate mode-wise story shears and moments
    for mode in modal_reactions:
        for story, nodes in story_nodes.items():
            story_shear_x = 0.0
            story_shear_y = 0.0
            story_moment_x = 0.0
            story_moment_y = 0.0
            
            for node_tag in nodes:
                if node_tag in modal_reactions[mode]:
                    # Get node coordinates
                    coords = ops.nodeCoord(node_tag)
                    z = coords[2] if len(coords) > 2 else 0.0
                    
                    # Get modal reactions [Fx, Fy, Fz, Mx, My, Mz]
                    reactions = modal_reactions[mode][node_tag]
                    Fx = reactions[0]
                    Fy = reactions[1]
                    
                    # Sum shear forces
                    story_shear_x += Fx
                    story_shear_y += Fy
                    
                    # Sum moments about base (z=0)
                    story_moment_x += Fy * z
                    story_moment_y += Fx * z
            
            shear_results['story_shears']['modal'][mode][story] = {
                'Fx': story_shear_x,
                'Fy': story_shear_y
            }
            
            shear_results['story_overturning_moments']['modal'][mode][story] = {
                'Mx': story_moment_x,
                'My': story_moment_y
            }
        
        # Calculate cumulative story shears for this mode (from top down)
        cumulative_shear_x = 0.0
        cumulative_shear_y = 0.0
        
        for story in sorted_stories:
            if story in shear_results['story_shears']['modal'][mode]:
                cumulative_shear_x += shear_results['story_shears']['modal'][mode][story]['Fx']
                cumulative_shear_y += shear_results['story_shears']['modal'][mode][story]['Fy']
                
                shear_results['story_shears']['modal'][mode][story]['cumulative_Fx'] = cumulative_shear_x
                shear_results['story_shears']['modal'][mode][story]['cumulative_Fy'] = cumulative_shear_y
    
    # Save to JSON
    json_filepath = os.path.join(output_dir, json_filepath)
    with open(json_filepath, 'w') as f:
        json.dump(shear_results, f, indent=4)
    
    return shear_results



def extract_and_combine_forces_multiple_sections(output_dir, JSON_FOLDER, section_properties, Tn, Sa, direction, eigs, dmp, scalf, 
                                               num_sections=10, json_filepath='RSA_Forces_MultiSection.json'):
    """
    Extract all member forces from RSA at multiple integration points and perform CQC combination
    
    Args:
        output_dir (str): Output directory path
        Tn (list): List of periods for response spectrum
        Sa (list): List of spectral accelerations
        direction (int): Excitation direction (1=X, 2=Y, 3=Z)
        eigs (list): Eigenvalues from modal analysis
        dmp (list): Damping ratios for each mode
        scalf (list): Scaling factors for each mode
        JSON_FOLDER (str): Path to folder containing element_data.json
        num_sections (int): Number of evaluation points along elements (default=10)
        json_filepath (str): Path to save JSON results
        csv_filepath (str): Path to save CSV results
    """
    # Initialize dictionaries to store all forces at all sections
    modal_forces = {
        'P': {}, 'Vy': {}, 'Vz': {}, 
        'T': {}, 'My': {}, 'Mz': {}
    }
    element_tags = ops.getEleTags()
    
    # Extract forces for each mode
    for mode in range(1, len(eigs)+1):
        ops.responseSpectrumAnalysis(direction, '-Tn', *Tn, '-Sa', *Sa, '-mode', mode)
        
        # Initialize mode entries
        for force_type in modal_forces:
            modal_forces[force_type][mode] = {}
        
        # Get forces for all elements at all sections
        for ele_tag in element_tags:
            ele_type = ops.eleType(ele_tag)
            # if 'beam' in ele_type.lower() or 'ForceBeamColumn' in ele_type:
            # Initialize element entry
            for force_type in modal_forces:
                modal_forces[force_type][mode][ele_tag] = {}

            try:
                if 'beam' in ele_type.lower() or 'ForceBeamColumn' in ele_type:
                    # Use extract_beam_results to get forces at multiple points
                    beam_results = extract_beam_results(ele_tag, num_sections, section_properties)
                    
                    # Distribute forces to sections
                    for section_idx, force_data in enumerate(beam_results['forces'], 1):
                        modal_forces['P'][mode][ele_tag][section_idx] = force_data['N']
                        modal_forces['Vy'][mode][ele_tag][section_idx] = force_data['Vy']
                        modal_forces['Vz'][mode][ele_tag][section_idx] = force_data['Vz']
                        modal_forces['T'][mode][ele_tag][section_idx] = force_data['T']
                        modal_forces['My'][mode][ele_tag][section_idx] = force_data['My']
                        modal_forces['Mz'][mode][ele_tag][section_idx] = force_data['Mz']
            except Exception as e:
                print(f"Warning: Could not extract forces for element {ele_tag} in mode {mode}: {str(e)}")
                # Initialize with zeros if extraction fails
                for section in range(1, num_sections+1):
                    modal_forces['P'][mode][ele_tag][section] = 0.0
                    modal_forces['Vy'][mode][ele_tag][section] = 0.0
                    modal_forces['Vz'][mode][ele_tag][section] = 0.0
                    modal_forces['T'][mode][ele_tag][section] = 0.0
                    modal_forces['My'][mode][ele_tag][section] = 0.0
                    modal_forces['Mz'][mode][ele_tag][section] = 0.0
    
    # Perform CQC combination for each section
    cqc_forces = {
        'P': {}, 'Vy': {}, 'Vz': {}, 
        'T': {}, 'My': {}, 'Mz': {}
    }
    
    for ele_tag in element_tags:
        ele_type = ops.eleType(ele_tag)
        # if 'beam' in ele_type.lower() or 'ForceBeamColumn' in ele_type:
        # Initialize element entry in CQC results
        for force_type in cqc_forces:
            cqc_forces[force_type][ele_tag] = {}

        if 'beam' in ele_type.lower() or 'ForceBeamColumn' in ele_type:
            # CQC combination for each section
            for section in range(1, num_sections+1):
                # Extract modal forces for this element and section
                P_modes = [modal_forces['P'][m][ele_tag][section] for m in range(1, len(eigs)+1)]
                Vy_modes = [modal_forces['Vy'][m][ele_tag][section] for m in range(1, len(eigs)+1)]
                Vz_modes = [modal_forces['Vz'][m][ele_tag][section] for m in range(1, len(eigs)+1)]
                T_modes = [modal_forces['T'][m][ele_tag][section] for m in range(1, len(eigs)+1)]
                My_modes = [modal_forces['My'][m][ele_tag][section] for m in range(1, len(eigs)+1)]
                Mz_modes = [modal_forces['Mz'][m][ele_tag][section] for m in range(1, len(eigs)+1)]
                
                # Perform CQC combination
                cqc_forces['P'][ele_tag][section] = CQC(P_modes, eigs, dmp, scalf)
                cqc_forces['Vy'][ele_tag][section] = CQC(Vy_modes, eigs, dmp, scalf)
                cqc_forces['Vz'][ele_tag][section] = CQC(Vz_modes, eigs, dmp, scalf)
                cqc_forces['T'][ele_tag][section] = CQC(T_modes, eigs, dmp, scalf)
                cqc_forces['My'][ele_tag][section] = CQC(My_modes, eigs, dmp, scalf)
                cqc_forces['Mz'][ele_tag][section] = CQC(Mz_modes, eigs, dmp, scalf)

    # Find critical forces (max absolute values across sections)
    critical_forces = {}
    for ele_tag in element_tags:
        ele_type = ops.eleType(ele_tag)
        if 'beam' in ele_type.lower() or 'ForceBeamColumn' in ele_type:
            critical_forces[ele_tag] = {
                'P': max(abs(f) for f in cqc_forces['P'][ele_tag].values()),
                'Vy': max(abs(f) for f in cqc_forces['Vy'][ele_tag].values()),
                'Vz': max(abs(f) for f in cqc_forces['Vz'][ele_tag].values()),
                'T': max(abs(f) for f in cqc_forces['T'][ele_tag].values()),
                'My': max(abs(f) for f in cqc_forces['My'][ele_tag].values()),
                'Mz': max(abs(f) for f in cqc_forces['Mz'][ele_tag].values())
            }

    # Save to JSON
    json_filepath = os.path.join(output_dir, json_filepath)
    with open(json_filepath, 'w') as f:
        json.dump({
            'modal_forces': modal_forces,
            'cqc_forces': cqc_forces,
            'critical_forces': critical_forces,
            'eigenvalues': eigs,
            'damping': dmp,
            'scaling_factors': scalf,
            'num_sections': num_sections
        }, f, indent=4)
    
    
    return modal_forces, cqc_forces, critical_forces


def generate_structural_plots(OUTPUT_FOLDER,
    load_combination,
    section_properties,
    elastic_section,
    aggregator_section,
    beam_integrations,
    frame_elements):
    """Generate structural model plots after analysis"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import opsvis as opsv
    
    # Simple filepath structure
    plots_folder = f"{OUTPUT_FOLDER}_{load_combination}_plots"
    os.makedirs(plots_folder, exist_ok=True)
    
    # Model plot with shell elements
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get all shell elements
    shell_elements = ops.getEleTags()

    for ele_tag in shell_elements:
        # Get node coordinates of the element
        ele_nodes = ops.eleNodes(ele_tag)
        node_coords = np.array([ops.nodeCoord(node) for node in ele_nodes])
        
        # Create a filled polygon for shell elements (if they have more than 2 nodes)
        if len(node_coords) > 2:
            poly = Poly3DCollection([node_coords], alpha=0.5, linewidth=1, edgecolor='k')
            poly.set_facecolor('yellow')  # Single color for all elements
            ax.add_collection3d(poly)

    # Overlay the original model edges
    opsv.plot_model(element_labels=0, node_labels=0, ax=ax, fmt_model={'color': 'k', 'linewidth': 1})
    plt.title(f"Model - {load_combination}")
    filepath = os.path.join(plots_folder, f"model_{load_combination}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Deformation plot
    plt.figure(figsize=(10, 8))
    opsv.plot_model()
    plt.title(f"Deformation - {load_combination}")
    filepath = os.path.join(plots_folder, f"deformation_{load_combination}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Load plot
    plt.figure(figsize=(10, 8))
    opsv.plot_load()
    plt.title(f"Load - {load_combination}")
    filepath = os.path.join(plots_folder, f"load_{load_combination}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    

    ele_shapes = get_element_shapes(section_properties, elastic_section,aggregator_section, beam_integrations, frame_elements)
 
    print("ele_shapes11")
    print(ele_shapes)

    # Get all element tags
    element_tags = ops.getEleTags()

    # Plot the extruded shapes
    plt.figure(figsize=(10, 8))
    opsv.plot_extruded_shapes_3d(ele_shapes)
    plt.title("Extruded Shapes")
    filepath = os.path.join(plots_folder, f"extruded_shapes.png")
    plt.savefig(filepath)
    plt.close()

    print(f"All plots saved to {plots_folder}")
    
    return plots_folder


def calculate_Cs(S, T, TB, TC, TD, xi, Z, I, R):
    # Calculate the damping correction factor mu
    mu = (10 / (5 + xi)) ** 0.5
    # Ensure mu is not smaller than 0.55
    mu = max(mu, 0.55)

    # Initialize Cs
    Cs = 0

    # Calculate Cs based on the given conditions
    if 0 <= T <= TB:
        Cs = S * (1 + (T / TB) * (2.5 * mu - 1))
    elif TB < T <= TC:  # Fixed: changed from TB <= T to TB < T
        Cs = 2.5 * S * mu
    elif TC < T <= TD:  # Fixed: changed from TC <= T to TC < T
        Cs = 2.5 * S * mu * (TC / T)
    elif TD < T <= 4:   # Fixed: changed from TD <= T to TD < T
        Cs = 2.5 * S * mu * (TC * TD / T ** 2)

    Sa = (2 / 3) * (Z * I / R) * Cs
    return Cs, Sa

def CQC(mu, lambdas, dmp, scalf):
    u = 0.0
    ne = len(lambdas)
    for i in range(ne):
        for j in range(ne):
            di = dmp[i]
            dj = dmp[j]
            bij = lambdas[i]/lambdas[j]
            rho = ((8.0*math.sqrt(di*dj)*(di+bij*dj)*(bij**(3.0/2.0))) /
                ((1.0-bij**2.0)**2.0 + 4.0*di*dj*bij*(1.0+bij**2.0) + 
                4.0*(di**2.0 + dj**2.0)*bij**2.0))
            u += scalf[i]*mu[i] * scalf[j]*mu[j] * rho
    return math.sqrt(u)

def calculate_reinforcement_with_spacing(Mu: float, d: float, fc: float, fy: float) -> Tuple[float, float, str]:
    """
    Calculate required reinforcement and spacing per ACI 318 (FPS units).
    
    Args:
        Mu: Factored moment per unit width (lb-in/ft)
        d: Effective depth (in)
        fc: Concrete compressive strength (psi)
        fy: Steel yield strength (psi)
        
    Returns:
        Tuple containing:
        - Required reinforcement area (in²/ft)
        - Center-to-center bar spacing (in)
        - Selected bar size
    """
    # Constants
    phi = 0.9  # Strength reduction factor for flexure
    beta1 = 0.85 if fc <= 4000 else max(0.65, 0.85 - 0.05*(fc-4000)/1000)
    b = 12  # Unit width for calculation (12 in = 1 ft)
    
    # Calculate required Rn (moment parameter)
    Rn = Mu / (phi * b * d**2)
    
    # Calculate required steel ratio
    rho = (0.85 * fc / fy) * (1 - math.sqrt(1 - (2 * Rn) / (0.85 * fc)))
    
    # Check minimum reinforcement (ACI 7.6.1.1)
    rho_min = max(0.0018, 3 * math.sqrt(fc) / fy)
    rho = max(rho, rho_min)
    
    # Calculate required steel area (in²/ft)
    As_req = rho * b * d
    
    # Check maximum reinforcement (ACI 21.2.2)
    epsilon_t = 0.005  # Net tensile strain
    rho_max = (0.85 * beta1 * fc / fy) * (epsilon_t / (epsilon_t + 0.002))
    if rho > rho_max:
        raise ValueError("Error: Reinforcement exceeds maximum allowed by ACI")
    
    # Available bar sizes (#3 to #6)
    bar_sizes = {
        '#3': 0.11,  # in²
        # '#4': 0.20,
        # '#5': 0.31,
        # '#6': 0.44
    }
    
    # Find the most economical bar size and spacing
    selected_bar = None
    for bar_size, bar_area in sorted(bar_sizes.items(), key=lambda x: x[1], reverse=True):
        spacing = (bar_area * b) / As_req
        
        # Check minimum spacing requirements (ACI 25.2.1)
        min_spacing = max(1,  # 1 inch minimum
                         {'#3': 0.375, '#4': 0.5, '#5': 0.625, '#6': 0.75}[bar_size],  # Bar diameter
                         1.33 * 0.75)  # 1.33 * max aggregate size (assume 3/4")
        
        if spacing >= min_spacing:
            selected_bar = (bar_size, spacing)
            break
    
    if not selected_bar:
        # If no bar satisfies spacing requirements, use smallest bar at minimum spacing
        bar_size = '#3'
        bar_area = bar_sizes[bar_size]
        min_spacing = max(1, 0.375, 1.33 * 0.75)
        As_provided = (bar_area * b) / min_spacing
        return (As_provided, min_spacing, bar_size)
    
    # Round spacing to nearest 1/2 inch for practicality
    practical_spacing = round(selected_bar[1] * 2) / 2
    
    return (As_req, practical_spacing, selected_bar[0])

def extract_shell_results(ele_tag):
    """
    Extract forces, stresses, and strains for a shell element
    
    Args:
        ele_tag: Element tag
        
    Returns:
        Dictionary containing shell results
    """
    # Initialize results
    results = {
        'forces': None,
        'stresses': [],
        'strains': []
    }
    
    try:
        # Get element responses
        forces = ops.eleResponse(ele_tag, 'forces')       # Membrane and bending forces
        stresses = ops.eleResponse(ele_tag, 'stresses')   # Stresses at integration points
        strains = ops.eleResponse(ele_tag, 'strains')     # Strains at integration points
        
        # Organize forces (stress resultants)
        if forces:
            # Different shell elements might return different numbers of forces
            force_results = {}
            force_components = ['Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qxz', 'Qyz']
            
            for i, component in enumerate(force_components):
                if i < len(forces):
                    force_results[component] = float(forces[i])
                else:
                    force_results[component] = 0.0  # Default value if not available
            
            results['forces'] = force_results
        
        # Organize stresses (typically at integration points through thickness)
        if stresses:
            # Stresses are usually reported for each integration point
            # Format can vary - we'll handle different cases
            num_stress_components = len(stresses)
            num_integration_points = num_stress_components // 5  # Most common case
            
            for ip in range(num_integration_points):
                start_idx = ip * 5
                end_idx = start_idx + 5
                
                if end_idx <= num_stress_components:
                    stress_data = {
                        'integration_point': ip + 1,
                        'σ_xx': float(stresses[start_idx]),
                        'σ_yy': float(stresses[start_idx + 1]),
                        'τ_xy': float(stresses[start_idx + 2]),
                        'τ_xz': float(stresses[start_idx + 3]) if (start_idx + 3) < num_stress_components else 0.0,
                        'τ_yz': float(stresses[start_idx + 4]) if (start_idx + 4) < num_stress_components else 0.0
                    }
                    results['stresses'].append(stress_data)
        
        # Organize strains
        if strains:
            num_strain_components = len(strains)
            num_integration_points = num_strain_components // 5  # Most common case
            
            for ip in range(num_integration_points):
                start_idx = ip * 5
                end_idx = start_idx + 5
                
                if end_idx <= num_strain_components:
                    strain_data = {
                        'integration_point': ip + 1,
                        'ε_xx': float(strains[start_idx]),
                        'ε_yy': float(strains[start_idx + 1]),
                        'γ_xy': float(strains[start_idx + 2]),
                        'γ_xz': float(strains[start_idx + 3]) if (start_idx + 3) < num_strain_components else 0.0,
                        'γ_yz': float(strains[start_idx + 4]) if (start_idx + 4) < num_strain_components else 0.0
                    }
                    results['strains'].append(strain_data)
                    
    except Exception as e:
        print(f"Error processing shell element {ele_tag}: {str(e)}")
        # Return partial results if available
        pass
    
    return results


def extract_all_results(section_props, output_folder, num_points=5):
    """
    Extract complete analysis results including nodal reactions/displacements and element forces/stresses/strains
    
    Args:
        section_props: Dictionary or list of section properties for beam elements
        output_folder: Path to folder where results will be saved
        num_points: Number of evaluation points per element for force/stress/strain calculations
        
    Returns:
        Dictionary containing all analysis results
    Raises:
        ValueError: If model contains no nodes or elements
        OSError: If output folder cannot be created
    """
    # Validate and create output folder
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        raise OSError(f"Cannot create output folder {output_folder}: {str(e)}")

    # Get all nodes and elements
    all_node_tags = ops.getNodeTags()
    all_element_tags = ops.getEleTags()

    if not all_node_tags or not all_element_tags:
        raise ValueError("Model contains no nodes or elements")

    # Initialize results dictionary
    results = {
        "metadata": {
            "node_count": len(all_node_tags),
            "element_count": len(all_element_tags),
            "force_points_per_element": num_points,
            "timestamp": str(datetime.datetime.now())
        },
        "nodal_results": {
            "reactions": {},
            "displacements": {}
        },
        "element_results": {
            "beam": {
                "forces": {},
                "stresses": {},
                "strains": {}
            },
            "shell": {
                "forces": {},
                "stresses": {},
                "strains": {}
            },
            "other": {
                "forces": {},
                "stresses": {},
                "strains": {}
            }
        }
    }

    # Nodal results
    print("Extracting nodal results...")
    for node_tag in all_node_tags:
        try:
            results["nodal_results"]["reactions"][node_tag] = {
                "FX": float(ops.nodeReaction(node_tag)[0]),
                "FY": float(ops.nodeReaction(node_tag)[1]),
                "FZ": float(ops.nodeReaction(node_tag)[2]),
                "MX": float(ops.nodeReaction(node_tag)[3]),
                "MY": float(ops.nodeReaction(node_tag)[4]),
                "MZ": float(ops.nodeReaction(node_tag)[5])
            }
            results["nodal_results"]["displacements"][node_tag] = {
                "UX": float(ops.nodeDisp(node_tag)[0]),
                "UY": float(ops.nodeDisp(node_tag)[1]),
                "UZ": float(ops.nodeDisp(node_tag)[2]),
                "RX": float(ops.nodeDisp(node_tag)[3]),
                "RY": float(ops.nodeDisp(node_tag)[4]),
                "RZ": float(ops.nodeDisp(node_tag)[5])
            }
        except Exception as e:
            print(f"Error processing node {node_tag}: {str(e)}")
            continue

    # Element results
    print("Extracting element results...")
    for elem_tag in all_element_tags:
        ele_type = ops.eleType(elem_tag)
        
        try:
            # Process based on element type
            if 'beam' in ele_type.lower() or 'ForceBeamColumn' in ele_type:
                beam_res = extract_beam_results(
                    ele_tag=elem_tag,
                    nep=num_points,
                    section_properties=section_props
                )
                
                results["element_results"]["beam"]["forces"][elem_tag] = beam_res['forces']
                results["element_results"]["beam"]["stresses"][elem_tag] = beam_res['stresses']
                results["element_results"]["beam"]["strains"][elem_tag] = beam_res['strains']
                
            elif 'shell' in ele_type.lower() or 'Shell' in ele_type:
                shell_res = extract_shell_results(elem_tag)
                
                results["element_results"]["shell"]["forces"][elem_tag] = shell_res['forces']
                results["element_results"]["shell"]["stresses"][elem_tag] = shell_res['stresses']
                results["element_results"]["shell"]["strains"][elem_tag] = shell_res['strains']
                
            else:
                print(f"Unsupported element type {ele_type} for element {elem_tag}")
                continue
                
        except Exception as e:
            print(f"Error processing element {elem_tag} ({ele_type}): {str(e)}")
            continue

    # Save results to JSON file
    output_path = os.path.join(output_folder, "analysis_results.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {str(e)}")
        raise

    return results

def calculate_slab_reinforcement_from_shell_forces(
    results,
    section_properties, 
    output_folder="postprocessing_folder", 
    num_points=5, 
    load_combination="combo2"
):
    """
    Calculate slab reinforcement from shell forces according to ACI 318 (FPS units).
    Uses results from extract_all_results() function.
    """
    # Create output directories
    json_folder = os.path.join(output_folder, "json_files", load_combination, "slab_reinforcement")
    os.makedirs(json_folder, exist_ok=True)

    # Extract shell forces from the results
    shell_forces = {}
    for elem_tag, force_data in results["element_results"]["shell"]["forces"].items():
        if force_data:  # Only process if force data exists
            shell_forces[elem_tag] = force_data

    # Material properties (FPS units - lb, in, psi)
    phi = 0.9  # Strength reduction factor
    fy = 60000  # psi (yield strength of reinforcement)
    fc = 4000  # psi (concrete compressive strength)
    slab_thickness = 6  # in (total slab thickness)
    cover = 0.75  # in (clear cover to reinforcement)

    # Calculate effective depth (assuming #4 bars - 0.5 in diameter)
    d = slab_thickness - cover - 0.25  # in (0.25 = half of #4 bar diameter)

    reinforcement_results = []

    for ele_tag, force in shell_forces.items():
        try:
            # Get moments (convert to lb-in/ft if needed)
            Mx = force.get("Mxx", 0) * 12  # Convert lb-ft/ft to lb-in/ft if necessary
            My = force.get("Myy", 0) * 12
            Mxy = force.get("Mxy", 0) * 12

            # Calculate reinforcement
            As_x_b, spacing_x_b, bar_size_x_b = calculate_reinforcement_with_spacing(Mx, d, fc, fy)
            As_y_b, spacing_y_b, bar_size_y_b = calculate_reinforcement_with_spacing(My, d, fc, fy)
            As_x_t, spacing_x_t, bar_size_x_t = calculate_reinforcement_with_spacing(-Mx, d, fc, fy)
            As_y_t, spacing_y_t, bar_size_y_t = calculate_reinforcement_with_spacing(-My, d, fc, fy)

            reinforcement_results.append({
                "element_id": ele_tag,
                "moments": {
                    "Mxx": Mx/12,  # Return as lb-ft/ft
                    "Myy": My/12,
                    "Mxy": Mxy/12
                },
                "reinforcement": {
                    "bottom_x": {"As": As_x_b, "spacing": spacing_x_b, "bar_size": bar_size_x_b},
                    "bottom_y": {"As": As_y_b, "spacing": spacing_y_b, "bar_size": bar_size_y_b},
                    "top_x": {"As": As_x_t, "spacing": spacing_x_t, "bar_size": bar_size_x_t},
                    "top_y": {"As": As_y_t, "spacing": spacing_y_t, "bar_size": bar_size_y_t}
                }
            })

        except Exception as e:
            print(f"Error processing element {ele_tag}: {str(e)}")
            continue

    # Save results
    output_path = os.path.join(json_folder, "slab_reinforcement.json")
    with open(output_path, 'w') as f:
        json.dump(reinforcement_results, f, indent=2)

    print(f"Slab reinforcement results saved to {output_path}")
    return reinforcement_results

# def response_spectrum_analysis(section_properties, Tn, Sa, direction=1, num_modes=7, output_dir="RSA_Results", JSON_FOLDER=None):
#     """
#     Complete Response Spectrum Analysis implementation with CQC combination, member forces, drifts, and plots
    
#     Args:
#         Tn (list): List of periods for response spectrum
#         Sa (list): List of spectral accelerations
#         num_modes (int): Number of modes to consider
#         output_dir (str): Directory to save output files
#         JSON_FOLDER (str): Path to folder containing element data (for member forces)
    
#     Returns:
#         dict: Dictionary containing all analysis results
#     """
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # =============================================
#     # STEP 1: SETUP AND MODAL ANALYSIS
#     # =============================================
    
#     # Analysis settings for modal analysis
#     ops.constraints("Transformation")
#     ops.numberer("RCM")
#     ops.system("UmfPack")
#     ops.test("NormUnbalance", 0.0001, 10)
#     ops.algorithm("Linear")
#     ops.integrator("LoadControl", 0.0)
#     ops.analysis("Static")
    
#     # Run eigenvalue analysis
#     eigs = ops.eigen("-genBandArpack", num_modes)
    
#     # Get modal properties
#     modal_props = ops.modalProperties("-return")
    
#     # Calculate natural periods
#     periods = []
#     for eig in eigs:
#         if eig > 0:
#             omega = math.sqrt(eig)
#             period = 2 * math.pi / omega
#         else:
#             period = 0.0
#         periods.append(period)
    
#     print("\nModal Properties:")
#     for i in range(num_modes):
#         print(f"Mode {i+1}: T = {periods[i]:.4f} s, ω = {math.sqrt(eigs[i]):.4f} rad/s")
    
#     # =============================================
#     # STEP 2: RESPONSE SPECTRUM ANALYSIS SETUP
#     # =============================================
    
#     # Define story heights (this should be customized for your structure)
#     story_heights = {
#         1: 0.0,      # Ground level
#         2: 4.0,      # First floor at 4m
#         3: 8.0,      # Second floor at 8m
#         4: 12.0      # Third floor at 12m
#     }
    
#     # Damping settings for CQC
#     dmp = [0.05] * num_modes  # 5% damping for all modes
#     scalf = [1.0] * num_modes  # Scaling factors
    
#     # Create response spectrum time series
#     ops.timeSeries("Path", 100, "-time", *Tn, "-values", *Sa)
    
#     # =============================================
#     # STEP 3: EXTRACT NODE AND ELEMENT DATA
#     # =============================================
    
#     # Get all node coordinates
#     node_coords = {}
#     node_tags = ops.getNodeTags()
#     for node in node_tags:
#         node_coords[node] = ops.nodeCoord(node)
    
#     # Calculate floor masses and stiffness properties
#     floor_masses = calculate_floor_masses(node_coords)
#     floor_stiffness, floor_stiffness_values = calculate_floor_stiffness(
#         node_coords, modal_props, eigs, floor_masses)
    
#     # =============================================
#     # STEP 4: PERFORM RESPONSE SPECTRUM ANALYSIS
#     # =============================================
    
#     # Initialize dictionaries to store results
#     modal_displacements = {}
#     modal_reactions = {}
    
#     # Run RSA for each mode
#     for mode in range(1, num_modes + 1):
#         ops.responseSpectrumAnalysis(1, '-Tn', *Tn, '-Sa', *Sa, '-mode', mode)
#         ops.reactions()
        
#         # Store modal displacements
#         modal_displacements[mode] = {}
#         for node in node_tags:
#             modal_displacements[mode][node] = ops.nodeDisp(node)
        
#         # Store modal reactions
#         modal_reactions[mode] = {}
#         for node in node_tags:
#             modal_reactions[mode][node] = ops.nodeReaction(node)
    
#     # =============================================
#     # STEP 5: CQC COMBINATION OF RESULTS
#     # =============================================
    
#     # Combine displacements using CQC
#     cqc_displacements = {}
#     for node in node_tags:
#         cqc_displacements[node] = {
#             'Ux': CQC([modal_displacements[m][node][0] for m in range(1, num_modes+1)], eigs, dmp, scalf),
#             'Uy': CQC([modal_displacements[m][node][1] for m in range(1, num_modes+1)], eigs, dmp, scalf),
#             'Uz': CQC([modal_displacements[m][node][2] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_displacements[1][node]) > 2 else 0.0,
#             'Rx': CQC([modal_displacements[m][node][3] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_displacements[1][node]) > 3 else 0.0,
#             'Ry': CQC([modal_displacements[m][node][4] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_displacements[1][node]) > 4 else 0.0,
#             'Rz': CQC([modal_displacements[m][node][5] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_displacements[1][node]) > 5 else 0.0
#         }
    
#     # Combine reactions using CQC
#     cqc_reactions = {}
#     for node in node_tags:
#         cqc_reactions[node] = {
#             'Fx': CQC([modal_reactions[m][node][0] for m in range(1, num_modes+1)], eigs, dmp, scalf),
#             'Fy': CQC([modal_reactions[m][node][1] for m in range(1, num_modes+1)], eigs, dmp, scalf),
#             'Fz': CQC([modal_reactions[m][node][2] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_reactions[1][node]) > 2 else 0.0,
#             'Mx': CQC([modal_reactions[m][node][3] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_reactions[1][node]) > 3 else 0.0,
#             'My': CQC([modal_reactions[m][node][4] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_reactions[1][node]) > 4 else 0.0,
#             'Mz': CQC([modal_reactions[m][node][5] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_reactions[1][node]) > 5 else 0.0
#         }
    
#     # =============================================
#     # STEP 6: EXTRACT MEMBER FORCES AND DRIFTS
#     # =============================================
    
#     # Extract and combine member forces (if element data is available)
#     if JSON_FOLDER is not None:
#         modal_forces, cqc_forces, critical_forces = extract_and_combine_forces_multiple_sections(output_dir, JSON_FOLDER, section_properties, Tn, Sa, direction, eigs, dmp, scalf, 
#                                                num_sections=10, json_filepath='RSA_Forces_MultiSection.json')
#     else:
#         print("Warning: No element data provided - skipping member force extraction")
    
#     # Calculate story drifts
#     story_drifts = extract_story_drifts(cqc_displacements, node_tags, story_heights)
    
#     # Calculate base and story shears
#     shear_results = calculate_base_and_story_shears(
#         output_dir, modal_reactions, cqc_reactions, story_heights, eigs)
    
#     # =============================================
#     # STEP 7: GENERATE PLOTS AND VISUALIZATIONS
#     # =============================================
    
#     # Plot model and mode shapes
#     opsv.plot_model()
#     plt.savefig(os.path.join(output_dir, "model_plot.png"))
#     plt.close()
    
#     for mode in range(1, num_modes + 1):
#         opsv.plot_mode_shape(mode, endDispFlag=0, fig_wi_he=(18, 18))
#         plt.title(f"Mode Shape {mode}", fontsize=16)
#         plt.savefig(os.path.join(output_dir, f"mode{mode}.png"), dpi=300)
#         plt.close()
    
#     # Plot response spectrum
#     plt.figure(figsize=(10, 6))
#     plt.plot(Tn, Sa, 'b-', linewidth=2, label='Design Spectrum')
#     plt.xlabel('Period (s)')
#     plt.ylabel('Spectral Acceleration (g)')
#     plt.title('Response Spectrum')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig(os.path.join(output_dir, 'response_spectrum.png'))
#     plt.close()
    
#     # =============================================
#     # STEP 8: SAVE RESULTS
#     # =============================================
#     print(f'---------+++++++++++-------')
#     # print(modal_props)
#     print(f'---------+++++++++++-------')

#     # Save modal properties
#     with open(os.path.join(output_dir, "modal_properties.json"), "w") as f:
#         json.dump({
#             "periods": periods,
#             "eigenvalues": eigs,
#             "modal_participation_factors": {
#                 "MX": modal_props["partiFactorMX"],
#                 "MY": modal_props["partiFactorMY"],
#                 "RMZ": modal_props["partiFactorRMZ"]
#             },
#             "effective_masses": {
#                 "MX": modal_props["partiMassMX"],  # Changed from effMassaX to partiMassMX
#                 "MY": modal_props["partiMassMY"],  # Changed from effMassaY to partiMassMY
#                 "RMZ": modal_props["partiMassRMZ"]  # Changed from effMassaRotZ to partiMassRMZ
#             },
#             "mass_ratios": {
#                 "MX": modal_props["partiMassRatiosMX"],
#                 "MY": modal_props["partiMassRatiosMY"],
#                 "RMZ": modal_props["partiMassRatiosRMZ"]
#             },
#             "cumulative_mass_ratios": {
#                 "MX": modal_props["partiMassRatiosCumuMX"],
#                 "MY": modal_props["partiMassRatiosCumuMY"],
#                 "RMZ": modal_props["partiMassRatiosCumuRMZ"]
#             }
#         }, f, indent=4)
    
#     # Save nodal responses
#     with open(os.path.join(output_dir, "nodal_responses.json"), "w") as f:
#         json.dump({
#             "modal_displacements": modal_displacements,
#             "modal_reactions": modal_reactions,
#             "cqc_displacements": cqc_displacements,
#             "cqc_reactions": cqc_reactions,
#             "story_drifts": story_drifts,
#             "shear_results": shear_results
#         }, f, indent=4)
    
#     # Save floor properties
#     floor_data = []
#     for z in sorted(floor_masses.keys()):
#         com_x, com_y, _, mass = floor_masses[z]
#         if z in floor_stiffness:
#             cos_x, cos_y, _ = floor_stiffness[z]
#             Kx, Ky, Kr = floor_stiffness_values[z]
#         else:
#             cos_x, cos_y = com_x, com_y
#             Kx, Ky, Kr = 0, 0, 0

#         floor_data.append({
#             "Floor_Z": round(z, 2),
#             "COM_X": round(com_x, 3),
#             "COM_Y": round(com_y, 3),
#             "Mass": round(mass, 3),
#             "COS_X": round(cos_x, 3),
#             "COS_Y": round(cos_y, 3),
#             "Kx": round(Kx, 3),
#             "Ky": round(Ky, 3),
#             "Kr": round(Kr, 3)
#         })
    
#     with open(os.path.join(output_dir, "floor_properties.json"), "w") as f:
#         json.dump(floor_data, f, indent=4)
    

    
#     return {
#         "modal_properties": modal_props,
#         "periods": periods,
#         "eigenvalues": eigs,
#         "modal_displacements": modal_displacements,
#         "modal_reactions": modal_reactions,
#         "cqc_displacements": cqc_displacements,
#         "cqc_reactions": cqc_reactions,
#         "story_drifts": story_drifts,
#         "shear_results": shear_results,
#         "floor_properties": floor_data
#     }



# def gravity_analysis(
#     node_loads,
#     element_uniform_loads,
#     shell_pressure_loads,
#     section_properties,
#     elastic_section,
#     aggregator_section,
#     beam_integrations,
#     frame_elements,
#     num_points=5,
#     OUTPUT_FOLDER="postprocessing_folder",
#     load_combination="gravity"
# ):
    
#     # Apply loads and run analysis
#     ops.timeSeries("Linear", 1)
#     ops.pattern("Plain", 1, 1)
    
#     for load in node_loads:
#         ops.load(*load[1:])
    
#     for load in element_uniform_loads:
#         ops.eleLoad("-ele", load[1], "-type", "-beamUniform", *load[2:])
    
#     for load in shell_pressure_loads:
#         ops.eleLoad("-ele", load[1], "-type", "-surfaceLoad", *load[2:])

#     # Analysis settings
#     ops.constraints("Transformation")
#     ops.numberer("RCM")
#     ops.system("UmfPack")
#     ops.test("NormUnbalance", 0.0001, 10)
#     ops.algorithm("Linear")
#     ops.integrator("LoadControl", 1.0)
#     ops.analysis("Static")
#     ops.analyze(1)
#     ops.reactions()

#     output_dir = OUTPUT_FOLDER

#     try:
#         results = extract_all_results(
#             section_props=section_properties,
#             output_folder=output_dir,
#             num_points=7
#         )
#         print("Analysis completed successfully!")
#     except Exception as e:
#         print(f"Analysis failed: {str(e)}")

#     calculate_slab_reinforcement_from_shell_forces(
#         results,
#         section_properties, 
#         output_folder="postprocessing_folder", 
#         num_points=5, 
#         load_combination="combo2"
#     )
#     print(f"Results saved to 'analysis_results.json' with {num_points} points per element")
    
#     generate_structural_plots(OUTPUT_FOLDER,
#     load_combination,
#     section_properties,
#     elastic_section,
#     aggregator_section,
#     beam_integrations,
#     frame_elements)

         
#     return results



# def create_structural_model():
#     """Create complete structural model with nodes, elements, loads, and shell elements"""
    
#     # =============================================
#     # 1. MATERIAL PROPERTIES
#     # ["Elastic", material tag, E - Young's modulus in Pa]
#     # ["ENT", material tag, stiffness value in N/m]
#     # =============================================
#     materials = [
#         ["Elastic", 2, 938000000.0],  # Shear material
#         ["ENT", 101, 1.0e6],          # Spring in X direction (1MN/m)
#         ["ENT", 102, 1.0e6],          # Spring in Y direction
#         ["ENT", 103, 1.0e6]           # Spring in Z direction
#     ]

#     # nD Materials (for shells)
#     # ["ElasticIsotropic", material tag, E - Young's modulus in Pa, v - Poisson's ratio]
#     nd_materials = [
#         ["ElasticIsotropic", 10, 30000000000.0, 0.2]  # E=30GPa, v=0.2
#     ]
    
#     # =============================================
#     # 2. SECTION PROPERTIES
#     # Store section properties as lists with the format:
#     # [section_tag, type, A, Iy, Iz, J, B, H, t]
#     # =============================================
#     section_properties = [
#         # tag, type,       A,    Iy,       Iz,       J,        B,    H,    t
#         [1,    'rectangular', 0.09, 0.000675, 0.000675, 0.00114075, 0.3, 0.3, None],
#         [3,    'rectangular', 0.09, 0.000675, 0.000675, 0.00114075, 0.3, 0.3, None]  # Aggregator uses same properties
#     ]
    
#     # Elastic section definition using the properties from the list
#     elastic_section = ["Elastic", 1, 30000000000.0, 
#                       section_properties[0][2],  # A
#                       section_properties[0][4],  # Iz
#                       section_properties[0][3],  # Iy
#                       12500000000.0,            # G
#                       section_properties[0][5]] # J
    
#     # Aggregator section
#     aggregator_section = [
#         "Aggregator", 3, 
#         2, "Vy", 2, "Vz", "-section", 1
#     ]
    
#     # Shell section
#     shell_section = [
#         "PlateFiber", 20, 10, 0.15  # 15cm thick shell
#     ]
    
#     # =============================================
#     # 3. NODE DEFINITIONS
#     # [nodeTag, x-coord in m, y-coord in m, z-coord in m, mass [mx, my, mz, mr1, mr2, mr3]]
#     # =============================================
#     nodes = [
#         [1, 0, 0, 0, None],                    # Base nodes
#         [2, 0, 0, 3, [200, 200, 200, 0, 0, 0]],
#         [3, 4, 0, 3, [200, 200, 200, 0, 0, 0]],
#         [4, 4, 0, 0, None],
#         [5, 0, 0, 6, [200, 200, 200, 0, 0, 0]],
#         [6, 4, 0, 6, [200, 200, 200, 0, 0, 0]],
#         [7, 4, 3, 6, [200, 200, 200, 0, 0, 0]],
#         [8, 0, 3, 6, [200, 200, 200, 0, 0, 0]],
#         [9, 0, 3, 3, [200, 200, 200, 0, 0, 0]],
#         [10, 0, 3, 0, None],
#         [11, 4, 3, 3, [200, 200, 200, 0, 0, 0]],
#         [12, 4, 3, 0, None],
#         [13, 2, 1.5, 6, None],  # Diaphragm masters
#         [14, 2, 1.5, 3, None],
#     ]
    
#     # =============================================
#     # 4. GEOMETRIC TRANSFORMATIONS
#     # ["Linear", transformation tag, vecxzX, vecxzY, vecxzZ]
#     # =============================================
#     transformations = [
#         ["Linear", 1, 1.0, 0.0, -0.0],
#         ["Linear", 2, 0.0, 0.0, 1.0],
#         ["Linear", 3, 1.0, 0.0, -0.0],
#         ["Linear", 4, 1.0, 0.0, -0.0],
#         ["Linear", 5, 0.0, 0.0, 1.0],
#         ["Linear", 6, 0.0, 0.0, 1.0],
#         ["Linear", 7, 0.0, 0.0, 1.0],
#         ["Linear", 8, 0.0, 0.0, 1.0],
#         ["Linear", 9, 0.0, 0.0, 1.0],
#         ["Linear", 10, 1.0, 0.0, -0.0],
#         ["Linear", 11, 1.0, 0.0, -0.0],
#         ["Linear", 12, 1.0, 0.0, -0.0],
#         ["Linear", 13, 0.0, 0.0, 1.0],
#         ["Linear", 14, 0.0, 0.0, 1.0],
#         ["Linear", 15, 1.0, 0.0, -0.0],
#         ["Linear", 16, 1.0, 0.0, -0.0],
#         ["Linear", 20, 0.0, 0.0, 1.0]  # For shell elements
#     ]
    
#     # =============================================
#     # 5. BEAM INTEGRATION
#     # ["Lobatto", integration tag, section tag, Np - number of integration points]
#     # =============================================
#     beam_integrations = [
#         ["Lobatto", 1, 3, 5]
#     ]
    
#     # =============================================
#     # 6. ELEMENT CONNECTIONS
#     # ["forceBeamColumn", element tag, iNode, jNode, transformation tag, integration tag]
#     # =============================================
#     frame_elements = [
#         ["forceBeamColumn", 1, 1, 2, 1, 1],
#         ["forceBeamColumn", 2, 2, 3, 2, 1],
#         ["forceBeamColumn", 3, 4, 3, 3, 1],
#         ["forceBeamColumn", 4, 2, 5, 4, 1],
#         ["forceBeamColumn", 5, 5, 6, 5, 1],
#         ["forceBeamColumn", 6, 7, 6, 6, 1],
#         ["forceBeamColumn", 7, 8, 7, 7, 1],
#         ["forceBeamColumn", 8, 9, 2, 8, 1],
#         ["forceBeamColumn", 9, 8, 5, 9, 1],
#         ["forceBeamColumn", 10, 10, 9, 10, 1],
#         ["forceBeamColumn", 11, 3, 6, 11, 1],
#         ["forceBeamColumn", 12, 11, 7, 12, 1],
#         ["forceBeamColumn", 13, 11, 3, 13, 1],
#         ["forceBeamColumn", 14, 9, 11, 14, 1],
#         ["forceBeamColumn", 15, 12, 11, 15, 1],
#         ["forceBeamColumn", 16, 9, 8, 16, 1]
#     ]
    
#     # Shell elements format: [type, tag, node1, node2, node3, node4, secTag]
#     shell_elements = [
#         ["ShellMITC4", 101, 2, 3, 11, 9, 20],
#         # ["ShellMITC4", 102, 15, 4, 3, 17, 20],
#         # ["ShellMITC4", 103, 17, 3, 11, 9, 20],
#         # ["ShellMITC4", 104, 2, 17, 9, 10, 20]
#     ]
    

#     # =============================================
#     # 7. BOUNDARY CONDITIONS
#     # [nodeTag, fixX, fixY, fixZ, fixRX, fixRY, fixRZ] (1=fixed, 0=free)
#     # =============================================
#     fixities = [
#         [1, 1, 1, 1, 1, 1, 1],
#         [10, 1, 1, 1, 1, 1, 1],
#         [4, 1, 1, 1, 1, 1, 1],
#         [12, 1, 1, 1, 1, 1, 1],
#         [13, 0, 0, 1, 1, 1, 0],
#         [14, 0, 0, 1, 1, 1, 0]
#     ]
    
#     # =============================================
#     # 8. RIGID DIAPHRAGMS
#     # Format: [perpDirn, masterNode, *slaveNodes]
#     # =============================================
#     diaphragms = [
#         [3, 14, 2, 3, 9, 11],
#         [3, 13, 5, 6, 7, 8]
#     ]
    
#     # =============================================
#     # 9. LOAD DEFINITIONS
#     # [loadTag, nodeTag, Fx, Fy, Fz, Mx, My, Mz]
#     # =============================================
#     node_loads = [
#         [1, 5, 0, 0, -10000, 0, 0, 0],  # 10kN vertical load at node 5
#         [2, 6, 0, 0, -10000, 0, 0, 0]   # 10kN vertical load at node 6
#     ]
    
#     element_uniform_loads = [
#         [1, 1, 0, -5000, 0],  # 5kN/m vertical load on element 1
#         [2, 2, 0, -5000, 0]   # 5kN/m vertical load on element 2
#     ]
    
#     shell_pressure_loads = [
#         # [101, 101, -2000],  # 2kPa pressure on shell 101
#         # [102, 102, -2000]   # 2kPa pressure on shell 102
#     ]

#     # Zero length elements
#     zero_length_elements = [
#         # [2001, 1, 1001, 101, 102, 103],  # Base spring
#         # [2002, 4, 1004, 101, 102, 103],
#         # [2003, 10, 1010, 101, 102, 103],
#         # [2004, 12, 1012, 101, 102, 103]
#     ]
    
#     # =============================================
#     # BUILD THE MODEL
#     # =============================================
#     ops.wipe()
#     ops.model('basic', '-ndm', 3, '-ndf', 6)
    
#     # Create materials
#     for mat in materials:
#         ops.uniaxialMaterial(*mat)
    
#     for mat in nd_materials:
#         ops.nDMaterial(*mat)

#     # Create sections
#     ops.section(*elastic_section)
#     ops.section(*aggregator_section)
#     ops.section(*shell_section)
    
#     # Create nodes
#     for node in nodes:
#         if node[4] is not None:
#             ops.node(node[0], node[1], node[2], node[3], '-mass', *node[4])
#         else:
#             ops.node(node[0], node[1], node[2], node[3])
    
#     # Create transformations
#     for transf in transformations:
#         ops.geomTransf(*transf)
    
#     # Create beam integration
#     for integ in beam_integrations:
#         ops.beamIntegration(*integ)
    
#     # Create frame elements
#     for elem in frame_elements:
#         ops.element(*elem)
    
#     # Create shell elements
#     for elem in shell_elements:
#         ops.element(*elem)
    
#     # Apply boundary conditions
#     for fix in fixities:
#         ops.fix(*fix)
    
#     # Create rigid diaphragms
#     for diaph in diaphragms:
#         ops.rigidDiaphragm(*diaph)
    
#     # Create zeroLength elements
#     for elem in zero_length_elements:
#         spring_node_tag = elem[2]
#         base_node_tag = elem[1]
#         base_node_coords = ops.nodeCoord(base_node_tag)
#         x, y, z = base_node_coords[0], base_node_coords[1], base_node_coords[2]
        
#         ops.node(spring_node_tag, x, y, z)
#         ops.fix(spring_node_tag, 1, 1, 1, 1, 1, 1)
        
#         ops.element("zeroLength", elem[0], elem[1], elem[2], 
#                 "-mat", elem[3], elem[4], elem[5], 
#                 "-dir", 1, 2, 3)

#     print("Structural model with loads created successfully")
#     return node_loads, element_uniform_loads, shell_pressure_loads, section_properties, elastic_section,aggregator_section, beam_integrations, frame_elements

def create_structural_model(materials, nd_materials, section_properties, elastic_section, 
                          aggregator_section, shell_section, nodes, transformations, 
                          beam_integrations, frame_elements, shell_elements, fixities, 
                          diaphragms, node_loads, element_uniform_loads, shell_pressure_loads, 
                          zero_length_elements):
    """Create complete structural model with nodes, elements, loads, and shell elements"""
    
    # =============================================
    # BUILD THE MODEL
    # =============================================
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    # Create materials
    for mat in materials:
        ops.uniaxialMaterial(*mat)
    
    for mat in nd_materials:
        ops.nDMaterial(*mat)

    # Create sections
    ops.section(*elastic_section)
    ops.section(*aggregator_section)
    ops.section(*shell_section)
    
    # Create nodes
    for node in nodes:
        if node[4] is not None:
            ops.node(node[0], node[1], node[2], node[3], '-mass', *node[4])
        else:
            ops.node(node[0], node[1], node[2], node[3])
    
    # Create transformations
    for transf in transformations:
        ops.geomTransf(*transf)
    
    # Create beam integration
    for integ in beam_integrations:
        ops.beamIntegration(*integ)
    
    # Create frame elements
    for elem in frame_elements:
        ops.element(*elem)
    
    # Create shell elements
    for elem in shell_elements:
        ops.element(*elem)
    
    # Apply boundary conditions
    for fix in fixities:
        ops.fix(*fix)
    
    # Create rigid diaphragms
    for diaph in diaphragms:
        ops.rigidDiaphragm(*diaph)
    
    # Create zeroLength elements
    for elem in zero_length_elements:
        spring_node_tag = elem[2]
        base_node_tag = elem[1]
        base_node_coords = ops.nodeCoord(base_node_tag)
        x, y, z = base_node_coords[0], base_node_coords[1], base_node_coords[2]
        
        ops.node(spring_node_tag, x, y, z)
        ops.fix(spring_node_tag, 1, 1, 1, 1, 1, 1)
        
        ops.element("zeroLength", elem[0], elem[1], elem[2], 
                "-mat", elem[3], elem[4], elem[5], 
                "-dir", 1, 2, 3)

    print("Structural model with loads created successfully")
    return node_loads, element_uniform_loads, shell_pressure_loads, section_properties, elastic_section, aggregator_section, beam_integrations, frame_elements

def gravity_analysis(model_data, num_points=5, OUTPUT_FOLDER="postprocessing_folder", load_combination="gravity"):
    """Run gravity analysis using model data"""
    
    # First create the structural model
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
        model_data['zero_length_elements']
    )
    
    # Apply loads and run analysis
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    
    for load in node_loads:
        ops.load(*load[1:])
    
    for load in element_uniform_loads:
        ops.eleLoad("-ele", load[1], "-type", "-beamUniform", *load[2:])
    
    for load in shell_pressure_loads:
        ops.eleLoad("-ele", load[1], "-type", "-surfaceLoad", *load[2:])

    # Analysis settings
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormUnbalance", 0.0001, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    ops.analyze(1)
    ops.reactions()

    output_dir = OUTPUT_FOLDER

    try:
        results = extract_all_results(
            section_props=section_properties,
            output_folder=output_dir,
            num_points=7
        )
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")

    calculate_slab_reinforcement_from_shell_forces(
        results,
        section_properties, 
        output_folder="postprocessing_folder", 
        num_points=5, 
        load_combination="combo2"
    )
    print(f"Results saved to 'analysis_results.json' with {num_points} points per element")
    
    generate_structural_plots(OUTPUT_FOLDER,
    load_combination,
    section_properties,
    elastic_section,
    aggregator_section,
    beam_integrations,
    frame_elements)

         
    return results



def response_spectrum_analysis(model_data, Tn, Sa, direction=1, num_modes=7, output_dir="RSA_Results", JSON_FOLDER=None):
    """
    Complete Response Spectrum Analysis implementation with CQC combination, member forces, drifts, and plots
    
    Args:
        model_data (dict): Dictionary containing all model data
        Tn (list): List of periods for response spectrum
        Sa (list): List of spectral accelerations
        num_modes (int): Number of modes to consider
        output_dir (str): Directory to save output files
        JSON_FOLDER (str): Path to folder containing element data (for member forces)
    
    Returns:
        dict: Dictionary containing all analysis results
    """
    # First create the structural model
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
        model_data['zero_length_elements']
    )
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # =============================================
    # STEP 1: SETUP AND MODAL ANALYSIS
    # =============================================
    
    # Analysis settings for modal analysis
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormUnbalance", 0.0001, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 0.0)
    ops.analysis("Static")
    
    # Run eigenvalue analysis
    eigs = ops.eigen("-genBandArpack", num_modes)
    
    # Get modal properties
    modal_props = ops.modalProperties("-return")
    
    # Calculate natural periods
    periods = []
    for eig in eigs:
        if eig > 0:
            omega = math.sqrt(eig)
            period = 2 * math.pi / omega
        else:
            period = 0.0
        periods.append(period)
    
    print("\nModal Properties:")
    for i in range(num_modes):
        print(f"Mode {i+1}: T = {periods[i]:.4f} s, ω = {math.sqrt(eigs[i]):.4f} rad/s")
    
    # =============================================
    # STEP 2: RESPONSE SPECTRUM ANALYSIS SETUP
    # =============================================
    
    # Define story heights (this should be customized for your structure)
    story_heights = {
        1: 0.0,      # Ground level
        2: 4.0,      # First floor at 4m
        3: 8.0,      # Second floor at 8m
        4: 12.0      # Third floor at 12m
    }
    
    # Damping settings for CQC
    dmp = [0.05] * num_modes  # 5% damping for all modes
    scalf = [1.0] * num_modes  # Scaling factors
    
    # Create response spectrum time series
    ops.timeSeries("Path", 100, "-time", *Tn, "-values", *Sa)
    
    # =============================================
    # STEP 3: EXTRACT NODE AND ELEMENT DATA
    # =============================================
    
    # Get all node coordinates
    node_coords = {}
    node_tags = ops.getNodeTags()
    for node in node_tags:
        node_coords[node] = ops.nodeCoord(node)
    
    # Calculate floor masses and stiffness properties
    floor_masses = calculate_floor_masses(node_coords)
    floor_stiffness, floor_stiffness_values = calculate_floor_stiffness(
        node_coords, modal_props, eigs, floor_masses)
    
    # =============================================
    # STEP 4: PERFORM RESPONSE SPECTRUM ANALYSIS
    # =============================================
    
    # Initialize dictionaries to store results
    modal_displacements = {}
    modal_reactions = {}
    
    # Run RSA for each mode
    for mode in range(1, num_modes + 1):
        ops.responseSpectrumAnalysis(1, '-Tn', *Tn, '-Sa', *Sa, '-mode', mode)
        ops.reactions()
        
        # Store modal displacements
        modal_displacements[mode] = {}
        for node in node_tags:
            modal_displacements[mode][node] = ops.nodeDisp(node)
        
        # Store modal reactions
        modal_reactions[mode] = {}
        for node in node_tags:
            modal_reactions[mode][node] = ops.nodeReaction(node)
    
    # =============================================
    # STEP 5: CQC COMBINATION OF RESULTS
    # =============================================
    
    # Combine displacements using CQC
    cqc_displacements = {}
    for node in node_tags:
        cqc_displacements[node] = {
            'Ux': CQC([modal_displacements[m][node][0] for m in range(1, num_modes+1)], eigs, dmp, scalf),
            'Uy': CQC([modal_displacements[m][node][1] for m in range(1, num_modes+1)], eigs, dmp, scalf),
            'Uz': CQC([modal_displacements[m][node][2] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_displacements[1][node]) > 2 else 0.0,
            'Rx': CQC([modal_displacements[m][node][3] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_displacements[1][node]) > 3 else 0.0,
            'Ry': CQC([modal_displacements[m][node][4] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_displacements[1][node]) > 4 else 0.0,
            'Rz': CQC([modal_displacements[m][node][5] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_displacements[1][node]) > 5 else 0.0
        }
    
    # Combine reactions using CQC
    cqc_reactions = {}
    for node in node_tags:
        cqc_reactions[node] = {
            'Fx': CQC([modal_reactions[m][node][0] for m in range(1, num_modes+1)], eigs, dmp, scalf),
            'Fy': CQC([modal_reactions[m][node][1] for m in range(1, num_modes+1)], eigs, dmp, scalf),
            'Fz': CQC([modal_reactions[m][node][2] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_reactions[1][node]) > 2 else 0.0,
            'Mx': CQC([modal_reactions[m][node][3] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_reactions[1][node]) > 3 else 0.0,
            'My': CQC([modal_reactions[m][node][4] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_reactions[1][node]) > 4 else 0.0,
            'Mz': CQC([modal_reactions[m][node][5] for m in range(1, num_modes+1)], eigs, dmp, scalf) if len(modal_reactions[1][node]) > 5 else 0.0
        }
    
    # =============================================
    # STEP 6: EXTRACT MEMBER FORCES AND DRIFTS
    # =============================================
    
    # Extract and combine member forces (if element data is available)
    if JSON_FOLDER is not None:
        modal_forces, cqc_forces, critical_forces = extract_and_combine_forces_multiple_sections(output_dir, JSON_FOLDER, section_properties, Tn, Sa, direction, eigs, dmp, scalf, 
                                               num_sections=10, json_filepath='RSA_Forces_MultiSection.json')
    else:
        print("Warning: No element data provided - skipping member force extraction")
    
    # Calculate story drifts
    story_drifts = extract_story_drifts(cqc_displacements, node_tags, story_heights)
    
    # Calculate base and story shears
    shear_results = calculate_base_and_story_shears(
        output_dir, modal_reactions, cqc_reactions, story_heights, eigs)
    
    # =============================================
    # STEP 7: GENERATE PLOTS AND VISUALIZATIONS
    # =============================================
    
    # Plot model and mode shapes
    opsv.plot_model()
    plt.savefig(os.path.join(output_dir, "model_plot.png"))
    plt.close()
    
    for mode in range(1, num_modes + 1):
        opsv.plot_mode_shape(mode, endDispFlag=0, fig_wi_he=(18, 18))
        plt.title(f"Mode Shape {mode}", fontsize=16)
        plt.savefig(os.path.join(output_dir, f"mode{mode}.png"), dpi=300)
        plt.close()
    
    # Plot response spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(Tn, Sa, 'b-', linewidth=2, label='Design Spectrum')
    plt.xlabel('Period (s)')
    plt.ylabel('Spectral Acceleration (g)')
    plt.title('Response Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'response_spectrum.png'))
    plt.close()
    
    # =============================================
    # STEP 8: SAVE RESULTS
    # =============================================
    print(f'---------+++++++++++-------')
    # print(modal_props)
    print(f'---------+++++++++++-------')

    # Save modal properties
    with open(os.path.join(output_dir, "modal_properties.json"), "w") as f:
        json.dump({
            "periods": periods,
            "eigenvalues": eigs,
            "modal_participation_factors": {
                "MX": modal_props["partiFactorMX"],
                "MY": modal_props["partiFactorMY"],
                "RMZ": modal_props["partiFactorRMZ"]
            },
            "effective_masses": {
                "MX": modal_props["partiMassMX"],  # Changed from effMassaX to partiMassMX
                "MY": modal_props["partiMassMY"],  # Changed from effMassaY to partiMassMY
                "RMZ": modal_props["partiMassRMZ"]  # Changed from effMassaRotZ to partiMassRMZ
            },
            "mass_ratios": {
                "MX": modal_props["partiMassRatiosMX"],
                "MY": modal_props["partiMassRatiosMY"],
                "RMZ": modal_props["partiMassRatiosRMZ"]
            },
            "cumulative_mass_ratios": {
                "MX": modal_props["partiMassRatiosCumuMX"],
                "MY": modal_props["partiMassRatiosCumuMY"],
                "RMZ": modal_props["partiMassRatiosCumuRMZ"]
            }
        }, f, indent=4)
    
    # Save nodal responses
    with open(os.path.join(output_dir, "nodal_responses.json"), "w") as f:
        json.dump({
            "modal_displacements": modal_displacements,
            "modal_reactions": modal_reactions,
            "cqc_displacements": cqc_displacements,
            "cqc_reactions": cqc_reactions,
            "story_drifts": story_drifts,
            "shear_results": shear_results
        }, f, indent=4)
    
    # Save floor properties
    floor_data = []
    for z in sorted(floor_masses.keys()):
        com_x, com_y, _, mass = floor_masses[z]
        if z in floor_stiffness:
            cos_x, cos_y, _ = floor_stiffness[z]
            Kx, Ky, Kr = floor_stiffness_values[z]
        else:
            cos_x, cos_y = com_x, com_y
            Kx, Ky, Kr = 0, 0, 0

        floor_data.append({
            "Floor_Z": round(z, 2),
            "COM_X": round(com_x, 3),
            "COM_Y": round(com_y, 3),
            "Mass": round(mass, 3),
            "COS_X": round(cos_x, 3),
            "COS_Y": round(cos_y, 3),
            "Kx": round(Kx, 3),
            "Ky": round(Ky, 3),
            "Kr": round(Kr, 3)
        })
    
    with open(os.path.join(output_dir, "floor_properties.json"), "w") as f:
        json.dump(floor_data, f, indent=4)
    

    
    return {
        "modal_properties": modal_props,
        "periods": periods,
        "eigenvalues": eigs,
        "modal_displacements": modal_displacements,
        "modal_reactions": modal_reactions,
        "cqc_displacements": cqc_displacements,
        "cqc_reactions": cqc_reactions,
        "story_drifts": story_drifts,
        "shear_results": shear_results,
        "floor_properties": floor_data
    }




# the response spectrum function
Tn = [0.0, 0.06, 0.1, 0.12, 0.18, 0.24, 0.3, 0.36, 0.4, 0.42, 
    0.48, 0.54, 0.6, 0.66, 0.72, 0.78, 0.84, 0.9, 0.96, 1.02, 
    1.08, 1.14, 1.2, 1.26, 1.32, 1.38, 1.44, 1.5, 1.56, 1.62, 
    1.68, 1.74, 1.8, 1.86, 1.92, 1.98, 2.04, 2.1, 2.16, 2.22, 
    2.28, 2.34, 2.4, 2.46, 2.52, 2.58, 2.64, 2.7, 2.76, 2.82, 
    2.88, 2.94, 3.0, 3.06, 3.12, 3.18, 3.24, 3.3, 3.36, 3.42, 
    3.48, 3.54, 3.6, 3.66, 3.72, 3.78, 3.84, 3.9, 3.96, 4.02, 
    4.08, 4.14, 4.2, 4.26, 4.32, 4.38, 4.44, 4.5, 4.56, 4.62, 
    4.68, 4.74, 4.8, 4.86, 4.92, 4.98, 5.04, 5.1, 5.16, 5.22, 
    5.28, 5.34, 5.4, 5.46, 5.52, 5.58, 5.64, 5.7, 5.76, 5.82, 
    5.88, 5.94, 6.0]


Sa = [1.9612, 3.72628, 4.903, 4.903, 4.903, 4.903, 4.903, 4.903, 4.903, 4.6696172, 
    4.0861602, 3.6321424, 3.2683398, 2.971218, 2.7241068, 2.5142584, 2.3348086, 2.1788932, 2.0425898, 1.9229566, 
    1.8160712, 1.7199724, 1.6346602, 1.5562122, 1.485609, 1.4208894, 1.3620534, 1.3071398, 1.2571292, 1.211041, 
    1.166914, 1.1267094, 1.0894466, 1.054145, 1.0217852, 0.990406, 0.960988, 0.9335312, 0.9080356, 0.8835206, 
    0.8599862, 0.838413, 0.8168398, 0.7972278, 0.7785964, 0.759965, 0.7432948, 0.7266246, 0.710935, 0.6952454, 
    0.6805364, 0.666808, 0.6540602, 0.6285646, 0.6040496, 0.5814958, 0.5609032, 0.5403106, 0.5206986, 0.5030478, 
    0.485397, 0.4697074, 0.4540178, 0.4393088, 0.4255804, 0.411852, 0.3991042, 0.3863564, 0.3755698, 0.3638026, 
    0.353016, 0.34321, 0.333404, 0.3245786, 0.3157532, 0.3069278, 0.2981024, 0.2902576, 0.2833934, 0.2755486, 
    0.2686844, 0.2618202, 0.254956, 0.2490724, 0.2431888, 0.2373052, 0.2314216, 0.2265186, 0.220635, 0.215732, 
    0.210829, 0.205926, 0.2020036, 0.1971006, 0.1931782, 0.1892558, 0.1853334, 0.181411, 0.1774886, 0.1735662, 
    0.1706244, 0.166702, 0.1637602]




