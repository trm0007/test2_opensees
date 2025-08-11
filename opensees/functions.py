import ast
from subprocess import PIPE, Popen
from venv import logger
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import Http404
import matplotlib
import numpy as np

from opensees.load_combinations import apply_structural_loads, get_combination_loads
from opensees.functions import *
# from opensees.wall_meshing import generate_model_with_shells
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from myapp.models import Project, Task
from .forms import *
from .utils import *
import json
import xlwt
from datetime import datetime





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
from django.shortcuts import get_object_or_404
from myapp.models import Project, Task
from django.conf import settings

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
    Extract forces, stresses, strains, deflections, and slopes for a beam element using section properties list
    
    Args:
        ele_tag: Element tag
        nep: Number of evaluation points
        section_properties: List of section properties [tag, type, A, Iy, Iz, J, B, H, t]
        
    Returns:
        Dictionary containing comprehensive beam results including forces, stresses, strains, 
        deflections, slopes, and beam properties
        
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
    section_tag = section[0]
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
    
    # ====== NEW: DEFLECTION AND SLOPE CALCULATIONS ======
    
    # Get nodal displacements and rotations
    node1_disp = ops.nodeDisp(node_tags[0])
    node2_disp = ops.nodeDisp(node_tags[1])
    
    # Extract displacements and rotations at nodes
    # Assuming 6 DOF per node: [ux, uy, uz, rx, ry, rz]
    delta1_y = node1_disp[1]  # Y-displacement at start
    delta1_z = node1_disp[2]  # Z-displacement at start
    theta1_y = node1_disp[4]  # Y-rotation at start
    theta1_z = node1_disp[5]  # Z-rotation at start
    delta1_x = node1_disp[0]  # X-displacement at start (axial)
    
    delta2_y = node2_disp[1]  # Y-displacement at end
    delta2_z = node2_disp[2]  # Z-displacement at end
    delta2_x = node2_disp[0]  # X-displacement at end (axial)
    
    # Calculate flexural stiffnesses
    EIy = E * Iy  # Flexural stiffness about Y-axis
    EIz = E * Iz  # Flexural stiffness about Z-axis
    EA = E * A    # Axial stiffness
    
    # Extract distributed loads from eload_data
    w1_y = w2_y = 0.0  # Default to no distributed load
    w1_z = w2_z = 0.0
    p1 = p2 = 0.0      # Axial distributed loads
    
    if eload_data:
        for load in eload_data:
            if load[0] == '-beamUniform':
                w1_y = w2_y = load[2]  # Uniform load in Y-direction
                w1_z = w2_z = load[3]  # Uniform load in Z-direction
    
    # Calculate deflections and slopes at each evaluation point
    deflection_results = []
    slope_results = []
    
    # Lists to track max/min values
    deflections_y = []
    deflections_z = []
    deflections_x = []
    slopes_y = []
    slopes_z = []
    
    for i in range(len(xl)):
        x = xl[i]
        
        # Current internal forces at this location
        V1_y = s[i,1]    # Shear force Y
        V1_z = s[i,2]    # Shear force Z  
        M1_y = s[i,4]    # Moment about Y
        M1_z = s[i,5]    # Moment about Z
        P1 = s[i,0]      # Axial force
        
        # Calculate deflections using beam theory functions
        deflection_y = beam_deflection_y(x, V1_y, M1_z, P1, w1_y, w2_y, 
                                       theta1_z, delta1_y, L, EIz, P_delta=False)
        
        deflection_z = beam_deflection_z(x, V1_z, M1_y, P1, w1_z, w2_z, 
                                       theta1_y, delta1_z, L, EIy, P_delta=False)
        
        axial_deflection = beam_axial_deflection(x, delta1_x, P1, p1, p2, L, EA)
        
        # Calculate slopes
        slope_y = beam_slope_y(x, V1_y, M1_z, P1, w1_y, w2_y, 
                             theta1_z, delta1_y, L, EIz, P_delta=False)
        
        slope_z = beam_slope_z(x, V1_z, M1_y, P1, w1_z, w2_z, 
                             theta1_y, delta1_z, L, EIy, P_delta=False)
        
        # Store results
        deflection_results.append({
            'position': float(x),
            'deflection_y': float(deflection_y),
            'deflection_z': float(deflection_z),
            'deflection_x': float(axial_deflection),
            'total_deflection': float(np.sqrt(deflection_y**2 + deflection_z**2))
        })
        
        slope_results.append({
            'position': float(x),
            'slope_y': float(slope_y),
            'slope_z': float(slope_z),
            'total_slope': float(np.sqrt(slope_y**2 + slope_z**2))
        })
        
        # Collect for max/min calculations
        deflections_y.append(deflection_y)
        deflections_z.append(deflection_z)
        deflections_x.append(axial_deflection)
        slopes_y.append(slope_y)
        slopes_z.append(slope_z)
    
    # Calculate relative deflections (relative to start of beam)
    relative_deflection_results = []
    for i, defl in enumerate(deflection_results):
        relative_deflection_results.append({
            'position': defl['position'],
            'relative_deflection_y': float(defl['deflection_y'] - delta1_y),
            'relative_deflection_z': float(defl['deflection_z'] - delta1_z),
            'relative_deflection_x': float(defl['deflection_x'] - delta1_x),
            'relative_total_deflection': float(np.sqrt(
                (defl['deflection_y'] - delta1_y)**2 + 
                (defl['deflection_z'] - delta1_z)**2
            ))
        })
    
    # Calculate maximum and minimum deflections
    max_min_deflections = {
        'max_deflection_y': float(max(deflections_y)),
        'min_deflection_y': float(min(deflections_y)),
        'max_deflection_z': float(max(deflections_z)),
        'min_deflection_z': float(min(deflections_z)),
        'max_deflection_x': float(max(deflections_x)),
        'min_deflection_x': float(min(deflections_x)),
        'max_total_deflection': float(max([d['total_deflection'] for d in deflection_results])),
        'max_relative_deflection_y': float(max([d['relative_deflection_y'] for d in relative_deflection_results])),
        'min_relative_deflection_y': float(min([d['relative_deflection_y'] for d in relative_deflection_results])),
        'max_relative_deflection_z': float(max([d['relative_deflection_z'] for d in relative_deflection_results])),
        'min_relative_deflection_z': float(min([d['relative_deflection_z'] for d in relative_deflection_results])),
        'max_relative_total_deflection': float(max([d['relative_total_deflection'] for d in relative_deflection_results]))
    }
    
    # Calculate maximum and minimum slopes
    max_min_slopes = {
        'max_slope_y': float(max(slopes_y)),
        'min_slope_y': float(min(slopes_y)),
        'max_slope_z': float(max(slopes_z)),
        'min_slope_z': float(min(slopes_z)),
        'max_total_slope': float(max([s['total_slope'] for s in slope_results]))
    }
    
    # Beam properties
    beam_properties = {
        'element_tag': int(ele_tag),
        'section_tag': int(section_tag),
        'node_tags': [int(node_tags[0]), int(node_tags[1])],
        'length': float(L),
        'section_type': section_type,
        'cross_sectional_area': float(A),
        'moment_of_inertia_y': float(Iy),
        'moment_of_inertia_z': float(Iz),
        'torsional_constant': float(J),
        'width': float(B),
        'height': float(H),
        'thickness': float(t) if t is not None else None,
        'material_properties': {
            'youngs_modulus': float(E),
            'shear_modulus': float(G),
            'flexural_stiffness_y': float(EIy),
            'flexural_stiffness_z': float(EIz),
            'axial_stiffness': float(EA)
        },
        'boundary_conditions': {
            'start_node_displacement': {
                'x': float(delta1_x),
                'y': float(delta1_y),
                'z': float(delta1_z)
            },
            'start_node_rotation': {
                'y': float(theta1_y),
                'z': float(theta1_z)
            },
            'end_node_displacement': {
                'x': float(delta2_x),
                'y': float(delta2_y),
                'z': float(delta2_z)
            }
        },
        'distributed_loads': {
            'transverse_y': float(w1_y),
            'transverse_z': float(w1_z),
            'axial_start': float(p1),
            'axial_end': float(p2)
        }
    }
    
    return {
        'beam_properties': beam_properties,
        'length': float(L),
        'section_type': section_type,
        'forces': force_results,
        'stresses': stress_results,
        'strains': strain_results,
        'deflections': deflection_results,
        'relative_deflections': relative_deflection_results,
        'slopes': slope_results,
        'max_min_deflections': max_min_deflections,
        'max_min_slopes': max_min_slopes
    }


# Helper functions for deflection calculations
def beam_slope_y(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta=False):
    """Returns the slope of the elastic curve at any point `x` along the segment (Y-direction)."""
    if P_delta:
        delta_x = beam_deflection_y(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta)
        return theta1 + (-V1*x**2/2 - w1*x**3/6 + x*(-M1 - P1*delta1 + P1*delta_x) + x**4*(w1 - w2)/(24*L))/EI
    else:
        return theta1 + (-V1*x**2/2 - w1*x**3/6 + x*(-M1) + x**4*(w1 - w2)/(24*L))/EI


def beam_slope_z(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta=False):
    """Returns the slope of the elastic curve at any point `x` along the segment (Z-direction)."""
    if P_delta:
        delta_x = beam_deflection_z(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta)
        theta_x = theta1 - (-V1*x**2/2 - w1*x**3/6 + x*(M1 - P1*delta1 + P1*delta_x) + x**4*(w1 - w2)/(24*L))/EI
    else:
        theta_x = theta1 - (-V1*x**2/2 - w1*x**3/6 + x*M1 + x**4*(w1 - w2)/(24*L))/EI
    return theta_x


def beam_deflection_y(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta=False):
    """Returns the deflection at a location on the segment (Y-direction)."""
    if P_delta:
        delta_x = delta1
        d_delta = 1
        while d_delta > 0.01: 
            delta_last = delta_x
            delta_x = delta1 - theta1*x + V1*x**3/(6*EI) + w1*x**4/(24*EI) - x**2*(-M1 - P1*delta1 + P1*delta_x)/(2*EI) - x**5*(w1 - w2)/(120*EI*L)
            if delta_last != 0:
                d_delta = abs(delta_x/delta_last - 1)
            else:
                if delta1 - delta_x == 0:
                    break
        return delta_x
    else:
        return delta1 - theta1*x + V1*x**3/(6*EI) + w1*x**4/(24*EI) - x**2*(-M1)/(2*EI) - x**5*(w1 - w2)/(120*EI*L)


def beam_deflection_z(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta=False):
    """Returns the deflection at a location on the segment (Z-direction)."""
    if P_delta:
        delta_x = delta1
        d_delta = 1
        while d_delta > 0.01:
            delta_last = delta_x
            delta_x = delta1 + theta1*x + V1*x**3/(6*EI) + w1*x**4/(24*EI) + x**2*(-M1 + P1*delta1 - P1*delta_x)/(2*EI) + x**5*(-w1 + w2)/(120*EI*L)
            if delta_last != 0:
                d_delta = abs(delta_x/delta_last - 1)
            else:
                if delta1 - delta_x == 0:
                    break
        return delta_x
    else:
        return delta1 + theta1*x + V1*x**3/(6*EI) + w1*x**4/(24*EI) + x**2*(-M1)/(2*EI) + x**5*(-w1 + w2)/(120*EI*L)


def beam_axial_deflection(x, delta_x1, P1, p1, p2, L, EA):
    """Returns the axial deflection at a location on the segment."""
    return delta_x1 - 1/EA*(P1*x + p1*x**2/2 + (p2 - p1)*x**3/(6*L))

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

def save_opensees_script(filename, data, output_folder="post_processing"):
    """Save analysis files based on extension"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create full filepath
    filepath = os.path.join(output_folder, filename)
    
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".json":
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    elif ext == ".txt":
        with open(filepath, 'w') as f:
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(str(data))

    elif ext in [".xls", ".xlsx"]:
        wb = xlwt.Workbook()
        ws = wb.add_sheet('Data')
        if isinstance(data, dict):
            for row_idx, (key, value) in enumerate(data.items()):
                ws.write(row_idx, 0, key)
                ws.write(row_idx, 1, str(value))
        wb.save(filepath)

    elif ext in [".png", ".jpg", ".jpeg"]:
        if isinstance(data, plt.Figure):
            data.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(data)
        else:
            raise ValueError("Invalid image data")

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return filepath

def calculate_base_and_story_shears(modal_reactions, cqc_reactions, story_heights, eigs, output_folder="post_processing"):
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
    
    # Save to JSON using the proper file saving function
    json_filename = 'RSA_Base_Story_Shears.json'
    json_path = save_opensees_script(json_filename, shear_results, output_folder)
    
    return shear_results, json_path



def extract_and_combine_forces_multiple_sections(section_properties, Tn, Sa, direction, eigs, dmp, scalf, num_sections=10, output_folder="post_processing"):
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
    json_filename = 'RSA_Forces_MultiSection.json'
    json_path = save_opensees_script(
        json_filename, {
            'modal_forces': modal_forces,
            'cqc_forces': cqc_forces,
            'critical_forces': critical_forces,
            'eigenvalues': eigs,
            'damping': dmp,
            'scaling_factors': scalf,
            'num_sections': num_sections
        },
        output_folder
    )
    
    return modal_forces, cqc_forces, critical_forces, json_path




def generate_structural_plots(section_properties, elastic_section, aggregator_section, 
                            beam_integrations, frame_elements, output_folder="post_processing"):
    """Generate structural model plots after analysis with proper file saving"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import opsvis as opsv
    import numpy as np
    
    # Create figures
    saved_images = []
    
    # =============================================
    # 1. Model plot with shell elements (working)
    # =============================================
    fig1 = plt.figure(figsize=(10, 8))
    ax = fig1.add_subplot(111, projection='3d')

    # Get all shell elements
    shell_elements = ops.getEleTags()

    for ele_tag in shell_elements:
        # Get node coordinates of the element
        ele_nodes = ops.eleNodes(ele_tag)
        node_coords = np.array([ops.nodeCoord(node) for node in ele_nodes])
        
        # Create a filled polygon for shell elements
        if len(node_coords) > 2:
            poly = Poly3DCollection([node_coords], alpha=0.5, linewidth=1, edgecolor='k')
            poly.set_facecolor('yellow')
            ax.add_collection3d(poly)

    # Overlay the original model edges
    opsv.plot_model(element_labels=0, node_labels=0, ax=ax, fmt_model={'color': 'k', 'linewidth': 1})
    plt.title(f"Model")
    
    # Save model plot
    model_plot_path = save_opensees_script("model_plot.png", fig1, output_folder)
    saved_images.append(model_plot_path)
    plt.close(fig1)
    
    # =============================================
    # 2. Deformation plot - FIXED
    # =============================================
    # Need to first run a deformation analysis to get results
    try:
        opsv.plot_defo()
        plt.title("Deformation")

        # Save deformation plot
        deformation_plot_path = save_opensees_script("deformation_plot.png", plt.gcf(), output_folder)
        saved_images.append(deformation_plot_path)
    except Exception as e:
        print(f"Could not generate deformation plot: {str(e)}")
        deformation_plot_path = None

    plt.close()

    
    # =============================================
    # 3. Load plot - FIXED
    # =============================================
    try:
        opsv.plot_load()
        plt.title("Loads")

        # Save load plot
        load_plot_path = save_opensees_script("load_plot.png", plt.gcf(), output_folder)
        saved_images.append(load_plot_path)
    except Exception as e:
        print(f"Could not generate load plot: {str(e)}")
        load_plot_path = None

    plt.close()


    # =============================================
    # 4. Extruded shapes plot - FIXED
    # =============================================
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    
    try:
        ele_shapes = get_element_shapes(
            section_properties, 
            elastic_section,
            aggregator_section, 
            beam_integrations, 
            frame_elements
        )
        
        # Plot the extruded shapes
        opsv.plot_extruded_shapes_3d(ele_shapes, ax=ax4)
        plt.title("Extruded Shapes")
        
        # Save extruded shapes plot
        extruded_shapes_path = save_opensees_script("extruded_shapes.png", fig4, output_folder)
        saved_images.append(extruded_shapes_path)
    except Exception as e:
        print(f"Could not generate extruded shapes plot: {str(e)}")
        extruded_shapes_path = None
    
    plt.close(fig4)

    # Filter out None values from failed plots
    saved_images = [img for img in saved_images if img is not None]
    print(f"Successfully generated {len(saved_images)} structural plots")
    
    return saved_images


def extract_all_results(section_props, num_points=5, output_folder="post_processing"):
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
            # Method 2: Or use setdefault (more concise)
            results["element_results"]["beam"].setdefault("deflections", {})
            results["element_results"]["beam"].setdefault("relative_deflections", {})
            results["element_results"]["beam"].setdefault("slopes", {})
            results["element_results"]["beam"].setdefault("max_min_deflections", {})
            results["element_results"]["beam"].setdefault("max_min_slopes", {})
            results["element_results"]["beam"].setdefault("beam_properties", {})
            results["element_results"]["beam"].setdefault("element_info", {})

            # Process based on element type
            if 'beam' in ele_type.lower() or 'ForceBeamColumn' in ele_type:
                beam_res = extract_beam_results(
                    ele_tag=elem_tag,
                    nep=num_points,
                    section_properties=section_props
                )
                
                # results["element_results"]["beam"]["forces"][elem_tag] = beam_res['forces']
                # results["element_results"]["beam"]["stresses"][elem_tag] = beam_res['stresses']
                # results["element_results"]["beam"]["strains"][elem_tag] = beam_res['strains']
                # Store all beam analysis results
                results["element_results"]["beam"]["forces"][elem_tag] = beam_res['forces']
                results["element_results"]["beam"]["stresses"][elem_tag] = beam_res['stresses']
                results["element_results"]["beam"]["strains"][elem_tag] = beam_res['strains']

                # Store new deflection and slope results
                results["element_results"]["beam"]["deflections"][elem_tag] = beam_res['deflections']
                results["element_results"]["beam"]["relative_deflections"][elem_tag] = beam_res['relative_deflections']
                results["element_results"]["beam"]["slopes"][elem_tag] = beam_res['slopes']

                # Store maximum and minimum values for easy access
                results["element_results"]["beam"]["max_min_deflections"][elem_tag] = beam_res['max_min_deflections']
                results["element_results"]["beam"]["max_min_slopes"][elem_tag] = beam_res['max_min_slopes']

                # Store comprehensive beam properties
                results["element_results"]["beam"]["beam_properties"][elem_tag] = beam_res['beam_properties']
                
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

    # Save results
    json_filename = 'analysis_results.json'
    json_path = save_opensees_script(json_filename, results, output_folder)
    
    return results, json_path



def calculate_slab_reinforcement_from_shell_forces(results, section_properties, num_points=5, output_folder="post_processing"):
    """
    Calculate slab reinforcement from shell forces according to ACI 318 (FPS units).
    Uses results from extract_all_results() function.
    """

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
    json_filename = 'slab_reinforcement.json'
    json_path = save_opensees_script(json_filename, reinforcement_results, output_folder)
    
    return reinforcement_results, json_path

def plot_mode_shapes_alternative(num_modes, periods, output_folder):
    """Alternative method using direct OpenSees commands"""
    import matplotlib.pyplot as plt
    import opsvis as opsv
    import openseespy.opensees as ops
    
    saved_plots = []
    
    # Get node tags and coordinates
    node_tags = ops.getNodeTags()
    if not node_tags:
        print("No nodes found for plotting")
        return saved_plots
    
    # Plot model
    try:
        fig = plt.figure(figsize=(10, 8))
        
        # Manual model plotting
        for node in node_tags:
            coords = ops.nodeCoord(node)
            if len(coords) >= 2:
                plt.plot(coords[0], coords[1], 'bo', markersize=4)
        
        # Plot elements
        ele_tags = ops.getEleTags()
        for ele in ele_tags:
            try:
                nodes = ops.eleNodes(ele)
                if len(nodes) >= 2:
                    node1_coords = ops.nodeCoord(nodes[0])
                    node2_coords = ops.nodeCoord(nodes[1])
                    plt.plot([node1_coords[0], node2_coords[0]], 
                            [node1_coords[1], node2_coords[1]], 'b-', linewidth=1)
            except:
                continue
        
        plt.title("Structural Model")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        model_path = os.path.join(output_folder, "model_plot.png")
        plt.savefig(model_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(model_path)
        print(f"Model plot saved: {model_path}")
        
    except Exception as e:
        print(f"Error in manual model plotting: {e}")
        plt.close()
    
    # For mode shapes, try opsvis with recorder approach
    for mode in range(1, num_modes + 1):
        try:
            # Record mode shape
            ops.recorder('Node', '-file', f'mode{mode}.out', '-node', *node_tags, '-dof', 1, 2, 'eigen', mode)
            ops.record()
            
            fig = plt.figure(figsize=(10, 8))
            
            # Try simple opsvis call
            opsv.plot_mode_shape(mode)
            plt.title(f"Mode Shape {mode} (T = {periods[mode-1]:.4f} s)")
            
            mode_path = os.path.join(output_folder, f"mode_shape_{mode}.png")
            plt.savefig(mode_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(mode_path)
            print(f"Mode {mode} plot saved: {mode_path}")
            
        except Exception as e:
            print(f"Error plotting mode {mode}: {e}")
            plt.close()
    
    return saved_plots


def response_spectrum_analysis(section_properties, Tn, Sa, direction=1, num_modes=7, output_folder="post_processing"):
    """
    Complete Response Spectrum Analysis with proper file saving
    
    Args:
        section_properties: Dictionary of section properties
        Tn: List of periods for response spectrum
        Sa: List of spectral accelerations
        direction: Excitation direction (1=X, 2=Y, 3=Z)
        num_modes: Number of modes to consider
        output_folder: Output folder name for saving files
    
    Returns:
        dict: Dictionary containing all analysis results and saved file paths
    """
    import matplotlib.pyplot as plt
    import opsvis as opsv
    import math
    import json
    import os
    
    print(f"\n[RSA Analysis] Starting Response Spectrum Analysis with {num_modes} modes")
    print(f"Direction: {direction}, Output folder: {output_folder}")
    
    try:
        # Create output folder for RSA
        rsa_output_folder = os.path.join(output_folder, "response_spectrum_analysis")
        os.makedirs(rsa_output_folder, exist_ok=True)
        
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
        print(f"[RSA Analysis] Running eigenvalue analysis for {num_modes} modes...")
        eigs = ops.eigen("-genBandArpack", num_modes)
        
        # Get modal properties
        modal_props = ops.modalProperties("-return")
        
        # Calculate natural periods
        periods = [2 * math.pi / math.sqrt(eig) if eig > 0 else 0.0 for eig in eigs]
        
        print("\nModal Properties:")
        for i in range(num_modes):
            print(f"Mode {i+1}: T = {periods[i]:.4f} s, ω = {math.sqrt(eigs[i]):.4f} rad/s")
        
        # =============================================
        # STEP 2: RESPONSE SPECTRUM ANALYSIS SETUP
        # =============================================
        
        # Define story heights (customize for your structure)
        story_heights = {1: 0.0, 2: 4.0, 3: 8.0, 4: 12.0}  # Example values
        
        # Damping settings for CQC
        dmp = [0.05] * num_modes  # 5% damping for all modes
        scalf = [1.0] * num_modes  # Scaling factors
        
        # Create response spectrum time series
        ops.timeSeries("Path", 100, "-time", *Tn, "-values", *Sa)
        
        # =============================================
        # STEP 3: EXTRACT NODE AND ELEMENT DATA
        # =============================================
        
        node_tags = ops.getNodeTags()
        node_coords = {node: ops.nodeCoord(node) for node in node_tags}
        
        # Calculate floor properties
        floor_masses = calculate_floor_masses(node_coords)
        floor_stiffness, floor_stiffness_values = calculate_floor_stiffness(
            node_coords, modal_props, eigs, floor_masses)
        
        # =============================================
        # STEP 4: PERFORM RESPONSE SPECTRUM ANALYSIS
        # =============================================
        
        print(f"[RSA Analysis] Performing response spectrum analysis...")
        modal_displacements = {}
        modal_reactions = {}
        
        for mode in range(1, num_modes + 1):
            ops.responseSpectrumAnalysis(direction, '-Tn', *Tn, '-Sa', *Sa, '-mode', mode)
            ops.reactions()
            
            modal_displacements[mode] = {node: ops.nodeDisp(node) for node in node_tags}
            modal_reactions[mode] = {node: ops.nodeReaction(node) for node in node_tags}
        
        # =============================================
        # STEP 5: CQC COMBINATION OF RESULTS
        # =============================================
        
        def get_cqc_values(modes_data, node, index, default=0.0):
            """Helper function for CQC combination of nodal results"""
            modal_values = []
            for m in range(1, num_modes + 1):
                if node in modes_data[m] and len(modes_data[m][node]) > index:
                    modal_values.append(modes_data[m][node][index])
                else:
                    modal_values.append(default)
            return CQC(modal_values, eigs, dmp, scalf)
        
        print(f"[RSA Analysis] Combining modal results using CQC...")
        
        # Combine displacements using CQC
        cqc_displacements = {}
        for node in node_tags:
            cqc_displacements[node] = {
                'Ux': get_cqc_values(modal_displacements, node, 0),
                'Uy': get_cqc_values(modal_displacements, node, 1),
                'Uz': get_cqc_values(modal_displacements, node, 2),
                'Rx': get_cqc_values(modal_displacements, node, 3),
                'Ry': get_cqc_values(modal_displacements, node, 4),
                'Rz': get_cqc_values(modal_displacements, node, 5)
            }
        
        # Combine reactions using CQC
        cqc_reactions = {}
        for node in node_tags:
            cqc_reactions[node] = {
                'Fx': get_cqc_values(modal_reactions, node, 0),
                'Fy': get_cqc_values(modal_reactions, node, 1),
                'Fz': get_cqc_values(modal_reactions, node, 2),
                'Mx': get_cqc_values(modal_reactions, node, 3),
                'My': get_cqc_values(modal_reactions, node, 4),
                'Mz': get_cqc_values(modal_reactions, node, 5)
            }
        
        # =============================================
        # STEP 6: EXTRACT MEMBER FORCES AND DRIFTS
        # =============================================
        
        print(f"[RSA Analysis] Extracting member forces and story drifts...")
        # section_properties = [prop for prop in section_properties if prop[1] != 'shell']
        # Extract and combine member forces
        modal_forces, cqc_forces, critical_forces, forces_path = extract_and_combine_forces_multiple_sections(
            section_properties, Tn, Sa, direction, eigs, dmp, scalf, 
            num_sections=10, output_folder=rsa_output_folder
        )
        
        # Calculate story drifts and shears
        story_drifts = extract_story_drifts(cqc_displacements, node_tags, story_heights)
        shear_results, shears_path = calculate_base_and_story_shears(
            modal_reactions, cqc_reactions, story_heights, eigs, output_folder=rsa_output_folder
        )
        
        # =============================================
        # STEP 7: GENERATE PLOTS AND VISUALIZATIONS
        # =============================================
        
        print(f"[RSA Analysis] Generating plots and visualizations...")
        saved_plots = []
        
        saved_plots = plot_mode_shapes_alternative(num_modes, periods, output_folder)
        # =============================================
        # STEP 8: SAVE RESULTS TO JSON
        # =============================================
        
        print(f"[RSA Analysis] Saving results to files...")
        
        # Prepare results dictionary
        results = {
            "modal_properties": {
                "periods": periods,
                "eigenvalues": eigs,
                "modal_participation_factors": {
                    "MX": modal_props["partiFactorMX"],
                    "MY": modal_props["partiFactorMY"],
                    "RMZ": modal_props["partiFactorRMZ"]
                },
                "effective_masses": {
                    "MX": modal_props["partiMassMX"],
                    "MY": modal_props["partiMassMY"],
                    "RMZ": modal_props["partiMassRMZ"]
                }
            },
            "nodal_responses": {
                "modal_displacements": modal_displacements,
                "modal_reactions": modal_reactions,
                "cqc_displacements": cqc_displacements,
                "cqc_reactions": cqc_reactions
            },
            "story_drifts": story_drifts,
            "shear_results": shear_results,
            "member_forces": {
                "modal_forces": modal_forces,
                "cqc_forces": cqc_forces,
                "critical_forces": critical_forces
            },
            "floor_properties": [{
                "Floor_Z": round(z, 2),
                "COM_X": round(floor_masses[z][0], 3),
                "COM_Y": round(floor_masses[z][1], 3),
                "Mass": round(floor_masses[z][3], 3),
                "Kx": round(floor_stiffness_values.get(z, (0,0,0))[0], 3),
                "Ky": round(floor_stiffness_values.get(z, (0,0,0))[1], 3),
                "Kr": round(floor_stiffness_values.get(z, (0,0,0))[2], 3)
            } for z in sorted(floor_masses.keys())]
        }
        
        # Save main results to JSON file
        results_path = os.path.join(rsa_output_folder, "rsa_analysis_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        results["saved_files"] = {
            "analysis_results": results_path,
            "member_forces": forces_path,
            "shear_results": shears_path,
            "plots": saved_plots
        }
        
        print(f"[RSA Analysis] Completed successfully. Results saved to {rsa_output_folder}")
        return results

    except Exception as e:
        print(f"\n[ERROR] Response Spectrum Analysis failed: {str(e)}")
        raise  # Re-raise the exception after logging



def gravity_analysis(node_loads, element_uniform_loads, shell_pressure_loads, 
                    section_properties, elastic_section, aggregator_section, 
                    beam_integrations, frame_elements, load_cases, load_combinations,
                    num_points=5, output_folder="post_processing"):
    
    # Initial debug prints
    print(f"\n[Gravity Analysis] Starting with {len(frame_elements)} frame elements")
    print(f"Load counts - Nodes: {len(node_loads)}, Elements: {len(element_uniform_loads)}, Shells: {len(shell_pressure_loads)}")

    try:
        # Get unique combination names from the load_combinations structure
        # Structure: [id, combo_name, load_case, factor]
        combos_to_apply = list(set([combo[1] for combo in load_combinations if len(combo) >= 2]))
        print(f"Found {len(combos_to_apply)} load combinations to apply: {combos_to_apply}")
        
        # Set analysis parameters once before the loop
        ops.constraints("Transformation")
        ops.numberer("RCM") 
        ops.system("UmfPack")
        ops.test("NormUnbalance", 0.0001, 10)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        output_folder = os.path.join(output_folder, "gravity_analysis")
        # Apply each combination in a loop
        for i, combo_name in enumerate(combos_to_apply, start=1):
            print(f"\nApplying combination: {combo_name}")
            
            # Create output folder for this combination
            combo_output_folder = os.path.join(output_folder,  combo_name)
            os.makedirs(combo_output_folder, exist_ok=True)
            
            applied_loads = apply_structural_loads(
                load_cases=load_cases,
                load_combinations=load_combinations,
                combo_name=combo_name,
                pattern_tag=i,  # Unique pattern tag for each combination
                time_series_tag=i  # Unique time series tag for each combination
            )
            ops.analyze(1)
            ops.reactions()
            # Analysis settings
            print(f"\n[Analysis] Setting up solver for {combo_name}...")
            # ops.constraints("Transformation")
            # ops.numberer("RCM")
            # ops.system("UmfPack")
            # ops.test("NormUnbalance", 0.0001, 10)
            # ops.algorithm("Linear")
            # ops.integrator("LoadControl", 1.0)
            # ops.analysis("Static")
            
            # print(f"[Analysis] Running analysis for {combo_name}...")
            # ops.analyze(1)
            
            # 1. First, set all solver parameters
            # ops.constraints("Transformation")
            # ops.numberer("RCM")
            # ops.system("UmfPack")
            # ops.test("NormUnbalance", 0.0001, 10)
            # ops.algorithm("Linear")
            # ops.integrator("LoadControl", 1.0)

            # # 2. THEN create the analysis object
            # ops.analysis("Static")

            # 3. Finally, run the analysis
            # ops.analyze(1)
            # ops.reactions()
            print(f"[Analysis] Completed successfully for {combo_name}")

            # Extract results for this combination
            print(f"\n[Post-processing] Extracting results for {combo_name}...")
            results, results_path = extract_all_results(section_properties, num_points, output_folder=combo_output_folder)
            print(f"Extracted {len(results)} result points to {results_path}")

            # Calculate reinforcement for this combination
            reinforcement, reinforcement_path = calculate_slab_reinforcement_from_shell_forces(
                results, section_properties, num_points, output_folder=combo_output_folder)
            if reinforcement_path:
                print(f"Generated reinforcement results at {reinforcement_path}")

            # Generate plots for this combination
            plot_paths = generate_structural_plots(
                section_properties, elastic_section, aggregator_section, 
                beam_integrations, frame_elements, output_folder=combo_output_folder)
            print(f"Generated {len(plot_paths)} plots for {combo_name}")
            
            # Clear the pattern before applying the next combination
            ops.remove('loadPattern', i)
            print(f"Cleared load pattern {i} for {combo_name}")

        # Return results from the last combination (you might want to modify this logic)
        return results

    except Exception as e:
        print(f"\n[ERROR] Gravity analysis failed: {str(e)}")
        raise  # Re-raise the exception after logging



def create_structural_model(materials, nd_materials, section_properties, elastic_section, aggregator_section, shell_section, nodes, transformations, beam_integrations, frame_elements, shell_elements, fixities, diaphragms, node_loads, element_uniform_loads, shell_pressure_loads, zero_length_elements ):
    """Create complete structural model with nodes, elements, loads, and shell elements"""
    
    # materials, nd_materials, section_properties, elastic_section, aggregator_section, shell_section, nodes, transformations, beam_integrations, frame_elements, shell_elements, fixities, diaphragms, node_loads, element_uniform_loads, shell_pressure_loads, zero_length_elements = model_input_data()
    
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
    return node_loads, element_uniform_loads, shell_pressure_loads, section_properties, elastic_section,aggregator_section, beam_integrations, frame_elements

