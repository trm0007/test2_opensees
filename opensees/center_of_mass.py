import numpy as np
from collections import defaultdict
import pandas as pd

# =============================================
# INPUT DATA DEFINITIONS
# =============================================

# Node definitions: [nodeTag, x-coord, y-coord, z-coord, mass]
nodes = [
    [1, 0, 0, 0, None],                    # Base nodes
    [2, 0, 0, 3, [200, 200, 200, 0, 0, 0]],
    [3, 4, 0, 3, [200, 200, 200, 0, 0, 0]],
    [4, 4, 0, 0, None],
    [5, 0, 0, 6, [200, 200, 200, 0, 0, 0]],
    [6, 4, 0, 6, [200, 200, 200, 0, 0, 0]],
    [7, 4, 3, 6, [200, 200, 200, 0, 0, 0]],
    [8, 0, 3, 6, [200, 200, 200, 0, 0, 0]],
    [9, 0, 3, 3, [200, 200, 200, 0, 0, 0]],
    [10, 0, 3, 0, None],
    [11, 4, 3, 3, [200, 200, 200, 0, 0, 0]],
    [12, 4, 3, 0, None],
    [13, 2, 1.5, 6, None],  # Diaphragm masters
    [14, 2, 1.5, 3, None],
]

# Frame elements: [type, tag, iNode, jNode, transformation, integration]
frame_elements = [
    ["forceBeamColumn", 1, 1, 2, 1, 1],
    ["forceBeamColumn", 2, 2, 3, 2, 1],
    ["forceBeamColumn", 3, 4, 3, 3, 1],
    ["forceBeamColumn", 4, 2, 5, 4, 1],
    ["forceBeamColumn", 5, 5, 6, 5, 1],
    ["forceBeamColumn", 6, 7, 6, 6, 1],
    ["forceBeamColumn", 7, 8, 7, 7, 1],
    ["forceBeamColumn", 8, 9, 2, 8, 1],
    ["forceBeamColumn", 9, 8, 5, 9, 1],
    ["forceBeamColumn", 10, 10, 9, 10, 1],
    ["forceBeamColumn", 11, 3, 6, 11, 1],
    ["forceBeamColumn", 12, 11, 7, 12, 1],
    ["forceBeamColumn", 13, 11, 3, 13, 1],
    ["forceBeamColumn", 14, 9, 11, 14, 1],
    ["forceBeamColumn", 15, 12, 11, 15, 1],
    ["forceBeamColumn", 16, 9, 8, 16, 1]
]

# Shell elements: [type, tag, node1, node2, node3, node4, secTag]
shell_elements = [
    ["ShellMITC4", 101, 2, 3, 11, 9, 20],
    ["ShellMITC4", 102, 5, 6, 7, 8, 20],  # Additional shell element
    ["ShellMITC4", 103, 2, 5, 8, 9, 20],  # Additional shell element
]

# Load cases
load_cases = [
    # Dead Load (DL) - Node Loads
    [1, "DL", "node_loads", 5, 0, 0, -5000, 0, 0, 0],
    [2, "DL", "node_loads", 6, 0, 0, -5000, 0, 0, 0],
    
    # Dead Load (DL) - Element Uniform Loads
    [3, "DL", "element_uniform_loads", 1, 0, -2500, 0],
    [4, "DL", "element_uniform_loads", 2, 0, -2500, 0],
    
    # Dead Load (DL) - Element Point Loads
    [5, "DL", "element_point_loads", 1, 1000, -500, 0, 0.5],
    [6, "DL", "element_point_loads", 2, 0, 0, -2000, 0.7],
    
    # Shell Pressure Loads
    [14, "WL", "shell_pressure_loads", 101, -500],
    [17, "DL", "shell_pressure_loads", 102, -300],
    [18, "LL", "shell_pressure_loads", 103, -400],
    
    # Live Load (LL) - Node Loads
    [8, "LL", "node_loads", 5, 0, 0, -5000, 0, 0, 0],
    [9, "LL", "node_loads", 6, 0, 0, -5000, 0, 0, 0],
    
    # Live Load (LL) - Element Uniform Loads
    [10, "LL", "element_uniform_loads", 1, 0, -2500, 0],
    [11, "LL", "element_uniform_loads", 2, 0, -2500, 0],
    
    # Wind Load (WL) - Node Loads
    [12, "WL", "node_loads", 5, 5000, 0, 0, 0, 0, 0],
    [13, "WL", "node_loads", 6, 5000, 0, 0, 0, 0, 0],
    
    # Seismic Load (EQ) - Node Loads
    [15, "EQ", "node_loads", 5, 10000, 0, 0, 0, 0, 0],
    [16, "EQ", "node_loads", 6, 10000, 0, 0, 0, 0, 0]
]

# Load combinations
load_combinations = [
    # Combo1: 1.4DL
    [1, "combo1", "DL", 1.4],
    
    # Combo2: 1.2DL + 1.6LL
    [2, "combo2", "DL", 1.2],
    [3, "combo2", "LL", 1.6],
    
    # Combo3: 1.2DL + 1.0LL + 1.0WL
    [4, "combo3", "DL", 1.2],
    [5, "combo3", "LL", 1.0],
    [6, "combo3", "WL", 1.0],
    
    # Combo4: 1.2DL + 1.0LL + 1.0EQ
    [7, "combo4", "DL", 1.2],
    [8, "combo4", "LL", 1.0],
    [9, "combo4", "EQ", 1.0],
    
    # Combo5: 0.9DL + 1.0WL
    [10, "combo5", "DL", 0.9],
    [11, "combo5", "WL", 1.0],
    
    # Combo6: 0.9DL + 1.0EQ
    [12, "combo6", "DL", 0.9],
    [13, "combo6", "EQ", 1.0],

    # unfactored: 1.0DL + 1.0LL + 1.0WL + 1.0EQ
    [14, "unfactored", "DL", 1.0],
    [15, "unfactored", "EQ", 1.0],
    [16, "unfactored", "LL", 1.0],  # Fixed typo
    [17, "unfactored", "WL", 1.0]
]

def clean_small_values(array, tolerance=1e-6):
    """Clean up small floating point values"""
    return np.where(np.abs(array) < tolerance, 0.0, array)

def surface_element_calculator(coords, pressure):
    """
    Convert a uniformly distributed load (UDL) to equivalent nodal loads
    for a surface element with 6 DOF per node.
    """
    coords = np.array(coords)
    num_nodes = len(coords)
    nodal_loads = [np.zeros(6) for _ in range(num_nodes)]
    
    # Calculate element centroid
    centroid = np.mean(coords, axis=0)
    
    # For quadrilateral elements, use 2x2 Gauss integration
    g = 1/np.sqrt(3)
    gauss_points = [np.array([-g, -g]), np.array([g, -g]), 
                    np.array([g, g]), np.array([-g, g])]
    weights = [1.0, 1.0, 1.0, 1.0]
    
    area = 0.0
    
    for gp, w in zip(gauss_points, weights):
        xi, eta = gp[0], gp[1]
        
        # Shape functions for quadrilateral element
        N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), 
                            (1+xi)*(1+eta), (1-xi)*(1+eta)])
        
        # Derivatives with respect to natural coordinates
        dN_dxi = 0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
        dN_deta = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
        
        # Calculate Jacobian matrix components
        dx_dxi = np.dot(dN_dxi, coords[:, 0])
        dy_dxi = np.dot(dN_dxi, coords[:, 1])
        dz_dxi = np.dot(dN_dxi, coords[:, 2])
        
        dx_deta = np.dot(dN_deta, coords[:, 0])
        dy_deta = np.dot(dN_deta, coords[:, 1])
        dz_deta = np.dot(dN_deta, coords[:, 2])
        
        # Tangent vectors
        t1 = np.array([dx_dxi, dy_dxi, dz_dxi])
        t2 = np.array([dx_deta, dy_deta, dz_deta])
        
        # Normal vector (cross product)
        normal = np.cross(t1, t2)
        detJ = np.linalg.norm(normal)
        
        if detJ < 1e-10:
            continue
            
        # Normalize normal vector
        normal = normal / detJ
        
        # Accumulate area
        area += w * detJ
        
        # Pressure contribution
        p_vec = pressure * normal * w * detJ
        
        # Add to nodal forces
        for i in range(num_nodes):
            nodal_loads[i][:3] += N[i] * p_vec  # Fx, Fy, Fz
            # Clean up small values with higher precision
            nodal_loads[i] = clean_small_values(nodal_loads[i])
    
    return {
        'nodal_loads': nodal_loads,
        'area': area,
        'centroid': centroid.tolist()
    }

def process_loads(nodes, shell_elements, load_cases):
    """
    Process all loads and combine shell pressure loads with existing nodal loads
    """
    shell_dict = {shell[1]: shell for shell in shell_elements}
    
    # Initialize storage for combined loads
    combined_nodal_loads = defaultdict(lambda: defaultdict(lambda: np.zeros(6)))
    nodal_loads_from_pressure = defaultdict(lambda: defaultdict(lambda: np.zeros(6)))
    
    # First, collect all existing nodal loads
    for load in load_cases:
        if load[2] == "node_loads":
            case = load[1]
            node = load[3]
            load_vector = np.array(load[4:10], dtype=float)
            combined_nodal_loads[case][node] += load_vector
    
    # Then, process shell pressures and add them to existing loads
    for load in load_cases:
        if load[2] == "shell_pressure_loads":
            case = load[1]
            shell_tag = load[3]
            pressure = load[4]
            
            shell = shell_dict[shell_tag]
            node_tags = shell[2:6]
            node_coords = []
            
            for tag in node_tags:
                node = next(n for n in nodes if n[0] == tag)
                node_coords.append(node[1:4])
            
            # Convert pressure to nodal loads
            result = surface_element_calculator(node_coords, pressure)
            
            # Add pressure loads to existing nodal loads
            for i, tag in enumerate(node_tags):
                converted_load = result['nodal_loads'][i]
                nodal_loads_from_pressure[case][tag] += converted_load
                combined_nodal_loads[case][tag] += converted_load
    
    # Clean up all combined loads for precision
    for case in combined_nodal_loads:
        for node in combined_nodal_loads[case]:
            combined_nodal_loads[case][node] = clean_small_values(combined_nodal_loads[case][node])
    
    # Create modified load cases (only for non-shell loads)
    modified_load_cases = []
    for load in load_cases:
        if load[2] != "shell_pressure_loads":
            modified_load_cases.append(load.copy())
    
    return {
        'nodal_loads_from_pressure': dict(nodal_loads_from_pressure),
        'combined_nodal_loads': dict(combined_nodal_loads),
        'modified_load_cases': modified_load_cases
    }


def generate_report(results):
    """Generate simplified report with shell pressures and floor analysis only"""
    modified_load_cases = []
    
    # Process each load case and update nodal loads where needed
    for load in results['modified_load_cases']:
        new_load = load.copy()
        
        # Only modify node loads
        if new_load[2] == "node_loads":
            case = new_load[1]
            node = new_load[3]
            
            # If this node has combined loads, use the combined values
            if (case in results['combined_nodal_loads'] and 
                node in results['combined_nodal_loads'][case]):
                combined_loads = results['combined_nodal_loads'][case][node]
                # Update the load values with the combined loads (clean precision)
                new_load[4:10] = [float(clean_small_values(np.array([x]))[0]) for x in combined_loads]
        
        modified_load_cases.append(new_load)
    
    # Build simplified report
    report_data =  modified_load_cases
    
    
    return report_data




def calculate_center_of_mass(nodes, frame_elements, load_cases, load_combinations, report):
    # Step 1: Identify floors based on z-coordinates
    z_coords = sorted({node[3] for node in nodes if node[3] != 0})  # Exclude base level (z=0)
    floors = {i+1: z for i, z in enumerate(z_coords)}
    
    # Step 2: Filter nodes for each floor (excluding nodes with None mass)
    floor_nodes = defaultdict(list)
    for node in nodes:
        z = node[3]
        if z in floors.values() and node[4] is not None:  # Only nodes with mass
            floor_num = [k for k, v in floors.items() if v == z][0]
            floor_nodes[floor_num].append(node)
    
    # Step 3: Filter frame elements for each floor (connected to floor nodes)
    floor_elements = defaultdict(list)
    for elem in frame_elements:
        iNode = elem[2]
        jNode = elem[3]
        
        # Find which floor(s) these nodes belong to
        elem_floors = set()
        for floor_num, nodes_in_floor in floor_nodes.items():
            node_tags_in_floor = [n[0] for n in nodes_in_floor]
            if iNode in node_tags_in_floor or jNode in node_tags_in_floor:
                elem_floors.add(floor_num)
        
        # Add element to all relevant floors
        for floor_num in elem_floors:
            floor_elements[floor_num].append(elem)
    
    # Step 4: Get the "unfactored" load combination
    unfactored_loads = defaultdict(list)
    for combo in load_combinations:
        if combo[1] == "unfactored":
            load_case = combo[2]
            factor = combo[3]
            
            # Find all loads for this case in the report (now report is directly the list)
            for load in report:
                if load[1] == load_case:
                    unfactored_loads[load_case].append((load, factor))
    
    # Step 5: Calculate total mass and moment for each floor
    floor_com = {}
    
    for floor_num in floor_nodes:
        total_mass = 0
        weighted_x = 0
        weighted_y = 0
        weighted_z = 0
        
        # Process node masses (from node definition)
        for node in floor_nodes[floor_num]:
            x, y, z = node[1:4]
            mass_values = node[4]  # [mx, my, mz, ...], we'll use mx (assume same for all directions)
            node_mass = mass_values[0] if mass_values else 0
            total_mass += node_mass
            weighted_x += node_mass * x
            weighted_y += node_mass * y
            weighted_z += node_mass * z
        
        # Process loads from unfactored combination (convert to mass equivalents)
        for load_case, loads in unfactored_loads.items():
            for load_info, factor in loads:
                load_type = load_info[2]
                
                if load_type == "node_loads":
                    node_tag = load_info[3]
                    # Check if this node is on current floor
                    if node_tag in [n[0] for n in floor_nodes[floor_num]]:
                        # Use Fz component only (assuming vertical loads contribute to mass)
                        fz = load_info[7] * factor
                        # Convert force to mass equivalent (divide by g=1.0 m/s²)
                        mass_equivalent = fz / 1.0
                        node = next(n for n in nodes if n[0] == node_tag)
                        x, y, z = node[1:4]
                        total_mass += mass_equivalent
                        weighted_x += mass_equivalent * x
                        weighted_y += mass_equivalent * y
                        weighted_z += mass_equivalent * z
                
                elif load_type == "element_uniform_loads" and load_info[1] in [elem[1] for elem in floor_elements[floor_num]]:
                    # Uniform loads on frame elements - convert to nodal loads then mass
                    elem_tag = load_info[3]
                    wy = load_info[4] * factor  # Uniform load in y direction
                    wz = load_info[5] * factor  # Uniform load in z direction
                    
                    # Find the element
                    elem = next(e for e in frame_elements if e[1] == elem_tag)
                    iNode_tag, jNode_tag = elem[2], elem[3]
                    
                    # Get node coordinates
                    iNode = next(n for n in nodes if n[0] == iNode_tag)
                    jNode = next(n for n in nodes if n[0] == jNode_tag)
                    
                    # Calculate element length
                    L = np.sqrt((jNode[1]-iNode[1])**2 + 
                               (jNode[2]-iNode[2])**2 + 
                               (jNode[3]-iNode[3])**2)
                    
                    # Convert uniform load to equivalent nodal loads (simple beam theory)
                    # We're only interested in vertical (z) components for mass calculation
                    fz_i = wz * L / 2
                    fz_j = wz * L / 2
                    
                    # Convert to mass equivalents
                    mass_i = fz_i / 1.0
                    mass_j = fz_j / 1.0
                    
                    total_mass += mass_i + mass_j
                    weighted_x += mass_i * iNode[1] + mass_j * jNode[1]
                    weighted_y += mass_i * iNode[2] + mass_j * jNode[2]
                    weighted_z += mass_i * iNode[3] + mass_j * jNode[3]
                
                elif load_type == "element_point_loads" and load_info[1] in [elem[1] for elem in floor_elements[floor_num]]:
                    # Point loads on frame elements - convert to nodal loads then mass
                    elem_tag = load_info[3]
                    fx = load_info[4] * factor
                    fy = load_info[5] * factor
                    fz = load_info[6] * factor
                    xi = load_info[7]  # Relative position along element
                    
                    # Find the element
                    elem = next(e for e in frame_elements if e[1] == elem_tag)
                    iNode_tag, jNode_tag = elem[2], elem[3]
                    
                    # Get node coordinates
                    iNode = next(n for n in nodes if n[0] == iNode_tag)
                    jNode = next(n for n in nodes if n[0] == jNode_tag)
                    
                    # Convert point load to equivalent nodal loads (simple beam theory)
                    # We're only interested in vertical (z) components for mass calculation
                    fz_i = fz * (1 - xi)
                    fz_j = fz * xi
                    
                    # Convert to mass equivalents
                    mass_i = fz_i / 1.0
                    mass_j = fz_j / 1.0
                    
                    total_mass += mass_i + mass_j
                    weighted_x += mass_i * iNode[1] + mass_j * jNode[1]
                    weighted_y += mass_i * iNode[2] + mass_j * jNode[2]
                    weighted_z += mass_i * iNode[3] + mass_j * jNode[3]
        
        # Calculate center of mass
        if total_mass > 0:
            com_x = weighted_x / total_mass
            com_y = weighted_y / total_mass
            com_z = weighted_z / total_mass
        else:
            com_x = com_y = com_z = 0
        
        floor_com[floor_num] = {
            'total_mass': total_mass,
            'com_x': com_x,
            'com_y': com_y,
            'com_z': com_z,
            'floor_z': floors[floor_num]
        }
    
    return floor_com
# Process all loads
# results = process_loads(nodes, shell_elements, load_cases)



# # Generate simplified report
# report = generate_report(results)
# print(100*"-")
# print(100*"-")
# print("=== STRUCTURAL ANALYSIS REPORT ===")
# print(report)

# # Calculate center of mass for each floor
# floor_com_results = calculate_center_of_mass(nodes, frame_elements, load_cases, load_combinations, report)
# print(100*"-")
# print(100*"-")
# print("floor_com_results:")
# print(floor_com_results)
