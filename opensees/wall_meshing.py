

from collections import defaultdict
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from shapely.ops import triangulate
import openseespy.opensees as ops
import opsvis as opsv
from django.conf import settings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
from datetime import datetime


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import os
import tempfile

import numpy as np
from scipy.spatial import ConvexHull






def collect_shell_nodes(building):
    """Extract all unique nodes from shell elements"""
    shell_nodes = set()
    
    for floor_data in building.get('shells', []):
        for elem_id, elem_data in floor_data.get('elements', {}).items():
            nodes = elem_data.get('nodes', [])
            shell_nodes.update(nodes)
    
    return shell_nodes

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
        shell_node_ids = collect_shell_nodes(building)
        
        # Add shell nodes that aren't already in frame_nodes
        for node_id in shell_node_ids:
            if node_id not in all_nodes:
                # For missing shell nodes, we need to find their coordinates
                # Check if building has 'all_nodes' which contains both frame and shell nodes
                if 'all_nodes' in building and node_id in building['all_nodes']:
                    all_nodes[node_id] = building['all_nodes'][node_id]
                else:
                    # Try to find coordinates in shell floor data
                    found = False
                    for floor_data in building.get('shells', []):
                        if node_id in floor_data.get('nodes', {}):
                            all_nodes[node_id] = np.array(floor_data['nodes'][node_id])
                            found = True
                            break
                    
                    if not found:
                        print(f"Warning: Shell node {node_id} not found in frame_nodes or shell node data")
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

def convert_beams_format(beams_dict):
    """
    Convert beams from dictionary format to the desired list format.
    Input format: {beam_id: [iNode, jNode], ...}
    Output format: [["forceBeamColumn", element_tag, iNode, jNode, transformation_tag, integration_tag], ...]
    """
    beams_list = []
    
    for beam_id, (i_node, j_node) in beams_dict.items():
        beams_list.append([
            "forceBeamColumn",
            int(beam_id),
            int(i_node),
            int(j_node),
            1,  # transformation tag (default)
            1   # integration tag (default)
        ])
    
    return beams_list

def convert_shells_format(shells_list):
    """
    Convert shells from the building format to the desired list format.
    Input format: [{'floor_level': x, 'elements': {elem_id: {'nodes': [...], 'type': '...', ...}, ...}}, ...]
    Output format: [["ShellMITC4", element_tag, node1, node2, node3, node4, section_tag], ...]
    """
    shells_formatted = []
    
    for floor_data in shells_list:
        for elem_id, elem_data in floor_data.get('elements', {}).items():
            nodes = elem_data.get('nodes', [])
            
            # Ensure we have 4 nodes for ShellMITC4 elements
            if len(nodes) >= 4:
                shells_formatted.append([
                    "ShellMITC4",
                    int(elem_id),
                    int(nodes[0]),
                    int(nodes[1]),
                    int(nodes[2]),
                    int(nodes[3]),
                    20  # section tag (default)
                ])
            else:
                print(f"Warning: Shell element {elem_id} has {len(nodes)} nodes, expected 4")
    
    return shells_formatted

# def convert_building_format(building):
#     """
#     Convert entire building data to the desired format with shell nodes included in frame_nodes
#     """
#     # Convert nodes including shell nodes
#     converted_frame_nodes = convert_frame_nodes(building.get('frame_nodes', {}), building)
    
#     # Convert beams to frame elements format
#     converted_beams = convert_beams_format(building.get('beams', {}))
    
#     # Convert shells to the new format
#     converted_shells = convert_shells_format(building.get('shells', []))
    
#     # Create the converted structure
#     converted = {
#         'nodes': converted_frame_nodes,
#         'frame_elements': converted_beams,
#         'shell_elements': converted_shells,
#         # Preserve original data for backward compatibility
#         'frame_nodes': building.get('frame_nodes', {}),
#         'beams': building.get('beams', {}),
#         'shells': building.get('shells', []),
#         'image_filepaths': building.get('image_filepaths', [])
#     }
    
#     return converted

def synchronize_element_coordinates(elements, all_nodes):
    """
    Ensures element coordinates match their node references
    """
    for elem_id, elem_data in elements.items():
        updated_coords = []
        for node_id in elem_data['nodes']:
            if node_id in all_nodes:
                coord = all_nodes[node_id]
                # Convert numpy arrays to lists for consistency
                if hasattr(coord, 'tolist'):
                    coord = coord.tolist()
                updated_coords.append(coord)
            else:
                raise ValueError(f"Node {node_id} not found in all_nodes")
        elem_data['coordinates'] = updated_coords
    return elements

def get_unified_id_offsets(floor_number):
    """
    Centralized ID offset calculation
    """
    node_id_offset = 1000 + (floor_number - 1) * 1000
    element_id_offset = 10000 + (floor_number - 1) * 1000
    opensees_shell_offset = 100000 + (floor_number - 1) * 1000
    return node_id_offset, element_id_offset, opensees_shell_offset

def validate_building_consistency(building):
    """
    Validates that building model is internally consistent
    """
    issues = []
    
    # Check that all beam nodes exist in frame_nodes
    for beam_id, nodes in building['beams'].items():
        for node_id in nodes:
            if node_id not in building['frame_nodes']:
                issues.append(f"Beam {beam_id} references non-existent node {node_id}")
    
    # Check shell element consistency
    for floor_idx, floor_data in enumerate(building['shells']):
        for elem_id, elem_data in floor_data['elements'].items():
            # Check node count matches coordinate count
            if len(elem_data['nodes']) != len(elem_data['coordinates']):
                issues.append(f"Floor {floor_idx}, Element {elem_id}: node count mismatch")
            
            # Check that all nodes exist somewhere
            for node_id in elem_data['nodes']:
                if (node_id not in building['all_nodes'] and 
                    node_id not in floor_data['nodes']):
                    issues.append(f"Floor {floor_idx}, Element {elem_id}: references missing node {node_id}")
    
    if issues:
        print("Building consistency issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    return True

def validate_coordinates_match(elements, all_nodes):
    """
    Validates that element coordinates match node coordinates
    """
    for elem_id, elem_data in elements.items():
        for i, node_id in enumerate(elem_data['nodes']):
            if node_id in all_nodes:
                stored_coord = elem_data['coordinates'][i]
                actual_coord = all_nodes[node_id]
                
                # Convert to numpy for comparison
                stored = np.array(stored_coord)
                actual = np.array(actual_coord)
                
                if not np.allclose(stored, actual, atol=1e-6):
                    print(f"WARNING: Element {elem_id}, Node {node_id} coordinate mismatch:")
                    print(f"  Stored: {stored_coord}")
                    print(f"  Actual: {actual_coord}")
                    return False
    return True

def plot_unified_shell_elements(building, ax):
    """
    Unified plotting function that uses consistent coordinate source
    """
    for floor_data in building['shells']:
        floor_elements = floor_data.get('elements', {})
        
        for shell_id, shell_data in floor_elements.items():
            try:
                # Get coordinates from the unified all_nodes dictionary
                node_coords = []
                for node_id in shell_data['nodes']:
                    if node_id in building['all_nodes']:
                        coord = building['all_nodes'][node_id]
                        if hasattr(coord, 'tolist'):
                            coord = coord.tolist()
                        node_coords.append(coord)
                    else:
                        # Fallback to stored coordinates if node not in all_nodes
                        coord_idx = shell_data['nodes'].index(node_id)
                        if coord_idx < len(shell_data['coordinates']):
                            node_coords.append(shell_data['coordinates'][coord_idx])
                
                if len(node_coords) >= 3:
                    poly = Poly3DCollection([node_coords], alpha=0.5, linewidth=1, edgecolor='k')
                    poly.set_facecolor('lightblue')
                    ax.add_collection3d(poly)
                    
            except Exception as e:
                print(f"Error plotting shell element {shell_id}: {e}")



def generate_building_model(num_bays_x, num_bays_y, num_floors, bay_width_x, bay_width_y,
                          story_heights, num_x_div, num_y_div, 
                          add_nodes=None, add_beams=None, add_shells=None,
                          remove_nodes=None, remove_beams=None, remove_shells=None,
                          save_plots=True, output_dir="building_plots", 
                          show_plots=False, dpi=300, figsize=(12, 8)):
    """
    Generate complete building model with nodes, beams, and floor slabs with meshing.
    
    Args:
        num_bays_x: Number of bays in X direction
        num_bays_y: Number of bays in Y direction  
        num_floors: Number of floors
        bay_width_x: List of bay widths in X direction
        bay_width_y: List of bay widths in Y direction
        story_heights: List of story heights
        num_x_div: Number of mesh divisions in X
        num_y_div: Number of mesh divisions in Y
        add_nodes: Dictionary of {node_id: [x,y,z]} to add additional nodes
        add_beams: Dictionary of {beam_id: [node1, node2]} to add additional beams
        add_shells: Dictionary of {shell_id: [node1, node2, node3, node4]} to add additional shells
        remove_nodes: List of node IDs to remove
        remove_beams: List of beam IDs to remove
        remove_shells: List of shell IDs to remove
        save_plots: Boolean, whether to save floor mesh plots
        output_dir: String, directory to save plot images
        show_plots: Boolean, whether to display plots
        dpi: Integer, resolution for saved images
        figsize: Tuple, figure size in inches
    
    Returns:
        Dictionary containing:
        - frame_nodes: Original structural nodes (integer IDs)
        - all_nodes: Combined frame and shell nodes (integer IDs)
        - beams: Beam connectivity (integer IDs)
        - shells: Shell elements data as LIST format
        - image_filepaths: List of paths to saved mesh plot images
    """
    # Initialize default values for optional parameters
    add_nodes = add_nodes or {}
    add_beams = add_beams or {}
    add_shells = add_shells or {}
    remove_nodes = remove_nodes or []
    remove_beams = remove_beams or []
    remove_shells = remove_shells or []
    print("generate_building_model activated")
    # Initialize list to store image filepaths
    image_filepaths = []

    # Validate input dimensions
    if len(bay_width_x) != num_bays_x:
        raise ValueError("Length of bay_width_x must match num_bays_x")
    if len(bay_width_y) != num_bays_y:
        raise ValueError("Length of bay_width_y must match num_bays_y")
    if len(story_heights) != num_floors:
        raise ValueError("Length of story_heights must match num_floors")

    # Initialize data structures with integer IDs
    frame_nodes = {}
    beams = {}
    shells = []
    all_nodes = {}
    node_id = 1
    beam_id = 1

    # 1. Generate frame nodes with integer IDs
    for floor in range(num_floors + 1):
        z = sum(story_heights[:floor]) if floor > 0 else 0
        for i in range(num_bays_x + 1):
            for j in range(num_bays_y + 1):
                frame_nodes[node_id] = np.array([
                    sum(bay_width_x[:i]), 
                    sum(bay_width_y[:j]), 
                    z
                ])
                all_nodes[node_id] = frame_nodes[node_id]
                node_id += 1

    # 2. Generate beams with integer IDs
    nodes_per_floor = (num_bays_x + 1) * (num_bays_y + 1)
    
    for floor in range(num_floors + 1):
        # Vertical members (columns)
        if floor < num_floors:
            for i in range(num_bays_x + 1):
                for j in range(num_bays_y + 1):
                    lower_node = 1 + floor * nodes_per_floor + i * (num_bays_y + 1) + j
                    upper_node = lower_node + nodes_per_floor
                    beams[beam_id] = [lower_node, upper_node]
                    beam_id += 1
        
        # Horizontal members (beams)
        if floor <= num_floors:
            # X-direction beams
            for i in range(num_bays_x):
                for j in range(num_bays_y + 1):
                    node1 = 1 + floor * nodes_per_floor + i * (num_bays_y + 1) + j
                    node2 = node1 + (num_bays_y + 1)
                    beams[beam_id] = [node1, node2]
                    beam_id += 1
            
            # Y-direction beams
            for i in range(num_bays_x + 1):
                for j in range(num_bays_y):
                    node1 = 1 + floor * nodes_per_floor + i * (num_bays_y + 1) + j
                    node2 = node1 + 1
                    beams[beam_id] = [node1, node2]
                    beam_id += 1

    # 3. Generate floor meshes and shells
    for floor in range(1, num_floors + 1):
        # Get all nodes for this floor
        start_id = 1 + floor * nodes_per_floor
        end_id = start_id + nodes_per_floor
        
        # Create predefined_points dictionary for this floor
        predefined_points = {
            node_id: frame_nodes[node_id]
            for node_id in range(start_id, end_id)
            if node_id in frame_nodes
        }
        
        # Get boundary points using convex hull
        floor_points = list(predefined_points.values())
        points_array = np.array(floor_points)
        hull = ConvexHull(points_array[:, :2])  # 2D hull using x,y coordinates
        boundary_points = [floor_points[i] for i in hull.vertices]
        print("create_proper_mesh_for_closed_area_3d running")
        # Generate mesh for this floor with plotting
        floor_mesh, image_filepath = create_proper_mesh_for_closed_area_3d(
            points=boundary_points,
            predefined_points=predefined_points,
            num_x_div=num_x_div,
            num_y_div=num_y_div,
            floor_number=floor,
            save_plot=save_plots,
            output_dir=output_dir,
            show_plot=show_plots,
            dpi=dpi,
            figsize=figsize
        )
        print("create_proper_mesh_for_closed_area_3d successful")
        # Add image filepath to list if it was saved
        if image_filepath:
            image_filepaths.append(image_filepath)
        
        # Create shell data
        floor_shell_data = {
            'floor_level': floor,
            'elements': {},
            'nodes': {}
        }
        
        # Process elements ensuring 'coordinates' exists
        for elem_id, elem_data in floor_mesh['elements'].items():
            elem_id_int = int(elem_id)
            
            # Ensure coordinates exist in element data
            if 'coordinates' not in elem_data:
                # If coordinates not provided, get them from nodes
                elem_coords = []
                for node_id in elem_data['nodes']:
                    if node_id in floor_mesh['nodes']:
                        elem_coords.append(floor_mesh['nodes'][node_id]['coordinates'])
                    elif node_id in predefined_points:
                        elem_coords.append(predefined_points[node_id])
                elem_data['coordinates'] = elem_coords
            
            floor_shell_data['elements'][elem_id_int] = {
                'type': elem_data['type'],
                'nodes': [int(node_id) for node_id in elem_data['nodes']],
                'coordinates': [list(coord) for coord in elem_data['coordinates']]
            }
        
        # Process nodes
        for node_id, node_data in floor_mesh['nodes'].items():
            node_id_int = int(node_id)
            all_nodes[node_id_int] = node_data['coordinates']
            floor_shell_data['nodes'][node_id_int] = node_data['coordinates']
        
        shells.append(floor_shell_data)

    # 4. Handle additional nodes, beams, and shells
    # Add additional nodes
    for node_id, coords in add_nodes.items():
        node_id_int = int(node_id)
        frame_nodes[node_id_int] = np.array(coords)
        all_nodes[node_id_int] = np.array(coords)
    
    # Add additional beams
    for beam_id, nodes in add_beams.items():
        beam_id_int = int(beam_id)
        nodes_int = [int(n) for n in nodes]
        # Validate nodes exist
        for node_id in nodes_int:
            if node_id not in all_nodes:
                raise ValueError(f"Node {node_id} referenced in beam {beam_id} does not exist")
        beams[beam_id_int] = nodes_int
    
    # Add additional shells (added to first floor by default)
    if add_shells and len(shells) > 0:
        first_floor = shells[0]
        for shell_id, nodes in add_shells.items():
            shell_id_int = int(shell_id)
            nodes_int = [int(n) for n in nodes]
            # Validate nodes exist
            for node_id in nodes_int:
                if node_id not in all_nodes:
                    raise ValueError(f"Node {node_id} referenced in shell {shell_id} does not exist")
            
            # Get coordinates for the shell
            shell_coords = [all_nodes[n] for n in nodes_int]
            
            first_floor['elements'][shell_id_int] = {
                'type': 'quad' if len(nodes_int) == 4 else 'triangle',
                'nodes': nodes_int,
                'coordinates': shell_coords
            }
            
            # Add nodes to floor if not already present
            for node_id in nodes_int:
                if node_id not in first_floor['nodes']:
                    first_floor['nodes'][node_id] = all_nodes[node_id]

    # 5. Handle removals
    # Remove nodes
    for node_id in remove_nodes:
        node_id_int = int(node_id)
        if node_id_int in frame_nodes:
            del frame_nodes[node_id_int]
        if node_id_int in all_nodes:
            del all_nodes[node_id_int]
        
        # Remove from shells
        for floor_data in shells:
            if node_id_int in floor_data['nodes']:
                del floor_data['nodes'][node_id_int]
            
            # Remove elements that reference this node
            elements_to_remove = []
            for elem_id, elem_data in floor_data['elements'].items():
                if node_id_int in elem_data['nodes']:
                    elements_to_remove.append(elem_id)
            
            for elem_id in elements_to_remove:
                del floor_data['elements'][elem_id]
    
    # Remove beams
    for beam_id in remove_beams:
        beam_id_int = int(beam_id)
        if beam_id_int in beams:
            del beams[beam_id_int]
    
    # Remove shells
    for shell_id in remove_shells:
        shell_id_int = int(shell_id)
        for floor_data in shells:
            if shell_id_int in floor_data['elements']:
                del floor_data['elements'][shell_id_int]

    # result = {
    #     'frame_nodes': frame_nodes,
    #     'all_nodes': all_nodes,
    #     'beams': beams,
    #     'shells': shells,
    #     'image_filepaths': image_filepaths
    # }

    # Ensure all shell elements have consistent coordinates
    for floor_data in shells:
        floor_data['elements'] = synchronize_element_coordinates(
            floor_data['elements'], 
            all_nodes
        )


    result = {
        'frame_nodes': frame_nodes,
        'all_nodes': all_nodes,
        'beams': beams,
        'shells': shells,
        'image_filepaths': image_filepaths
    }
    # Validate the building model
    validate_building_consistency(result)

    return result



def consolidate_duplicate_nodes(building, tolerance=1e-10):
    """
    Consolidates duplicate nodes between frame and shell nodes by finding nodes with identical coordinates
    and updating references to use the frame node IDs when possible.
    
    Now works with LIST format shells and ensures integer IDs.
    """
    import numpy as np
    
    def coords_equal(coord1, coord2, tol=tolerance):
        """Check if two coordinates are equal within tolerance"""
        # Convert to numpy arrays for easier comparison
        if hasattr(coord1, 'tolist'):
            coord1 = coord1.tolist()
        if hasattr(coord2, 'tolist'):
            coord2 = coord2.tolist()
        
        arr1 = np.array(coord1)
        arr2 = np.array(coord2)
        return np.allclose(arr1, arr2, atol=tol)
    
    # Create mapping from shell nodes to frame nodes (integer IDs)
    node_mapping = {}
    
    # Process each floor slab (LIST format)
    for floor_data in building['shells']:
        # Check all shell nodes against frame nodes
        for shell_node_id, shell_coords in floor_data['nodes'].items():
            shell_node_id = int(shell_node_id)  # Ensure integer
            # Look for matching node in frame nodes
            for frame_node_id, frame_coords in building['frame_nodes'].items():
                frame_node_id = int(frame_node_id)  # Ensure integer
                if coords_equal(shell_coords, frame_coords):
                    node_mapping[shell_node_id] = frame_node_id
                    break
        
        # Update shell elements with consolidated node IDs (integers)
        for element_id, shell_data in floor_data['elements'].items():
            # Update node references with integer IDs
            shell_data['nodes'] = [
                int(node_mapping.get(int(node_id), int(node_id))) 
                for node_id in shell_data['nodes']
            ]
            
            # Synchronize coordinates after node consolidation
            shell_data = synchronize_element_coordinates(
                {element_id: shell_data},
                building['all_nodes']
            )[element_id]
        
        # Remove consolidated nodes from floor_data (integer IDs)
        nodes_to_remove = [int(node_id) for node_id in node_mapping.keys() 
                          if int(node_id) in floor_data['nodes']]
        for node_id in nodes_to_remove:
            del floor_data['nodes'][node_id]
    
    return building



def create_structural_image_from_output_data(output_data, predefined_points=None, show_element_ids=True, show_node_ids=True):
    """
    Creates a 3D structural visualization from the output data of create_proper_mesh_for_closed_area_3d.
    
    Args:
        output_data: The output dictionary from create_proper_mesh_for_closed_area_3d
        predefined_points: Dictionary of predefined points (if not included in output_data)
        show_element_ids: Whether to display element IDs
        show_node_ids: Whether to display node IDs
    
    Returns:
        str: Filepath to the saved image
    """
    # Extract data from output
    elements = output_data["elements"]
    nodes = output_data["nodes"]
    
    # Get predefined points if not provided
    if predefined_points is None:
        predefined_points = {node_id: node_data["coordinates"] 
                           for node_id, node_data in nodes.items() 
                           if node_data.get("is_predefined", False)}
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Structural Mesh Visualization')
    
    # Plot mesh elements
    for element_id, elem_data in elements.items():
        if elem_data['type'] == 'rectangle':
            color = 'cyan'
            alpha = 0.4
        else:
            color = 'lightgreen'
            alpha = 0.6
        
        verts = [elem_data['coordinates'] + [elem_data['coordinates'][0]]]
        poly = Poly3DCollection(verts, alpha=alpha, facecolors=color, 
                              edgecolors='blue', linewidths=1)
        ax.add_collection3d(poly)
        
        if show_element_ids:
            centroid = np.mean(elem_data['coordinates'], axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], str(element_id), 
                    ha='center', va='center', fontsize=8, weight='bold')
    
    # Plot nodes
    replaced_nodes = set(predefined_points.keys())
    for node_id, node_data in nodes.items():
        coord = node_data["coordinates"]
        
        if node_id in predefined_points:
            # Plot predefined points
            ax.scatter([coord[0]], [coord[1]], [coord[2]], c='blue', s=100, marker='*')
            if show_node_ids:
                ax.text(coord[0], coord[1], coord[2], str(node_id), 
                        ha='left', va='top', fontsize=10, color='darkblue', weight='bold')
        else:
            # Plot regular nodes
            ax.scatter([coord[0]], [coord[1]], [coord[2]], c='red', s=50)
            if show_node_ids:
                ax.text(coord[0], coord[1], coord[2], str(node_id), 
                        ha='right', va='bottom', fontsize=8, color='darkred')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Create legend
    rect_patch = mpatches.Patch(color='cyan', label='Rectangular Elements')
    tri_patch = mpatches.Patch(color='lightgreen', label='Triangular Elements')
    node_patch = mpatches.Patch(color='red', label='Nodes (non-predefined)')
    predef_patch = mpatches.Patch(color='blue', label='Predefined Points')
    ax.legend(handles=[rect_patch, tri_patch, node_patch, predef_patch])
    
    plt.tight_layout()
    
    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"structural_mesh_{os.getpid()}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return filepath




def create_proper_mesh_for_closed_area_3d(points, predefined_points, num_x_div=4, num_y_div=4, floor_number=1, 
                                         save_plot=True, output_dir="mesh_plots", 
                                         show_plot=False, dpi=300, figsize=(12, 8)):
    """
    Create a 3D mesh for a closed area with optional plotting and saving.
    
    Parameters:
    -----------
    points : list
        List of 3D points defining the closed area
    predefined_points : dict
        Dictionary of predefined points
    num_x_div : int, default=4
        Number of divisions in x direction
    num_y_div : int, default=4
        Number of divisions in y direction
    floor_number : int, default=1
        Floor number for ID offset calculation
    save_plot : bool, default=True
        Whether to save the plot as an image
    output_dir : str, default="mesh_plots"
        Directory to save the plot image
    show_plot : bool, default=False
        Whether to display the plot
    dpi : int, default=300
        Resolution for saved image
    figsize : tuple, default=(12, 8)
        Figure size in inches
    
    Returns:
    --------
    tuple : (output_data, image_filepath)
        output_data: Dictionary containing mesh elements and nodes
        image_filepath: Path to saved image file (None if save_plot=False)
    """
    
    # Calculate ID offsets using unified function
    node_id_offset, element_id_offset, _ = get_unified_id_offsets(floor_number)
    
    # Calculate the plane equation ax + by + cz + d = 0
    p0, p1, p2 = np.array(points[0]), np.array(points[1]), np.array(points[2])
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = -np.dot(normal, p0)
    
    # Find two orthogonal vectors in the plane (basis vectors)
    if abs(a) > 0.1 or abs(b) > 0.1:
        u = np.array([b, -a, 0])  # Orthogonal to normal in XY plane
    else:
        u = np.array([0, c, -b])  # Orthogonal to normal in YZ plane
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Function to project 3D points to 2D plane coordinates
    def project_to_plane(points_3d):
        projected = []
        for p in points_3d:
            vec = p - p0
            x_proj = np.dot(vec, u)
            y_proj = np.dot(vec, v)
            projected.append((x_proj, y_proj))
        return projected
    
    # Function to ensure counter-clockwise ordering
    def ensure_counter_clockwise(nodes, coords):
        if len(nodes) < 3:
            return nodes, coords
        
        # Calculate normal vector for the polygon
        if len(nodes) == 3:
            # For triangles
            v1 = np.array(coords[1]) - np.array(coords[0])
            v2 = np.array(coords[2]) - np.array(coords[0])
            cross = np.cross(v1, v2)
            normal = cross  # For triangles, use the cross product as the normal
        else:
            # For polygons with more than 3 points
            # Use Newell's method to compute normal
            normal = np.zeros(3)
            for i in range(len(coords)):
                current = np.array(coords[i])
                next_point = np.array(coords[(i+1)%len(coords)])
                normal[0] += (current[1] - next_point[1]) * (current[2] + next_point[2])
                normal[1] += (current[2] - next_point[2]) * (current[0] + next_point[0])
                normal[2] += (current[0] - next_point[0]) * (current[1] + next_point[1])
            cross = normal
        
        # Project onto plane normal to check winding
        dot_product = np.dot(cross, normal)
        
        # If winding is clockwise (dot product negative), reverse the order
        if dot_product < 0:
            nodes = nodes[::-1]
            coords = coords[::-1]
        
        return nodes, coords
    
    # Project original points to 2D plane coordinates
    points_2d = project_to_plane(points)
    main_poly = ShapelyPolygon(points_2d)
    
    # Get bounding box of the polygon in plane coordinates
    min_x, min_y, max_x, max_y = main_poly.bounds
    
    # Calculate step sizes
    x_step = (max_x - min_x) / num_x_div
    y_step = (max_y - min_y) / num_y_div
    
    # Create dictionaries to store mesh and node information
    mesh_elements = {}
    node_positions = {}  # Stores {internal_id: (actual_node_id, coordinates)}
    node_counter = 1
    mesh_counter = 1
    
    # First pass: create rectangular elements clipped to the polygon
    for i in range(num_x_div):
        for j in range(num_y_div):
            x1 = min_x + i * x_step
            x2 = x1 + x_step
            y1 = min_y + j * y_step
            y2 = y1 + y_step
            
            # Create rectangle in plane coordinates and clip it
            rect = ShapelyPolygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            clipped = rect.intersection(main_poly)
            
            if clipped.is_empty or not isinstance(clipped, (ShapelyPolygon, MultiPolygon)):
                continue
                
            if isinstance(clipped, MultiPolygon):
                polygons = list(clipped.geoms)
            else:
                polygons = [clipped]
            
            for poly in polygons:
                if not isinstance(poly, ShapelyPolygon):
                    continue
                    
                ext_coords = list(poly.exterior.coords)
                
                if len(ext_coords) >= 3:  # At least 3 points needed for a polygon
                    # Convert back to 3D coordinates
                    node_indices = []
                    coords_3d = []
                    for coord in ext_coords[:-1]:
                        x_proj, y_proj = coord
                        point_3d = p0 + x_proj * u + y_proj * v
                        
                        # Check if this 3D point already exists
                        found = False
                        for internal_id, (existing_node_id, existing_point) in node_positions.items():
                            if np.linalg.norm(point_3d - existing_point) < 1e-6:
                                node_indices.append(existing_node_id)
                                coords_3d.append(existing_point)
                                found = True
                                break
                        
                        if not found:
                            node_id = node_counter + node_id_offset
                            node_positions[node_counter] = (node_id, point_3d)
                            node_indices.append(node_id)
                            coords_3d.append(point_3d)
                            node_counter += 1
                    
                    # Ensure counter-clockwise ordering
                    node_indices, coords_3d = ensure_counter_clockwise(node_indices, coords_3d)
                    
                    # Handle polygons with more than 4 points by triangulating them
                    if len(node_indices) > 4:
                        # Convert to 2D coordinates for triangulation
                        poly_2d = ShapelyPolygon(ext_coords)
                        triangles = triangulate(poly_2d)
                        
                        for triangle in triangles:
                            tri_coords = list(triangle.exterior.coords)
                            tri_node_indices = []
                            tri_coords_3d = []
                            
                            for coord in tri_coords[:-1]:
                                x_proj, y_proj = coord
                                point_3d = p0 + x_proj * u + y_proj * v
                                
                                # Find or create nodes for this triangle
                                found = False
                                for nid, coord_3d in zip(node_indices, coords_3d):
                                    if np.linalg.norm(point_3d - coord_3d) < 1e-6:
                                        tri_node_indices.append(nid)
                                        tri_coords_3d.append(coord_3d)
                                        found = True
                                        break
                                
                                if not found:
                                    node_id = node_counter + node_id_offset
                                    node_positions[node_counter] = (node_id, point_3d)
                                    tri_node_indices.append(node_id)
                                    tri_coords_3d.append(point_3d)
                                    node_counter += 1
                            
                            # Ensure counter-clockwise ordering for triangles
                            tri_node_indices, tri_coords_3d = ensure_counter_clockwise(tri_node_indices, tri_coords_3d)
                            
                            element_id = mesh_counter + element_id_offset
                            mesh_elements[element_id] = {
                                'type': 'triangle',
                                'nodes': tri_node_indices,
                                'coordinates': tri_coords_3d,
                                'id': element_id
                            }
                            mesh_counter += 1
                    else:
                        element_id = mesh_counter + element_id_offset
                        elem_type = 'rectangle' if len(node_indices) == 4 else 'triangle'
                        mesh_elements[element_id] = {
                            'type': elem_type,
                            'nodes': node_indices,
                            'coordinates': coords_3d,
                            'id': element_id
                        }
                        mesh_counter += 1
    
    # Second pass: triangulate remaining areas
    covered_area = ShapelyPolygon()
    for mesh in mesh_elements.values():
        projected = project_to_plane(mesh['coordinates'])
        covered_area = covered_area.union(ShapelyPolygon(projected))
    
    remaining_area = main_poly.difference(covered_area)
    
    if not remaining_area.is_empty and isinstance(remaining_area, (ShapelyPolygon, MultiPolygon)):
        if isinstance(remaining_area, MultiPolygon):
            remaining_polys = list(remaining_area.geoms)
        else:
            remaining_polys = [remaining_area]
        
        for poly in remaining_polys:
            if not isinstance(poly, ShapelyPolygon):
                continue
                
            ext_coords = list(poly.exterior.coords)
            coords = ext_coords[:-1]
            
            # Check if this is a simple polygon we can handle
            if len(coords) <= 4:
                # Handle as either triangle or rectangle
                node_indices = []
                coords_3d = []
                for coord in coords:
                    x_proj, y_proj = coord
                    point_3d = p0 + x_proj * u + y_proj * v
                    
                    found = False
                    for internal_id, (existing_node_id, existing_point) in node_positions.items():
                        if np.linalg.norm(point_3d - existing_point) < 1e-6:
                            node_indices.append(existing_node_id)
                            coords_3d.append(existing_point)
                            found = True
                            break
                    
                    if not found:
                        node_id = node_counter + node_id_offset
                        node_positions[node_counter] = (node_id, point_3d)
                        node_indices.append(node_id)
                        coords_3d.append(point_3d)
                        node_counter += 1
                
                # Ensure counter-clockwise ordering
                node_indices, coords_3d = ensure_counter_clockwise(node_indices, coords_3d)
                
                element_id = mesh_counter + element_id_offset
                elem_type = 'rectangle' if len(node_indices) == 4 else 'triangle'
                mesh_elements[element_id] = {
                    'type': elem_type,
                    'nodes': node_indices,
                    'coordinates': coords_3d,
                    'id': element_id
                }
                mesh_counter += 1
            else:
                # Complex polygon - triangulate it
                triangles = triangulate(poly)
                for triangle in triangles:
                    tri_coords = list(triangle.exterior.coords)
                    tri_node_indices = []
                    tri_coords_3d = []
                    
                    for coord in tri_coords[:-1]:
                        x_proj, y_proj = coord
                        point_3d = p0 + x_proj * u + y_proj * v
                        
                        # Find or create nodes for this triangle
                        found = False
                        for internal_id, (existing_node_id, existing_point) in node_positions.items():
                            if np.linalg.norm(point_3d - existing_point) < 1e-6:
                                tri_node_indices.append(existing_node_id)
                                tri_coords_3d.append(existing_point)
                                found = True
                                break
                        
                        if not found:
                            node_id = node_counter + node_id_offset
                            node_positions[node_counter] = (node_id, point_3d)
                            tri_node_indices.append(node_id)
                            tri_coords_3d.append(point_3d)
                            node_counter += 1
                    
                    # Ensure counter-clockwise ordering for triangles
                    tri_node_indices, tri_coords_3d = ensure_counter_clockwise(tri_node_indices, tri_coords_3d)
                    
                    element_id = mesh_counter + element_id_offset
                    mesh_elements[element_id] = {
                        'type': 'triangle',
                        'nodes': tri_node_indices,
                        'coordinates': tri_coords_3d,
                        'id': element_id
                    }
                    mesh_counter += 1
    
    # Create a comprehensive node mapping system
    all_node_coords = {}  # Maps all node IDs to coordinates
    node_replacement_map = {}  # Maps old node IDs to new node IDs

    # First, populate all_node_coords with mesh nodes
    for internal_id, (node_id, coord) in node_positions.items():
        all_node_coords[node_id] = coord

    # Find and map closest mesh nodes to predefined nodes
    for predefined_id, predefined_coord in predefined_points.items():
        min_dist = float('inf')
        closest_node_id = None
        
        for node_id, mesh_coord in all_node_coords.items():
            dist = np.linalg.norm(np.array(predefined_coord) - np.array(mesh_coord))
            if dist < min_dist and dist < 1e-3:  # Tolerance for "close enough"
                min_dist = dist
                closest_node_id = node_id
        
        if closest_node_id is not None:
            # Map the mesh node to the predefined node
            node_replacement_map[closest_node_id] = predefined_id
            # Update coordinates to exact predefined location
            all_node_coords[predefined_id] = predefined_coord
            # Remove the old mesh node
            if closest_node_id in all_node_coords:
                del all_node_coords[closest_node_id]

    # Update all mesh elements with new node mappings
    for element_id, elem_data in mesh_elements.items():
        # Update node references
        updated_nodes = []
        updated_coords = []
        
        for node_id in elem_data['nodes']:
            # Use replacement if available, otherwise keep original
            final_node_id = node_replacement_map.get(node_id, node_id)
            updated_nodes.append(final_node_id)
            updated_coords.append(all_node_coords[final_node_id])
        
        elem_data['nodes'] = updated_nodes
        elem_data['coordinates'] = updated_coords

    # Update the final output
    node_names = all_node_coords  # This now contains the correct mapping

    # Create the output structure with elements and nodes including IDs
    output_data = {
        "elements": {},
        "nodes": {},
    }

    # Add all nodes with their coordinates and IDs
    for node_id, coord in node_names.items():
        output_data["nodes"][node_id] = {
            "id": node_id,
            "coordinates": [float(coord[0]), float(coord[1]), float(coord[2])],
            "is_predefined": node_id in predefined_points
        }

    # Add all elements in the requested format with IDs
    for element_id, elem_data in mesh_elements.items():
        output_data["elements"][element_id] = {
            "id": elem_data['id'],
            "type": elem_data['type'],
            "nodes": elem_data['nodes'],
            "coordinates": [[float(coord[0]), float(coord[1]), float(coord[2])] 
                           for coord in elem_data['coordinates']]
        }

    # PLOTTING SECTION
    image_filepath = None
    
    if save_plot or show_plot:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D Mesh Elements with Nodes (excluding predefined points)')
        
        # Plot original shape
        original_verts = [points + [points[0]]]
        original_poly = Poly3DCollection(original_verts, alpha=0.3, 
                                       facecolors='red', linewidths=2, 
                                       edgecolors='red', linestyles='--')
        ax.add_collection3d(original_poly)
        
        # Use the output_data structure for plotting
        # Get all node coordinates from output_data
        node_coords = {}
        for node_id, node_data in output_data['nodes'].items():
            node_coords[node_id] = np.array(node_data['coordinates'])
        
        # Identify predefined nodes
        predefined_node_ids = set()
        for node_id, node_data in output_data['nodes'].items():
            if node_data.get('is_predefined', False):
                predefined_node_ids.add(node_id)
        
        # Plot mesh elements using output_data
        for element_id, elem_data in output_data['elements'].items():
            if elem_data['type'] == 'rectangle':
                color = 'cyan'
                alpha = 0.4
            else:
                color = 'lightgreen'
                alpha = 0.6
            
            # Get coordinates from the element data
            coordinates = [np.array(coord) for coord in elem_data['coordinates']]
            verts = [coordinates + [coordinates[0]]]
            poly = Poly3DCollection(verts, alpha=alpha, facecolors=color, 
                                  edgecolors='blue', linewidths=1)
            ax.add_collection3d(poly)
            
            centroid = np.mean(coordinates, axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], element_id, 
                    ha='center', va='center', fontsize=8, weight='bold')
            
            # Plot nodes that are not predefined
            for node_id in elem_data['nodes']:
                if node_id not in predefined_node_ids:
                    coord = node_coords[node_id]
                    ax.scatter([coord[0]], [coord[1]], [coord[2]], c='red', s=50)
                    ax.text(coord[0], coord[1], coord[2], f'{node_id}', 
                            ha='right', va='bottom', fontsize=8, color='darkred')
        
        # Plot predefined points
        for node_id in predefined_node_ids:
            coord = node_coords[node_id]
            ax.scatter([coord[0]], [coord[1]], [coord[2]], c='blue', s=100, marker='*')
            ax.text(coord[0], coord[1], coord[2], f'{node_id}', 
                    ha='left', va='top', fontsize=10, color='darkblue', weight='bold')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        rect_patch = mpatches.Patch(color='cyan', label='Rectangular Elements')
        tri_patch = mpatches.Patch(color='lightgreen', label='Triangular Elements')
        node_patch = mpatches.Patch(color='red', label='Nodes (non-predefined)')
        predef_patch = mpatches.Patch(color='blue', label='Predefined Points')
        ax.legend(handles=[rect_patch, tri_patch, node_patch, predef_patch])
        
        plt.tight_layout()
        
        if save_plot:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate filename with timestamp

            filename = f"mesh_floor_{floor_number}.png"
            image_filepath = os.path.join(output_dir, filename)
            
            # Save the plot
            plt.savefig(image_filepath, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # Close figure if not showing to save memory

    return output_data, image_filepath

def create_and_visualize_model(building, num_bays_x, num_bays_y, num_floors, 
                             bay_width_x, bay_width_y, story_heights, 
                             num_x_div=1, num_y_div=1, output_dir=None, filename=None):
    """
    Create and visualize the complete building model in OpenSees
    
    Now works with LIST format shells and integer IDs consistently.
    """
    # 1. Consolidate duplicate nodes
    building = consolidate_duplicate_nodes(building)

    # Validate building consistency before creating model
    if not validate_building_consistency(building):
        print("WARNING: Building model has consistency issues")

    # Synchronize all coordinates before OpenSees model creation
    for floor_data in building['shells']:
        floor_data['elements'] = synchronize_element_coordinates(
            floor_data['elements'], 
            building['all_nodes']
        )

    # 2. Initialize OpenSees model
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    # 3. Process all nodes (frame + shell) - ensure integer IDs
    all_nodes = {}
    
    # Add frame nodes (ensure integer IDs)
    for k, v in building['frame_nodes'].items():
        all_nodes[int(k)] = v
    
    # Add shell nodes from each floor (LIST format)
    for floor_data in building['shells']:
        for k, v in floor_data['nodes'].items():
            all_nodes[int(k)] = v
    
    # 4. Create nodes in OpenSees (integer IDs)
    for node_id, coords in all_nodes.items():
        try:
            node_id_int = int(node_id)  # Ensure integer for OpenSees
            clean_coords = [float(c) for c in coords]  # Ensure coordinates are floats
            ops.node(node_id_int, *clean_coords)
        except Exception as e:
            print(f"Failed to create node {node_id}: {e}")
            continue

    # 5. Define geometric transformations
    vecxz_col = [1, 0, 1]  # Local x-z plane for columns
    ops.geomTransf('Linear', 1, *vecxz_col)
    vecxz_beamx = [0, 1, 1]  # Local x-z plane for X-direction beams
    ops.geomTransf('Linear', 2, *vecxz_beamx)
    vecxz_beamy = [1, 0, 1]  # Local x-z plane for Y-direction beams
    ops.geomTransf('Linear', 3, *vecxz_beamy)
    
    # 6. Define material properties (example values - adjust as needed)
    E = 200e9       # Pa (200 GPa for steel)
    G = 77e9        # Pa (77 GPa for steel)
    A = 0.01        # m² (100 cm² cross-sectional area)
    J = 1e-5        # m⁴ (torsional constant)
    Iy = 1e-5       # m⁴ (moment of inertia about local y-axis)
    Iz = 1e-5       # m⁴ (moment of inertia about local z-axis)
    
    # 7. Create beam elements (integer IDs)
    for beam_id, nodes in building['beams'].items():
        try:
            # Ensure integer IDs
            beam_id_int = int(beam_id)
            node1, node2 = [int(n) for n in nodes]
            
            # Get coordinates for orientation check
            node1_coords = building['frame_nodes'][nodes[0]]
            node2_coords = building['frame_nodes'][nodes[1]]
            
            # Determine transformation based on orientation
            if abs(node1_coords[2] - node2_coords[2]) > 0.1:  # Column (vertical)
                transfTag = 1
            elif abs(node1_coords[0] - node2_coords[0]) > 0.1:  # X-direction beam
                transfTag = 2
            else:  # Y-direction beam
                transfTag = 3
                
            ops.element('elasticBeamColumn', beam_id_int, node1, node2, 
                       A, E, G, J, Iy, Iz, transfTag)
        except Exception as e:
            print(f"Failed to create beam {beam_id}: {e}")

    # 8. Define shell material and section
    ops.nDMaterial('ElasticIsotropic', 10, 30e9, 0.2)  # E=30GPa, v=0.2
    ops.section('PlateFiber', 20, 10, 0.15)  # 15cm thick shell

    # 9. Process shell elements (LIST format with integer IDs)
    _, _, shell_element_counter = get_unified_id_offsets(1)  # Use floor 1 as base
    if 'shells' in building:
        for floor_data in building['shells']:
            floor_level = floor_data.get('floor_level', 'unknown')
            shell_elements = floor_data.get('elements', {})
            
            for shell_id, shell_data in shell_elements.items():
                try:
                    # Ensure integer IDs
                    shell_id_int = int(shell_id)
                    nodes = [int(n) for n in shell_data['nodes']]
                    
                    if shell_data['type'] == 'rectangle' and len(nodes) == 4:
                        ops.element('ShellMITC4', shell_element_counter, *nodes, 20)
                    elif shell_data['type'] == 'triangle' and len(nodes) == 3:
                        ops.element('ShellDKGQ', shell_element_counter, *nodes, 20)
                    
                    shell_element_counter += 1
                except Exception as e:
                    print(f"Failed to create shell {shell_id}: {e}")

    # 10. Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot shell elements using unified function
    plot_unified_shell_elements(building, ax)
    
    # Plot frame model
    try:
        opsv.plot_model(
            element_labels=1, 
            node_labels=1, 
            ax=ax, 
            fmt_model={'color': 'darkred', 'linewidth': 1.5}
        )
    except Exception as e:
        print(f"Error plotting frame model: {e}")
    
    # Configure plot
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Building Model Visualization')
    
    # Save visualization
    if filename:
        os.makedirs(output_dir or "visualizations", exist_ok=True)
        filepath = os.path.join(output_dir or "visualizations", f"{filename}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        filepath = None
        plt.show()
    
    return building, filepath


def modify_building_model(building, add_nodes, delete_nodes, add_beams, delete_beams, add_shells, delete_shells):
    """
    Modifies a building model dictionary based on user-provided modifications.
    
    Now ensures all IDs are integers and works with LIST format shells.
    """
    # Validate input types
    if not isinstance(building, dict):
        raise ValueError("Building must be a dictionary")
    if not all(isinstance(x, (dict, list)) for x in [add_nodes, delete_nodes, add_beams, delete_beams, add_shells, delete_shells]):
        raise ValueError("All modification inputs must be dictionaries or lists")
    
    # Make a deep copy to avoid modifying the original
    import copy
    building = copy.deepcopy(building)
    
    # Ensure required keys exist
    building.setdefault('frame_nodes', {})
    building.setdefault('beams', {})
    building.setdefault('shells', [])
    
    # Helper function to ensure integer ID
    def to_int(id_val):
        try:
            return int(id_val)
        except (ValueError, TypeError):
            raise ValueError(f"ID must be convertible to integer: {id_val}")
    
    # 1. Process node modifications (integer IDs)
    try:
        # Add new nodes
        for node_id, coords in add_nodes.items():
            node_id_int = to_int(node_id)
            if not isinstance(coords, (list, tuple)) or len(coords) != 3:
                raise ValueError(f"Coordinates for node {node_id} must be [x,y,z]")
            building['frame_nodes'][node_id_int] = [float(x) for x in coords]
            building['all_nodes'][node_id_int] = [float(x) for x in coords]
        
        # Delete nodes (integer IDs)
        for node_id in delete_nodes:
            node_id_int = to_int(node_id)
            if node_id_int in building['frame_nodes']:
                del building['frame_nodes'][node_id_int]
                # Remove beams connected to deleted nodes
                building['beams'] = {
                    beam_id: nodes for beam_id, nodes in building['beams'].items() 
                    if node_id_int not in [int(n) for n in nodes]
                }
    except Exception as e:
        raise ValueError(f"Node modification error: {str(e)}")
    
    # 2. Process beam modifications (integer IDs)
    try:
        # Add new beams
        for beam_id, nodes in add_beams.items():
            beam_id_int = to_int(beam_id)
            if not isinstance(nodes, (list, tuple)) or len(nodes) != 2:
                raise ValueError(f"Beam {beam_id} must connect exactly 2 nodes")
            nodes_int = [to_int(n) for n in nodes]
            if any(n not in building['frame_nodes'] for n in nodes_int):
                raise ValueError(f"Beam {beam_id} references non-existent nodes")
            building['beams'][beam_id_int] = nodes_int
        
        # Delete beams (integer IDs)
        for beam_id in delete_beams:
            beam_id_int = to_int(beam_id)
            if beam_id_int in building['beams']:
                del building['beams'][beam_id_int]
    except Exception as e:
        raise ValueError(f"Beam modification error: {str(e)}")
    
    # 3. Process shell modifications (LIST format with integer IDs)
    try:
        if building['shells']:
            # Use first floor shell data or create new one
            if len(building['shells']) > 0:
                shell_data = building['shells'][0]
            else:
                shell_data = {'floor_level': 1, 'elements': {}, 'nodes': {}}
                building['shells'].append(shell_data)
                
            shell_data.setdefault('elements', {})
            shell_data.setdefault('nodes', {})
            
            # Add new shells (integer IDs)
            for shell_id, nodes in add_shells.items():
                shell_id_int = to_int(shell_id)
                if not isinstance(nodes, (list, tuple)) or len(nodes) != 4:
                    raise ValueError(f"Shell {shell_id} must connect exactly 4 nodes")
                nodes_int = [to_int(n) for n in nodes]
                if any(n not in building['frame_nodes'] for n in nodes_int):
                    raise ValueError(f"Shell {shell_id} references non-existent nodes")
                
                shell_data['elements'][shell_id_int] = {
                    'type': 'quad',
                    'nodes': nodes_int,
                    'coordinates': [building['frame_nodes'].get(n, [0.0,0.0,0.0]) for n in nodes_int],
                    'id': shell_id_int
                }
                
                # Update shell nodes (integer IDs)
                for node_id in nodes_int:
                    if node_id in building['frame_nodes']:
                        shell_data['nodes'][node_id] = building['frame_nodes'][node_id]
                        building['all_nodes'][node_id] = building['frame_nodes'][node_id]
            
            # Delete shells (integer IDs)
            for shell_id in delete_shells:
                shell_id_int = to_int(shell_id)
                if shell_id_int in shell_data['elements']:
                    del shell_data['elements'][shell_id_int]
    except Exception as e:
        raise ValueError(f"Shell modification error: {str(e)}")
    
    # Synchronize coordinates after modifications
    for floor_data in building['shells']:
        floor_data['elements'] = synchronize_element_coordinates(
            floor_data['elements'], 
            building['all_nodes']
        )
    
    return building


