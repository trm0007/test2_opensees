

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



def add_new_shells(mesh_elements, node_ids_dict, add_shell):
    """
    Add new shells to the mesh based on node IDs
    
    Parameters:
    -----------
    mesh_elements: dict
        Dictionary containing the current mesh elements
    node_ids_dict: dict
        Dictionary mapping node IDs to their coordinates {id: [x, y, z]}
    add_shell: dict
        Dictionary with shell names as keys and lists of node IDs as values
        
    Returns:
    --------
    dict: Updated mesh elements with new shells added
    """
    updated_mesh = mesh_elements.copy()
    
    # Determine starting mesh ID
    mesh_counter = 1 if not mesh_elements else max(elem_data['id'] for elem_data in mesh_elements.values()) + 1
    
    # Process each shell to be added
    for shell_name, node_ids in add_shell.items():
        # Determine element type based on number of points
        if len(node_ids) == 3:  # Triangle
            element_id = mesh_counter
            mesh_name = f"T{element_id}"
            shell_type = 'triangle'
        elif len(node_ids) == 4:  # Quadrilateral
            element_id = mesh_counter
            mesh_name = f"R{element_id}"
            shell_type = 'rectangle'
        else:
            print(f"Warning: Shell {shell_name} has {len(node_ids)} points, only triangles (3) and quads (4) are supported")
            continue
        
        # Extract coordinates for the node IDs
        coords = []
        valid_nodes = []
        
        for node_id in node_ids:
            if node_id in node_ids_dict:
                valid_nodes.append(node_id)
                coords.append(node_ids_dict[node_id])
            else:
                print(f"Warning: Node ID {node_id} not found in mesh nodes")
        
        # Only create shell if we have all required nodes
        if len(valid_nodes) == len(node_ids):
            updated_mesh[shell_name] = {
                'type': shell_type,
                'nodes': valid_nodes,
                'coordinates': coords,
                'id': element_id
            }
            mesh_counter += 1
    
    return updated_mesh


def remove_shells(mesh_elements, remove_shell):
    """
    Remove specified shells from the mesh
    
    Parameters:
    -----------
    mesh_elements: dict
        Dictionary containing the current mesh elements
    remove_shell: list
        List of shell names to be removed
        
    Returns:
    --------
    dict: Updated mesh elements with specified shells removed
    """
    final_mesh = mesh_elements.copy()
    
    for shell_name in remove_shell:
        if shell_name in final_mesh:
            del final_mesh[shell_name]
            print(f"Removed shell {shell_name}")
        else:
            print(f"Warning: Shell {shell_name} not found in mesh")
    
    return final_mesh


def sort_nodes_anticlockwise(nodes, coords):
    """
    Sort nodes in anti-clockwise order around their centroid.
    
    Args:
        nodes: List of node IDs
        coords: List of corresponding 3D coordinates
        
    Returns:
        Tuple of (sorted_nodes, sorted_coords)
    """
    if len(nodes) <= 3:  # Triangles are already planar
        return nodes, coords
        
    # Calculate centroid
    centroid = np.mean(coords, axis=0)
    
    # Center the points
    centered = np.array(coords) - centroid
    
    # Compute normal vector using first 3 points
    normal = np.cross(centered[1] - centered[0], centered[2] - centered[0])
    normal = normal / np.linalg.norm(normal)
    
    # Create basis vectors for projection
    if abs(normal[0]) > 0.1 or abs(normal[1]) > 0.1:
        u = np.array([normal[1], -normal[0], 0])  # Orthogonal to normal in XY plane
    else:
        u = np.array([0, normal[2], -normal[1]])  # Orthogonal to normal in YZ plane
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Project points to 2D plane
    projected = []
    for point in centered:
        x_proj = np.dot(point, u)
        y_proj = np.dot(point, v)
        projected.append((x_proj, y_proj))
    
    # Calculate angles from centroid and sort
    angles = np.arctan2([p[1] for p in projected], [p[0] for p in projected])
    positive_angles = np.where(angles < 0, angles + 2*np.pi, angles)
    sorted_indices = np.argsort(positive_angles)
    
    # Return sorted nodes and coordinates
    sorted_nodes = [nodes[i] for i in sorted_indices]
    sorted_coords = [coords[i] for i in sorted_indices]
    
    return sorted_nodes, sorted_coords


def convert_surface_configurations(surface_configurations, node_ids_dict):
    """
    Convert surface configurations to use coordinates from node IDs
    
    Parameters:
    -----------
    surface_configurations: dict
        Dictionary with surface configuration data
    node_ids_dict: dict
        Dictionary mapping node IDs to coordinates {id: [x, y, z]}
    
    Returns:
    --------
    dict: Converted surface configurations
    """
    new_configurations = {}
    for key, config in surface_configurations.items():
        point_ids = config.get("points", [])
        if isinstance(point_ids[0], list):
            continue  # skip if already converted to coordinates

        points = [node_ids_dict[node_id] for node_id in point_ids if node_id in node_ids_dict]
        updated_points = [[x, y, z] for x, y, z in points]
        
        predefined_ids = list(config.get("predefined_points", []))
        updated_predefined = {
            node_id: np.array(node_ids_dict[node_id])
            for node_id in predefined_ids
            if node_id in node_ids_dict
        }

        new_configurations[key] = {
            "points": updated_points,
            "add_shell": config["add_shell"],
            "remove_shell": config["remove_shell"],
            "predefined_points": updated_predefined,
            "num_x_div": config["num_x_div"],
            "num_y_div": config["num_y_div"],
        }
    return new_configurations


def find_existing_node_id(new_coord, node_ids_dict, tolerance=1e-6):
    """Check if a node with the same coordinates already exists."""
    for node_id, coord in node_ids_dict.items():
        if np.linalg.norm(np.array(new_coord) - np.array(coord)) < tolerance:
            return node_id
    return None


def create_proper_mesh_for_closed_area_3d1(points, predefined_points, num_x_div, num_y_div, numbering, add_shell, remove_shell):
    """
    Creates a proper mesh for a closed 3D area using only node IDs (no names) and no JSON dependencies.
    
    Parameters:
        points: List of 3D points defining the closed area
        predefined_points: Dictionary of {node_id: coordinates} for predefined points
        num_x_div: Number of divisions in x-direction
        num_y_div: Number of divisions in y-direction
        numbering: Numbering offset for nodes and elements
        add_shell: Dictionary of additional shells to add
        remove_shell: List of shell IDs to remove
    
    Returns:
        Dictionary containing:
        - 'elements': Dictionary of generated shell elements
        - 'nodes': Dictionary of node coordinates
    """
    
    # Calculate ID offsets based on numbering parameter
    node_id_offset = 10000 + (numbering - 1) * 1000
    element_id_offset = 10000 + (numbering - 1) * 1000
    
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
        return [(np.dot(p - p0, u), np.dot(p - p0, v)) for p in points_3d]
    
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
    
    # Function to add or find node
    def add_or_find_node(point_3d):
        nonlocal node_counter
        # Check if this 3D point already exists
        for internal_id, (existing_node_id, existing_point) in node_positions.items():
            if np.linalg.norm(point_3d - existing_point) < 1e-6:
                return existing_node_id, existing_point, False
        
        # Node doesn't exist, create new one
        node_id = node_counter + node_id_offset
        node_positions[node_counter] = (node_id, point_3d)
        node_counter += 1
        return node_id, point_3d, True
    
    # Function to process clipped polygon
    def process_polygon(poly, poly_type='clipped'):
        nonlocal mesh_counter
        
        if not isinstance(poly, ShapelyPolygon):
            return
            
        ext_coords = list(poly.exterior.coords)
        
        if len(ext_coords) < 3:  # Need at least 3 points
            return
            
        # Convert back to 3D coordinates
        node_indices = []
        coords_3d = []
        for coord in ext_coords[:-1]:  # Skip last point (same as first)
            x_proj, y_proj = coord
            point_3d = p0 + x_proj * u + y_proj * v
            node_id, coord_3d, _ = add_or_find_node(point_3d)
            node_indices.append(node_id)
            coords_3d.append(coord_3d)
        
        # Handle simple polygons (triangles/rectangles) vs complex ones
        if len(node_indices) <= 4:
            element_id = mesh_counter + element_id_offset
            elem_type = 'quad' if len(node_indices) == 4 else 'tri'
            mesh_elements[element_id] = {
                'type': elem_type,
                'nodes': node_indices,
                'coordinates': coords_3d,
                'id': element_id
            }
            mesh_counter += 1
        else:
            # Triangulate complex polygon
            triangles = triangulate(poly)
            for triangle in triangles:
                tri_coords = list(triangle.exterior.coords)
                tri_node_indices = []
                tri_coords_3d = []
                
                for coord in tri_coords[:-1]:
                    x_proj, y_proj = coord
                    point_3d = p0 + x_proj * u + y_proj * v
                    node_id, coord_3d, _ = add_or_find_node(point_3d)
                    tri_node_indices.append(node_id)
                    tri_coords_3d.append(coord_3d)
                
                element_id = mesh_counter + element_id_offset
                mesh_elements[element_id] = {
                    'type': 'tri',
                    'nodes': tri_node_indices,
                    'coordinates': tri_coords_3d,
                    'id': element_id
                }
                mesh_counter += 1
    
    # First pass: create mesh from grid
    for i in range(num_x_div):
        for j in range(num_y_div):
            x1 = min_x + i * x_step
            x2 = x1 + x_step
            y1 = min_y + j * y_step
            y2 = y1 + y_step
            
            # Create rectangle in plane coordinates and clip it
            rect = ShapelyPolygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            clipped = rect.intersection(main_poly)
            
            if clipped.is_empty:
                continue
                
            if isinstance(clipped, MultiPolygon):
                for poly in clipped.geoms:
                    process_polygon(poly)
            else:
                process_polygon(clipped)
    
    # Second pass: triangulate remaining areas
    covered_area = ShapelyPolygon()
    for mesh in mesh_elements.values():
        projected = project_to_plane(mesh['coordinates'])
        covered_area = covered_area.union(ShapelyPolygon(projected))
    
    remaining_area = main_poly.difference(covered_area)
    
    if not remaining_area.is_empty:
        if isinstance(remaining_area, MultiPolygon):
            for poly in remaining_area.geoms:
                process_polygon(poly, 'remaining')
        else:
            process_polygon(remaining_area, 'remaining')
    
    # Create a mapping of all node IDs to their coordinates
    node_coords = {node_id: coord for internal_id, (node_id, coord) in node_positions.items()}
    
    # Find closest mesh points to predefined points
    closest_mesh_points = {}
    for predefined_id, predefined_point in predefined_points.items():
        min_dist = float('inf')
        closest_node = None
        for node_id, mesh_point in node_coords.items():
            dist = np.linalg.norm(np.array(predefined_point) - np.array(mesh_point))
            if dist < min_dist:
                min_dist = dist
                closest_node = node_id
        if closest_node is not None:
            closest_mesh_points[predefined_id] = closest_node
    
    # Replace closest mesh points with predefined points
    for p_id, node_id in closest_mesh_points.items():
        predefined_point = predefined_points[p_id]
        # Update node_coords
        node_coords[node_id] = predefined_point
        # Update coordinates in mesh elements
        for elem in mesh_elements.values():
            for i, n in enumerate(elem['nodes']):
                if n == node_id:
                    elem['coordinates'][i] = predefined_point
    
    # Clean up unused nodes (except predefined points)
    used_nodes = set()
    for element in mesh_elements.values():
        used_nodes.update(element['nodes'])
    
    predefined_node_ids = set(closest_mesh_points.values())
    unused_nodes = set(node_coords.keys()) - used_nodes
    
    for node_id in unused_nodes:
        if node_id not in predefined_node_ids:
            del node_coords[node_id]
    
    # Sort nodes anti-clockwise for all elements
    for elem_id, elem_data in mesh_elements.items():
        nodes = elem_data['nodes']
        coords = elem_data['coordinates']
        
        if len(nodes) >= 3:
            sorted_nodes, sorted_coords = sort_nodes_anticlockwise(nodes, coords)
            elem_data['nodes'] = sorted_nodes
            elem_data['coordinates'] = sorted_coords

    # Handle additional shells
    if add_shell:
        for shell_id, shell_points in add_shell.items():
            # Convert points to coordinates if needed
            if isinstance(shell_points[0], int):  # Assume these are node IDs
                shell_coords = [node_coords[pid] for pid in shell_points]
            else:  # Assume these are coordinates
                shell_coords = shell_points
                # Create new nodes for these coordinates
                shell_points = []
                for coord in shell_coords:
                    node_id = max(node_coords.keys()) + 1 if node_coords else 1
                    node_coords[node_id] = coord
                    shell_points.append(node_id)
            
            element_id = max(mesh_elements.keys()) + 1 if mesh_elements else 1
            elem_type = 'quad' if len(shell_points) == 4 else 'tri'
            mesh_elements[element_id] = {
                'type': elem_type,
                'nodes': shell_points,
                'coordinates': shell_coords,
                'id': element_id
            }
    
    # Handle shell removal
    if remove_shell:
        for shell_id in remove_shell:
            if shell_id in mesh_elements:
                del mesh_elements[shell_id]
    
    return {
        'elements': mesh_elements,
        'nodes': node_coords
    }



def generate_building_model(num_bays_x, num_bays_y, num_floors, bay_width_x, bay_width_y, story_heights, num_x_div=1, num_y_div=1):
    """
    Generate a complete building model with nodes, beams, and floor slabs.
    Returns frame nodes and shell nodes separately.
    
    Parameters:
        num_bays_x: Number of bays in x-direction
        num_bays_y: Number of bays in y-direction
        num_floors: Number of floors
        bay_width_x: List of bay widths in x-direction
        bay_width_y: List of bay widths in y-direction
        story_heights: List of story heights
        num_x_div: Number of mesh divisions in x-direction for slabs (default=1)
        num_y_div: Number of mesh divisions in y-direction for slabs (default=1)
    
    Returns:
        Dictionary containing:
        - 'frame_nodes': Dictionary of frame node coordinates {node_id: [x,y,z]}
        - 'beams': Dictionary of beam elements {beam_id: [node1, node2]}
        - 'shells': List of shell element dictionaries for each floor with their own nodes
    """
    
    # Validate input dimensions
    if len(bay_width_x) != num_bays_x:
        raise ValueError("Length of bay_width_x must match num_bays_x")
    if len(bay_width_y) != num_bays_y:
        raise ValueError("Length of bay_width_y must match num_bays_y")
    if len(story_heights) != num_floors:
        raise ValueError("Length of story_heights must match num_floors")
    
    # Frame node generation (for beams and columns)
    frame_nodes = {}
    node_id = 1
    for floor in range(num_floors + 1):
        z = sum(story_heights[:floor]) if floor > 0 else 0
        for i in range(num_bays_x + 1):
            for j in range(num_bays_y + 1):
                frame_nodes[node_id] = [
                    sum(bay_width_x[:i]), 
                    sum(bay_width_y[:j]), 
                    z
                ]
                node_id += 1

    # Beam generation
    beams = {}
    beam_id = 1
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

    # Create floor slabs with separate shell nodes
    shells = []
    shell_node_start_id = node_id  # Start shell nodes after frame nodes
    
    for floor in range(1, num_floors + 1):
        z_level = sum(story_heights[:floor])
        
        # Get corner nodes for this floor
        corner_nodes = [
            [0, 0, z_level],
            [sum(bay_width_x), 0, z_level],
            [sum(bay_width_x), sum(bay_width_y), z_level],
            [0, sum(bay_width_y), z_level]
        ]
        
        # Generate mesh for this floor with separate node IDs
        floor_mesh = create_proper_mesh_for_closed_area_3d1(
            points=corner_nodes,
            predefined_points={},  # No predefined points for basic floor
            num_x_div=num_x_div,
            num_y_div=num_y_div,
            numbering=shell_node_start_id,  # Start numbering shell nodes separately
            add_shell={},  # No additional shells
            remove_shell=[]  # No shells to remove
        )
        
        # Update shell node start ID for next floor
        if floor_mesh['nodes']:
            shell_node_start_id = max(floor_mesh['nodes'].keys()) + 1
        
        # Add floor slab to results (keeping shell nodes separate)
        shells.append({
            'floor_level': floor,
            'elements': floor_mesh['elements'],
            'nodes': floor_mesh['nodes']  # These are separate from frame_nodes
        })

    return {
        'frame_nodes': frame_nodes,  # Only frame nodes (for beams/columns)
        'beams': beams,
        'shells': shells  # Each shell has its own nodes
    }


def consolidate_duplicate_nodes(building, tolerance=1e-10):
    """
    Consolidates duplicate nodes between frame and shell nodes by finding nodes with identical coordinates
    and updating references to use the frame node IDs when possible.
    
    Modified version for separate frame and shell nodes structure.
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
    
    # Create mapping from shell nodes to frame nodes
    node_mapping = {}
    
    # Process each floor slab
    for floor_data in building['shells']:
        # Check all shell nodes against frame nodes
        for shell_node_id, shell_coords in floor_data['nodes'].items():
            # Look for matching node in frame nodes
            for frame_node_id, frame_coords in building['frame_nodes'].items():
                if coords_equal(shell_coords, frame_coords):
                    node_mapping[shell_node_id] = frame_node_id
                    break
        
        # Update shell elements with consolidated node IDs
        for element_id, shell_data in floor_data['elements'].items():
            # Update node references
            shell_data['nodes'] = [
                node_mapping.get(node_id, node_id) 
                for node_id in shell_data['nodes']
            ]
            
            # Update coordinates array to match the consolidated nodes
            updated_coordinates = []
            for node_id in shell_data['nodes']:
                if node_id in building['frame_nodes']:
                    # Use coordinates from frame nodes
                    coords = building['frame_nodes'][node_id]
                    if hasattr(coords, 'tolist'):
                        coords = coords.tolist()
                    updated_coordinates.append(np.array(coords))
                else:
                    # Use coordinates from shell nodes (for unmapped nodes)
                    coords = floor_data['nodes'][node_id]
                    if hasattr(coords, 'tolist'):
                        coords = coords.tolist()
                    updated_coordinates.append(np.array(coords))
            
            shell_data['coordinates'] = updated_coordinates
        
        # Remove consolidated nodes from floor_data
        nodes_to_remove = [node_id for node_id in node_mapping.keys() 
                          if node_id in floor_data['nodes']]
        for node_id in nodes_to_remove:
            del floor_data['nodes'][node_id]
    
    return building


def create_and_visualize_model(building, num_bays_x, num_bays_y, num_floors, bay_width_x, bay_width_y, story_heights, num_x_div=1, num_y_div=1, output_dir=None, filename=None):
    # Generate the building model with separate frame and shell nodes


    # Consolidate duplicate nodes (nodes that exist in both frame and shells)
    building = consolidate_duplicate_nodes(building)

    # Create OpenSees model
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    # Add ALL nodes first (both frame and shell)
    all_nodes = {}
    all_nodes.update(building['frame_nodes'])
    
    # Add shell nodes from each floor
    for floor_data in building['shells']:
        all_nodes.update(floor_data['nodes'])
    
    # Create all nodes in OpenSees
    for node_id, coords in all_nodes.items():
        ops.node(node_id, *coords)
        print(f"Created node {node_id} at {coords}")

    # Define geometric transformations
    vecxz_col = [1, 0, 1]  # Local x-z plane for columns
    ops.geomTransf('Linear', 1, *vecxz_col)
    vecxz_beamx = [0, 1, 1]  # Local x-z plane for X-direction beams
    ops.geomTransf('Linear', 2, *vecxz_beamx)
    vecxz_beamy = [1, 0, 1]  # Local x-z plane for Y-direction beams
    ops.geomTransf('Linear', 3, *vecxz_beamy)
    
    # Define material properties
    E = 200e9       # Pa (200 GPa for steel)
    G = 77e9        # Pa (77 GPa for steel)
    A = 0.01        # m² (100 cm² cross-sectional area)
    J = 1e-5        # m⁴ (torsional constant)
    Iy = 1e-5       # m⁴ (moment of inertia about local y-axis)
    Iz = 1e-5       # m⁴ (moment of inertia about local z-axis)
    
    # Add beam elements with proper transformations
    for beam_id, nodes in building['beams'].items():
        node1_coords = building['frame_nodes'][nodes[0]]
        node2_coords = building['frame_nodes'][nodes[1]]
        
        # Determine transformation based on orientation
        if abs(node1_coords[2] - node2_coords[2]) > 0.1:  # Column (vertical)
            transfTag = 1
        elif abs(node1_coords[0] - node2_coords[0]) > 0.1:  # Beam in X direction
            transfTag = 2
        else:  # Beam in Y direction
            transfTag = 3
            
        ops.element('elasticBeamColumn', beam_id, *nodes, A, E, G, J, Iy, Iz, transfTag)

    # Define shell material and section properties
    ops.nDMaterial('ElasticIsotropic', 10, 30000000000.0, 0.2)  # E=30GPa, v=0.2
    ops.section('PlateFiber', 20, 10, 0.15)  # 15cm thick shell

    # Process shell elements
    if 'shells' in building:
        shell_floors = building['shells']
        print(f"Found {len(shell_floors)} floor slabs in building model")

        for floor_data in shell_floors:
            floor_level = floor_data.get('floor_level', 'unknown')
            print(f"\nProcessing floor {floor_level}:")
            
            shell_elements = floor_data['elements']
            shell_nodes = floor_data['nodes']
            
            print(f"Floor {floor_level} has {len(shell_elements)} shell elements")
            print(f"Floor {floor_level} has {len(shell_nodes)} nodes")

            # Create shell elements
            for shell_id, shell_data in shell_elements.items():
                nodes = shell_data['nodes']
                
                # Verify all nodes exist
                missing_nodes = [n for n in nodes if n not in all_nodes]
                if missing_nodes:
                    print(f"Error: Shell {shell_id} references missing nodes: {missing_nodes}")
                    continue
                    
                try:
                    if shell_data['type'] == 'quad' and len(nodes) == 4:
                        ops.element('ShellMITC4', shell_id, *nodes, 20)
                        print(f"Created quad shell {shell_id} with nodes {nodes}")
                    elif shell_data['type'] == 'tri' and len(nodes) == 3:
                        ops.element('ShellDKGQ', shell_id, *nodes, 20)
                        print(f"Created tri shell {shell_id} with nodes {nodes}")
                    else:
                        print(f"Warning: Shell {shell_id} has invalid configuration - "
                            f"type: {shell_data['type']}, nodes: {nodes}")
                except Exception as e:
                    print(f"Failed to create shell {shell_id}: {str(e)}")
                    continue

    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot shell elements
    shell_elements = ops.getEleTags()
    for ele_tag in shell_elements:
        ele_nodes = ops.eleNodes(ele_tag)
        node_coords = np.array([ops.nodeCoord(node) for node in ele_nodes])
        
        poly = Poly3DCollection([node_coords], alpha=0.5, linewidth=1, edgecolor='k')
        poly.set_facecolor('yellow')
        ax.add_collection3d(poly)

    # Overlay the frame model
    opsv.plot_model(element_labels=1, node_labels=1, ax=ax, fmt_model={'color': 'k', 'linewidth': 1})

    # Save plot
    folder = "STRUCTURAL_MODEL_FOLDER"
    os.makedirs(folder, exist_ok=True)
    filepath = f"{filename}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    return building, filepath


def modify_building_model(building, add_nodes, delete_nodes, add_beams, delete_beams, add_shells, delete_shells):
    """
    Modifies a building model dictionary based on user-provided modifications.
    
    Args:
        building: The base building model dictionary
        add_nodes: Dict of {node_id: [x,y,z]} to add
        delete_nodes: List of node IDs to remove
        add_beams: Dict of {beam_id: [i_node,j_node]} to add
        delete_beams: List of beam IDs to remove
        add_shells: Dict of {shell_id: [i,j,k,l]} to add
        delete_shells: List of shell IDs to remove
    
    Returns:
        Modified building dictionary
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
    building.setdefault('shells', [{}])
    
    # Helper function to convert ID to int if possible
    def to_int(id):
        try:
            return int(id)
        except (ValueError, TypeError):
            return id
    
    # 1. Process node modifications
    try:
        # Add new nodes
        for node_id, coords in add_nodes.items():
            node_id = to_int(node_id)
            if not isinstance(coords, (list, tuple)) or len(coords) != 3:
                raise ValueError(f"Coordinates for node {node_id} must be [x,y,z]")
            building['frame_nodes'][node_id] = [float(x) for x in coords]
        
        # Delete nodes
        for node_id in delete_nodes:
            node_id = to_int(node_id)
            if node_id in building['frame_nodes']:
                del building['frame_nodes'][node_id]
                # Remove beams connected to deleted nodes
                building['beams'] = {
                    beam_id: nodes for beam_id, nodes in building['beams'].items() 
                    if node_id not in nodes
                }
    except Exception as e:
        raise ValueError(f"Node modification error: {str(e)}")
    
    # 2. Process beam modifications
    try:
        # Add new beams
        for beam_id, nodes in add_beams.items():
            beam_id = to_int(beam_id)
            if not isinstance(nodes, (list, tuple)) or len(nodes) != 2:
                raise ValueError(f"Beam {beam_id} must connect exactly 2 nodes")
            if any(n not in building['frame_nodes'] for n in nodes):
                raise ValueError(f"Beam {beam_id} references non-existent nodes")
            building['beams'][beam_id] = [to_int(n) for n in nodes]
        
        # Delete beams
        for beam_id in delete_beams:
            beam_id = to_int(beam_id)
            if beam_id in building['beams']:
                del building['beams'][beam_id]
    except Exception as e:
        raise ValueError(f"Beam modification error: {str(e)}")
    
    # 3. Process shell modifications
    try:
        if building['shells']:
            shell_data = building['shells'][0]
            shell_data.setdefault('elements', {})
            shell_data.setdefault('nodes', {})
            
            # Add new shells
            for shell_id, nodes in add_shells.items():
                shell_id = to_int(shell_id)
                if not isinstance(nodes, (list, tuple)) or len(nodes) != 4:
                    raise ValueError(f"Shell {shell_id} must connect exactly 4 nodes")
                if any(n not in building['frame_nodes'] for n in nodes):
                    raise ValueError(f"Shell {shell_id} references non-existent nodes")
                
                shell_data['elements'][shell_id] = {
                    'type': 'quad',
                    'nodes': [to_int(n) for n in nodes],
                    'coordinates': [building['frame_nodes'].get(n, [0.0,0.0,0.0]) for n in nodes],
                    'id': shell_id
                }
                
                # Update shell nodes
                for node_id in nodes:
                    node_id = to_int(node_id)
                    if node_id in building['frame_nodes']:
                        shell_data['nodes'][node_id] = building['frame_nodes'][node_id]
            
            # Delete shells
            for shell_id in delete_shells:
                shell_id = to_int(shell_id)
                if shell_id in shell_data['elements']:
                    del shell_data['elements'][shell_id]
    except Exception as e:
        raise ValueError(f"Shell modification error: {str(e)}")
    
    return building