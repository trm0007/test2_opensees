import os
import sys
import re
import time
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


import sys
import os
import shutil



import openseespy.opensees as ops
# import opsvis1 as opsv

import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Path3DCollection

from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from shapely.ops import triangulate
from matplotlib.tri import Triangulation
from matplotlib import colors
import matplotlib.tri as mtri

# from opensees.wall_meshing import synchronize_element_coordinates


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




def add_new_shells(mesh_elements, node_names, add_shell):
    """
    Add new shells to the mesh based on node IDs
    
    Parameters:
    -----------
    mesh_elements: dict
        Dictionary containing the current mesh elements
    node_names: dict
        Dictionary mapping node IDs to their coordinates
    add_shell: dict
        Dictionary with shell IDs as keys and lists of node IDs as values
        
    Returns:
    --------
    dict: Updated mesh elements with new shells added
    """
    # Create a copy of the input mesh_elements to avoid modifying the original
    updated_mesh = mesh_elements.copy()
    mesh_counter = max([elem_data['id'] for elem_data in mesh_elements.values()]) + 1
    
    # Process each shell to be added
    for shell_id, node_ids in add_shell.items():
        if len(node_ids) == 3:  # Triangle
            element_id = mesh_counter
            shell_type = 'triangle'
        elif len(node_ids) == 4:  # Quadrilateral
            element_id = mesh_counter
            shell_type = 'rectangle'
        else:
            print(f"Warning: Shell {shell_id} has {len(node_ids)} nodes, only triangles (3) and quads (4) are supported")
            continue
        
        # Extract node IDs and coordinates
        nodes = []
        coords = []
        
        # Find node IDs corresponding to node IDs
        for node_id in node_ids:
            found = False
            for existing_node_id, node_info in node_names.items():
                if existing_node_id == node_id:
                    found = True
                    nodes.append(node_id)
                    coords.append(node_info)
                    break
            
            if not found:
                print(f"Warning: Node ID {node_id} not found in mesh nodes")
                # You could implement logic to create new nodes here if needed
        
        # Only create shell if we have all required nodes
        if len(nodes) == len(node_ids):
            updated_mesh[element_id] = {
                'type': shell_type,
                'nodes': nodes,
                'coordinates': coords,
                'id': element_id
            }
            mesh_counter += 1
            print(f"Added shell {element_id} with ID {element_id}")
    
    return updated_mesh


def remove_shells(mesh_elements, remove_shell):
    """
    Remove specified shells from the mesh
    
    Parameters:
    -----------
    mesh_elements: dict
        Dictionary containing the current mesh elements
    remove_shell: list
        List of shell IDs to be removed
        
    Returns:
    --------
    dict: Updated mesh elements with specified shells removed
    """
    # Create a copy of the input mesh_elements to avoid modifying the original
    final_mesh = mesh_elements.copy()
    
    for shell_id in remove_shell:
        if shell_id in final_mesh:
            del final_mesh[shell_id]
            print(f"Removed shell {shell_id}")
        else:
            print(f"Warning: Shell {shell_id} not found in mesh")
    
    return final_mesh

def get_unified_id_offsets(floor_number):
    """
    Centralized ID offset calculation
    """
    node_id_offset = 1000 + (floor_number - 1) * 1000
    element_id_offset = 10000 + (floor_number - 1) * 1000
    opensees_shell_offset = 100000 + (floor_number - 1) * 1000
    return node_id_offset, element_id_offset, opensees_shell_offset

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import os
import tempfile

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

def create_proper_mesh_for_closed_area_3d(points, predefined_points, num_x_div=4, num_y_div=4, floor_number=1):
    # Calculate ID offsets based on floor_number parameter
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
    
    # [Rest of the function remains the same...]
    # (The rest of the function including plotting and JSON output is unchanged)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Mesh Elements with Nodes (excluding predefined points)')
    
    # Plot original shape
    original_verts = [points + [points[0]]]
    original_poly = Poly3DCollection(original_verts, alpha=0.3, 
                                   facecolors='red', linewidths=2, 
                                   edgecolors='red', linestyles='--')
    ax.add_collection3d(original_poly)
    
    # Plot mesh elements
    for element_id, data in mesh_elements.items():
        if data['type'] == 'rectangle':
            color = 'cyan'
            alpha = 0.4
        else:
            color = 'lightgreen'
            alpha = 0.6
        
        verts = [data['coordinates'] + [data['coordinates'][0]]]
        poly = Poly3DCollection(verts, alpha=alpha, facecolors=color, 
                              edgecolors='blue', linewidths=1)
        ax.add_collection3d(poly)
        
        centroid = np.mean(data['coordinates'], axis=0)
        ax.text(centroid[0], centroid[1], centroid[2], str(element_id), 
                ha='center', va='center', fontsize=8, weight='bold')
        
        
        # Initialize replaced_nodes as an empty set at the beginning of the plotting section
        # Create a mapping of all node IDs to their coordinates
        node_names = {node_id: coord for internal_id, (node_id, coord) in node_positions.items()}

        # Initialize replaced_nodes as an empty set
        replaced_nodes = set()

        # Find closest mesh points to predefined points
        predefined_points_list = list(predefined_points.values())
        closest_mesh_points = {}

        for i, predefined_point in enumerate(predefined_points_list):
            min_dist = float('inf')
            closest_node = None
            for node_id, mesh_point in node_names.items():
                dist = np.linalg.norm(predefined_point - mesh_point)
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node_id
            if closest_node is not None:
                predefined_ids = list(predefined_points.keys())
                closest_mesh_points[predefined_ids[i]] = closest_node

        # Replace closest mesh points with predefined points
        for predefined_id, node_id in closest_mesh_points.items():
            predefined_point = predefined_points[predefined_id]
            # Mark this node as replaced
            replaced_nodes.add(node_id)
            # Update node_names
            node_names[node_id] = predefined_point
            # Update coordinates in mesh elements after node replacement
            for elem in mesh_elements.values():
                for i, n in enumerate(elem['nodes']):
                    if n == node_id:
                        elem['coordinates'][i] = predefined_point

            # ADD THIS NEW CODE:
            # Ensure all element coordinates are synchronized
            mesh_elements = synchronize_element_coordinates(mesh_elements, node_names)

        # Only plot nodes that weren't replaced by predefined points
        for node_num in data['nodes']:
            if node_num not in replaced_nodes:
                coord = node_names[node_num]
                ax.scatter([coord[0]], [coord[1]], [coord[2]], c='red', s=50)
                ax.text(coord[0], coord[1], coord[2], str(node_num), 
                        ha='right', va='bottom', fontsize=8, color='darkred')
    
    # Plot predefined points
    for predefined_id, p_coord in predefined_points.items():
        ax.scatter([p_coord[0]], [p_coord[1]], [p_coord[2]], c='blue', s=100, marker='*')
        ax.text(p_coord[0], p_coord[1], p_coord[2], str(predefined_id), 
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
    # plt.show()
    
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
            "is_predefined": node_id in replaced_nodes
        }

    # Create a mapping of replaced node IDs to predefined point IDs
    replaced_nodes_mapping = {node_id: predefined_id for predefined_id, node_id in closest_mesh_points.items()}

    # Add all elements in the requested format with IDs
    for element_id, elem_data in mesh_elements.items():
        # Replace node names with predefined point IDs where applicable
        node_ids_in_elem = []
        for node_id in elem_data['nodes']:
            if node_id in replaced_nodes_mapping:
                node_ids_in_elem.append(replaced_nodes_mapping[node_id])
            else:
                node_ids_in_elem.append(node_id)
        
        output_data["elements"][element_id] = {
            "id": elem_data['id'],
            "type": elem_data['type'],
            "nodes": node_ids_in_elem,
            # "nodes_coordinate": [[float(coord[0]), float(coord[1]), float(coord[2])] 
            #                     for coord in elem_data['coordinates']]
        }


    return output_data

# def create_proper_mesh_for_closed_area_3d(points, predefined_points, num_x_div=4, num_y_div=4, floor_number=1):
#     """
#     Fixed version that ensures 'coordinates' key is properly included in element data
#     """
#     # Calculate ID offsets based on floor_number parameter
#     node_id_offset = 1 + (floor_number - 1) * 1000
#     element_id_offset = 10000 + (floor_number - 1) * 1000
    
#     # Calculate the plane equation ax + by + cz + d = 0
#     p0, p1, p2 = np.array(points[0]), np.array(points[1]), np.array(points[2])
#     v1 = p1 - p0
#     v2 = p2 - p0
#     normal = np.cross(v1, v2)
#     a, b, c = normal
#     d = -np.dot(normal, p0)
    
#     # Find two orthogonal vectors in the plane (basis vectors)
#     if abs(a) > 0.1 or abs(b) > 0.1:
#         u = np.array([b, -a, 0])  # Orthogonal to normal in XY plane
#     else:
#         u = np.array([0, c, -b])  # Orthogonal to normal in YZ plane
#     u = u / np.linalg.norm(u)
#     v = np.cross(normal, u)
#     v = v / np.linalg.norm(v)
    
#     # Function to project 3D points to 2D plane coordinates
#     def project_to_plane(points_3d):
#         projected = []
#         for p in points_3d:
#             vec = p - p0
#             x_proj = np.dot(vec, u)
#             y_proj = np.dot(vec, v)
#             projected.append((x_proj, y_proj))
#         return projected
    
#     # Function to ensure counter-clockwise ordering
#     def ensure_counter_clockwise(nodes, coords):
#         if len(nodes) < 3:
#             return nodes, coords
        
#         # Calculate normal vector for the polygon
#         if len(nodes) == 3:
#             # For triangles
#             v1 = np.array(coords[1]) - np.array(coords[0])
#             v2 = np.array(coords[2]) - np.array(coords[0])
#             cross = np.cross(v1, v2)
#             normal = cross  # For triangles, use the cross product as the normal
#         else:
#             # For polygons with more than 3 points
#             # Use Newell's method to compute normal
#             normal = np.zeros(3)
#             for i in range(len(coords)):
#                 current = np.array(coords[i])
#                 next_point = np.array(coords[(i+1)%len(coords)])
#                 normal[0] += (current[1] - next_point[1]) * (current[2] + next_point[2])
#                 normal[1] += (current[2] - next_point[2]) * (current[0] + next_point[0])
#                 normal[2] += (current[0] - next_point[0]) * (current[1] + next_point[1])
#             cross = normal
        
#         # Project onto plane normal to check winding
#         dot_product = np.dot(cross, normal)
        
#         # If winding is clockwise (dot product negative), reverse the order
#         if dot_product < 0:
#             nodes = nodes[::-1]
#             coords = coords[::-1]
        
#         return nodes, coords
    
#     # Project original points to 2D plane coordinates
#     points_2d = project_to_plane(points)
#     main_poly = ShapelyPolygon(points_2d)
    
#     # Get bounding box of the polygon in plane coordinates
#     min_x, min_y, max_x, max_y = main_poly.bounds
    
#     # Calculate step sizes
#     x_step = (max_x - min_x) / num_x_div
#     y_step = (max_y - min_y) / num_y_div
    
#     # Create dictionaries to store mesh and node information
#     mesh_elements = {}
#     node_positions = {}  # Stores {internal_id: (actual_node_id, coordinates)}
#     node_counter = 1
#     mesh_counter = 1
    
#     # First pass: create rectangular elements clipped to the polygon
#     for i in range(num_x_div):
#         for j in range(num_y_div):
#             x1 = min_x + i * x_step
#             x2 = x1 + x_step
#             y1 = min_y + j * y_step
#             y2 = y1 + y_step
            
#             # Create rectangle in plane coordinates and clip it
#             rect = ShapelyPolygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
#             clipped = rect.intersection(main_poly)
            
#             if clipped.is_empty or not isinstance(clipped, (ShapelyPolygon, MultiPolygon)):
#                 continue
                
#             if isinstance(clipped, MultiPolygon):
#                 polygons = list(clipped.geoms)
#             else:
#                 polygons = [clipped]
            
#             for poly in polygons:
#                 if not isinstance(poly, ShapelyPolygon):
#                     continue
                    
#                 ext_coords = list(poly.exterior.coords)
                
#                 if len(ext_coords) >= 3:  # At least 3 points needed for a polygon
#                     # Convert back to 3D coordinates
#                     node_indices = []
#                     coords_3d = []
#                     for coord in ext_coords[:-1]:
#                         x_proj, y_proj = coord
#                         point_3d = p0 + x_proj * u + y_proj * v
                        
#                         # Check if this 3D point already exists
#                         found = False
#                         for internal_id, (existing_node_id, existing_point) in node_positions.items():
#                             if np.linalg.norm(point_3d - existing_point) < 1e-6:
#                                 node_indices.append(existing_node_id)
#                                 coords_3d.append(existing_point)
#                                 found = True
#                                 break
                        
#                         if not found:
#                             node_id = node_counter + node_id_offset
#                             node_positions[node_counter] = (node_id, point_3d)
#                             node_indices.append(node_id)
#                             coords_3d.append(point_3d)
#                             node_counter += 1
                    
#                     # Ensure counter-clockwise ordering
#                     node_indices, coords_3d = ensure_counter_clockwise(node_indices, coords_3d)
                    
#                     # Handle polygons with more than 4 points by triangulating them
#                     if len(node_indices) > 4:
#                         # Convert to 2D coordinates for triangulation
#                         poly_2d = ShapelyPolygon(ext_coords)
#                         triangles = triangulate(poly_2d)
                        
#                         for triangle in triangles:
#                             tri_coords = list(triangle.exterior.coords)
#                             tri_node_indices = []
#                             tri_coords_3d = []
                            
#                             for coord in tri_coords[:-1]:
#                                 x_proj, y_proj = coord
#                                 point_3d = p0 + x_proj * u + y_proj * v
                                
#                                 # Find or create nodes for this triangle
#                                 found = False
#                                 for nid, coord_3d in zip(node_indices, coords_3d):
#                                     if np.linalg.norm(point_3d - coord_3d) < 1e-6:
#                                         tri_node_indices.append(nid)
#                                         tri_coords_3d.append(coord_3d)
#                                         found = True
#                                         break
                                
#                                 if not found:
#                                     node_id = node_counter + node_id_offset
#                                     node_positions[node_counter] = (node_id, point_3d)
#                                     tri_node_indices.append(node_id)
#                                     tri_coords_3d.append(point_3d)
#                                     node_counter += 1
                            
#                             # Ensure counter-clockwise ordering for triangles
#                             tri_node_indices, tri_coords_3d = ensure_counter_clockwise(tri_node_indices, tri_coords_3d)
                            
#                             element_id = mesh_counter + element_id_offset
#                             mesh_elements[element_id] = {
#                                 'type': 'triangle',
#                                 'nodes': tri_node_indices,
#                                 'coordinates': tri_coords_3d,  # ENSURE coordinates are included
#                                 'id': element_id
#                             }
#                             mesh_counter += 1
#                     else:
#                         element_id = mesh_counter + element_id_offset
#                         elem_type = 'rectangle' if len(node_indices) == 4 else 'triangle'
#                         mesh_elements[element_id] = {
#                             'type': elem_type,
#                             'nodes': node_indices,
#                             'coordinates': coords_3d,  # ENSURE coordinates are included
#                             'id': element_id
#                         }
#                         mesh_counter += 1
    
#     # Second pass: triangulate remaining areas
#     covered_area = ShapelyPolygon()
#     for mesh in mesh_elements.values():
#         projected = project_to_plane(mesh['coordinates'])
#         covered_area = covered_area.union(ShapelyPolygon(projected))
    
#     remaining_area = main_poly.difference(covered_area)
    
#     if not remaining_area.is_empty and isinstance(remaining_area, (ShapelyPolygon, MultiPolygon)):
#         if isinstance(remaining_area, MultiPolygon):
#             remaining_polys = list(remaining_area.geoms)
#         else:
#             remaining_polys = [remaining_area]
        
#         for poly in remaining_polys:
#             if not isinstance(poly, ShapelyPolygon):
#                 continue
                
#             ext_coords = list(poly.exterior.coords)
#             coords = ext_coords[:-1]
            
#             # Check if this is a simple polygon we can handle
#             if len(coords) <= 4:
#                 # Handle as either triangle or rectangle
#                 node_indices = []
#                 coords_3d = []
#                 for coord in coords:
#                     x_proj, y_proj = coord
#                     point_3d = p0 + x_proj * u + y_proj * v
                    
#                     found = False
#                     for internal_id, (existing_node_id, existing_point) in node_positions.items():
#                         if np.linalg.norm(point_3d - existing_point) < 1e-6:
#                             node_indices.append(existing_node_id)
#                             coords_3d.append(existing_point)
#                             found = True
#                             break
                    
#                     if not found:
#                         node_id = node_counter + node_id_offset
#                         node_positions[node_counter] = (node_id, point_3d)
#                         node_indices.append(node_id)
#                         coords_3d.append(point_3d)
#                         node_counter += 1
                
#                 # Ensure counter-clockwise ordering
#                 node_indices, coords_3d = ensure_counter_clockwise(node_indices, coords_3d)
                
#                 element_id = mesh_counter + element_id_offset
#                 elem_type = 'rectangle' if len(node_indices) == 4 else 'triangle'
#                 mesh_elements[element_id] = {
#                     'type': elem_type,
#                     'nodes': node_indices,
#                     'coordinates': coords_3d,  # ENSURE coordinates are included
#                     'id': element_id
#                 }
#                 mesh_counter += 1
#             else:
#                 # Complex polygon - triangulate it
#                 triangles = triangulate(poly)
#                 for triangle in triangles:
#                     tri_coords = list(triangle.exterior.coords)
#                     tri_node_indices = []
#                     tri_coords_3d = []
                    
#                     for coord in tri_coords[:-1]:
#                         x_proj, y_proj = coord
#                         point_3d = p0 + x_proj * u + y_proj * v
                        
#                         # Find or create nodes for this triangle
#                         found = False
#                         for internal_id, (existing_node_id, existing_point) in node_positions.items():
#                             if np.linalg.norm(point_3d - existing_point) < 1e-6:
#                                 tri_node_indices.append(existing_node_id)
#                                 tri_coords_3d.append(existing_point)
#                                 found = True
#                                 break
                        
#                         if not found:
#                             node_id = node_counter + node_id_offset
#                             node_positions[node_counter] = (node_id, point_3d)
#                             tri_node_indices.append(node_id)
#                             tri_coords_3d.append(point_3d)
#                             node_counter += 1
                    
#                     # Ensure counter-clockwise ordering for triangles
#                     tri_node_indices, tri_coords_3d = ensure_counter_clockwise(tri_node_indices, tri_coords_3d)
                    
#                     element_id = mesh_counter + element_id_offset
#                     mesh_elements[element_id] = {
#                         'type': 'triangle',
#                         'nodes': tri_node_indices,
#                         'coordinates': tri_coords_3d,  # ENSURE coordinates are included
#                         'id': element_id
#                     }
#                     mesh_counter += 1
    
#     # Create a mapping of all node IDs to their coordinates
#     node_names = {node_id: coord for internal_id, (node_id, coord) in node_positions.items()}

#     # Initialize replaced_nodes as an empty set
#     replaced_nodes = set()

#     # Find closest mesh points to predefined points
#     predefined_points_list = list(predefined_points.values())
#     closest_mesh_points = {}

#     for i, predefined_point in enumerate(predefined_points_list):
#         min_dist = float('inf')
#         closest_node = None
#         for node_id, mesh_point in node_names.items():
#             dist = np.linalg.norm(predefined_point - mesh_point)
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_node = node_id
#         if closest_node is not None:
#             predefined_ids = list(predefined_points.keys())
#             closest_mesh_points[predefined_ids[i]] = closest_node

#     # Replace closest mesh points with predefined points
#     for predefined_id, node_id in closest_mesh_points.items():
#         predefined_point = predefined_points[predefined_id]
#         # Mark this node as replaced
#         replaced_nodes.add(node_id)
#         # Update node_names
#         node_names[node_id] = predefined_point
#         # Update coordinates in mesh elements
#         for elem in mesh_elements.values():
#             for i, n in enumerate(elem['nodes']):
#                 if n == node_id:
#                     elem['coordinates'][i] = predefined_point

#     # Create the output structure with elements and nodes including IDs
#     output_data = {
#         "elements": {},
#         "nodes": {},
#     }

#     # Add all nodes with their coordinates and IDs
#     for node_id, coord in node_names.items():
#         output_data["nodes"][node_id] = {
#             "id": node_id,
#             "coordinates": [float(coord[0]), float(coord[1]), float(coord[2])],
#             "is_predefined": node_id in replaced_nodes
#         }

#     # Create a mapping of replaced node IDs to predefined point IDs
#     replaced_nodes_mapping = {node_id: predefined_id for predefined_id, node_id in closest_mesh_points.items()}

#     # Add all elements in the requested format with IDs and COORDINATES
#     for element_id, elem_data in mesh_elements.items():
#         # Replace node names with predefined point IDs where applicable
#         node_ids_in_elem = []
#         for node_id in elem_data['nodes']:
#             if node_id in replaced_nodes_mapping:
#                 node_ids_in_elem.append(replaced_nodes_mapping[node_id])
#             else:
#                 node_ids_in_elem.append(node_id)
        
#         output_data["elements"][element_id] = {
#             "id": elem_data['id'],
#             "type": elem_data['type'],
#             "nodes": node_ids_in_elem,
#             "coordinates": elem_data['coordinates']  # CRITICAL: Include coordinates in output
#         }

#     return output_data

# predefined_points = {
#     30001: np.array([0, 0, 0]),
#     30002: np.array([3, 0, 0]),
#     30003: np.array([3, 0, 2]),
#     # 10004: np.array([0, 0, 2]),
#     # 10005: np.array([1.5, 0, 1]),
#     # 10006: np.array([2, 0, 1.5]),
#     # 10007: np.array([1, 0, 1.5]),
#     # 10008: np.array([0.5, 0, 0.5]),
#     # 10009: np.array([2.5, 0, 0.5]),
#     # 10010: np.array([2.5, 0, 1.5]),
#     # 10011: np.array([0.5, 0, 1.5])
# }

# # Define the add_shell and remove_shell dictionaries
# add_shell = {
#     10001: [10001, 10002, 10003, 10004],  # Quadrilateral shell using node IDs
#     10002: [10005, 10006, 10007],         # Triangular shell using node IDs
#     10003: [10008, 10009, 10010, 10011]   # Another quadrilateral using node IDs
# }

# remove_shell = [10004, 10005]  # Shell IDs to be removed

# # Define all the point sets
# horizontal_points = [
#     [0, 0, 0],
#     [2, 0, 0],
#     [2.5, 1.5, 0],
#     [1.5, 2.5, 0],
#     [0.5, 2, 0],
#     [-0.5, 1, 0]
# ]

# vertical_points = [
#     [0, 0, 0],    # Bottom-front
#     [3, 0, 0],    # Bottom-back
#     [3, 0, 2],    # Top-back
#     [0, 0, 2]     # Top-front
# ]

# inclined_points = [
#     [0, 0, 0],    # Base point 1
#     [3, 0, 0],    # Base point 2
#     [2, 2, 2],    # Top point 1
#     [0, 2, 2]     # Top point 2
# ]

# complex_inclined_points = [
#     [0, 0, 0],     # Base point
#     [4, 0, 1],     # Right point (slightly elevated)
#     [3, 3, 3],     # Top point
#     [1, 3, 2],     # Left point
#     [0, 2, 1.5]    # Front point
# ]

# ==================================================================
# 1. Demonstrate create_proper_mesh_for_closed_area_3d with all point sets
# ==================================================================

# predefined_points1 = {
#             10: np.array([0., 0., 3.]),
#             11: np.array([0., 4., 3.]),
#             12: np.array([0., 8., 3.]),
#             13: np.array([5., 0., 3.]),
#             14: np.array([5., 4., 3.]),
#             15: np.array([5., 8., 3.]),
#             16: np.array([10., 0., 3.]),
#             17: np.array([10., 4., 3.]),
#             18: np.array([10., 8., 3.])
#         }
# boundary_points1 = [[0.0, 0.0, 3.0], [10.0, 0.0, 3.0], [10.0, 8.0, 3.0], [0.0, 8.0, 3.0]]

# print("="*80)
# print("Creating mesh for HORIZONTAL surface:")
# horizontal_mesh = create_proper_mesh_for_closed_area_3d(
#     points=boundary_points1,
#     predefined_points=predefined_points1,
#     num_x_div=4,
#     num_y_div=4,
#     floor_number=1
# )

# print("\n" + "="*80)
# print("Creating mesh for VERTICAL surface:")
# vertical_mesh = create_proper_mesh_for_closed_area_3d(
#     points=vertical_points,
#     predefined_points=predefined_points,
#     num_x_div=4,
#     num_y_div=4,
#     floor_number=2
# )

# print("\n" + "="*80)
# print("Creating mesh for INCLINED surface:")
# inclined_mesh = create_proper_mesh_for_closed_area_3d(
#     points=inclined_points,
#     predefined_points=predefined_points,
#     num_x_div=4,
#     num_y_div=4,
#     floor_number=3
# )

# print("\n" + "="*80)
# print("Creating mesh for COMPLEX INCLINED surface:")
# complex_inclined_mesh = create_proper_mesh_for_closed_area_3d(
#     points=complex_inclined_points,
#     predefined_points=predefined_points,
#     num_x_div=4,
#     num_y_div=4,
#     floor_number=4
# )

