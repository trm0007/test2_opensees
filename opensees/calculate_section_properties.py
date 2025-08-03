# Import Statements ===========================================================
import os
import time
# 1. Add missing imports at the top:
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Tuple
from shapely import Polygon
from sectionproperties.pre import Geometry, CompoundGeometry, Material
from sectionproperties.analysis import Section
from concreteproperties.material import Concrete, SteelBar
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
    SteelElasticPlastic
)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar to native Python type
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    return obj

def save_plots(sec: Section, output_dir: str = "outputs") -> list[str]:
    """
    Saves all section property plots to files with guaranteed image content.

    Parameters:
        sec: Section object with calculated properties
        output_dir: Directory to save plots (default: "outputs")

    Returns:
        List of file paths to the saved plots
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # 1. Mesh Plot
    mesh_path = os.path.join(output_dir, "mesh_plot.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        sec.plot_mesh(ax=ax)
        fig.savefig(mesh_path, bbox_inches='tight', dpi=300, format='png')
        plt.close(fig)
        if os.path.exists(mesh_path) and os.path.getsize(mesh_path) > 0:
            saved_files.append(mesh_path)
    except Exception as e:
        plt.close(fig)
        print(f"Failed to save mesh plot: {str(e)}")

    # 2. Centroid Plot
    centroid_path = os.path.join(output_dir, "centroid_plot.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        sec.plot_centroids(ax=ax)
        fig.savefig(centroid_path, bbox_inches='tight', dpi=300, format='png')
        plt.close(fig)
        if os.path.exists(centroid_path) and os.path.getsize(centroid_path) > 0:
            saved_files.append(centroid_path)
    except Exception as e:
        plt.close(fig)
        print(f"Failed to save centroid plot: {str(e)}")

    # 3. Stress Plot
    stress_path = os.path.join(output_dir, "stress_plot.png")
    try:
        fig = plt.figure(figsize=(10, 8))
        fig.savefig(stress_path, bbox_inches='tight', dpi=300, format='png')
        plt.close(fig)
        if os.path.exists(stress_path) and os.path.getsize(stress_path) > 0:
            saved_files.append(stress_path)
    except Exception as e:
        plt.close('all')
        print(f"Failed to save stress plot: {str(e)}")

    return saved_files

def extract_section_properties(section: Section, fmt: str = "8.6e") -> dict:
    """
    Extracts all calculated section properties and organizes them in a dictionary.
    Converts numpy types to native Python types for better JSON serialization.
    
    Parameters:
        section (Section): Section object with calculated properties
        fmt (str): Number formatting string (default: "8.6e")
        
    Returns:
        dict: Dictionary containing all available section properties with native Python types
    """
    props = {
        "geometric": {},
        "warping": {},
        "plastic": {},
        "composite": {},
        "material_info": {},
        "centroids": {}
    }
    
    # Check if material properties exist (composite analysis)
    has_materials = len(section.materials) > 1  # More than just default material
    
    # Helper function to safely get properties and convert numpy types
    def safe_get(getter, *args, **kwargs):
        try:
            result = getter(*args, **kwargs)
            return convert_numpy_types(result)
        except Exception:
            return None
    
    # =====================
    # Geometric Properties
    # =====================
    props["geometric"]["area"] = safe_get(section.get_area)
    props["geometric"]["perimeter"] = safe_get(section.get_perimeter)
    props["geometric"]["first_moments"] = safe_get(section.get_q)
    props["geometric"]["global_second_moments"] = safe_get(section.get_ig)
    props["geometric"]["elastic_centroid"] = safe_get(section.get_c)
    props["geometric"]["centroidal_second_moments"] = safe_get(section.get_ic)
    props["geometric"]["elastic_section_moduli"] = safe_get(section.get_z)
    props["geometric"]["centroidal_radii_gyration"] = safe_get(section.get_rc)
    props["geometric"]["principal_second_moments"] = safe_get(section.get_ip)
    props["geometric"]["principal_bending_angle"] = safe_get(section.get_phi)
    props["geometric"]["principal_section_moduli"] = safe_get(section.get_zp)
    props["geometric"]["principal_radii_gyration"] = safe_get(section.get_rp)
    
    # =====================
    # Warping Properties
    # =====================
    props["warping"]["torsion_constant"] = safe_get(section.get_j)
    props["warping"]["centroidal_shear_center"] = safe_get(section.get_sc)
    props["warping"]["principal_shear_center"] = safe_get(section.get_sc_p)
    props["warping"]["trefftz_shear_center"] = safe_get(section.get_sc_t)
    props["warping"]["warping_constant"] = safe_get(section.get_gamma)
    props["warping"]["centroidal_shear_area"] = safe_get(section.get_as)
    props["warping"]["principal_shear_area"] = safe_get(section.get_as_p)
    props["warping"]["global_monosymmetry"] = safe_get(section.get_beta)
    props["warping"]["principal_monosymmetry"] = safe_get(section.get_beta_p)
    
    # =====================
    # Plastic Properties
    # =====================
    props["plastic"]["centroidal_plastic_centroid"] = safe_get(section.get_pc)
    props["plastic"]["principal_plastic_centroid"] = safe_get(section.get_pc_p)
    props["plastic"]["centroidal_plastic_moduli"] = safe_get(section.get_s)
    props["plastic"]["principal_plastic_moduli"] = safe_get(section.get_sp)
    props["plastic"]["centroidal_shape_factors"] = safe_get(section.get_sf)
    props["plastic"]["principal_shape_factors"] = safe_get(section.get_sf_p)
    
    # =====================
    # Composite Properties (if materials exist)
    # =====================
    if has_materials:
        props["composite"]["mass"] = safe_get(section.get_mass)
        props["composite"]["axial_rigidity"] = safe_get(section.get_ea)
        props["composite"]["mod_weighted_first_moments"] = safe_get(section.get_eq)
        props["composite"]["mod_weighted_global_moments"] = safe_get(section.get_eig)
        props["composite"]["mod_weighted_centroidal_moments"] = safe_get(section.get_eic)
        props["composite"]["mod_weighted_section_moduli"] = safe_get(section.get_ez)
        props["composite"]["yield_moment_centroidal"] = safe_get(section.get_my)
        props["composite"]["mod_weighted_principal_moments"] = safe_get(section.get_eip)
        props["composite"]["mod_weighted_principal_moduli"] = safe_get(section.get_ezp)
        props["composite"]["yield_moment_principal"] = safe_get(section.get_my_p)
        props["composite"]["effective_poissons_ratio"] = safe_get(section.get_nu_eff)
        props["composite"]["effective_elastic_modulus"] = safe_get(section.get_e_eff)
        props["composite"]["effective_shear_modulus"] = safe_get(section.get_g_eff)
        props["composite"]["mod_weighted_torsion_constant"] = safe_get(section.get_ej)
        props["composite"]["mod_weighted_warping_constant"] = safe_get(section.get_egamma)
        props["composite"]["mod_weighted_shear_area"] = safe_get(section.get_eas)
        props["composite"]["mod_weighted_principal_shear_area"] = safe_get(section.get_eas_p)
        props["composite"]["centroidal_plastic_moment"] = safe_get(section.get_mp)
        props["composite"]["principal_plastic_moment"] = safe_get(section.get_mp_p)
    
    # =====================
    # Material Information
    # =====================
    if has_materials:
        props["material_info"] = convert_numpy_types({
            "material_count": len(section.materials),
            "materials": [{
                "name": mat.name,
                "elastic_modulus": mat.elastic_modulus,
                "poissons_ratio": mat.poissons_ratio,
                "yield_strength": mat.yield_strength,
                "density": mat.density,
                "color": mat.color
            } for mat in section.materials if mat.name != "default"]
        })
    
    # =====================
    # Centroids Information
    # =====================
    try:
        # Get centroids if they've been calculated
        if hasattr(section, 'section_props'):
            sp = section.section_props
            props["centroids"] = convert_numpy_types({
                "elastic_centroid": (sp.cx, sp.cy),
                "shear_center": (sp.x_se, sp.y_se) if hasattr(sp, 'x_se') else None,
                "plastic_centroid": (sp.x_pc, sp.y_pc) if hasattr(sp, 'x_pc') else None,
                "principal_plastic_centroid": (sp.x11_pc, sp.y22_pc) if hasattr(sp, 'x11_pc') else None,
                "principal_axes_angle": sp.phi if hasattr(sp, 'phi') else None
            })
    except Exception:
        props["centroids"] = "Centroid data not available"
    
    # Remove None values for cleaner output
    for category in list(props.keys()):
        if isinstance(props[category], dict):
            props[category] = {k: v for k, v in props[category].items() if v is not None}
            if not props[category]:  # Remove empty categories
                del props[category]
    
    return props



def extract_max_stresses(stress_post) -> Dict[str, Dict[str, float]]:
    """
    Extracts the maximum stress values from a stress analysis result.
    Returns only the peak values for each stress type, not all mesh stresses.
    
    Parameters:
        stress_post: StressPost object from sectionproperties analysis
        
    Returns:
        Dictionary containing maximum stress values organized by stress type
    """
    max_stresses = {}
    
    # Get all stress data grouped by material
    stress_data = stress_post.get_stress()
    
    # Initialize storage for all stress types
    all_stress_types = [
        'sig_zz_n', 'sig_zz_mxx', 'sig_zz_myy', 'sig_zz_m11', 'sig_zz_m22',
        'sig_zz_m', 'sig_zx_mzz', 'sig_zy_mzz', 'sig_zxy_mzz', 'sig_zx_vx',
        'sig_zy_vx', 'sig_zxy_vx', 'sig_zx_vy', 'sig_zy_vy', 'sig_zxy_vy',
        'sig_zx_v', 'sig_zy_v', 'sig_zxy_v', 'sig_zz', 'sig_zx', 'sig_zy',
        'sig_zxy', 'sig_11', 'sig_33', 'sig_vm'
    ]
    
    # Initialize result dictionary
    for stress_type in all_stress_types:
        max_stresses[stress_type] = {
            'max_value': -np.inf,
            'min_value': np.inf,
            'abs_max': 0.0,
            'material': None
        }
    
    # Process each material's stress data
    for material_data in stress_data:
        material_name = material_data['material']
        
        for stress_type in all_stress_types:
            if stress_type in material_data:
                stresses = material_data[stress_type]
                
                # Skip if no stresses for this type
                if len(stresses) == 0:
                    continue
                
                current_max = np.max(stresses)
                current_min = np.min(stresses)
                current_abs_max = np.max(np.abs(stresses))
                
                # Update global max values
                if current_max > max_stresses[stress_type]['max_value']:
                    max_stresses[stress_type]['max_value'] = current_max
                    max_stresses[stress_type]['material'] = material_name
                
                if current_min < max_stresses[stress_type]['min_value']:
                    max_stresses[stress_type]['min_value'] = current_min
                
                if current_abs_max > max_stresses[stress_type]['abs_max']:
                    max_stresses[stress_type]['abs_max'] = current_abs_max
    
    # Clean up - remove stress types that had no data
    result = {
        k: v for k, v in max_stresses.items() 
        if v['max_value'] != -np.inf or v['min_value'] != np.inf
    }
    
    # Add combined summary information
    result['summary'] = {
        'max_normal_stress': max(
            result.get('sig_zz', {}).get('max_value', 0),
            result.get('sig_11', {}).get('max_value', 0)
        ),
        'max_shear_stress': max(
            result.get('sig_zxy', {}).get('max_value', 0),
            result.get('sig_zxy_v', {}).get('max_value', 0),
            result.get('sig_zxy_mzz', {}).get('max_value', 0)
        ),
        'max_von_mises': result.get('sig_vm', {}).get('max_value', 0)
    }
        # Clean up - remove stress types that had no data
    result = {
        k: convert_numpy_types(v) for k, v in max_stresses.items() 
        if v['max_value'] != -np.inf or v['min_value'] != np.inf
    }
    return result


# def model_generation(
#     materials: dict,
#     advanced_materials: dict,
#     geometry_definitions: dict,
#     rebar_definitions: list,
#     mesh_size: float = 8,
#     plot_title: str = "Integrated Cross-Section"
# ) -> Section:
#     """
#     Creates and analyzes a composite section with all specified components.
    
#     Args:
#         materials: Dictionary of simple material properties
#         advanced_materials: Dictionary of advanced material objects
#         geometry_definitions: Dictionary defining all geometries
#         rebar_definitions: List of rebar definitions
#         mesh_size: Size for finite element mesh
#         plot_title: Title for the output plot
        
#     Returns:
#         Section: Analyzed section object
#     """
#     import numpy as np
#     from shapely import Polygon
#     from sectionproperties.pre import Geometry, CompoundGeometry, Material
#     from sectionproperties.analysis import Section

#     # Helper functions within main function
#     def get_circle_points(radius: float, n: int, center: tuple[float, float]) -> list[tuple[float, float]]:
#         """Returns circle boundary points"""
#         cx, cy = center
#         theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
#         return [(cx + float(radius * np.cos(t)), cy + float(radius * np.sin(t))) for t in theta]

#     def create_geometry(geom_def: dict):
#         """Creates geometry from definition"""
#         if geom_def["type"] == "circle":
#             points = get_circle_points(
#                 radius=geom_def["radius"],
#                 n=geom_def.get("n_points", 30),
#                 center=geom_def["center"]
#             )
#             return Polygon(points)
#         elif geom_def["type"] in ["polygon", "rectangle"]:
#             return Polygon(geom_def["points"])
#         raise ValueError(f"Unknown geometry type: {geom_def['type']}")



#     def create_materials(materials: dict = None, advanced_materials: dict = None) -> dict:
#         """Creates material objects with color compatibility"""
#         materials_dict = {}

#         # Ensure both inputs are dictionaries
#         materials = materials or {}
#         advanced_materials = advanced_materials or {}

#         # Create simple materials
#         for name, props in materials.items():
#             materials_dict[name] = Material(**props)

#         # Add advanced materials
#         materials_dict.update(advanced_materials)

#         # Normalize color attribute
#         for mat in materials_dict.values():
#             if hasattr(mat, 'colour') and not hasattr(mat, 'color'):
#                 mat.color = mat.colour

#         return materials_dict

#     def add_rebars(rebar_definitions: list, geometries: list, material_objects: dict) -> None:
#         """Adds rebars to geometry list"""
#         for rebar in rebar_definitions:
#             rebar_points = get_circle_points(
#                 radius=rebar["radius"],
#                 n=rebar["n_points"],
#                 center=rebar["position"]
#             )
#             rebar_geom = Polygon(rebar_points)
#             geometries.append(Geometry(
#                 geom=rebar_geom,
#                 material=material_objects[rebar["material"]]
#             ))

#     # Main execution flow
#     material_objects = create_materials(materials, advanced_materials)
    
#     geometries = []
#     for geom_def in geometry_definitions.values():
#         geom = create_geometry(geom_def)
#         for hole_def in geom_def.get("holes", []):
#             geom = geom.difference(create_geometry(hole_def))
#         geometries.append(Geometry(
#             geom=geom,
#             material=material_objects[geom_def["material"]]
#         ))
    
#     add_rebars(rebar_definitions, geometries, material_objects)
    
#     compound_geom = CompoundGeometry(geoms=geometries)
#     compound_geom.create_mesh(mesh_sizes=[mesh_size])
#     section = Section(geometry=compound_geom)
#     section.plot_mesh(materials=True, title=plot_title)
    
#     return section

def model_generation(
    materials: dict,
    advanced_materials: dict,
    geometry_definitions: dict,
    rebar_definitions: list,
    mesh_size: float = 8,
    plot_title: str = "Integrated Cross-Section"
) -> Section:
    """
    Creates and analyzes a composite section with all specified components.
    
    Args:
        materials: Dictionary of simple material properties
        advanced_materials: Dictionary of advanced material objects
        geometry_definitions: Dictionary defining all geometries
        rebar_definitions: List of rebar definitions
        mesh_size: Size for finite element mesh
        plot_title: Title for the output plot
        
    Returns:
        Section: Analyzed section object
    """
    import numpy as np
    from shapely import Polygon
    from sectionproperties.pre import Geometry, CompoundGeometry, Material
    from sectionproperties.analysis import Section

    # Helper functions within main function
    def get_circle_points(radius: float, n: int, center: tuple[float, float]) -> list[tuple[float, float]]:
        """Returns circle boundary points"""
        cx, cy = center
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return [(cx + float(radius * np.cos(t)), cy + float(radius * np.sin(t))) for t in theta]

    def create_geometry(geom_def: dict):
        """Creates geometry from definition"""
        if geom_def["type"] == "circle":
            points = get_circle_points(
                radius=geom_def["radius"],
                n=geom_def.get("n_points", 30),
                center=geom_def["center"]
            )
            return Polygon(points)
        elif geom_def["type"] in ["polygon", "rectangle"]:
            return Polygon(geom_def["points"])
        raise ValueError(f"Unknown geometry type: {geom_def['type']}")

    def create_materials(materials: dict = None, advanced_materials: dict = None) -> dict:
        """Creates material objects with color compatibility"""
        materials_dict = {}

        # Ensure both inputs are dictionaries
        materials = materials or {}
        advanced_materials = advanced_materials or {}

        # Create simple materials
        for name, props in materials.items():
            materials_dict[name] = Material(**props)

        # Add advanced materials
        materials_dict.update(advanced_materials)

        # Normalize color attribute
        for mat in materials_dict.values():
            if hasattr(mat, 'colour') and not hasattr(mat, 'color'):
                mat.color = mat.colour

        return materials_dict

    def resolve_overlapping_geometries(geometry_definitions: dict) -> dict:
        """
        Resolve overlapping geometries by removing overlaps from mother sections.
        Priority: steel sections > concrete sections (steel cuts through concrete)
        Returns dictionary with lists of polygons to handle MultiPolygon cases.
        """
        print("Resolving overlapping geometries...")
        
        # Create all base geometries first
        base_geometries = {}
        for name, geom_def in geometry_definitions.items():
            geom = create_geometry(geom_def)
            
            # Subtract holes if they exist
            for hole_def in geom_def.get("holes", []):
                hole_geom = create_geometry(hole_def)
                geom = geom.difference(hole_geom)
            
            base_geometries[name] = {
                'geometry': geom,
                'material': geom_def["material"],
                'priority': 1 if 'steel' in geom_def["material"] else 0  # Steel has higher priority
            }
        
        # Sort by priority (higher priority geometries cut through lower priority ones)
        sorted_items = sorted(base_geometries.items(), key=lambda x: x[1]['priority'], reverse=True)
        
        resolved_geometries = {}
        processed_geometries = []
        
        for name, geom_data in sorted_items:
            current_geom = geom_data['geometry']
            
            # For lower priority geometries, subtract all higher priority geometries
            if geom_data['priority'] == 0:  # Concrete sections
                for processed_geom in processed_geometries:
                    if current_geom.intersects(processed_geom):
                        intersection_area = current_geom.intersection(processed_geom).area
                        total_area = current_geom.area
                        overlap_percentage = (intersection_area / total_area) * 100 if total_area > 0 else 0
                        
                        if overlap_percentage > 0.1:  # Only process significant overlaps (>0.1%)
                            print(f"  - Removing {overlap_percentage:.1f}% overlap from {name}")
                            current_geom = current_geom.difference(processed_geom)
                            
                            # Ensure geometry is still valid after subtraction
                            if not current_geom.is_valid:
                                from shapely.validation import make_valid
                                current_geom = make_valid(current_geom)
            
            # Handle different geometry types after processing
            polygon_list = []
            if current_geom.is_empty:
                print(f"  - Warning: {name} became empty after overlap removal, skipping")
            elif hasattr(current_geom, 'geoms'):  # MultiPolygon
                print(f"  - {name} split into {len(current_geom.geoms)} parts after overlap removal")
                for geom_part in current_geom.geoms:
                    if geom_part.area > 1e-6:  # Only keep significant parts
                        polygon_list.append(geom_part)
            else:  # Single Polygon
                if current_geom.area > 1e-6:  # Only keep geometries with significant area
                    polygon_list.append(current_geom)
            
            if polygon_list:  # Only add if we have valid polygons
                resolved_geometries[name] = polygon_list
                processed_geometries.extend(polygon_list)
        
        return resolved_geometries

    def add_rebars(rebar_definitions: list, geometries: list, material_objects: dict) -> None:
        """Adds rebars to geometry list"""
        for rebar in rebar_definitions:
            rebar_points = get_circle_points(
                radius=rebar["radius"],
                n=rebar["n_points"],
                center=rebar["position"]
            )
            rebar_geom = Polygon(rebar_points)
            geometries.append(Geometry(
                geom=rebar_geom,
                material=material_objects[rebar["material"]]
            ))

    # Main execution flow
    material_objects = create_materials(materials, advanced_materials)
    
    # Resolve overlapping geometries first
    resolved_geoms = resolve_overlapping_geometries(geometry_definitions)
    
    # Create geometries from resolved shapes
    geometries = []
    for name, polygon_list in resolved_geoms.items():
        material_name = geometry_definitions[name]["material"]
        for i, polygon in enumerate(polygon_list):
            geometries.append(Geometry(
                geom=polygon,
                material=material_objects[material_name]
            ))
    
    add_rebars(rebar_definitions, geometries, material_objects)
    
    compound_geom = CompoundGeometry(geoms=geometries)
    compound_geom.create_mesh(mesh_sizes=[mesh_size])
    section = Section(geometry=compound_geom)
    section.plot_mesh(materials=True, title=plot_title)
    
    return section


def analyze_section(
    section: Section,
    output_dir: str = "output",
    load_data: dict = None
) -> tuple:
    print("Starting analysis...")

    elastic_modulus = 35e3  # MPa
    poissons_ratio = 0.2
    shear_modulus = elastic_modulus / (2 * (1 + poissons_ratio))
    print(f"Elastic modulus: {elastic_modulus}, Poisson's ratio: {poissons_ratio}, Shear modulus: {shear_modulus}")

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created or already exists: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")

    start = time.time()

    try:
        print(50*"-")
        section.calculate_geometric_properties()
        print("Geometric properties calculated.")
        section.calculate_plastic_properties()
        print("Plastic properties calculated.")
        section.calculate_warping_properties()
        print("Warping properties calculated.")
    except Exception as e:
        print(f"Error during section property calculations: {e}")

    try:
        section_props = extract_section_properties(section)
        print("Section properties extracted.")
    except Exception as e:
        print(f"Error extracting section properties: {e}")
        section_props = {}

    print(f"Analysis completed in {time.time() - start:.2f} seconds")

    stress_result = {}
    max_stresses = {}
    if load_data is not None:
        try:
            print(f"Calculating stress with load data: {load_data}")
            stress_result = section.calculate_stress(**load_data)
            max_stresses = extract_max_stresses(stress_result)
            print("Stress calculation completed.")
        except Exception as e:
            print(f"Error during stress calculation: {e}")

    try:
        filepaths = save_plots(section, output_dir=output_dir)
        print(f"Plots saved to: {filepaths}")
    except Exception as e:
        print(f"Error saving plots: {e}")
        filepaths = []

    return section_props, max_stresses, filepaths


# Main Execution ==============================================================
# Input Variables =============================================================
# Original Material Definitions
materials = {
    "main_concrete": {
        "name": "C40 Concrete",
        "elastic_modulus": 30000,
        "poissons_ratio": 0.2,
        "density": 2400,
        "yield_strength": 40,
        "color": "lightgray"
    },
    "structural_steel": {
        "name": "S355 Steel",
        "elastic_modulus": 210000,
        "poissons_ratio": 0.3,
        "density": 7850,
        "yield_strength": 355,
        "color": "steelblue"
    },
    "aluminum_alloy": {
        "name": "6061 Aluminum",
        "elastic_modulus": 69000,
        "poissons_ratio": 0.33,
        "density": 2700,
        "yield_strength": 240,
        "color": "silver"
    },
    "rebar_steel": {
        "name": "B500B Rebar",
        "elastic_modulus": 200000,
        "poissons_ratio": 0.3,
        "density": 7850,
        "yield_strength": 500,
        "color": "red"
    }
}

advanced_materials = {
    "32mpa_concrete": Concrete(
        name="32 MPa Concrete",
        density=2.4e-6,
        stress_strain_profile=ConcreteLinear(elastic_modulus=30.1e3),
        ultimate_stress_strain_profile=RectangularStressBlock(
            compressive_strength=32,
            alpha=0.802,
            gamma=0.89,
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=3.4,
        colour="lightgrey",  # Note British spelling
    ),
    "500mpa_steel": SteelBar(
        name="500 MPa Steel",
        density=7.85e-6,
        stress_strain_profile=SteelElasticPlastic(
            yield_strength=500,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="red",  # Note British spelling
    )
}


# Geometry Definitions
geometry_definitions = {
    "main_rect": {
        "type": "polygon",
        "points": [(0, 0), (300, 0), (300, 200), (0, 200)],
        "material": "32mpa_concrete",  # Using advanced concrete
        "holes": [
            {"type": "circle", "radius": 12, "center": (150, 40), "n_points": 20},
            {"type": "circle", "radius": 8, "center": (150, 120), "n_points": 16},
            {"type": "circle", "radius": 10, "center": (50, 100), "n_points": 18},
            {"type": "polygon", "points": [(250, 140), (265, 125), (280, 140), (265, 155)]}
        ]
    },
    "steel_column": {
        "type": "circle",
        "radius": 40,
        "center": (80, 60),
        "n_points": 30,
        "material": "500mpa_steel",  # Using advanced steel
        "holes": [
            {"type": "polygon", "points": [(70, 55), (90, 55), (80, 70)]}
        ]
    },
    "aluminum_hex": {
        "type": "polygon",
        "points": [(220+35*np.cos(a), 60+35*np.sin(a)) for a in np.linspace(0, 2*np.pi, 7)[:-1]],
        "material": "aluminum_alloy",
        "holes": [
            {"type": "rectangle", "points": [(210, 50), (230, 50), (230, 70), (210, 70)]}
        ]
    },
    "inner_steel": {
        "type": "circle",
        "radius": 15,
        "center": (220, 60),
        "n_points": 20,
        "material": "structural_steel"  # Using original steel
    }
}

# Rebar Definitions
rebar_definitions = [
    {"position": (20, 20), "radius": 6, "n_points": 12, "material": "500mpa_steel"},
    {"position": (280, 20), "radius": 6, "n_points": 12, "material": "500mpa_steel"},
    {"position": (20, 180), "radius": 6, "n_points": 12, "material": "500mpa_steel"},
    {"position": (280, 180), "radius": 6, "n_points": 12, "material": "500mpa_steel"},
    {"position": (150, 20), "radius": 5, "n_points": 10, "material": "500mpa_steel"}
]

load_data = {
        "n": 50000,
        "mxx": -5000000,
        "m22": 2500000,
        "mzz": 500000,
        "vx": 10000,
        "vy": 5000
    }

# # Example usage (would be outside this function):
# section = model_generation(
#     materials=materials,
#     advanced_materials=advanced_materials,
#     geometry_definitions=geometry_definitions,
#     rebar_definitions=rebar_definitions,
#     mesh_size=8,
#     plot_title="My Custom Section"
# )

# # Then analyze it with your load case
# section_properties, max_stresses, plot_paths = analyze_section(
#     section=section,
#     output_dir="analysis_results",
#     load_data=load_data
# )