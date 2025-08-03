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
from concreteproperties import (
    Concrete,
    ConcreteLinear,
    ConcreteSection,
    RectangularStressBlock,
    SteelBar,
    SteelElasticPlastic,
)
# from concreteproperties.design_codes import NZS3101
from concreteproperties.design_codes import ACI318
from concreteproperties.post import si_kn_m, si_n_mm

from sectionproperties.pre import Geometry, CompoundGeometry, Material
from sectionproperties.analysis import Section
from concreteproperties.material import Concrete, SteelBar
from concreteproperties.stress_strain_profile import (
    ConcreteLinear,
    RectangularStressBlock,
    SteelElasticPlastic
)
from concreteproperties import (
    ConcreteSection,
    Concrete,
    SteelBar,
)
from concreteproperties.design_codes import AS3600
from concreteproperties.post import si_kn_m
from concreteproperties.results import MomentInteractionResults, BiaxialBendingResults

# ========================
# INPUT VARIABLES
# ========================

# Main Execution ==============================================================
# Input Variables ==============================================================




# advanced_materials = {
#     "32mpa_concrete": Concrete(
#         name="32 MPa Concrete",
#         density=2.4e-6,
#         stress_strain_profile=ConcreteLinear(elastic_modulus=30.1e3),
#         ultimate_stress_strain_profile=RectangularStressBlock(
#             compressive_strength=32,
#             alpha=0.802,
#             gamma=0.89,
#             ultimate_strain=0.003,
#         ),
#         flexural_tensile_strength=3.4,
#         colour="lightgrey",
#     ),
#     "500mpa_steel": SteelBar(
#         name="500 MPa Steel",
#         density=7.85e-6,
#         stress_strain_profile=SteelElasticPlastic(
#             yield_strength=500,
#             elastic_modulus=200e3,
#             fracture_strain=0.05,
#         ),
#         colour="red",
#     )
# }

# # Geometry Definitions
# geometry_definitions = {
#     "main_rect": {
#         "type": "polygon",
#         "points": [(0, 0), (300, 0), (300, 450), (0, 450)],
#         "material": "32mpa_concrete",
#         "holes": [
#             # {"type": "circle", "radius": 12, "center": (150, 40), "n_points": 20},
#             # {"type": "circle", "radius": 8, "center": (150, 120), "n_points": 16},
#             # {"type": "circle", "radius": 10, "center": (50, 100), "n_points": 18},
#             # {"type": "polygon", "points": [(250, 140), (265, 125), (280, 140), (265, 155)]}
#         ]
#     },
#     # "steel_column": {
#     #     "type": "circle",
#     #     "radius": 40,
#     #     "center": (80, 60),
#     #     "n_points": 30,
#     #     "material": "500mpa_steel",
#     #     "holes": [
#     #         {"type": "polygon", "points": [(70, 55), (90, 55), (80, 70)]}
#     #     ]
#     # },
#     # "aluminum_hex": {
#     #     "type": "polygon",
#     #     "points": [(220+35*np.cos(a), 60+35*np.sin(a)) for a in np.linspace(0, 2*np.pi, 7)[:-1]],
#     #     "material": "32mpa_concrete",  # replaced aluminum_alloy
#     #     "holes": [
#     #         {"type": "rectangle", "points": [(210, 50), (230, 50), (230, 70), (210, 70)]}
#     #     ]
#     # },
#     # "inner_steel": {
#     #     "type": "circle",
#     #     "radius": 15,
#     #     "center": (220, 60),
#     #     "n_points": 20,
#     #     "material": "500mpa_steel"  # replaced structural_steel
#     # }
# }

# # Rebar Definitions
# rebar_definitions = [
#     {"position": (20, 20), "radius": 10, "n_points": 12, "material": "500mpa_steel"},
#     {"position": (280, 20), "radius": 10, "n_points": 12, "material": "500mpa_steel"},
#     {"position": (20, 410), "radius": 10, "n_points": 12, "material": "500mpa_steel"},
#     {"position": (280, 410), "radius": 10, "n_points": 12, "material": "500mpa_steel"},
#     {"position": (150, 20), "radius": 12.5, "n_points": 10, "material": "500mpa_steel"},
#     {"position": (150, 410), "radius": 12.5, "n_points": 10, "material": "500mpa_steel"}
# ]

# load_data = {
#     "n": 50000,
#     "mxx": -5000000,
#     "m22": 2500000,
#     "mzz": 500000,
#     "vx": 10000,
#     "vy": 5000
# }
# mesh_size = 8



def concrete_model_generation(
    materials: Dict[str, Any],
    advanced_materials: Dict[str, Any],
    geometry_definitions: Dict[str, Any],
    rebar_definitions: List[Dict[str, Any]],
    mesh_size: float = 8,
    plot_title: str = "Integrated Cross-Section"
) -> Tuple[Section, CompoundGeometry]:
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
        Tuple of (Section object, CompoundGeometry object)
    """
    
    # Helper functions within main function
    def get_circle_points(radius: float, n: int, center: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Generate points for a circular boundary"""
        cx, cy = center
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return [(cx + float(radius * np.cos(t)), cy + float(radius * np.sin(t))) for t in theta]

    def create_geometry(geom_def: Dict[str, Any]) -> Polygon:
        """Create a Shapely Polygon from geometry definition"""
        if geom_def["type"] == "circle":
            points = get_circle_points(
                radius=geom_def["radius"],
                n=geom_def.get("n_points", 30),
                center=geom_def["center"]
            )
            return Polygon(points)
        elif geom_def["type"] in ["polygon", "rectangle"]:
            return Polygon(geom_def["points"])
        else:
            raise ValueError(f"Unknown geometry type: {geom_def['type']}")

    def create_materials(materials: Dict[str, Any] = None, 
                        advanced_materials: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create material objects with color compatibility"""
        materials_dict = {}
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

    def resolve_overlapping_geometries(geometry_definitions: Dict[str, Any]) -> Dict[str, List[Polygon]]:
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

    def add_rebars(rebar_definitions: List[Dict[str, Any]], 
                   geometries: List[Geometry], 
                   material_objects: Dict[str, Any]) -> None:
        """Add rebar geometries to the geometry list"""
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
    try:
        # Create materials
        material_objects = create_materials(materials, advanced_materials)
        
        # Resolve overlapping geometries first
        resolved_geoms = resolve_overlapping_geometries(geometry_definitions)
        
        # Create geometries from resolved shapes
        geometries = []
        for name, polygon_list in resolved_geoms.items():
            material_name = geometry_definitions[name]["material"]
            for i, polygon in enumerate(polygon_list):
                part_name = f"{name}_{i+1}" if len(polygon_list) > 1 else name
                geometries.append(Geometry(
                    geom=polygon,
                    material=material_objects[material_name]
                ))
        
        # Collect all resolved geometry polygons for rebar overlap checking
        all_resolved_polygons = []
        for polygon_list in resolved_geoms.values():
            all_resolved_polygons.extend(polygon_list)
        
        # Check for rebar overlaps and resolve them
        print("Checking rebar overlaps...")
        processed_rebars = []
        for i, rebar in enumerate(rebar_definitions):
            rebar_points = get_circle_points(
                radius=rebar["radius"],
                n=rebar["n_points"],
                center=rebar["position"]
            )
            rebar_geom = Polygon(rebar_points)
            
            # Check if rebar overlaps with main geometries
            rebar_embedded = False
            for name, polygon_list in resolved_geoms.items():
                for j, main_geom in enumerate(polygon_list):
                    if rebar_geom.intersects(main_geom):
                        intersection_area = rebar_geom.intersection(main_geom).area
                        rebar_area = rebar_geom.area
                        overlap_percentage = (intersection_area / rebar_area) * 100 if rebar_area > 0 else 0
                        
                        if overlap_percentage > 50:  # If more than 50% of rebar is inside a section
                            # Remove rebar area from the main geometry it's embedded in
                            updated_geom = main_geom.difference(rebar_geom)
                            
                            # Handle the result of difference operation
                            if updated_geom.is_empty:
                                print(f"  - Warning: Rebar {i+1} completely removed {name} part {j+1}")
                                resolved_geoms[name].pop(j)
                            elif hasattr(updated_geom, 'geoms'):  # MultiPolygon result
                                print(f"  - Rebar {i+1} split {name} part {j+1} into {len(updated_geom.geoms)} pieces")
                                resolved_geoms[name][j] = list(updated_geom.geoms)[0]  # Take the largest piece
                                for extra_geom in list(updated_geom.geoms)[1:]:
                                    if extra_geom.area > 1e-6:
                                        resolved_geoms[name].append(extra_geom)
                            else:  # Single polygon result
                                resolved_geoms[name][j] = updated_geom
                                print(f"  - Rebar {i+1} embedded in {name}, removing overlap")
                            
                            rebar_embedded = True
                            break
                if rebar_embedded:
                    break
            
            processed_rebars.append(rebar_geom)
        
        # Rebuild geometries after rebar processing
        geometries = []
        total_parts = 0
        for name, polygon_list in resolved_geoms.items():
            material_name = geometry_definitions[name]["material"]
            valid_polygons = [p for p in polygon_list if not p.is_empty and p.area > 1e-6]
            
            for i, polygon in enumerate(valid_polygons):
                part_name = f"{name}_{i+1}" if len(valid_polygons) > 1 else name
                geometries.append(Geometry(
                    geom=polygon,
                    material=material_objects[material_name]
                ))
                total_parts += 1
        
        # Add rebars as separate geometries
        for i, (rebar, rebar_geom) in enumerate(zip(rebar_definitions, processed_rebars)):
            geometries.append(Geometry(
                geom=rebar_geom,
                material=material_objects[rebar["material"]]
            ))
        
        print(f"Final section contains {len(geometries)} geometries ({total_parts} main parts + {len(processed_rebars)} rebars)")
        
        # Create compound geometry and mesh
        compound_geom = CompoundGeometry(geoms=geometries)
        compound_geom.create_mesh(mesh_sizes=[mesh_size])
        
        # Create section and plot
        section = Section(geometry=compound_geom)
        section.plot_mesh(materials=True, title=plot_title)
        
        return section, compound_geom
        
    except Exception as e:
        print(f"Error generating section: {e}")
        raise


# check_ultimate(conc_sec)

design_code = ACI318()


import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from concreteproperties.design_codes import AS3600
from concreteproperties.post import si_kn_m
from typing import Tuple, Dict, List



def analyze_concrete_section(
    concrete_section: ConcreteSection,
    compressive_strength: float = 40,
    steel_yield_strength: float = 500,
    n_design: float = None,
    plot_results: bool = True,
    file_directory: str = "media_folder",
    design_code: str = "AS3600"  # Accepts string "AS3600" or "ACI318"
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Perform concrete section analysis according to specified design code.
    """
    # Create output directory
    output_dir = Path(file_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    filepaths = []
    
    # Initialize the appropriate design code object
    if design_code.upper() == "AS3600":
        from concreteproperties.design_codes.as3600 import AS3600
        code = AS3600()  # Instantiate AS3600
        design_code_name = "AS3600"
    elif design_code.upper() == "ACI318":
        from concreteproperties.design_codes.aci import ACI318
        code = ACI318()  # Instantiate ACI318
        design_code_name = "ACI318"
    else:
        raise ValueError(f"Unsupported design code: {design_code}. Use 'AS3600' or 'ACI318'")
    
    # Store the design code name in results
    results['design_code'] = design_code_name  # Store the string name, don't call it
    
   # Create materials using the design code object
    concrete = code.create_concrete_material(compressive_strength)
    steel = code.create_steel_material()
    code.assign_concrete_section(concrete_section)
    
    # Store material properties - FIXED LINE BELOW
    results['materials'] = {
        'concrete': {
            'name': concrete.name,
            'density': float(concrete.density),
            'compressive_strength': float(compressive_strength),
            'flexural_tensile_strength': float(concrete.flexural_tensile_strength),
            'elastic_modulus': float(concrete.stress_strain_profile.elastic_modulus)
        },
        'steel': {
            'name': steel.name,
            'density': float(steel.density),
            'yield_strength': float(steel_yield_strength),
            'elastic_modulus': float(steel.stress_strain_profile.elastic_modulus)
        },
        'design_code': design_code_name  # FIXED: Use design_code_name instead of design_code()
    }
    
    # Calculate section properties
    gross_props = code.get_gross_properties()
    cracked_props = code.calculate_cracked_properties()
    
    # Ultimate bending capacity
    f_ult_res, ult_res, phi = code.ultimate_bending_capacity()
    
    # Ultimate capacity with axial load if provided
    if n_design is not None:
        f_ult_res_axial, ult_res_axial, phi_axial = code.ultimate_bending_capacity(
            n_design=n_design
        )
        results.update({
            'gross_props': gross_props,
            'cracked_props': cracked_props,
            'f_ult_res': f_ult_res,
            'ult_res': ult_res,
            'phi': phi,
            'f_ult_res_axial': f_ult_res_axial,
            'ult_res_axial': ult_res_axial,
            'phi_axial': phi_axial,
            'n_design': n_design
        })
    else:
        results.update({
            'gross_props': gross_props,
            'cracked_props': cracked_props,
            'f_ult_res': f_ult_res,
            'ult_res': ult_res,
            'phi': phi
        })
    
    # Generate plots if requested
    if plot_results:
        # Section plot
        fig, ax = plt.subplots(figsize=(10, 6))
        concrete_section.plot_section(ax=ax)
        section_path = output_dir / "section_geometry.png"
        fig.savefig(section_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        filepaths.append(str(section_path))
        
        # Moment interaction plot
        f_mi_res, mi_res, phis = code.moment_interaction_diagram(progress_bar=False)
        results.update({
            'f_mi_res': f_mi_res,
            'mi_res': mi_res,
            'phis': phis
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        MomentInteractionResults.plot_multiple_diagrams(
            [mi_res, f_mi_res],
            ["Unfactored", "Factored"],
            fmt="-",
            units=si_kn_m,
            ax=ax
        )
        mi_path = output_dir / "moment_interaction.png"
        fig.savefig(mi_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        filepaths.append(str(mi_path))
        
        # Phi vs axial load plot
        n_list, _ = mi_res.get_results_lists(moment="m_x")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.array(n_list) / 1e3, phis, "-x")
        ax.set_xlabel("Axial Force [kN]")
        ax.set_ylabel(r"$\phi$")
        ax.set_title("Capacity Reduction Factor vs Axial Load")
        ax.grid()
        phi_path = output_dir / "phi_vs_axial.png"
        fig.savefig(phi_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        filepaths.append(str(phi_path))
        
        # Biaxial bending plot
        f_bb_res, phis_bb = code.biaxial_bending_diagram(n_points=24, progress_bar=False)
        results.update({
            'f_bb_res': f_bb_res,
            'phis_bb': phis_bb
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        BiaxialBendingResults.plot_multiple_diagrams_2d(
            [f_bb_res],
            ["Factored"],
            units=si_kn_m,
            ax=ax
        )
        bb_path = output_dir / "biaxial_bending.png"
        fig.savefig(bb_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        filepaths.append(str(bb_path))
    
    return results, filepaths

# # Example usage:
# if __name__ == "__main__":
#     # 1. First create your concrete section
#     section, compound_geom = concrete_model_generation(
#         materials={},
#         advanced_materials=advanced_materials,
#         geometry_definitions=geometry_definitions,
#         rebar_definitions=rebar_definitions,
#         mesh_size=mesh_size,
#         plot_title="Generated Section"
#     )
    
#     # Convert to ConcreteSection
#     from concreteproperties import ConcreteSection
#     conc_sec = ConcreteSection(compound_geom)
    
#     # 2. Perform analysis
#     analysis_results, image_paths = analyze_concrete_section(
#         concrete_section=conc_sec,
#         compressive_strength=32,
#         steel_yield_strength=500,
#         n_design=1000e3,  # 1000 kN
#         plot_results=True,
#         file_directory="concrete_results",
#         design_code="AS3600"
#     )
    

