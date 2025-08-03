"""
ACI/AISC Code-Based Stress Checking Functions
Using Cross-Section Stresses from OpenSees Analysis

This module provides functions for checking all types of stresses
based on ACI 318 (concrete) and AISC 360 (steel) code provisions.
Focus: Pure stress verification without reinforcement design.
"""

import numpy as np
from math import sqrt, pi

# =============================================================================
# MATERIAL PROPERTIES AND CONSTANTS
# =============================================================================

class MaterialProperties:
    """Material properties for stress checking calculations"""
    
    def __init__(self, concrete_fc=4000, steel_fy=60000, steel_Es=29000000):
        # Concrete properties (psi)
        self.fc = concrete_fc  # Concrete compressive strength
        self.Ec = 57000 * sqrt(concrete_fc)  # Modulus of elasticity (ACI 318-19: 19.2.2.1)
        
        # Steel properties (psi)
        self.fy = steel_fy  # Steel yield strength
        self.Es = steel_Es  # Steel modulus of elasticity
        self.fu = 1.5 * steel_fy  # Ultimate strength (typical)
        
        # ACI 318 factors
        self.phi_flexure = 0.9  # Strength reduction factor for flexure
        self.phi_compression = 0.65  # For tied columns
        self.phi_shear = 0.75  # For shear

# =============================================================================
# 1. CONCRETE STRESS CHECKING - ACI 318
# =============================================================================

def check_concrete_normal_stresses(stresses, materials, safety_factor=1.4):
    """
    Check all normal stress components for concrete per ACI 318
    
    Args:
        stresses: Dictionary containing all stress components
        materials: MaterialProperties object
        safety_factor: Load factor for ultimate strength design
    
    Returns:
        dict: Comprehensive normal stress check results
    """
    results = {
        'axial_stress_check': {},
        'bending_stress_checks': {},
        'combined_normal_stress_check': {},
        'overall_status': 'PASS',
        'critical_stresses': {},
        'recommendations': []
    }
    
    # ACI 318-19: Allowable stresses
    max_compression_limit = 0.85 * materials.fc / safety_factor
    concrete_tensile_strength = 7.5 * sqrt(materials.fc)
    
    # 1. Axial stress check (sig_zz_n)
    if "sig_zz_n" in stresses:
        axial_stresses = np.array(stresses["sig_zz_n"])
        max_compression = abs(min(axial_stresses)) if len(axial_stresses) > 0 else 0
        max_tension = max(axial_stresses) if len(axial_stresses) > 0 else 0
        
        results['axial_stress_check'] = {
            'max_compression': max_compression,
            'max_tension': max_tension,
            'compression_status': 'PASS' if max_compression <= max_compression_limit else 'FAIL',
            'tension_status': 'PASS' if max_tension <= concrete_tensile_strength else 'FAIL',
            'compression_utilization': max_compression / max_compression_limit,
            'tension_utilization': max_tension / concrete_tensile_strength
        }
    
    # 2. Bending stress checks
    bending_components = ["sig_zz_mxx", "sig_zz_myy", "sig_zz_m11", "sig_zz_m22", "sig_zz_m"]
    
    for component in bending_components:
        if component in stresses:
            bending_stresses = np.array(stresses[component])
            max_compression = abs(min(bending_stresses)) if len(bending_stresses) > 0 else 0
            max_tension = max(bending_stresses) if len(bending_stresses) > 0 else 0
            
            results['bending_stress_checks'][component] = {
                'max_compression': max_compression,
                'max_tension': max_tension,
                'compression_status': 'PASS' if max_compression <= max_compression_limit else 'FAIL',
                'tension_status': 'PASS' if max_tension <= concrete_tensile_strength else 'FAIL',
                'compression_utilization': max_compression / max_compression_limit,
                'tension_utilization': max_tension / concrete_tensile_strength
            }
    
    # 3. Combined normal stress check (sig_zz)
    if "sig_zz" in stresses:
        combined_stresses = np.array(stresses["sig_zz"])
        max_compression = abs(min(combined_stresses)) if len(combined_stresses) > 0 else 0
        max_tension = max(combined_stresses) if len(combined_stresses) > 0 else 0
        
        results['combined_normal_stress_check'] = {
            'max_compression': max_compression,
            'max_tension': max_tension,
            'compression_status': 'PASS' if max_compression <= max_compression_limit else 'FAIL',
            'tension_status': 'PASS' if max_tension <= concrete_tensile_strength else 'FAIL',
            'compression_utilization': max_compression / max_compression_limit,
            'tension_utilization': max_tension / concrete_tensile_strength
        }
    
    # Determine overall status and critical stresses
    all_checks = [results['axial_stress_check']] + list(results['bending_stress_checks'].values()) + [results['combined_normal_stress_check']]
    
    for check in all_checks:
        if check and (check.get('compression_status') == 'FAIL' or check.get('tension_status') == 'FAIL'):
            results['overall_status'] = 'FAIL'
            break
    
    # Find critical utilization ratios
    max_compression_util = 0
    max_tension_util = 0
    
    for check in all_checks:
        if check:
            max_compression_util = max(max_compression_util, check.get('compression_utilization', 0))
            max_tension_util = max(max_tension_util, check.get('tension_utilization', 0))
    
    results['critical_stresses'] = {
        'max_compression_utilization': max_compression_util,
        'max_tension_utilization': max_tension_util
    }
    
    # Generate recommendations
    if max_compression_util > 1.0:
        results['recommendations'].append(f"Compressive stress utilization {max_compression_util:.2f} exceeds limit - increase concrete strength or section size")
    if max_tension_util > 1.0:
        results['recommendations'].append(f"Tensile stress utilization {max_tension_util:.2f} exceeds concrete capacity - provide adequate reinforcement")
    
    return results

def check_concrete_shear_stresses(stresses, materials):
    """
    Check all shear stress components for concrete per ACI 318
    
    Args:
        stresses: Dictionary containing shear stress components
        materials: MaterialProperties object
    
    Returns:
        dict: Comprehensive shear stress check results
    """
    results = {
        'torsional_shear_checks': {},
        'force_shear_checks': {},
        'combined_shear_checks': {},
        'overall_status': 'PASS',
        'critical_stresses': {},
        'recommendations': []
    }
    
    # ACI 318-19: Allowable shear stress (simplified)
    # For members without shear reinforcement: vc = 2*sqrt(f'c)
    allowable_shear = 2 * sqrt(materials.fc)
    
    # 1. Torsional shear stress checks
    torsional_components = ["sig_zx_mzz", "sig_zy_mzz", "sig_zxy_mzz"]
    
    for component in torsional_components:
        if component in stresses:
            shear_stresses = np.array(stresses[component])
            max_shear = max(np.abs(shear_stresses)) if len(shear_stresses) > 0 else 0
            
            results['torsional_shear_checks'][component] = {
                'max_shear': max_shear,
                'status': 'PASS' if max_shear <= allowable_shear else 'FAIL',
                'utilization': max_shear / allowable_shear
            }
    
    # 2. Force-induced shear stress checks
    force_shear_components = ["sig_zx_vx", "sig_zy_vx", "sig_zxy_vx", 
                             "sig_zx_vy", "sig_zy_vy", "sig_zxy_vy"]
    
    for component in force_shear_components:
        if component in stresses:
            shear_stresses = np.array(stresses[component])
            max_shear = max(np.abs(shear_stresses)) if len(shear_stresses) > 0 else 0
            
            results['force_shear_checks'][component] = {
                'max_shear': max_shear,
                'status': 'PASS' if max_shear <= allowable_shear else 'FAIL',
                'utilization': max_shear / allowable_shear
            }
    
    # 3. Combined shear stress checks
    combined_shear_components = ["sig_zx_v", "sig_zy_v", "sig_zxy_v", 
                                "sig_zx", "sig_zy", "sig_zxy"]
    
    for component in combined_shear_components:
        if component in stresses:
            shear_stresses = np.array(stresses[component])
            max_shear = max(np.abs(shear_stresses)) if len(shear_stresses) > 0 else 0
            
            results['combined_shear_checks'][component] = {
                'max_shear': max_shear,
                'status': 'PASS' if max_shear <= allowable_shear else 'FAIL',
                'utilization': max_shear / allowable_shear
            }
    
    # Determine overall status and find critical stresses
    all_checks = (list(results['torsional_shear_checks'].values()) + 
                 list(results['force_shear_checks'].values()) + 
                 list(results['combined_shear_checks'].values()))
    
    max_utilization = 0
    for check in all_checks:
        if check['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        max_utilization = max(max_utilization, check['utilization'])
    
    results['critical_stresses'] = {
        'max_shear_utilization': max_utilization
    }
    
    # Generate recommendations
    if max_utilization > 1.0:
        results['recommendations'].append(f"Shear stress utilization {max_utilization:.2f} exceeds limit - provide shear reinforcement or increase section size")
    
    return results

def check_concrete_principal_stresses(stresses, materials):
    """
    Check principal stresses for concrete
    
    Args:
        stresses: Dictionary containing principal stress components
        materials: MaterialProperties object
    
    Returns:
        dict: Principal stress check results
    """
    results = {
        'principal_stress_checks': {},
        'von_mises_check': {},
        'overall_status': 'PASS',
        'recommendations': []
    }
    
    # Allowable stresses
    max_compression_limit = 0.85 * materials.fc
    concrete_tensile_strength = 7.5 * sqrt(materials.fc)
    
    # Principal stress checks
    principal_components = ["sig_11", "sig_33"]
    
    for component in principal_components:
        if component in stresses:
            principal_stresses = np.array(stresses[component])
            max_compression = abs(min(principal_stresses)) if len(principal_stresses) > 0 else 0
            max_tension = max(principal_stresses) if len(principal_stresses) > 0 else 0
            
            results['principal_stress_checks'][component] = {
                'max_compression': max_compression,
                'max_tension': max_tension,
                'compression_status': 'PASS' if max_compression <= max_compression_limit else 'FAIL',
                'tension_status': 'PASS' if max_tension <= concrete_tensile_strength else 'FAIL',
                'compression_utilization': max_compression / max_compression_limit,
                'tension_utilization': max_tension / concrete_tensile_strength
            }
    
    # Von Mises stress check (for reference, though not typically used for concrete)
    if "sig_vm" in stresses:
        vm_stresses = np.array(stresses["sig_vm"])
        max_vm = max(vm_stresses) if len(vm_stresses) > 0 else 0
        
        results['von_mises_check'] = {
            'max_von_mises': max_vm,
            'status': 'PASS' if max_vm <= materials.fc else 'FAIL',
            'utilization': max_vm / materials.fc
        }
    
    # Determine overall status
    for component_check in results['principal_stress_checks'].values():
        if (component_check['compression_status'] == 'FAIL' or 
            component_check['tension_status'] == 'FAIL'):
            results['overall_status'] = 'FAIL'
    
    if results['von_mises_check'].get('status') == 'FAIL':
        results['overall_status'] = 'FAIL'
    
    return results

# =============================================================================
# 2. STEEL STRESS CHECKING - AISC 360
# =============================================================================

def check_steel_normal_stresses(stresses, steel_grade='A572'):
    """
    Check all normal stress components for steel per AISC 360
    
    Args:
        stresses: Dictionary containing stress components
        steel_grade: Steel grade designation
    
    Returns:
        dict: Comprehensive normal stress check results
    """
    # Steel properties based on grade
    steel_props = {
        'A572': {'Fy': 50000, 'Fu': 65000},  # Grade 50
        'A992': {'Fy': 50000, 'Fu': 65000},
        'A36': {'Fy': 36000, 'Fu': 58000}
    }
    
    Fy = steel_props[steel_grade]['Fy']
    Fu = steel_props[steel_grade]['Fu']
    
    results = {
        'axial_stress_check': {},
        'bending_stress_checks': {},
        'combined_normal_stress_check': {},
        'overall_status': 'PASS',
        'critical_stresses': {},
        'recommendations': []
    }
    
    # 1. Axial stress check (sig_zz_n)
    if "sig_zz_n" in stresses:
        axial_stresses = np.array(stresses["sig_zz_n"])
        max_stress = max(np.abs(axial_stresses)) if len(axial_stresses) > 0 else 0
        
        results['axial_stress_check'] = {
            'max_stress': max_stress,
            'status': 'PASS' if max_stress <= Fy else 'FAIL',
            'utilization': max_stress / Fy
        }
    
    # 2. Bending stress checks
    bending_components = ["sig_zz_mxx", "sig_zz_myy", "sig_zz_m11", "sig_zz_m22", "sig_zz_m"]
    
    for component in bending_components:
        if component in stresses:
            bending_stresses = np.array(stresses[component])
            max_stress = max(np.abs(bending_stresses)) if len(bending_stresses) > 0 else 0
            
            results['bending_stress_checks'][component] = {
                'max_stress': max_stress,
                'status': 'PASS' if max_stress <= Fy else 'FAIL',
                'utilization': max_stress / Fy
            }
    
    # 3. Combined normal stress check (sig_zz)
    if "sig_zz" in stresses:
        combined_stresses = np.array(stresses["sig_zz"])
        max_stress = max(np.abs(combined_stresses)) if len(combined_stresses) > 0 else 0
        
        results['combined_normal_stress_check'] = {
            'max_stress': max_stress,
            'status': 'PASS' if max_stress <= Fy else 'FAIL',
            'utilization': max_stress / Fy
        }
    
    # Determine overall status and critical stresses
    all_checks = [results['axial_stress_check']] + list(results['bending_stress_checks'].values()) + [results['combined_normal_stress_check']]
    
    max_utilization = 0
    for check in all_checks:
        if check and check.get('status') == 'FAIL':
            results['overall_status'] = 'FAIL'
        if check:
            max_utilization = max(max_utilization, check.get('utilization', 0))
    
    results['critical_stresses'] = {
        'max_normal_utilization': max_utilization
    }
    
    # Generate recommendations
    if max_utilization > 1.0:
        results['recommendations'].append(f"Normal stress utilization {max_utilization:.2f} exceeds yield strength - increase section size or use higher grade steel")
    
    return results

def check_steel_shear_stresses(stresses, steel_grade='A572'):
    """
    Check all shear stress components for steel per AISC 360
    
    Args:
        stresses: Dictionary containing shear stress components
        steel_grade: Steel grade designation
    
    Returns:
        dict: Comprehensive shear stress check results
    """
    steel_props = {
        'A572': {'Fy': 50000, 'Fu': 65000},
        'A992': {'Fy': 50000, 'Fu': 65000},
        'A36': {'Fy': 36000, 'Fu': 58000}
    }
    
    Fy = steel_props[steel_grade]['Fy']
    # AISC 360: Shear yielding limit = 0.6*Fy
    shear_limit = 0.6 * Fy
    
    results = {
        'torsional_shear_checks': {},
        'force_shear_checks': {},
        'combined_shear_checks': {},
        'overall_status': 'PASS',
        'critical_stresses': {},
        'recommendations': []
    }
    
    # 1. Torsional shear stress checks
    torsional_components = ["sig_zx_mzz", "sig_zy_mzz", "sig_zxy_mzz"]
    
    for component in torsional_components:
        if component in stresses:
            shear_stresses = np.array(stresses[component])
            max_shear = max(np.abs(shear_stresses)) if len(shear_stresses) > 0 else 0
            
            results['torsional_shear_checks'][component] = {
                'max_shear': max_shear,
                'status': 'PASS' if max_shear <= shear_limit else 'FAIL',
                'utilization': max_shear / shear_limit
            }
    
    # 2. Force-induced shear stress checks
    force_shear_components = ["sig_zx_vx", "sig_zy_vx", "sig_zxy_vx", 
                             "sig_zx_vy", "sig_zy_vy", "sig_zxy_vy"]
    
    for component in force_shear_components:
        if component in stresses:
            shear_stresses = np.array(stresses[component])
            max_shear = max(np.abs(shear_stresses)) if len(shear_stresses) > 0 else 0
            
            results['force_shear_checks'][component] = {
                'max_shear': max_shear,
                'status': 'PASS' if max_shear <= shear_limit else 'FAIL',
                'utilization': max_shear / shear_limit
            }
    
    # 3. Combined shear stress checks
    combined_shear_components = ["sig_zx_v", "sig_zy_v", "sig_zxy_v", 
                                "sig_zx", "sig_zy", "sig_zxy"]
    
    for component in combined_shear_components:
        if component in stresses:
            shear_stresses = np.array(stresses[component])
            max_shear = max(np.abs(shear_stresses)) if len(shear_stresses) > 0 else 0
            
            results['combined_shear_checks'][component] = {
                'max_shear': max_shear,
                'status': 'PASS' if max_shear <= shear_limit else 'FAIL',
                'utilization': max_shear / shear_limit
            }
    
    # Determine overall status and find critical stresses
    all_checks = (list(results['torsional_shear_checks'].values()) + 
                 list(results['force_shear_checks'].values()) + 
                 list(results['combined_shear_checks'].values()))
    
    max_utilization = 0
    for check in all_checks:
        if check['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        max_utilization = max(max_utilization, check['utilization'])
    
    results['critical_stresses'] = {
        'max_shear_utilization': max_utilization
    }
    
    # Generate recommendations
    if max_utilization > 1.0:
        results['recommendations'].append(f"Shear stress utilization {max_utilization:.2f} exceeds limit - increase section size or use higher grade steel")
    
    return results

def check_steel_principal_stresses(stresses, steel_grade='A572'):
    """
    Check principal stresses and von Mises stress for steel
    
    Args:
        stresses: Dictionary containing principal stress components
        steel_grade: Steel grade designation
    
    Returns:
        dict: Principal stress check results
    """
    steel_props = {
        'A572': {'Fy': 50000, 'Fu': 65000},
        'A992': {'Fy': 50000, 'Fu': 65000},
        'A36': {'Fy': 36000, 'Fu': 58000}
    }
    
    Fy = steel_props[steel_grade]['Fy']
    
    results = {
        'principal_stress_checks': {},
        'von_mises_check': {},
        'overall_status': 'PASS',
        'critical_stresses': {},
        'recommendations': []
    }
    
    # Principal stress checks
    principal_components = ["sig_11", "sig_33"]
    
    for component in principal_components:
        if component in stresses:
            principal_stresses = np.array(stresses[component])
            max_stress = max(np.abs(principal_stresses)) if len(principal_stresses) > 0 else 0
            
            results['principal_stress_checks'][component] = {
                'max_stress': max_stress,
                'status': 'PASS' if max_stress <= Fy else 'FAIL',
                'utilization': max_stress / Fy
            }
    
    # Von Mises stress check (most important for steel)
    if "sig_vm" in stresses:
        vm_stresses = np.array(stresses["sig_vm"])
        max_vm = max(vm_stresses) if len(vm_stresses) > 0 else 0
        
        results['von_mises_check'] = {
            'max_von_mises': max_vm,
            'status': 'PASS' if max_vm <= Fy else 'FAIL',
            'utilization': max_vm / Fy
        }
    
    # Determine overall status and find critical stresses
    max_utilization = 0
    
    for component_check in results['principal_stress_checks'].values():
        if component_check['status'] == 'FAIL':
            results['overall_status'] = 'FAIL'
        max_utilization = max(max_utilization, component_check['utilization'])
    
    if results['von_mises_check'].get('status') == 'FAIL':
        results['overall_status'] = 'FAIL'
    
    if results['von_mises_check']:
        max_utilization = max(max_utilization, results['von_mises_check']['utilization'])
    
    results['critical_stresses'] = {
        'max_stress_utilization': max_utilization
    }
    
    # Generate recommendations
    if max_utilization > 1.0:
        results['recommendations'].append(f"Stress utilization {max_utilization:.2f} exceeds yield strength - increase section size or use higher grade steel")
    
    return results

# =============================================================================
# 3. COMPREHENSIVE STRESS CHECK FUNCTIONS
# =============================================================================

def comprehensive_concrete_stress_check(stresses, materials, safety_factor=1.4):
    """
    Perform comprehensive stress checking for concrete elements
    
    Args:
        stresses: Dictionary containing all stress components
        materials: MaterialProperties object
        safety_factor: Load factor for ultimate strength design
    
    Returns:
        dict: Complete stress check results
    """
    results = {
        'normal_stresses': check_concrete_normal_stresses(stresses, materials, safety_factor),
        'shear_stresses': check_concrete_shear_stresses(stresses, materials),
        'principal_stresses': check_concrete_principal_stresses(stresses, materials),
        'overall_status': 'PASS',
        'summary': {}
    }
    
    # Determine overall status
    if (results['normal_stresses']['overall_status'] == 'FAIL' or
        results['shear_stresses']['overall_status'] == 'FAIL' or
        results['principal_stresses']['overall_status'] == 'FAIL'):
        results['overall_status'] = 'FAIL'
    
    # Create summary
    max_normal_util = results['normal_stresses']['critical_stresses'].get('max_compression_utilization', 0)
    max_shear_util = results['shear_stresses']['critical_stresses'].get('max_shear_utilization', 0)
    
    results['summary'] = {
        'max_normal_utilization': max_normal_util,
        'max_shear_utilization': max_shear_util,
        'critical_utilization': max(max_normal_util, max_shear_util),
        'material_grade': f"f'c = {materials.fc} psi"
    }
    
    return results

def comprehensive_steel_stress_check(stresses, steel_grade='A572'):
    """
    Perform comprehensive stress checking for steel elements
    
    Args:
        stresses: Dictionary containing all stress components
        steel_grade: Steel grade designation
    
    Returns:
        dict: Complete stress check results
    """
    results = {
        'normal_stresses': check_steel_normal_stresses(stresses, steel_grade),
        'shear_stresses': check_steel_shear_stresses(stresses, steel_grade),
        'principal_stresses': check_steel_principal_stresses(stresses, steel_grade),
        'overall_status': 'PASS',
        'summary': {}
    }
    
    # Determine overall status
    if (results['normal_stresses']['overall_status'] == 'FAIL' or
        results['shear_stresses']['overall_status'] == 'FAIL' or
        results['principal_stresses']['overall_status'] == 'FAIL'):
        results['overall_status'] = 'FAIL'
    
    # Create summary
    steel_props = {'A572': 50000, 'A992': 50000, 'A36': 36000}
    Fy = steel_props[steel_grade]
    
    max_normal_util = results['normal_stresses']['critical_stresses'].get('max_normal_utilization', 0)
    max_shear_util = results['shear_stresses']['critical_stresses'].get('max_shear_utilization', 0)
    max_principal_util = results['principal_stresses']['critical_stresses'].get('max_stress_utilization', 0)
    
    results['summary'] = {
        'max_normal_utilization': max_normal_util,
        'max_shear_utilization': max_shear_util,
        'max_principal_utilization': max_principal_util,
        'critical_utilization': max(max_normal_util, max_shear_util, max_principal_util),
        'material_grade': f"{steel_grade}, Fy = {Fy} psi"
    }
    
    return results

# =============================================================================
# 4. EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

def example_stress_checking():
    """
    Example showing how to use the stress checking functions
    """
    
    # Material properties
    concrete_materials = MaterialProperties(concrete_fc=4000, steel_fy=60000)
    
    # Example stress data (all components as specified)
    example_stresses = {
        # Normal stresses
        "sig_zz_n": [1000, -800, 600, -1200, 900],           # Axial load
        "sig_zz_mxx": [1500, -1200, 800, -1800, 1100],       # Bending Mxx
        "sig_zz_myy": [800, -600, 400, -900, 500],           # Bending Myy
        "sig_zz_m11": [1200, -900, 600, -1400, 800],         # Bending M11
        "sig_zz_m22": [900, -700, 500, -1100, 600],          # Bending M22
        "sig_zz_m": [1800, -1500, 1000, -2200, 1300],        # All bending
        
        # Torsional shear stresses
        "sig_zx_mzz": [150, 120, 80, 180, 100],              # Torsion x-component
        "sig_zy_mzz": [120, 100, 60, 150, 80],               # Torsion y-component
        "sig_zxy_mzz": [190, 156, 100, 234, 128],            # Resultant torsion
        
        # Shear force induced stresses - Vx
        "sig_zx_vx": [200, 160, 120, 240, 140],              # Vx x-component
        "sig_zy_vx": [80, 60, 40, 100, 50],                  # Vx y-component
        "sig_zxy_vx": [215, 171, 127, 260, 148],             # Resultant Vx
        
        # Shear force induced stresses - Vy
        "sig_zx_vy": [100, 80, 60, 120, 70],                 # Vy x-component
        "sig_zy_vy": [180, 140, 100, 200, 120],              # Vy y-component
        "sig_zxy_vy": [208, 162, 118, 233, 139],             # Resultant Vy
        
        # Combined shear force stresses
        "sig_zx_v": [300, 240, 180, 360, 210],               # Combined Vx
        "sig_zy_v": [260, 200, 140, 300, 170],               # Combined Vy
        "sig_zxy_v": [396, 312, 230, 469, 273],              # Resultant V
        
        # Combined stresses from all actions
        "sig_zz": [2800, -2300, 1600, -3400, 2100],          # Combined normal
        "sig_zx": [450, 360, 260, 540, 310],                 # Combined x-shear
        "sig_zy": [380, 300, 200, 450, 250],                 # Combined y-shear
        "sig_zxy": [586, 468, 330, 702, 403],                # Combined resultant shear
        
        # Principal stresses
        "sig_11": [2900, 400, 1700, 500, 2200],              # Major principal
        "sig_33": [-100, -2700, -100, -3900, -100],          # Minor principal
        "sig_vm": [2950, 2400, 1650, 3500, 2150]             # Von Mises
    }
    
    print("="*60)
    print("CONCRETE STRESS CHECKING EXAMPLE")
    print("="*60)
    
    # Perform comprehensive concrete stress check
    concrete_results = comprehensive_concrete_stress_check(example_stresses, concrete_materials)
    
    print(f"Overall Status: {concrete_results['overall_status']}")
    print(f"Material: {concrete_results['summary']['material_grade']}")
    print(f"Critical Utilization: {concrete_results['summary']['critical_utilization']:.3f}")
    print(f"Max Normal Utilization: {concrete_results['summary']['max_normal_utilization']:.3f}")
    print(f"Max Shear Utilization: {concrete_results['summary']['max_shear_utilization']:.3f}")
    
    print("\n" + "-"*40)
    print("NORMAL STRESS CHECK DETAILS:")
    normal_results = concrete_results['normal_stresses']
    
    if normal_results['combined_normal_stress_check']:
        combined = normal_results['combined_normal_stress_check']
        print(f"Combined Normal Stress - Max Compression: {combined['max_compression']:.0f} psi")
        print(f"Combined Normal Stress - Max Tension: {combined['max_tension']:.0f} psi")
        print(f"Compression Utilization: {combined['compression_utilization']:.3f}")
        print(f"Tension Utilization: {combined['tension_utilization']:.3f}")
    
    print("\n" + "-"*40)
    print("SHEAR STRESS CHECK DETAILS:")
    shear_results = concrete_results['shear_stresses']
    
    if 'sig_zxy' in shear_results['combined_shear_checks']:
        combined_shear = shear_results['combined_shear_checks']['sig_zxy']
        print(f"Combined Shear Stress: {combined_shear['max_shear']:.0f} psi")
        print(f"Shear Utilization: {combined_shear['utilization']:.3f}")
    
    if normal_results['recommendations']:
        print("\nCONCRETE RECOMMENDATIONS:")
        for rec in normal_results['recommendations']:
            print(f"- {rec}")
    
    print("\n" + "="*60)
    print("STEEL STRESS CHECKING EXAMPLE")
    print("="*60)
    
    # Perform comprehensive steel stress check
    steel_results = comprehensive_steel_stress_check(example_stresses, 'A572')
    
    print(f"Overall Status: {steel_results['overall_status']}")
    print(f"Material: {steel_results['summary']['material_grade']}")
    print(f"Critical Utilization: {steel_results['summary']['critical_utilization']:.3f}")
    print(f"Max Normal Utilization: {steel_results['summary']['max_normal_utilization']:.3f}")
    print(f"Max Shear Utilization: {steel_results['summary']['max_shear_utilization']:.3f}")
    print(f"Max Principal Utilization: {steel_results['summary']['max_principal_utilization']:.3f}")
    
    print("\n" + "-"*40)
    print("STEEL STRESS CHECK DETAILS:")
    
    if steel_results['normal_stresses']['combined_normal_stress_check']:
        combined = steel_results['normal_stresses']['combined_normal_stress_check']
        print(f"Combined Normal Stress: {combined['max_stress']:.0f} psi")
        print(f"Normal Stress Utilization: {combined['utilization']:.3f}")
    
    if steel_results['principal_stresses']['von_mises_check']:
        vm_check = steel_results['principal_stresses']['von_mises_check']
        print(f"Von Mises Stress: {vm_check['max_von_mises']:.0f} psi")
        print(f"Von Mises Utilization: {vm_check['utilization']:.3f}")
    
    if steel_results['normal_stresses']['recommendations']:
        print("\nSTEEL RECOMMENDATIONS:")
        for rec in steel_results['normal_stresses']['recommendations']:
            print(f"- {rec}")
    
    return concrete_results, steel_results

def detailed_stress_report(stresses, material_type='concrete', materials=None, steel_grade='A572'):
    """
    Generate detailed stress checking report for all stress components
    
    Args:
        stresses: Dictionary containing all stress components
        material_type: 'concrete' or 'steel'
        materials: MaterialProperties object (for concrete)
        steel_grade: Steel grade (for steel)
    
    Returns:
        dict: Detailed report with all stress checks
    """
    
    if material_type.lower() == 'concrete':
        if materials is None:
            materials = MaterialProperties()
        
        report = {
            'material_type': 'concrete',
            'material_properties': {
                'fc': materials.fc,
                'tensile_strength': 7.5 * sqrt(materials.fc),
                'allowable_compression': 0.85 * materials.fc / 1.4,
                'allowable_shear': 2 * sqrt(materials.fc)
            },
            'stress_checks': comprehensive_concrete_stress_check(stresses, materials),
            'detailed_component_analysis': {}
        }
        
        # Detailed analysis of each stress component
        all_components = [
            "sig_zz_n", "sig_zz_mxx", "sig_zz_myy", "sig_zz_m11", "sig_zz_m22", "sig_zz_m",
            "sig_zx_mzz", "sig_zy_mzz", "sig_zxy_mzz",
            "sig_zx_vx", "sig_zy_vx", "sig_zxy_vx", "sig_zx_vy", "sig_zy_vy", "sig_zxy_vy",
            "sig_zx_v", "sig_zy_v", "sig_zxy_v",
            "sig_zz", "sig_zx", "sig_zy", "sig_zxy",
            "sig_11", "sig_33", "sig_vm"
        ]
        
        for component in all_components:
            if component in stresses:
                stress_array = np.array(stresses[component])
                
                if component.startswith('sig_zz') or component in ['sig_11', 'sig_33']:
                    # Normal stress analysis
                    max_compression = abs(min(stress_array)) if len(stress_array) > 0 else 0
                    max_tension = max(stress_array) if len(stress_array) > 0 else 0
                    
                    report['detailed_component_analysis'][component] = {
                        'type': 'normal_stress',
                        'max_compression': max_compression,
                        'max_tension': max_tension,
                        'compression_utilization': max_compression / report['material_properties']['allowable_compression'],
                        'tension_utilization': max_tension / report['material_properties']['tensile_strength'],
                        'status': 'PASS'
                    }
                    
                    if (max_compression > report['material_properties']['allowable_compression'] or
                        max_tension > report['material_properties']['tensile_strength']):
                        report['detailed_component_analysis'][component]['status'] = 'FAIL'
                
                else:
                    # Shear stress analysis
                    max_shear = max(np.abs(stress_array)) if len(stress_array) > 0 else 0
                    
                    report['detailed_component_analysis'][component] = {
                        'type': 'shear_stress',
                        'max_shear': max_shear,
                        'utilization': max_shear / report['material_properties']['allowable_shear'],
                        'status': 'PASS' if max_shear <= report['material_properties']['allowable_shear'] else 'FAIL'
                    }
    
    elif material_type.lower() == 'steel':
        steel_props = {'A572': 50000, 'A992': 50000, 'A36': 36000}
        Fy = steel_props[steel_grade]
        
        report = {
            'material_type': 'steel',
            'material_properties': {
                'steel_grade': steel_grade,
                'fy': Fy,
                'allowable_normal': Fy,
                'allowable_shear': 0.6 * Fy
            },
            'stress_checks': comprehensive_steel_stress_check(stresses, steel_grade),
            'detailed_component_analysis': {}
        }
        
        # Detailed analysis of each stress component
        all_components = [
            "sig_zz_n", "sig_zz_mxx", "sig_zz_myy", "sig_zz_m11", "sig_zz_m22", "sig_zz_m",
            "sig_zx_mzz", "sig_zy_mzz", "sig_zxy_mzz",
            "sig_zx_vx", "sig_zy_vx", "sig_zxy_vx", "sig_zx_vy", "sig_zy_vy", "sig_zxy_vy",
            "sig_zx_v", "sig_zy_v", "sig_zxy_v",
            "sig_zz", "sig_zx", "sig_zy", "sig_zxy",
            "sig_11", "sig_33", "sig_vm"
        ]
        
        for component in all_components:
            if component in stresses:
                stress_array = np.array(stresses[component])
                
                if component.startswith('sig_zz') or component in ['sig_11', 'sig_33']:
                    # Normal stress analysis
                    max_stress = max(np.abs(stress_array)) if len(stress_array) > 0 else 0
                    
                    report['detailed_component_analysis'][component] = {
                        'type': 'normal_stress',
                        'max_stress': max_stress,
                        'utilization': max_stress / Fy,
                        'status': 'PASS' if max_stress <= Fy else 'FAIL'
                    }
                
                elif component == 'sig_vm':
                    # Von Mises stress
                    max_vm = max(stress_array) if len(stress_array) > 0 else 0
                    
                    report['detailed_component_analysis'][component] = {
                        'type': 'von_mises_stress',
                        'max_stress': max_vm,
                        'utilization': max_vm / Fy,
                        'status': 'PASS' if max_vm <= Fy else 'FAIL'
                    }
                
                else:
                    # Shear stress analysis
                    max_shear = max(np.abs(stress_array)) if len(stress_array) > 0 else 0
                    
                    report['detailed_component_analysis'][component] = {
                        'type': 'shear_stress',
                        'max_shear': max_shear,
                        'utilization': max_shear / report['material_properties']['allowable_shear'],
                        'status': 'PASS' if max_shear <= report['material_properties']['allowable_shear'] else 'FAIL'
                    }
    
    return report

def print_detailed_report(report):
    """
    Print a formatted detailed stress report
    """
    print("="*80)
    print(f"DETAILED STRESS ANALYSIS REPORT - {report['material_type'].upper()}")
    print("="*80)
    
    if report['material_type'] == 'concrete':
        props = report['material_properties']
        print(f"Concrete Strength (f'c): {props['fc']} psi")
        print(f"Tensile Strength: {props['tensile_strength']:.1f} psi")
        print(f"Allowable Compression: {props['allowable_compression']:.1f} psi")
        print(f"Allowable Shear: {props['allowable_shear']:.1f} psi")
    else:
        props = report['material_properties']
        print(f"Steel Grade: {props['steel_grade']}")
        print(f"Yield Strength (Fy): {props['fy']} psi")
        print(f"Allowable Shear: {props['allowable_shear']} psi")
    
    print(f"\nOverall Status: {report['stress_checks']['overall_status']}")
    print(f"Critical Utilization: {report['stress_checks']['summary']['critical_utilization']:.3f}")
    
    print("\n" + "="*80)
    print("COMPONENT-BY-COMPONENT ANALYSIS")
    print("="*80)
    
    # Group components by type
    normal_components = []
    shear_components = []
    special_components = []
    
    for component, analysis in report['detailed_component_analysis'].items():
        if analysis['type'] == 'normal_stress':
            normal_components.append((component, analysis))
        elif analysis['type'] == 'shear_stress':
            shear_components.append((component, analysis))
        else:
            special_components.append((component, analysis))
    
    # Print normal stress components
    if normal_components:
        print("\nNORMAL STRESS COMPONENTS:")
        print("-" * 40)
        for component, analysis in normal_components:
            status_symbol = "✓" if analysis['status'] == 'PASS' else "✗"
            print(f"{status_symbol} {component:12}")
            
            if report['material_type'] == 'concrete':
                print(f"    Max Compression: {analysis['max_compression']:8.0f} psi (Util: {analysis['compression_utilization']:.3f})")
                print(f"    Max Tension:     {analysis['max_tension']:8.0f} psi (Util: {analysis['tension_utilization']:.3f})")
            else:
                print(f"    Max Stress:      {analysis['max_stress']:8.0f} psi (Util: {analysis['utilization']:.3f})")
    
    # Print shear stress components
    if shear_components:
        print("\nSHEAR STRESS COMPONENTS:")
        print("-" * 40)
        for component, analysis in shear_components:
            status_symbol = "✓" if analysis['status'] == 'PASS' else "✗"
            print(f"{status_symbol} {component:12}")
            print(f"    Max Shear:       {analysis['max_shear']:8.0f} psi (Util: {analysis['utilization']:.3f})")
    
    # Print special components (Von Mises)
    if special_components:
        print("\nSPECIAL STRESS COMPONENTS:")
        print("-" * 40)
        for component, analysis in special_components:
            status_symbol = "✓" if analysis['status'] == 'PASS' else "✗"
            print(f"{status_symbol} {component:12}")
            print(f"    Max Stress:      {analysis['max_stress']:8.0f} psi (Util: {analysis['utilization']:.3f})")
    
    print("\n" + "="*80)























"""
Enhanced Steel Structure Code Checking - AISC 360
Including Buckling, Slenderness, and Advanced Steel Design Checks

This module extends the basic stress checking to include comprehensive
steel design provisions per AISC 360 including stability analysis.
"""

import numpy as np
from math import sqrt, pi, log

# =============================================================================
# STEEL SECTION PROPERTIES CLASS
# =============================================================================

class SteelSection:
    """Steel section properties for comprehensive design checks"""
    
    def __init__(self, section_type='W', **kwargs):
        self.section_type = section_type
        
        # Common properties for all sections
        self.A = kwargs.get('area', 0)           # Cross-sectional area
        self.Ix = kwargs.get('Ix', 0)            # Moment of inertia about x-axis
        self.Iy = kwargs.get('Iy', 0)            # Moment of inertia about y-axis
        self.Sx = kwargs.get('Sx', 0)            # Section modulus about x-axis
        self.Sy = kwargs.get('Sy', 0)            # Section modulus about y-axis
        self.Zx = kwargs.get('Zx', 0)            # Plastic section modulus about x-axis
        self.Zy = kwargs.get('Zy', 0)            # Plastic section modulus about y-axis
        self.rx = kwargs.get('rx', 0)            # Radius of gyration about x-axis
        self.ry = kwargs.get('ry', 0)            # Radius of gyration about y-axis
        self.J = kwargs.get('J', 0)              # Torsional constant
        self.Cw = kwargs.get('Cw', 0)            # Warping constant
        
        # W-shape specific properties
        if section_type == 'W':
            self.d = kwargs.get('depth', 0)          # Overall depth
            self.bf = kwargs.get('flange_width', 0)  # Flange width
            self.tw = kwargs.get('web_thickness', 0) # Web thickness
            self.tf = kwargs.get('flange_thickness', 0) # Flange thickness
            self.k1 = kwargs.get('k1', 0)            # Distance from outer face to web toe of fillet
            
        # HSS properties
        elif section_type in ['HSS_RECT', 'HSS_ROUND']:
            self.B = kwargs.get('width', 0)          # Width (rectangular HSS)
            self.H = kwargs.get('height', 0)         # Height (rectangular HSS)
            self.t = kwargs.get('thickness', 0)      # Wall thickness
            if section_type == 'HSS_ROUND':
                self.D = kwargs.get('diameter', 0)   # Outside diameter
    
    @classmethod
    def W_section(cls, depth, flange_width, web_thickness, flange_thickness, **other_props):
        """Create W-section with calculated properties"""
        d = depth
        bf = flange_width
        tw = web_thickness
        tf = flange_thickness
        
        # Calculate basic properties (simplified)
        A = 2 * bf * tf + (d - 2 * tf) * tw
        
        # More accurate calculations would use steel manual values
        Ix = (bf * d**3 - (bf - tw) * (d - 2*tf)**3) / 12
        Iy = 2 * tf * bf**3 / 12 + (d - 2*tf) * tw**3 / 12
        
        rx = sqrt(Ix / A)
        ry = sqrt(Iy / A)
        
        Sx = 2 * Ix / d
        Sy = 2 * Iy / bf
        
        # Approximate plastic section moduli
        Zx = Sx * 1.12  # Typical ratio for W-sections
        Zy = Sy * 1.5   # Typical ratio for W-sections
        
        props = {
            'area': A, 'Ix': Ix, 'Iy': Iy, 'Sx': Sx, 'Sy': Sy,
            'Zx': Zx, 'Zy': Zy, 'rx': rx, 'ry': ry,
            'depth': d, 'flange_width': bf, 'web_thickness': tw, 'flange_thickness': tf
        }
        props.update(other_props)
        
        return cls(section_type='W', **props)

# =============================================================================
# BUCKLING AND STABILITY CHECKS - AISC 360
# =============================================================================

def check_local_buckling(section, steel_grade='A572', loads=None):
    """
    Check local buckling of plate elements per AISC 360 Chapter B
    
    Args:
        section: SteelSection object
        steel_grade: Steel grade designation
        loads: Dictionary with load information for interaction
    
    Returns:
        dict: Local buckling check results
    """
    steel_props = {
        'A572': {'Fy': 50000, 'Fu': 65000},
        'A992': {'Fy': 50000, 'Fu': 65000},
        'A36': {'Fy': 36000, 'Fu': 58000}
    }
    
    Fy = steel_props[steel_grade]['Fy']
    E = 29000000  # psi
    
    results = {
        'flange_check': {},
        'web_check': {},
        'overall_classification': 'COMPACT',
        'local_buckling_factors': {},
        'recommendations': []
    }
    
    if section.section_type == 'W':
        # Flange local buckling check (AISC 360 Table B4.1a)
        if section.bf > 0 and section.tf > 0:
            bf_2tf = section.bf / (2 * section.tf)
            
            # Limiting width-to-thickness ratios
            lambda_p_flange = 0.38 * sqrt(E / Fy)  # Compact limit
            lambda_r_flange = 1.0 * sqrt(E / Fy)   # Noncompact limit
            
            results['flange_check'] = {
                'bf_2tf_ratio': bf_2tf,
                'lambda_p': lambda_p_flange,
                'lambda_r': lambda_r_flange,
                'classification': 'COMPACT'
            }
            
            if bf_2tf > lambda_p_flange:
                results['flange_check']['classification'] = 'NONCOMPACT'
                results['overall_classification'] = 'NONCOMPACT'
                
            if bf_2tf > lambda_r_flange:
                results['flange_check']['classification'] = 'SLENDER'
                results['overall_classification'] = 'SLENDER'
        
        # Web local buckling check
        if section.d > 0 and section.tw > 0 and section.tf > 0:
            h_tw = (section.d - 2 * section.tf) / section.tw
            
            # For flexure (AISC 360 Table B4.1a)
            lambda_p_web = 3.76 * sqrt(E / Fy)
            lambda_r_web = 5.70 * sqrt(E / Fy)
            
            # Adjust for axial compression if present
            if loads and 'axial_force' in loads and loads['axial_force'] < 0:
                Pr = abs(loads['axial_force'])
                Pc = Fy * section.A  # Simplified
                Pr_Pc = Pr / Pc
                
                if Pr_Pc > 0.125:
                    lambda_p_web = max(1.12 * sqrt(E / Fy) * sqrt(2.33 - Pr_Pc), 
                                      0.64 * sqrt(E / Fy))
                    lambda_r_web = lambda_p_web  # For compression, noncompact = slender
            
            results['web_check'] = {
                'h_tw_ratio': h_tw,
                'lambda_p': lambda_p_web,
                'lambda_r': lambda_r_web,
                'classification': 'COMPACT'
            }
            
            if h_tw > lambda_p_web:
                results['web_check']['classification'] = 'NONCOMPACT'
                if results['overall_classification'] == 'COMPACT':
                    results['overall_classification'] = 'NONCOMPACT'
                    
            if h_tw > lambda_r_web:
                results['web_check']['classification'] = 'SLENDER'
                results['overall_classification'] = 'SLENDER'
    
    # Calculate local buckling reduction factors
    if results['overall_classification'] == 'NONCOMPACT':
        # Linear interpolation between compact and noncompact limits
        if 'flange_check' in results and results['flange_check']['classification'] == 'NONCOMPACT':
            flange_ratio = results['flange_check']['bf_2tf_ratio']
            lambda_p = results['flange_check']['lambda_p']
            lambda_r = results['flange_check']['lambda_r']
            
            # Reduction factor for moment capacity
            results['local_buckling_factors']['Mn_factor'] = min(1.0, 
                1.0 - 0.3 * (flange_ratio - lambda_p) / (lambda_r - lambda_p))
    
    elif results['overall_classification'] == 'SLENDER':
        results['recommendations'].append("Slender elements require special analysis - consider section redesign")
        results['local_buckling_factors']['Mn_factor'] = 0.7  # Conservative estimate
    
    else:
        results['local_buckling_factors']['Mn_factor'] = 1.0
    
    return results

def check_lateral_torsional_buckling(section, steel_grade='A572', unbraced_length=120, 
                                   Cb=1.0, loads=None):
    """
    Check lateral-torsional buckling per AISC 360 Chapter F
    
    Args:
        section: SteelSection object
        steel_grade: Steel grade designation
        unbraced_length: Unbraced length for lateral-torsional buckling (inches)
        Cb: Lateral-torsional buckling modification factor
        loads: Load information
    
    Returns:
        dict: Lateral-torsional buckling check results
    """
    steel_props = {
        'A572': {'Fy': 50000, 'Fu': 65000},
        'A992': {'Fy': 50000, 'Fu': 65000},
        'A36': {'Fy': 36000, 'Fu': 58000}
    }
    
    Fy = steel_props[steel_grade]['Fy']
    E = 29000000  # psi
    G = 11200000  # psi
    
    results = {
        'ltb_limits': {},
        'moment_capacity': {},
        'ltb_classification': 'SHORT',
        'critical_moment': 0,
        'phi_Mn': 0,
        'recommendations': []
    }
    
    if section.section_type == 'W':
        # Calculate limiting unbraced lengths (AISC 360 Eq. F2-5 and F2-6)
        if section.ry > 0 and section.Sx > 0:
            Lp = 1.76 * section.ry * sqrt(E / Fy)  # Plastic limiting length
            
            # For doubly symmetric I-shapes
            if section.J > 0 and section.Cw > 0:
                c = 1.0  # For doubly symmetric sections
                Lr = 1.95 * section.ry * E / (0.7 * Fy) * sqrt(section.J * c / (section.Sx * section.d)) * \
                     sqrt(1 + sqrt(1 + 6.76 * (0.7 * Fy / E)**2 * (section.Sx * section.d / (section.J * c))**2))
            else:
                # Approximate for cases where J and Cw are not available
                Lr = 5.0 * Lp  # Conservative approximation
            
            results['ltb_limits'] = {
                'Lp': Lp,
                'Lr': Lr,
                'L': unbraced_length,
                'Cb': Cb
            }
            
            # Classify the beam
            if unbraced_length <= Lp:
                results['ltb_classification'] = 'SHORT'
                Mn = Fy * section.Zx  # Plastic moment
                
            elif unbraced_length <= Lr:
                results['ltb_classification'] = 'INTERMEDIATE'
                # Linear interpolation between Mp and Mr
                Mp = Fy * section.Zx
                Mr = 0.7 * Fy * section.Sx  # Limiting buckling moment
                
                Mn = Cb * (Mp - (Mp - Mr) * (unbraced_length - Lp) / (Lr - Lp))
                Mn = min(Mn, Mp)
                
            else:
                results['ltb_classification'] = 'LONG'
                # Elastic lateral-torsional buckling
                if section.J > 0 and section.Cw > 0:
                    Fcr = (Cb * pi**2 * E) / (unbraced_length / section.ry)**2 * \
                          sqrt(section.J * section.Cw / section.Ix) / section.Sx
                    Fcr = min(Fcr, 0.7 * Fy)
                else:
                    # Conservative estimate when torsional properties unavailable
                    Fcr = 0.5 * Fy
                
                Mn = Fcr * section.Sx
            
            results['critical_moment'] = Mn
            results['phi_Mn'] = 0.90 * Mn  # LRFD resistance factor
            
            results['moment_capacity'] = {
                'Mn': Mn,
                'phi_Mn': results['phi_Mn'],
                'classification': results['ltb_classification']
            }
            
            # Generate recommendations
            if results['ltb_classification'] == 'LONG':
                results['recommendations'].append(
                    f"Long unbraced length ({unbraced_length:.0f}\" > {Lr:.0f}\") - consider lateral bracing")
            
            if Cb < 1.0:
                results['recommendations'].append(
                    "Consider more accurate Cb calculation based on moment distribution")
    
    return results

def check_member_stability(section, steel_grade='A572', Lx=120, Ly=120, Kx=1.0, Ky=1.0,
                          axial_force=0, moments=None):
    """
    Check overall member stability per AISC 360 Chapter C (compression) and H (combined)
    
    Args:
        section: SteelSection object
        steel_grade: Steel grade designation
        Lx, Ly: Unbraced lengths about x and y axes (inches)
        Kx, Ky: Effective length factors
        axial_force: Applied axial force (negative for compression)
        moments: Dictionary with moments {'Mx': value, 'My': value}
    
    Returns:
        dict: Member stability check results
    """
    steel_props = {
        'A572': {'Fy': 50000, 'Fu': 65000},
        'A992': {'Fy': 50000, 'Fu': 65000},
        'A36': {'Fy': 36000, 'Fu': 58000}
    }
    
    Fy = steel_props[steel_grade]['Fy']
    E = 29000000
    
    results = {
        'compression_check': {},
        'flexure_check': {},
        'interaction_check': {},
        'overall_status': 'PASS',
        'critical_ratios': {},
        'recommendations': []
    }
    
    if moments is None:
        moments = {'Mx': 0, 'My': 0}
    
    # Compression member check (AISC 360 Chapter C)
    if axial_force < 0:  # Compression
        Pr = abs(axial_force)
        
        # Calculate slenderness ratios
        KL_r_x = (Kx * Lx) / section.rx if section.rx > 0 else 0
        KL_r_y = (Ky * Ly) / section.ry if section.ry > 0 else 0
        KL_r = max(KL_r_x, KL_r_y)
        
        # Elastic buckling stress
        Fe = pi**2 * E / (KL_r**2) if KL_r > 0 else float('inf')
        
        # Critical stress (AISC 360 Eq. C-2 and C-3)
        if KL_r <= 4.71 * sqrt(E / Fy):  # Inelastic buckling
            Fcr = (0.658**(Fy / Fe)) * Fy
        else:  # Elastic buckling
            Fcr = 0.877 * Fe
        
        # Nominal compressive strength
        Pn = Fcr * section.A
        phi_Pn = 0.90 * Pn  # LRFD resistance factor
        
        results['compression_check'] = {
            'KL_r_x': KL_r_x,
            'KL_r_y': KL_r_y,
            'KL_r_governing': KL_r,
            'Fe': Fe,
            'Fcr': Fcr,
            'Pn': Pn,
            'phi_Pn': phi_Pn,
            'Pr': Pr,
            'utilization': Pr / phi_Pn,
            'status': 'PASS' if Pr <= phi_Pn else 'FAIL'
        }
        
        if KL_r > 200:
            results['recommendations'].append(f"Slenderness ratio {KL_r:.0f} exceeds 200 - check serviceability")
    
    else:  # Tension
        if axial_force > 0:
            Pt = axial_force
            # Tensile yielding
            Pn_yielding = Fy * section.A
            phi_Pn = 0.90 * Pn_yielding
            
            results['compression_check'] = {
                'Pn': Pn_yielding,
                'phi_Pn': phi_Pn,
                'Pt': Pt,
                'utilization': Pt / phi_Pn,
                'status': 'PASS' if Pt <= phi_Pn else 'FAIL',
                'failure_mode': 'tension_yielding'
            }
    
    # Combined axial and flexure (AISC 360 Chapter H)
    if abs(axial_force) > 0 and (abs(moments['Mx']) > 0 or abs(moments['My']) > 0):
        # Get flexural capacities (simplified - assumes adequate lateral bracing)
        phi_Mnx = 0.90 * Fy * section.Zx
        phi_Mny = 0.90 * Fy * section.Zy
        
        if axial_force < 0:  # Compression + bending
            phi_Pc = results['compression_check']['phi_Pn']
            Pr = abs(axial_force)
            
            # Interaction equations (AISC 360 Eq. H1-1a and H1-1b)
            if Pr / phi_Pc >= 0.2:
                # Equation H1-1a
                ratio = Pr / phi_Pc + (8/9) * (abs(moments['Mx']) / phi_Mnx + abs(moments['My']) / phi_Mny)
            else:
                # Equation H1-1b  
                ratio = Pr / (2 * phi_Pc) + (abs(moments['Mx']) / phi_Mnx + abs(moments['My']) / phi_Mny)
            
            results['interaction_check'] = {
                'governing_equation': 'H1-1a' if Pr / phi_Pc >= 0.2 else 'H1-1b',
                'interaction_ratio': ratio,
                'status': 'PASS' if ratio <= 1.0 else 'FAIL',
                'components': {
                    'axial_ratio': Pr / phi_Pc,
                    'moment_x_ratio': abs(moments['Mx']) / phi_Mnx,
                    'moment_y_ratio': abs(moments['My']) / phi_Mny
                }
            }
        
        else:  # Tension + bending
            phi_Pt = results['compression_check']['phi_Pn']
            Pt = axial_force
            
            # Equation H2-1
            ratio = Pt / phi_Pt + (abs(moments['Mx']) / phi_Mnx + abs(moments['My']) / phi_Mny)
            
            results['interaction_check'] = {
                'governing_equation': 'H2-1',
                'interaction_ratio': ratio,
                'status': 'PASS' if ratio <= 1.0 else 'FAIL',
                'components': {
                    'axial_ratio': Pt / phi_Pt,
                    'moment_x_ratio': abs(moments['Mx']) / phi_Mnx,
                    'moment_y_ratio': abs(moments['My']) / phi_Mny
                }
            }
    
    # Determine overall status
    checks = [results.get('compression_check', {}), results.get('interaction_check', {})]
    for check in checks:
        if check.get('status') == 'FAIL':
            results['overall_status'] = 'FAIL'
    
    # Find critical ratios
    all_ratios = []
    if 'compression_check' in results:
        all_ratios.append(results['compression_check'].get('utilization', 0))
    if 'interaction_check' in results:
        all_ratios.append(results['interaction_check'].get('interaction_ratio', 0))
    
    if all_ratios:
        results['critical_ratios']['max_utilization'] = max(all_ratios)
    
    return results

# =============================================================================
# CONNECTION DESIGN CHECKS
# =============================================================================

def check_bolt_connection(bolt_diameter, steel_grade_bolt='A325', steel_grade_plate='A572',
                         plate_thickness=0.5, bolt_spacing=3.0, edge_distance=1.5,
                         shear_force=0, tension_force=0, num_bolts=4):
    """
    Basic bolt connection design check per AISC 360 Chapter J
    
    Args:
        bolt_diameter: Bolt diameter (inches)
        steel_grade_bolt: Bolt material grade
        steel_grade_plate: Connected plate material grade  
        plate_thickness: Plate thickness (inches)
        bolt_spacing: Center-to-center bolt spacing (inches)
        edge_distance: Distance from bolt center to plate edge (inches)
        shear_force: Applied shear force per bolt (lbs)
        tension_force: Applied tension force per bolt (lbs)
        num_bolts: Number of bolts in connection
    
    Returns:
        dict: Bolt connection check results
    """
    
    bolt_props = {
        'A325': {'Fnv': 54000, 'Fnt': 90000},  # Type N bolts
        'A490': {'Fnv': 68000, 'Fnt': 113000},
        'A307': {'Fnv': 27000, 'Fnt': 45000}
    }
    
    plate_props = {
        'A572': {'Fu': 65000, 'Fy': 50000},
        'A992': {'Fu': 65000, 'Fy': 50000},
        'A36': {'Fu': 58000, 'Fy': 36000}
    }
    
    results = {
        'bolt_shear_check': {},
        'bolt_tension_check': {},
        'bolt_interaction_check': {},
        'bearing_check': {},
        'tearout_check': {},
        'overall_status': 'PASS',
        'recommendations': []
    }
    
    # Bolt properties
    Ab = pi * (bolt_diameter**2) / 4  # Bolt area
    Fnv = bolt_props[steel_grade_bolt]['Fnv']
    Fnt = bolt_props[steel_grade_bolt]['Fnt']
    
    # Plate properties
    Fu_plate = plate_props[steel_grade_plate]['Fu']
    
    # Bolt shear strength (AISC 360 Eq. J3-1)
    phi_Rn_shear = 0.75 * Fnv * Ab
    
    results['bolt_shear_check'] = {
        'applied_shear': shear_force,
        'phi_Rn_shear': phi_Rn_shear,
        'utilization': shear_force / phi_Rn_shear if phi_Rn_shear > 0 else 0,
        'status': 'PASS' if shear_force <= phi_Rn_shear else 'FAIL'
    }
    
    # Bolt tension strength (AISC 360 Eq. J3-2)
    phi_Rn_tension = 0.75 * Fnt * Ab
    
    results['bolt_tension_check'] = {
        'applied_tension': tension_force,
        'phi_Rn_tension': phi_Rn_tension,
        'utilization': tension_force / phi_Rn_tension if phi_Rn_tension > 0 else 0,
        'status': 'PASS' if tension_force <= phi_Rn_tension else 'FAIL'
    }
    
    # Combined shear and tension (AISC 360 Eq. J3-3a)
    if shear_force > 0 and tension_force > 0:
        fnt_required = tension_force / Ab
        fnv_modified = 1.3 * Fnv - (Fnt / (0.75 * Ab)) * tension_force
        fnv_modified = max(fnv_modified, 0)
        
        interaction_ratio = shear_force / (0.75 * fnv_modified * Ab) if fnv_modified > 0 else float('inf')
        
        results['bolt_interaction_check'] = {
            'interaction_ratio': interaction_ratio,
            'fnv_modified': fnv_modified,
            'status': 'PASS' if interaction_ratio <= 1.0 else 'FAIL'
        }
    
    # Bearing strength on bolt (AISC 360 Eq. J3-6a)
    # Assumes clear distance between holes >= 2*bolt_diameter
    Lc = edge_distance - bolt_diameter/2  # Clear distance
    
    if Lc >= 1.5 * bolt_diameter:
        Rn_bearing = 2.4 * bolt_diameter * plate_thickness * Fu_plate
    else:
        Rn_bearing = 1.2 * Lc * plate_thickness * Fu_plate
    
    phi_Rn_bearing = 0.75 * Rn_bearing
    
    results['bearing_check'] = {
        'applied_bearing': shear_force,
        'phi_Rn_bearing': phi_Rn_bearing,
        'utilization': shear_force / phi_Rn_bearing if phi_Rn_bearing > 0 else 0,
        'status': 'PASS' if shear_force <= phi_Rn_bearing else 'FAIL'
    }
    
    # Check minimum spacing and edge distance requirements
    min_spacing = 2.67 * bolt_diameter  # AISC 360 Table J3.3
    min_edge_dist = 1.25 * bolt_diameter  # AISC 360 Table J3.4 (rolled edge)
    
    if bolt_spacing < min_spacing:
        results['recommendations'].append(f"Bolt spacing {bolt_spacing}\" < minimum {min_spacing:.2f}\"")
        
    if edge_distance < min_edge_dist:
        results['recommendations'].append(f"Edge distance {edge_distance}\" < minimum {min_edge_dist:.2f}\"")
    
    # Overall status
    checks = [results['bolt_shear_check'], results['bolt_tension_check'], 
              results.get('bolt_interaction_check', {}), results['bearing_check']]
    
    for check in checks:
        if check.get('status') == 'FAIL':
            results['overall_status'] = 'FAIL'
    
    return results

def check_weld_connection(weld_size, weld_length, steel_grade='A572', electrode='E70',
                         force_parallel=0, force_perpendicular=0, weld_type='fillet'):
    """
    Basic weld connection design check per AISC 360 Chapter J
    
    Args:
        weld_size: Weld size (inches) - leg size for fillet welds
        weld_length: Effective weld length (inches)
        steel_grade: Base metal grade
        electrode: Weld electrode designation
        force_parallel: Force parallel to weld axis (lbs)
        force_perpendicular: Force perpendicular to weld axis (lbs)
        weld_type: Type of weld ('fillet' or 'groove')
    
    Returns:
        dict: Weld connection check results
    """
    
    electrode_props = {
        'E60': {'FEXX': 60000},
        'E70': {'FEXX': 70000},
        'E80': {'FEXX': 80000}
    }
    
    base_metal_props = {
        'A572': {'Fu': 65000, 'Fy': 50000},
        'A992': {'Fu': 65000, 'Fy': 50000},
        'A36': {'Fu': 58000, 'Fy': 36000}
    }
    
    results = {
        'weld_strength_check': {},
        'base_metal_check': {},
        'overall_status': 'PASS',
        'recommendations': []
    }
    
    FEXX = electrode_props[electrode]['FEXX']
    Fu_base = base_metal_props[steel_grade]['Fu']
    
    if weld_type == 'fillet':
        # Fillet weld strength (AISC 360 Eq. J2-4)
        Fnw = 0.60 * FEXX
        
        # Effective throat area
        effective_throat = 0.707 * weld_size  # For equal leg fillet welds
        Aw = effective_throat * weld_length
        
        # Weld metal strength
        phi_Rn_weld = 0.75
                # Weld metal strength
        phi_Rn_weld = 0.75 * Fnw * Aw
        
        # Resultant force on weld
        resultant_force = sqrt(force_parallel**2 + force_perpendicular**2)
        
        results['weld_strength_check'] = {
            'applied_force': resultant_force,
            'phi_Rn_weld': phi_Rn_weld,
            'utilization': resultant_force / phi_Rn_weld if phi_Rn_weld > 0 else 0,
            'status': 'PASS' if resultant_force <= phi_Rn_weld else 'FAIL'
        }
        
        # Base metal shear rupture (AISC 360 Eq. J4-4)
        # Assuming shear occurs along the weld length
        phi_Rn_base = 0.75 * 0.6 * Fu_base * weld_size * weld_length
        
        results['base_metal_check'] = {
            'applied_force': resultant_force,
            'phi_Rn_base': phi_Rn_base,
            'utilization': resultant_force / phi_Rn_base if phi_Rn_base > 0 else 0,
            'status': 'PASS' if resultant_force <= phi_Rn_base else 'FAIL'
        }
    
    elif weld_type == 'groove':
        # Complete joint penetration groove weld strength (AISC 360 J2.5)
        # Assumes matching filler metal
        Fy = base_metal_props[steel_grade]['Fy']  # Get yield strength from material properties

        phi_Rn_weld = 0.90 * Fy * weld_size * weld_length
        
        results['weld_strength_check'] = {
            'applied_force': force_perpendicular,  # Most critical for groove welds
            'phi_Rn_weld': phi_Rn_weld,
            'utilization': force_perpendicular / phi_Rn_weld if phi_Rn_weld > 0 else 0,
            'status': 'PASS' if force_perpendicular <= phi_Rn_weld else 'FAIL'
        }
    
    # Check minimum weld size requirements (AISC 360 Table J2.4)
    if weld_type == 'fillet':
        min_weld_size = max(0.125, min(0.25, 0.0625 + 0.125 * weld_size))  # Simplified
        if weld_size < min_weld_size:
            results['recommendations'].append(
                f"Weld size {weld_size}\" < minimum {min_weld_size:.3f}\" per AISC 360 Table J2.4")
    
    # Overall status
    if (results['weld_strength_check'].get('status') == 'FAIL' or 
        results['base_metal_check'].get('status') == 'FAIL'):
        results['overall_status'] = 'FAIL'
    
    return results

# =============================================================================
# FATIGUE ANALYSIS - AISC 360 APPENDIX 3
# =============================================================================

def fatigue_check(stress_range, detail_category='A', cycles=1000000, 
                 threshold_stress=24, steel_grade='A572'):
    """
    Basic fatigue analysis per AISC 360 Appendix 3
    
    Args:
        stress_range: Maximum stress range (ksi)
        detail_category: Fatigue detail category from AISC 360 Table A-3.1
        cycles: Number of stress range cycles expected
        threshold_stress: Threshold stress range (ksi) for infinite life
        steel_grade: Steel material grade
    
    Returns:
        dict: Fatigue check results
    """
    # Fatigue constants from AISC 360 Table A-3.1
    fatigue_constants = {
        'A': {'C': 250, 'Cf': 1200, 'FTH': 24},
        'B': {'C': 120, 'Cf': 610, 'FTH': 16},
        'B\'': {'C': 61, 'Cf': 305, 'FTH': 12},
        'C': {'C': 44, 'Cf': 220, 'FTH': 10},
        'D': {'C': 22, 'Cf': 110, 'FTH': 7},
        'E': {'C': 11, 'Cf': 55, 'FTH': 4.5},
        'E\'': {'C': 3.9, 'Cf': 18, 'FTH': 2.6}
    }
    
    results = {
        'fatigue_life': {},
        'infinite_life_check': {},
        'recommendations': []
    }
    
    if detail_category not in fatigue_constants:
        raise ValueError(f"Invalid detail category. Must be one of: {list(fatigue_constants.keys())}")
    
    C = fatigue_constants[detail_category]['C']
    Cf = fatigue_constants[detail_category]['Cf']
    FTH = fatigue_constants[detail_category]['FTH']
    
    # Check for infinite life (AISC 360 Eq. A-3-1)
    if stress_range <= FTH:
        results['infinite_life_check'] = {
            'stress_range': stress_range,
            'threshold_stress': FTH,
            'status': 'PASS',
            'message': 'Stress range below threshold for infinite life'
        }
    else:
        results['infinite_life_check'] = {
            'stress_range': stress_range,
            'threshold_stress': FTH,
            'status': 'FAIL',
            'message': 'Stress range exceeds threshold for infinite life'
        }
    
    # Calculate fatigue life (AISC 360 Eq. A-3-2)
    if stress_range > 0:
        N = (Cf / stress_range)**3  # Number of cycles to failure
        
        results['fatigue_life'] = {
            'calculated_life_cycles': N,
            'required_life_cycles': cycles,
            'life_ratio': cycles / N if N > 0 else float('inf'),
            'status': 'PASS' if cycles <= N else 'FAIL'
        }
        
        if cycles > N:
            results['recommendations'].append(
                f"Detail category {detail_category} insufficient for {cycles:.0e} cycles at ΔF = {stress_range} ksi")
    
    return results

# =============================================================================
# COMPREHENSIVE STEEL DESIGN FUNCTION
# =============================================================================

def comprehensive_steel_design(section, steel_grade='A572', loads=None, 
                             connection_params=None, fatigue_params=None):
    """
    Comprehensive steel member design check including:
    - Local and global stability
    - Connection design
    - Fatigue analysis (if applicable)
    
    Args:
        section: SteelSection object
        steel_grade: Steel material grade
        loads: Dictionary of applied loads
        connection_params: Dictionary of connection parameters
        fatigue_params: Dictionary of fatigue parameters
    
    Returns:
        dict: Comprehensive design check results
    """
    if loads is None:
        loads = {}
    if connection_params is None:
        connection_params = {}
    if fatigue_params is None:
        fatigue_params = {}
    
    results = {
        'section_properties': vars(section),
        'material_properties': {'grade': steel_grade},
        'stability_checks': {},
        'connection_checks': {},
        'fatigue_checks': {},
        'overall_status': 'PASS',
        'critical_ratios': {},
        'recommendations': []
    }
    
    # Stability checks
    if 'axial_force' in loads or 'moments' in loads:
        moments = loads.get('moments', {'Mx': 0, 'My': 0})
        axial_force = loads.get('axial_force', 0)
        Lx = loads.get('unbraced_length_x', 120)
        Ly = loads.get('unbraced_length_y', 120)
        Kx = loads.get('effective_length_factor_x', 1.0)
        Ky = loads.get('effective_length_factor_y', 1.0)
        
        # Local buckling check
        local_buckling = check_local_buckling(section, steel_grade, loads)
        results['stability_checks']['local_buckling'] = local_buckling
        
        # LTB check if moments present
        if abs(moments['Mx']) > 0:
            ltb_length = loads.get('ltb_unbraced_length', Ly)
            Cb = loads.get('moment_modifier', 1.0)
            ltb_check = check_lateral_torsional_buckling(
                section, steel_grade, ltb_length, Cb, loads)
            results['stability_checks']['lateral_torsional_buckling'] = ltb_check
        
        # Member stability check
        member_stability = check_member_stability(
            section, steel_grade, Lx, Ly, Kx, Ky, axial_force, moments)
        results['stability_checks']['member_stability'] = member_stability
    
    # Connection checks
    if connection_params:
        # Remove the connection_type parameter before passing to check functions
        conn_params = connection_params.copy()
        conn_params.pop('connection_type', None)
        
        if connection_params.get('connection_type') == 'bolted':
            bolt_check = check_bolt_connection(**conn_params)
            results['connection_checks']['bolted_connection'] = bolt_check
        elif connection_params.get('connection_type') == 'welded':
            weld_check = check_weld_connection(**conn_params)
            results['connection_checks']['welded_connection'] = weld_check
    
    # Fatigue checks
    if fatigue_params and 'stress_range' in fatigue_params:
        fatigue_check_results = fatigue_check(**fatigue_params)
        # Store each fatigue check result separately
        results['fatigue_checks'] = {
            'fatigue_life': fatigue_check_results.get('fatigue_life', {}),
            'infinite_life_check': fatigue_check_results.get('infinite_life_check', {})
        }
    
    # Determine overall status
    for category in ['stability_checks', 'connection_checks']:
        for check_name, check in results[category].items():
            if isinstance(check, dict) and check.get('overall_status') == 'FAIL':
                results['overall_status'] = 'FAIL'
            if isinstance(check, dict) and 'utilization' in check:
                results['critical_ratios'].setdefault('utilizations', []).append(check['utilization'])
            if isinstance(check, dict) and 'interaction_ratio' in check:
                results['critical_ratios'].setdefault('interaction_ratios', []).append(check['interaction_ratio'])
    
    # Check fatigue results separately
    if results['fatigue_checks']:
        for check_name, check in results['fatigue_checks'].items():
            if isinstance(check, dict) and check.get('status') == 'FAIL':
                results['overall_status'] = 'FAIL'
            if isinstance(check, dict) and 'life_ratio' in check:
                results['critical_ratios'].setdefault('fatigue_ratios', []).append(check['life_ratio'])
    
    # Compile critical ratios
    if 'utilizations' in results['critical_ratios']:
        results['critical_ratios']['max_utilization'] = max(results['critical_ratios']['utilizations'])
    if 'interaction_ratios' in results['critical_ratios']:
        results['critical_ratios']['max_interaction'] = max(results['critical_ratios']['interaction_ratios'])
    if 'fatigue_ratios' in results['critical_ratios']:
        results['critical_ratios']['max_fatigue_ratio'] = max(results['critical_ratios']['fatigue_ratios'])
    
    return results

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_steel_design_report(results):
    """
    Generate a comprehensive design report from analysis results
    
    Args:
        results: Dictionary of design check results
    
    Returns:
        str: Formatted design report
    """
    report = []
    
    # Header
    report.append("="*80)
    report.append("STEEL STRUCTURAL DESIGN CHECK REPORT - AISC 360-16")
    report.append("="*80)
    report.append("")
    
    # Section properties
    report.append("SECTION PROPERTIES:")
    report.append("-"*40)
    for prop, value in results['section_properties'].items():
        # Handle both numeric and string properties
        if isinstance(value, (int, float)):
            report.append(f"{prop:>20}: {value:>10.3f}")
        else:
            report.append(f"{prop:>20}: {str(value):>10}")
    report.append(f"{'Material grade':>20}: {results['material_properties']['grade']:>10}")
    report.append("")
    
    # Stability checks
    if 'stability_checks' in results:
        report.append("STABILITY CHECKS:")
        report.append("-"*40)
        
        for check_name, check in results['stability_checks'].items():
            report.append(f"\n{check_name.replace('_', ' ').upper()}:")
            
            if 'classification' in check:
                report.append(f"  Classification: {check['classification']}")
            
            if 'utilization' in check:
                report.append(f"  Utilization ratio: {check['utilization']:.3f}")
            
            if 'interaction_ratio' in check:
                report.append(f"  Interaction ratio: {check['interaction_ratio']:.3f}")
            
            if 'status' in check:
                report.append(f"  Status: {check['status']}")
            
            if check.get('recommendations'):
                report.append("  Recommendations:")
                for rec in check['recommendations']:
                    report.append(f"    - {rec}")
    
    # Connection checks
    if 'connection_checks' in results:
        report.append("\nCONNECTION CHECKS:")
        report.append("-"*40)
        
        for check_name, check in results['connection_checks'].items():
            report.append(f"\n{check_name.replace('_', ' ').upper()}:")
            
            if 'utilization' in check:
                report.append(f"  Utilization ratio: {check['utilization']:.3f}")
            
            if 'interaction_ratio' in check:
                report.append(f"  Interaction ratio: {check['interaction_ratio']:.3f}")
            
            if 'status' in check:
                report.append(f"  Status: {check['status']}")
            
            if check.get('recommendations'):
                report.append("  Recommendations:")
                for rec in check['recommendations']:
                    report.append(f"    - {rec}")
    
    # Fatigue checks
    if 'fatigue_checks' in results:
        report.append("\nFATIGUE CHECKS:")
        report.append("-"*40)
        
        for check_name, check in results['fatigue_checks'].items():
            report.append(f"\n{check_name.replace('_', ' ').upper()}:")
            
            if 'life_ratio' in check:
                report.append(f"  Life ratio: {check['life_ratio']:.3f}")
            
            if 'status' in check:
                report.append(f"  Status: {check['status']}")
            
            if 'message' in check:
                report.append(f"  Note: {check['message']}")
    
    # Summary
    report.append("\nSUMMARY:")
    report.append("-"*40)
    report.append(f"Overall Design Status: {results['overall_status']}")
    
    if 'critical_ratios' in results:
        report.append("\nCritical Ratios:")
        for ratio_name, value in results['critical_ratios'].items():
            if ratio_name not in ['utilizations', 'interaction_ratios']:
                if isinstance(value, (list, tuple)):
                    report.append(f"  {ratio_name}: {max(value):.3f}")
                else:
                    report.append(f"  {ratio_name}: {value:.3f}")
    
    if results['recommendations']:
        report.append("\nDESIGN RECOMMENDATIONS:")
        for rec in results['recommendations']:
            report.append(f"  - {rec}")
    
    return "\n".join(report)
# =============================================================================
# USAGE EXAMPLES
# =============================================================================

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example W-section for steel design
    w14x90 = SteelSection.W_section(
        depth=14.0, flange_width=14.5, web_thickness=0.44, flange_thickness=0.71,
        Ix=999, Iy=362, J=4.06, Cw=12000
    )
    
    # Steel design loads
    design_loads = {
        'axial_force': -250000,  # Compression (negative)
        'moments': {'Mx': 500000, 'My': 100000},  # in-lb
        'unbraced_length_x': 180,  # inches
        'unbraced_length_y': 60,
        'ltb_unbraced_length': 60,
        'effective_length_factor_x': 1.0,
        'effective_length_factor_y': 1.0,
        'moment_modifier': 1.0
    }
    
    # Connection parameters
    connection = {
        'connection_type': 'bolted',
        'bolt_diameter': 0.875,  # 7/8" bolts
        'steel_grade_bolt': 'A325',
        'steel_grade_plate': 'A572',
        'plate_thickness': 0.625,
        'bolt_spacing': 3.0,
        'edge_distance': 1.5,
        'shear_force': 15000,  # per bolt
        'tension_force': 5000,  # per bolt
        'num_bolts': 8
    }
    
    # Fatigue parameters
    fatigue = {
        'stress_range': 18.0,  # ksi
        'detail_category': 'C',
        'cycles': 2000000
    }
    
    # Run comprehensive steel design check
    steel_results = comprehensive_steel_design(
        w14x90, steel_grade='A572', loads=design_loads,
        connection_params=connection, fatigue_params=fatigue
    )
    
    # Generate and print steel design report
    print("\n" + "="*80)
    print("STEEL DESIGN EXAMPLE REPORT")
    print("="*80)
    print(generate_steel_design_report(steel_results))
    
    # Example stress data for concrete detailed report
    example_stresses = {
        "sig_zz_n": [1000, -800, 600, -1200, 900],
        "sig_zz_mxx": [1500, -1200, 800, -1800, 1100],
        "sig_zz_myy": [800, -600, 400, -900, 500],
        "sig_zz_m": [1800, -1500, 1000, -2200, 1300],
        "sig_zxy_mzz": [190, 156, 100, 234, 128],
        "sig_zxy_v": [396, 312, 230, 469, 273],
        "sig_zz": [2800, -2300, 1600, -3400, 2100],
        "sig_zxy": [586, 468, 330, 702, 403],
        "sig_vm": [2950, 2400, 1650, 3500, 2150]
    }
    
    # Generate and print detailed report for concrete
    print("\n" + "="*80)
    print("CONCRETE STRESS CHECK EXAMPLE REPORT")
    print("="*80)
    materials = MaterialProperties(concrete_fc=4000)
    concrete_report = detailed_stress_report(example_stresses, 'concrete', materials)
    print_detailed_report(concrete_report)


