# ACI 318 Code Compliance Checks - Simple Examples

import math

# =============================================================================
# ACI 318 COMPLIANCE CHECK FUNCTIONS (Simplified)
# =============================================================================

def check_beam_strength(phi_Mn, required_Mu):
    """ACI 318: Ultimate Strength Check"""
    if phi_Mn >= required_Mu:
        return True, f"OK: φMn={phi_Mn/1e6:.1f} ≥ Mu={required_Mu/1e6:.1f} kN⋅m"
    else:
        increase_needed = (required_Mu / phi_Mn - 1) * 100
        return False, f"FAIL: Need {increase_needed:.0f}% more capacity"

def check_service_stress(max_stress, fc_prime=30):
    """ACI 318: Service Stress Limits"""
    allowable = 0.45 * fc_prime  # ACI 24.2.4
    
    if max_stress <= allowable:
        return True, f"OK: Max stress {max_stress:.1f} ≤ {allowable:.1f} MPa"
    else:
        return False, f"FAIL: Reduce loads or increase section size"

def check_deflection(calculated_deflection, span_L, beam_type="simply_supported"):
    """ACI 318: Deflection Limits"""
    limits = {
        "simply_supported": span_L / 250,
        "cantilever": span_L / 125,
        "continuous": span_L / 300
    }
    
    limit = limits.get(beam_type, span_L / 250)
    
    if calculated_deflection <= limit:
        return True, f"OK: Deflection {calculated_deflection:.1f} ≤ {limit:.1f} mm"
    else:
        return False, f"FAIL: Use compression steel or increase depth"

def check_minimum_steel(As_actual, b, d, fc_prime=30, fy=500):
    """ACI 318: Minimum Reinforcement"""
    As_min1 = (3 * (fc_prime**0.5) / fy) * b * d  # ACI Eq. 9.6.1.2a
    As_min2 = (200 / fy) * b * d                  # ACI Eq. 9.6.1.2b
    As_min = max(As_min1, As_min2)
    
    if As_actual >= As_min:
        return True, f"OK: As={As_actual:.0f} ≥ As,min={As_min:.0f} mm²"
    else:
        return False, f"FAIL: Add {As_min - As_actual:.0f} mm² more steel"

def check_maximum_steel(As_actual, b, d, fc_prime=30, fy=500):
    """ACI 318: Maximum Reinforcement (Tension-controlled)"""
    epsilon_t_min = 0.005  # ACI 9.3.3.1
    beta1 = 0.85 if fc_prime <= 28 else max(0.65, 0.85 - 0.05*(fc_prime-28)/7)
    
    # Maximum reinforcement ratio for tension-controlled
    rho_max = (0.85 * beta1 * fc_prime / fy) * (0.003 / (0.003 + epsilon_t_min))
    rho_actual = As_actual / (b * d)
    
    if rho_actual <= rho_max:
        return True, f"OK: Tension-controlled (ρ={rho_actual:.4f})"
    else:
        return False, f"FAIL: Over-reinforced. Reduce steel or increase section"

def check_shear_capacity(Vu, phi_Vn):
    """ACI 318: Shear Strength Check"""
    if phi_Vn >= Vu:
        return True, f"OK: φVn={phi_Vn/1000:.0f} ≥ Vu={Vu/1000:.0f} kN"
    else:
        return False, f"FAIL: Increase shear reinforcement"

# =============================================================================
# SIMPLE EXAMPLES WITH SAMPLE VALUES - CHANGE THESE VALUES AS NEEDED
# =============================================================================

def example_1_beam_checks():
    """Example 1: Basic Beam Checks"""
    print("EXAMPLE 1: BEAM DESIGN CHECKS")
    print("=" * 40)
    print("Beam: 300×600mm, fc'=30 MPa, fy=500 MPa")
    print("Steel: 4-20mm bars (1256 mm²)")
    print("-" * 40)
    
    # ========================================
    # CHANGE THESE INPUT VALUES AS NEEDED:
    # ========================================
    phi_Mn = 0.9 * 310e6        # Design moment capacity: 279 kN⋅m
    required_Mu = 250e6         # Required moment: 250 kN⋅m
    max_stress = 12.5           # Service stress: 12.5 MPa
    deflection = 18.0           # Calculated deflection: 18 mm
    span_L = 6000              # Span: 6m
    As_actual = 1256           # Steel area: 1256 mm²
    b, d = 300, 550            # Width, effective depth
    fc_prime = 30              # Concrete strength: 30 MPa
    fy = 500                   # Steel yield strength: 500 MPa
    # ========================================
    
    # Perform checks
    checks = [
        check_beam_strength(phi_Mn, required_Mu),
        check_service_stress(max_stress, fc_prime),
        check_deflection(deflection, span_L),
        check_minimum_steel(As_actual, b, d, fc_prime, fy),
        check_maximum_steel(As_actual, b, d, fc_prime, fy)
    ]
    
    # Print results
    check_names = ["Strength", "Service Stress", "Deflection", "Min Steel", "Max Steel"]
    for i, (passed, message) in enumerate(checks):
        status = "✓" if passed else "✗"
        print(f"{status} {check_names[i]}: {message}")
    
    print(f"\nResult: {'PASS' if all(c[0] for c in checks) else 'FAIL'}")

def example_2_shear_check():
    """Example 2: Shear Check"""
    print("\n" + "=" * 40)
    print("EXAMPLE 2: SHEAR CHECK")
    print("=" * 40)
    
    # ========================================
    # CHANGE THESE INPUT VALUES AS NEEDED:
    # ========================================
    Vu = 180e3          # Applied shear force: 180 kN
    Vc = 85e3           # Concrete shear capacity: 85 kN
    Vs = 120e3          # Steel shear capacity: 120 kN
    phi_v = 0.75        # Shear reduction factor
    # ========================================
    
    phi_Vn = phi_v * (Vc + Vs)
    
    print(f"Vu = {Vu/1000:.0f} kN")
    print(f"φVn = 0.75 × ({Vc/1000:.0f} + {Vs/1000:.0f}) = {phi_Vn/1000:.0f} kN")
    
    passed, message = check_shear_capacity(Vu, phi_Vn)
    status = "✓" if passed else "✗"
    print(f"{status} Shear: {message}")

def example_3_failure_cases():
    """Example 3: Failure Cases"""
    print("\n" + "=" * 40)
    print("EXAMPLE 3: FAILURE CASES")
    print("=" * 40)
    
    # ========================================
    # CHANGE THESE INPUT VALUES AS NEEDED:
    # ========================================
    
    print("A. Under-strength beam:")
    weak_capacity = 0.9 * 200e6  # Only 180 kN⋅m capacity
    required = 250e6             # Need 250 kN⋅m demand
    passed, msg = check_beam_strength(weak_capacity, required)
    print(f"   {'✓' if passed else '✗'} {msg}")
    
    print("\nB. High service stress:")
    high_stress = 18.0  # 18 MPa (exceeds 0.45fc' limit)
    fc_prime = 30       # 30 MPa concrete
    passed, msg = check_service_stress(high_stress, fc_prime)
    print(f"   {'✓' if passed else '✗'} {msg}")
    
    print("\nC. Insufficient steel:")
    low_steel = 600     # Only 600 mm² steel area
    b, d = 300, 550     # Beam dimensions
    fc_prime, fy = 30, 500  # Material properties
    passed, msg = check_minimum_steel(low_steel, b, d, fc_prime, fy)
    print(f"   {'✓' if passed else '✗'} {msg}")
    
    print("\nD. Excessive deflection:")
    high_deflection = 30.0  # 30 mm calculated deflection
    span = 6000            # 6m span length
    passed, msg = check_deflection(high_deflection, span)
    print(f"   {'✓' if passed else '✗'} {msg}")
    # ========================================

def example_4_individual_calls():
    """Example 4: Individual Function Calls"""
    print("\n" + "=" * 40)
    print("EXAMPLE 4: INDIVIDUAL CALLS")
    print("=" * 40)
    
    # ========================================
    # CHANGE THESE INPUT VALUES AS NEEDED:
    # ========================================
    
    # Individual function calls with different values
    print("Testing different scenarios:")
    
    # 1. Strength check - adequate capacity
    phi_Mn_test = 270e6      # Design capacity: 270 kN⋅m
    required_Mu_test = 250e6 # Required moment: 250 kN⋅m
    passed, msg = check_beam_strength(phi_Mn_test, required_Mu_test)
    print(f"1. {msg}")
    
    # 2. Service stress - OK stress level
    service_stress = 11.5    # Service stress: 11.5 MPa
    fc_prime = 30           # Concrete strength: 30 MPa
    passed, msg = check_service_stress(service_stress, fc_prime)
    print(f"2. {msg}")
    
    # 3. Deflection - cantilever beam
    calculated_defl = 15.0   # Calculated deflection: 15 mm
    cantilever_span = 3000   # Cantilever span: 3m
    passed, msg = check_deflection(calculated_defl, cantilever_span, "cantilever")
    print(f"3. {msg}")
    
    # 4. Min steel - small beam
    As_provided = 800        # Provided steel: 800 mm²
    beam_width = 250         # Beam width: 250 mm
    eff_depth = 400          # Effective depth: 400 mm
    fc_prime = 30           # Concrete strength: 30 MPa
    fy = 500                # Steel yield: 500 MPa
    passed, msg = check_minimum_steel(As_provided, beam_width, eff_depth, fc_prime, fy)
    print(f"4. {msg}")
    
    # 5. Max steel - check if over-reinforced
    As_heavy = 2500          # Heavy reinforcement: 2500 mm²
    beam_width = 300         # Beam width: 300 mm
    eff_depth = 500          # Effective depth: 500 mm
    fc_prime = 30           # Concrete strength: 30 MPa
    fy = 500                # Steel yield: 500 MPa
    passed, msg = check_maximum_steel(As_heavy, beam_width, eff_depth, fc_prime, fy)
    print(f"5. {msg}")
    # ========================================

def example_5_quick_calculations():
    """Example 5: Quick Manual Calculations"""
    print("\n" + "=" * 40)
    print("EXAMPLE 5: QUICK CALCULATIONS")
    print("=" * 40)
    
    # ========================================
    # CHANGE THESE INPUT VALUES AS NEEDED:
    # ========================================
    b, d = 300, 500             # Beam width and effective depth (mm)
    total_depth = d + 50        # Total depth including cover
    fc_prime, fy = 25, 500      # Concrete and steel strengths (MPa)
    span = 8000                 # Beam span (mm)
    # ========================================
    
    # Quick calculation examples
    # Calculate minimum steel
    As_min1 = (3 * math.sqrt(fc_prime) / fy) * b * d
    As_min2 = (200 / fy) * b * d
    As_min = max(As_min1, As_min2)
    
    print(f"For {b}×{total_depth}mm beam, fc'={fc_prime} MPa:")
    print(f"As,min = max({As_min1:.0f}, {As_min2:.0f}) = {As_min:.0f} mm²")
    
    # Service stress limit
    stress_limit = 0.45 * fc_prime
    print(f"Service stress limit = 0.45 × {fc_prime} = {stress_limit:.1f} MPa")
    
    # Deflection limit
    defl_limit = span / 250
    print(f"Deflection limit = L/250 = {span}/250 = {defl_limit:.1f} mm")

# =============================================================================
# RUN ALL EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("ACI 318 COMPLIANCE CHECKS - SIMPLE EXAMPLES")
    print("=" * 50)
    
    example_1_beam_checks()
    example_2_shear_check()
    example_3_failure_cases()
    example_4_individual_calls()
    example_5_quick_calculations()
    
    print("\n" + "=" * 50)
    print("USAGE SUMMARY:")
    print("• Call functions with your calculated values")
    print("• Each function returns (True/False, message)")
    print("• Use returned messages for design reports")
    print("• Modify parameters for different materials/codes")