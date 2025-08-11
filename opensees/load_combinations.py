import openseespy.opensees as ops

# load_cases = [
#         # Dead Load (DL) - Node Loads
#         [1, "DL", "node_loads", 5, 0, 0, -5000, 0, 0, 0],  # Node 5
#         [2, "DL", "node_loads", 6, 0, 0, -5000, 0, 0, 0],   # Node 6
        
#         # Dead Load (DL) - Element Uniform Loads
#         [3, "DL", "element_uniform_loads", 1, 0, -2500, 0],  # Element 1
#         [4, "DL", "element_uniform_loads", 2, 0, -2500, 0],  # Element 2
        
#         # Dead Load (DL) - Element Point Loads
#         [5, "DL", "element_point_loads", 1, 1000, -500, 0, 0.5],  # Element 1: Px=1000, Py=-500, Pz=0 at L=0.5
#         [6, "DL", "element_point_loads", 2, 0, 0, -2000, 0.7],    # Element 2: Pz=-2000 at L=0.7
        
#         # Dead Load (DL) - Shell Pressure Loads
#         [7, "DL", "shell_pressure_loads", 101, -1000],
        
#         # Live Load (LL) - Node Loads
#         [8, "LL", "node_loads", 5, 0, 0, -5000, 0, 0, 0],   # Node 5
#         [9, "LL", "node_loads", 6, 0, 0, -5000, 0, 0, 0],    # Node 6
        
#         # Live Load (LL) - Element Uniform Loads
#         [10, "LL", "element_uniform_loads", 1, 0, -2500, 0],  # Element 1
#         [11, "LL", "element_uniform_loads", 2, 0, -2500, 0],  # Element 2
        
#         # Wind Load (WL) - Node Loads
#         [12, "WL", "node_loads", 5, 5000, 0, 0, 0, 0, 0],    # Node 5
#         [13, "WL", "node_loads", 6, 5000, 0, 0, 0, 0, 0],    # Node 6
        
#         # Wind Load (WL) - Shell Pressure Loads
#         [14, "WL", "shell_pressure_loads", 101, -500],
        
#         # Seismic Load (EQ) - Node Loads
#         [15, "EQ", "node_loads", 5, 10000, 0, 0, 0, 0, 0],   # Node 5
#         [16, "EQ", "node_loads", 6, 10000, 0, 0, 0, 0, 0]    # Node 6
#     ]

# load_combinations = [
#         # Combo1: 1.4DL
#         [1, "combo1", "DL", 1.4],
        
#         # Combo2: 1.2DL + 1.6LL
#         [2, "combo2", "DL", 1.2],
#         [3, "combo2", "LL", 1.6],
        
#         # Combo3: 1.2DL + 1.0LL + 1.0WL
#         [4, "combo3", "DL", 1.2],
#         [5, "combo3", "LL", 1.0],
#         [6, "combo3", "WL", 1.0],
        
#         # Combo4: 1.2DL + 1.0LL + 1.0EQ
#         [7, "combo4", "DL", 1.2],
#         [8, "combo4", "LL", 1.0],
#         [9, "combo4", "EQ", 1.0],
        
#         # Combo5: 0.9DL + 1.0WL
#         [10, "combo5", "DL", 0.9],
#         [11, "combo5", "WL", 1.0],
        
#         # Combo6: 0.9DL + 1.0EQ
#         [12, "combo6", "DL", 0.9],
#         [13, "combo6", "EQ", 1.0]
#     ]

def define_load_cases(load_cases):
    """Define all individual load cases as flat lists with all load types"""
    
    return load_cases

def define_load_combinations(load_combinations):
    """Define load combinations as flat lists with factors"""
    
    return load_combinations

def process_load_case(load_case_name, all_load_cases):
    """Extract all loads for a specific case from the flat list"""
    node_loads = []
    element_uniform_loads = []
    element_point_loads = []
    shell_pressure_loads = []
    
    for load in all_load_cases:
        if load[1] == load_case_name:
            if load[2] == "node_loads":
                node_loads.append([load[3], load[4], load[5], load[6], load[7], load[8], load[9]])
            elif load[2] == "element_uniform_loads":
                element_uniform_loads.append([load[3], load[4], load[5]])
            elif load[2] == "element_point_loads":
                element_point_loads.append([load[3], load[4], load[5], load[6], load[7]])
            elif load[2] == "shell_pressure_loads":
                shell_pressure_loads.append([load[3], load[4]])
    
    return {
        "node_loads": node_loads,
        "element_uniform_loads": element_uniform_loads,
        "element_point_loads": element_point_loads,
        "shell_pressure_loads": shell_pressure_loads
    }

def apply_load_combinations(load_cases, load_combinations ):
    """Combine load cases with factors and sum loads on common nodes/elements"""
    all_load_cases = define_load_cases(load_cases)
    all_combinations = define_load_combinations(load_combinations)
    
    combined_results = {}
    
    current_combo_id = None
    for combo in all_combinations:
        combo_id = combo[0]
        combo_name = combo[1]
        case_name = combo[2]
        factor = combo[3]
        
        if combo_id != current_combo_id:
            if combo_name not in combined_results:
                combined_results[combo_name] = {
                    "node_loads": {},
                    "element_uniform_loads": {},
                    "element_point_loads": {},
                    "shell_pressure_loads": {}
                }
            current_combo_id = combo_id
        
        case_loads = process_load_case(case_name, all_load_cases)
        
        # Process node loads
        for node_load in case_loads["node_loads"]:
            node_tag = node_load[0]
            loads = [x * factor for x in node_load[1:]]
            
            if node_tag in combined_results[combo_name]["node_loads"]:
                existing = combined_results[combo_name]["node_loads"][node_tag]
                combined_results[combo_name]["node_loads"][node_tag] = [
                    existing[i] + loads[i] for i in range(len(loads))]
            else:
                combined_results[combo_name]["node_loads"][node_tag] = loads
        
        # Process element uniform loads
        for elem_load in case_loads["element_uniform_loads"]:
            elem_tag = elem_load[0]
            loads = [x * factor for x in elem_load[1:]]
            
            if elem_tag in combined_results[combo_name]["element_uniform_loads"]:
                existing = combined_results[combo_name]["element_uniform_loads"][elem_tag]
                combined_results[combo_name]["element_uniform_loads"][elem_tag] = [
                    existing[i] + loads[i] for i in range(len(loads))]
            else:
                combined_results[combo_name]["element_uniform_loads"][elem_tag] = loads
        
        # Process element point loads
        for point_load in case_loads["element_point_loads"]:
            elem_tag = point_load[0]
            location = point_load[4]
            key = (elem_tag, location)
            
            loads = [x * factor for x in point_load[1:4]]  # Px, Py, Pz
            
            if key in combined_results[combo_name]["element_point_loads"]:
                existing = combined_results[combo_name]["element_point_loads"][key]
                combined_results[combo_name]["element_point_loads"][key] = [
                    existing[i] + loads[i] for i in range(len(loads))]
            else:
                combined_results[combo_name]["element_point_loads"][key] = loads
        
        # Process shell pressure loads
        for shell_load in case_loads["shell_pressure_loads"]:
            shell_tag = shell_load[0]
            pressure = shell_load[1] * factor
            
            if shell_tag in combined_results[combo_name]["shell_pressure_loads"]:
                combined_results[combo_name]["shell_pressure_loads"][shell_tag] += pressure
            else:
                combined_results[combo_name]["shell_pressure_loads"][shell_tag] = pressure
    
    return combined_results

def get_combination_loads(load_cases, load_combinations , combo_name):
    """Get combined loads for a specific combination in analysis-ready format"""
    combined_results = apply_load_combinations(load_cases, load_combinations )
    
    if combo_name not in combined_results:
        raise ValueError(f"Combination {combo_name} not found")
    
    combo_data = combined_results[combo_name]
    
    # Convert to lists in expected format
    node_loads = [[tag] + loads for tag, loads in combo_data["node_loads"].items()]
    element_uniform_loads = [[tag] + loads for tag, loads in combo_data["element_uniform_loads"].items()]
    element_point_loads = [[elem_tag] + loads + [loc] for (elem_tag, loc), loads in combo_data["element_point_loads"].items()]
    shell_pressure_loads = [[tag, pressure] for tag, pressure in combo_data["shell_pressure_loads"].items()]
    
    return {
        "node_loads": node_loads,
        "element_uniform_loads": element_uniform_loads,
        "element_point_loads": element_point_loads,
        "shell_pressure_loads": shell_pressure_loads
    }

def apply_structural_loads(load_cases, load_combinations, combo_name, pattern_tag=1, time_series_tag=1):
    """
    Apply all loads for a combination to the OpenSees model
    
    Args:
        combo_name (str): Name of the load combination to apply
        pattern_tag (int): Tag for the load pattern (default=1)
        time_series_tag (int): Tag for the time series (default=1)
        
    Returns:
        dict: The load data that was applied
    """
    # Get the combined loads for this combination
    load_data = get_combination_loads(load_cases, load_combinations , combo_name)
    
    # Set up load pattern
    ops.timeSeries("Linear", time_series_tag)
    ops.pattern("Plain", pattern_tag, time_series_tag)
    
    # Apply node loads (Fx, Fy, Fz, Mx, My, Mz)
    for node_tag, *load_components in load_data["node_loads"]:
        ops.load(int(node_tag), *load_components)
    
    # Apply element uniform loads (Wy, Wz)
    for elem_tag, wy, wz in load_data["element_uniform_loads"]:
        ops.eleLoad("-ele", int(elem_tag), "-type", "-beamUniform", wy, wz)
    
    # Apply element point loads (Px, Py, Pz, location)
    # Note: OpenSees expects order: Pz, Py, location, Px
    for elem_tag, px, py, pz, loc in load_data["element_point_loads"]:
        ops.eleLoad("-ele", int(elem_tag), "-type", "-beamPoint", 
                   float(pz), float(py), float(loc), float(px))
    
    # Apply shell pressure loads
    # for shell_tag, pressure in load_data["shell_pressure_loads"]:
    #     ops.eleLoad("-ele", int(shell_tag), "-type", "-surfaceLoad", float(pressure))
    
    return load_data

# # Define your load cases and combinations (same as before)
# load_cases = [
#     [1, "DL", "node_loads", 5, 0, 0, -5000, 0, 0, 0],
#     [2, "DL", "node_loads", 6, 0, 0, -5000, 0, 0, 0],
#     [3, "DL", "element_uniform_loads", 1, 0, -2500, 0],
#     [4, "DL", "element_uniform_loads", 2, 0, -2500, 0],
#     [8, "LL", "node_loads", 5, 0, 0, -5000, 0, 0, 0],
#     [9, "LL", "node_loads", 6, 0, 0, -5000, 0, 0, 0],
# ]

# load_combinations = [
#     [1, "combo1", "DL", 1.4],
#     [2, "combo2", "DL", 1.2],
#     [3, "combo2", "LL", 1.6],
#     [4, "combo3", "DL", 1.2],
#     [5, "combo3", "LL", 1.0],
# ]

# # List of combinations you want to apply
# # Extract unique combination names from load_combinations
# combos_to_apply = list({combo[1] for combo in load_combinations})

# # Apply each combination in a loop
# for i, combo_name in enumerate(combos_to_apply, start=1):
#     print(f"\nApplying combination: {combo_name}")
    
#     applied_loads = apply_structural_loads(
#         load_cases=load_cases,
#         load_combinations=load_combinations,
#         combo_name=combo_name,
#         pattern_tag=i,  # Unique pattern tag for each combination
#         time_series_tag=i  # Unique time series tag for each combination
#     )
       
#     # Clear the pattern before applying the next one
#     ops.remove('loadPattern', i)