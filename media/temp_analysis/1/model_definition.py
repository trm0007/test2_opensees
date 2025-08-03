# Type of Analysis:
analysis_type = "gravity_analysis"
# analysis_type = "response_spectrum_analysis"
                    # the response spectrum function
Tn = [0.0, 0.06, 0.1, 0.12, 0.18, 0.24, 0.3, 0.36, 0.4, 0.42, 
    0.48, 0.54, 0.6, 0.66, 0.72, 0.78, 0.84, 0.9, 0.96, 1.02, 
    1.08, 1.14, 1.2, 1.26, 1.32, 1.38, 1.44, 1.5, 1.56, 1.62, 
    1.68, 1.74, 1.8, 1.86, 1.92, 1.98, 2.04, 2.1, 2.16, 2.22, 
    2.28, 2.34, 2.4, 2.46, 2.52, 2.58, 2.64, 2.7, 2.76, 2.82, 
    2.88, 2.94, 3.0, 3.06, 3.12, 3.18, 3.24, 3.3, 3.36, 3.42, 
    3.48, 3.54, 3.6, 3.66, 3.72, 3.78, 3.84, 3.9, 3.96, 4.02, 
    4.08, 4.14, 4.2, 4.26, 4.32, 4.38, 4.44, 4.5, 4.56, 4.62, 
    4.68, 4.74, 4.8, 4.86, 4.92, 4.98, 5.04, 5.1, 5.16, 5.22, 
    5.28, 5.34, 5.4, 5.46, 5.52, 5.58, 5.64, 5.7, 5.76, 5.82, 
    5.88, 5.94, 6.0]
Sa = [1.9612, 3.72628, 4.903, 4.903, 4.903, 4.903, 4.903, 4.903, 4.903, 4.6696172, 
    4.0861602, 3.6321424, 3.2683398, 2.971218, 2.7241068, 2.5142584, 2.3348086, 2.1788932, 2.0425898, 1.9229566, 
    1.8160712, 1.7199724, 1.6346602, 1.5562122, 1.485609, 1.4208894, 1.3620534, 1.3071398, 1.2571292, 1.211041, 
    1.166914, 1.1267094, 1.0894466, 1.054145, 1.0217852, 0.990406, 0.960988, 0.9335312, 0.9080356, 0.8835206, 
    0.8599862, 0.838413, 0.8168398, 0.7972278, 0.7785964, 0.759965, 0.7432948, 0.7266246, 0.710935, 0.6952454, 
    0.6805364, 0.666808, 0.6540602, 0.6285646, 0.6040496, 0.5814958, 0.5609032, 0.5403106, 0.5206986, 0.5030478, 
    0.485397, 0.4697074, 0.4540178, 0.4393088, 0.4255804, 0.411852, 0.3991042, 0.3863564, 0.3755698, 0.3638026, 
    0.353016, 0.34321, 0.333404, 0.3245786, 0.3157532, 0.3069278, 0.2981024, 0.2902576, 0.2833934, 0.2755486, 
    0.2686844, 0.2618202, 0.254956, 0.2490724, 0.2431888, 0.2373052, 0.2314216, 0.2265186, 0.220635, 0.215732, 
    0.210829, 0.205926, 0.2020036, 0.1971006, 0.1931782, 0.1892558, 0.1853334, 0.181411, 0.1774886, 0.1735662, 
    0.1706244, 0.166702, 0.1637602]
num_points = 5
# =============================================
# 1. MATERIAL PROPERTIES
# ["Elastic", material tag, E - Young's modulus in Pa]
# ["ENT", material tag, stiffness value in N/m]
# =============================================
materials = [
    ["Elastic", 2, 938000000.0],  # Shear material
    ["ENT", 101, 1.0e6],          # Spring in X direction (1MN/m)
    ["ENT", 102, 1.0e6],          # Spring in Y direction
    ["ENT", 103, 1.0e6]           # Spring in Z direction
]

# nD Materials (for shells)
# ["ElasticIsotropic", material tag, E - Young's modulus in Pa, v - Poisson's ratio]
nd_materials = [
    ["ElasticIsotropic", 10, 30000000000.0, 0.2]  # E=30GPa, v=0.2
]

# =============================================
# 2. SECTION PROPERTIES
# Store section properties as lists with the format:
# [section_tag, type, A, Iy, Iz, J, B, H, t]
# =============================================
section_properties = [
    # tag, type,       A,    Iy,       Iz,       J,        B,    H,    t
    [1,    'rectangular', 0.09, 0.000675, 0.000675, 0.00114075, 0.3, 0.3, None],
    [3,    'rectangular', 0.09, 0.000675, 0.000675, 0.00114075, 0.3, 0.3, None]  # Aggregator uses same properties
]

# Elastic section definition using the properties from the list
elastic_section = ["Elastic", 1, 30000000000.0, 
                    section_properties[0][2],  # A
                    section_properties[0][4],  # Iz
                    section_properties[0][3],  # Iy
                    12500000000.0,            # G
                    section_properties[0][5]] # J

# Aggregator section
aggregator_section = [
    "Aggregator", 3, 
    2, "Vy", 2, "Vz", "-section", 1
]

# Shell section
shell_section = [
    "PlateFiber", 20, 10, 0.15  # 15cm thick shell
]

# =============================================
# 3. NODE DEFINITIONS
# [nodeTag, x-coord in m, y-coord in m, z-coord in m, mass [mx, my, mz, mr1, mr2, mr3]]
# =============================================
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

# =============================================
# 4. GEOMETRIC TRANSFORMATIONS
# ["Linear", transformation tag, vecxzX, vecxzY, vecxzZ]
# =============================================
transformations = [
    ["Linear", 1, 1.0, 0.0, -0.0],
    ["Linear", 2, 0.0, 0.0, 1.0],
    ["Linear", 3, 1.0, 0.0, -0.0],
    ["Linear", 4, 1.0, 0.0, -0.0],
    ["Linear", 5, 0.0, 0.0, 1.0],
    ["Linear", 6, 0.0, 0.0, 1.0],
    ["Linear", 7, 0.0, 0.0, 1.0],
    ["Linear", 8, 0.0, 0.0, 1.0],
    ["Linear", 9, 0.0, 0.0, 1.0],
    ["Linear", 10, 1.0, 0.0, -0.0],
    ["Linear", 11, 1.0, 0.0, -0.0],
    ["Linear", 12, 1.0, 0.0, -0.0],
    ["Linear", 13, 0.0, 0.0, 1.0],
    ["Linear", 14, 0.0, 0.0, 1.0],
    ["Linear", 15, 1.0, 0.0, -0.0],
    ["Linear", 16, 1.0, 0.0, -0.0],
    ["Linear", 20, 0.0, 0.0, 1.0]  # For shell elements
]

# =============================================
# 5. BEAM INTEGRATION
# ["Lobatto", integration tag, section tag, Np - number of integration points]
# =============================================
beam_integrations = [
    ["Lobatto", 1, 3, 5]
]

# =============================================
# 6. ELEMENT CONNECTIONS
# ["forceBeamColumn", element tag, iNode, jNode, transformation tag, integration tag]
# =============================================
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

# Shell elements format: [type, tag, node1, node2, node3, node4, secTag]
shell_elements = [
    ["ShellMITC4", 101, 2, 3, 11, 9, 20],
    # ["ShellMITC4", 102, 15, 4, 3, 17, 20],
    # ["ShellMITC4", 103, 17, 3, 11, 9, 20],
    # ["ShellMITC4", 104, 2, 17, 9, 10, 20]
]


# =============================================
# 7. BOUNDARY CONDITIONS
# [nodeTag, fixX, fixY, fixZ, fixRX, fixRY, fixRZ] (1=fixed, 0=free)
# =============================================
fixities = [
    [1, 1, 1, 1, 1, 1, 1],
    [10, 1, 1, 1, 1, 1, 1],
    [4, 1, 1, 1, 1, 1, 1],
    [12, 1, 1, 1, 1, 1, 1],
    [13, 0, 0, 1, 1, 1, 0],
    [14, 0, 0, 1, 1, 1, 0]
]

# =============================================
# 8. RIGID DIAPHRAGMS
# Format: [perpDirn, masterNode, *slaveNodes]
# =============================================
diaphragms = [
    [3, 14, 2, 3, 9, 11],
    [3, 13, 5, 6, 7, 8]
]

# =============================================
# 9. LOAD DEFINITIONS
# [loadTag, nodeTag, Fx, Fy, Fz, Mx, My, Mz]
# =============================================
node_loads = [
    [1, 5, 0, 0, -10000, 0, 0, 0],  # 10kN vertical load at node 5
    [2, 6, 0, 0, -10000, 0, 0, 0]   # 10kN vertical load at node 6
]

element_uniform_loads = [
    [1, 1, 0, -5000, 0],  # 5kN/m vertical load on element 1
    [2, 2, 0, -5000, 0]   # 5kN/m vertical load on element 2
]

"""Define all individual load cases as flat lists with all load types"""
load_cases = [
        # Dead Load (DL) - Node Loads
        [1, "DL", "node_loads", 5, 0, 0, -5000, 0, 0, 0],  # Node 5
        [2, "DL", "node_loads", 6, 0, 0, -5000, 0, 0, 0],   # Node 6
        
        # Dead Load (DL) - Element Uniform Loads
        [3, "DL", "element_uniform_loads", 1, 0, -2500, 0],  # Element 1
        [4, "DL", "element_uniform_loads", 2, 0, -2500, 0],  # Element 2
        
        # Dead Load (DL) - Element Point Loads
        [5, "DL", "element_point_loads", 1, 1000, -500, 0, 0.5],  # Element 1: Px=1000, Py=-500, Pz=0 at L=0.5
        [6, "DL", "element_point_loads", 2, 0, 0, -2000, 0.7],    # Element 2: Pz=-2000 at L=0.7
        
        # Dead Load (DL) - Shell Pressure Loads
        [7, "DL", "shell_pressure_loads", 101, -1000],
        
        # Live Load (LL) - Node Loads
        [8, "LL", "node_loads", 5, 0, 0, -5000, 0, 0, 0],   # Node 5
        [9, "LL", "node_loads", 6, 0, 0, -5000, 0, 0, 0],    # Node 6
        
        # Live Load (LL) - Element Uniform Loads
        [10, "LL", "element_uniform_loads", 1, 0, -2500, 0],  # Element 1
        [11, "LL", "element_uniform_loads", 2, 0, -2500, 0],  # Element 2
        
        # Wind Load (WL) - Node Loads
        [12, "WL", "node_loads", 5, 5000, 0, 0, 0, 0, 0],    # Node 5
        [13, "WL", "node_loads", 6, 5000, 0, 0, 0, 0, 0],    # Node 6
        
        # Wind Load (WL) - Shell Pressure Loads
        [14, "WL", "shell_pressure_loads", 101, -500],
        
        # Seismic Load (EQ) - Node Loads
        [15, "EQ", "node_loads", 5, 10000, 0, 0, 0, 0, 0],   # Node 5
        [16, "EQ", "node_loads", 6, 10000, 0, 0, 0, 0, 0]    # Node 6
    ]

"""Define load combinations as flat lists with factors"""
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
        [13, "combo6", "EQ", 1.0]
    ]

shell_pressure_loads = [
    # [101, 101, -2000],  # 2kPa pressure on shell 101
    # [102, 102, -2000]   # 2kPa pressure on shell 102
]

# Zero length elements
zero_length_elements = [
    # [2001, 1, 1001, 101, 102, 103],  # Base spring
    # [2002, 4, 1004, 101, 102, 103],
    # [2003, 10, 1010, 101, 102, 103],
    # [2004, 12, 1012, 101, 102, 103]
]