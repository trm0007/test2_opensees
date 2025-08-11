import numpy as np
import math

from opensees.seismic import get_occupancy_category
# Table 6.2.12: Wind Directionality Factor, Kd
def wind_directionality_factor(structure_type):
    """
    Returns the wind directionality factor K_d based on the structure type.
    
    Parameters:
    structure_type (str): The type of structure as defined in Table 6.2.12
    
    Returns:
    float: The directionality factor K_d
    """
    kd_values = {
        # Buildings
        "Main Wind Force Resisting System": 0.85,
        "Components and Cladding": 0.85,
        "Arched Roofs": 0.85,
        
        # Chimneys, Tanks, and Similar Structures
        "Square": 0.90,
        "Hexagonal": 0.95,
        "Round": 0.95,
        
        # Solid Signs
        "Solid Signs": 0.85,
        
        # Open Signs and Lattice Framework
        "Open Signs and Lattice Framework": 0.85,
        
        # Trussed Towers
        "Triangular, square, rectangular": 0.85,
        "All other cross section": 0.95
    }
    
    # Try to get the value, return None if not found
    return kd_values.get(structure_type, None)

    def get_factor(data, search_type):
        """Recursive helper to find the factor"""
        if isinstance(data, dict):
            for key, value in data.items():
                if key == search_type:
                    if isinstance(value, (int, float)):
                        return value
                    else:
                        # Return default value if nested dict found
                        return 0.85
                elif isinstance(value, dict):
                    result = get_factor(value, search_type)
                    if result is not None:
                        return result
        return None

    factor = get_factor(factors, structure_type)
    return factor if factor is not None else 0.85  # Default value
# Table 6.2.12: Wind Directionality Factor, Kd

# 2.4.7.2
def calculate_topographic_factor(k1,k2,k3):
    """
    Calculate the topographic factor for wind speed-up effect.

    Parameters:
        k1 (float): Value from Figure 6.2.4
        k2 (float): Value from Figure 6.2.4
        k3 (float): Value from Figure 6.2.4

    Returns:
        float: Topographic factor
        # 2.4.7.2
    """
    kzt = (1 + k1*k2*k3)**2
    return kzt
# 2.4.7.2

# Table 6.2.8: Basic Wind Speeds, V, for Selected Locations in Bangladesh
def basic_wind_speed(location):
    # Table 6.2.8: Basic Wind Speeds, V, for Selected Locations in Bangladesh
    wind_speeds = {
        "Angarpota": 47.8, "Lalmonirhat": 63.7,
        "Bagerhat": 77.5, "Madaripur": 68.1,
        "Bandarban": 62.5, "Magura": 65.0,
        "Barguna": 80.0, "Manikganj": 58.2,
        "Barisal": 78.7, "Meherpur": 58.2,
        "Bhola": 69.5, "Maheshkhali": 80.0,
        "Bogra": 61.9, "Moulvibazar": 53.0,
        "Brahmanbaria": 56.7, "Munshiganj": 57.1,
        "Chandpur": 50.6, "Mymensingh": 67.4,
        "Chapai Nawabganj": 41.4, "Naogaon": 55.2,
        "Chittagong": 80.0, "Narail": 68.6,
        "Chuadanga": 61.9, "Narayanganj": 61.1,
        "Comilla": 61.4, "Narsinghdi": 59.7,
        "Cox’s Bazar": 80.0, "Natore": 61.9,
        "Dahagram": 47.8, "Netrokona": 65.6,
        "Dhaka": 65.7, "Nilphamari": 44.7,
        "Dinajpur": 41.4, "Noakhali": 57.1,
        "Faridpur": 63.1, "Pabna": 63.1,
        "Feni": 64.1, "Panchagarh": 41.4,
        "Gaibandha": 65.6, "Patuakhali": 80.0,
        "Gazipur": 66.5, "Pirojpur": 80.0,
        "Gopalganj": 74.5, "Rajbari": 59.1,
        "Habiganj": 54.2, "Rajshahi": 49.2,
        "Hatiya": 80.0, "Rangamati": 56.7,
        "Ishurdi": 69.5, "Rangpur": 65.3,
        "Joypurhat": 56.7, "Satkhira": 57.6,
        "Jamalpur": 56.7, "Shariatpur": 61.9,
        "Jessore": 64.1, "Sherpur": 62.5,
        "Jhalakati": 80.0, "Sirajganj": 50.6,
        "Jhenaidah": 65.0, "Srimangal": 50.6,
        "Khagrachhari": 56.7, "St. Martin’s Island": 80.0,
        "Khulna": 73.3, "Sunamganj": 61.1,
        "Kutubdia": 80.0, "Sylhet": 61.1,
        "Kishoreganj": 64.7, "Sandwip": 80.0,
        "Kurigram": 65.6, "Tangail": 50.6,
        "Kushtia": 66.9, "Teknaf": 80.0,
        "Lakshmipur": 51.2, "Thakurgaon": 41.4
    }

    # Check if the location is in the wind_speeds dictionary
    if location in wind_speeds:
        return wind_speeds[location]
    else:
        print("Location not found.")
        return None
# Table 6.2.8: Basic Wind Speeds, V, for Selected Locations in Bangladesh

# Table 6.2.9: Importance Factor, I (Wind Loads)
def importance_factor(occupancy_category, basic_wind_speed):
    # Table 6.2.9: Importance Factor, I (Wind Loads)
    # Importance factors for different occupancy categories based on basic wind speed ranges
    importance_factors = {
        "I": {"Non-Cyclone Prone Regions": 0.87, "Cyclone Prone Regions (V 5 38-44 m/s)": 0.87,
              "Cyclone Prone Regions (V > 44 m/s)": 0.77},
        "II": {"Non-Cyclone Prone Regions": 1.0, "Cyclone Prone Regions (V 5 38-44 m/s)": 1.0,
               "Cyclone Prone Regions (V > 44 m/s)": 1.0},
        "III": {"Non-Cyclone Prone Regions": 1.15, "Cyclone Prone Regions (V 5 38-44 m/s)": 1.15,
                "Cyclone Prone Regions (V > 44 m/s)": 1.15},
        "IV": {"Non-Cyclone Prone Regions": 1.15, "Cyclone Prone Regions (V 5 38-44 m/s)": 1.15,
               "Cyclone Prone Regions (V > 44 m/s)": 1.15}
    }

    # Check if the occupancy category is in the importance_factors dictionary
    if occupancy_category in importance_factors:
        if basic_wind_speed >= 44:
            cyclone_prone_region_key = "Cyclone Prone Regions (V > 44 m/s)"
        elif 38 <= basic_wind_speed <= 44:
            cyclone_prone_region_key = "Cyclone Prone Regions (V 5 38-44 m/s)"
        else:
            cyclone_prone_region_key = "Non-Cyclone Prone Regions"

        return importance_factors[occupancy_category][cyclone_prone_region_key]
    else:
        print( "Occupancy category not found." )
        return None
# Table 6.2.9: Importance Factor, I (Wind Loads)


def terrain_exposure_constants(exposure_category):
    """
    Retrieve terrain exposure constants for the specified exposure category.

    Args:
    exposure_category (str): The exposure category (A, B, or C).

    Returns:
    dict: Dictionary containing the terrain exposure constants.
    """
    # Define the table as a dictionary
    # Table 6.2.10: Terrain Exposure Constants
    constants = {
        "A": {
            "exposure": "A",
            "alpha": 7.0,           # ¯ˆ˘ (m): Height of the terrain (m)
            "zg": 365.76,           # ¸˝ ˛_: Terrain roughness (m)
            "a": 1/7,               # ¸_: Constant 'a' for terrain exposure
            "b": 0.84,              # c  (m): Constant 'b' for terrain exposure
            "a_var": 1/4.0,         # _ ˆ (m): Constant 'a_var' for terrain exposure
            "b_var": 0.45,          # ˆ: Constant 'b_var' for terrain exposure
            "c": 0.30,              # ˆ: Constant 'c' for terrain exposure
            "L": 97.54,             # L: Maximum fetch (m)
            "epsilon": 1/3.0,       # 1/α: Ratio of terrain roughness to structure height
            "z_min": 9.14           # z_min: Minimum height used to ensure that the equivalent height z is greater of 0.6*h or z_min
        },
        "B": {
            "exposure": "B",
            "alpha": 9.5,
            "zg": 274.32,
            "a": 1/9.5,
            "b": 1.00,
            "a_var": 1/6.5,
            "b_var": 0.65,
            "c": 0.20,
            "L": 152.4,
            "epsilon": 1/5.0,
            "z_min": 4.57
        },
        "C": {
            "exposure": "C",
            "alpha": 11.5,
            "zg": 213.36,
            "a": 1/11.5,
            "b": 1.07,
            "a_var": 1/9.0,
            "b_var": 0.80,
            "c": 0.15,
            "L": 198.12,
            "epsilon": 1/8.0,
            "z_min": 2.13
        }
    }

    # Check if the exposure category is in the constants dictionary
    if exposure_category in constants:
        return constants[exposure_category]
    else:
        print("Exposure category not found.")
        return None
# Table 6.2.10: Terrain Exposure Constants



def velocity_pressure_coefficient(z, zg, alpha, exposure_category):
    """
    Calculate the velocity pressure exposure coefficient Kz.
    
    Args:
    z (float): Height above ground level (m).
    zg (float): Height of the structure above ground level (m).
    alpha (float): Coefficient depending on exposure category.
    exposure_category (str): Exposure category (A, B, or C).
    
    Returns:
    float: Velocity pressure exposure coefficient Kz.
    """
    # Handle minimum heights based on exposure category
    if exposure_category == "A":
        z = max(z, 9.1)  # Minimum height for Exposure A
    elif exposure_category in ["B", "C"]:
        z = max(z, 4.57)  # Minimum height for Exposures B and C
    
    return 2.01 * (z / zg) ** (2 / alpha)



# exposure,alpha,zg,a,b,a_var,b_var,c,L,epsilon,z_min

def Compute_Gust_factor(T,z, z_min,c,l,epsilon,wind_speed,a_var,b_var,h,L,B):
    """
    Computes various parameters for flexible buildings or structures based on provided inputs.

    Parameters:
    n (float): Fundamental frequency of the structure.
    Vkph (float): Velocity in kilometers per hour.

    Returns:
    tuple: A tuple containing computed values for V2, R, RRB, RL, R9, I, Q, and G.
    """

    # Constants and initial values
    n1 = 1/T # n1 = building natural frequency
    B_damping_percentage = 0.01  # 1% for steel and 2% for concrete
    V = 0.2778 * wind_speed
    z_var = np.maximum(0.6*h, z_min)
    # print(f'z_var={z_var}')

    # 1. Compute V2
    Vz = b_var * (z_var / 10)**a_var * V
    # print("Vz:", Vz)

    # 2. Compute Rh, Rb, and Rl
    ah = (4.6 * n1 * h / Vz)
    ab = (4.6 * n1 * B / Vz)
    al = (15.4 * n1 * L / Vz)
    nh = (ah) ** -1
    nb = (ab) ** -1
    nl = (al) ** -1
    Rh = np.maximum( nh - 0.5 * nh * nh * (1 - np.exp( -2 * ah )), 0 )
    Rb = np.maximum( nb - 0.5 * nb * nb * (1 - np.exp( -2 * ab )), 0 )
    Rl = np.maximum( nl - 0.5 * nl * nl * (1 - np.exp( -2 * al )), 0 )

    # print("Rh:", Rh)
    # print("Rb:", Rb)
    # print("Rl:", Rl)

    # 3. Compute Resonant Response Factor (Rn)
    # Assuming a value for z, change according to requirement
    Lz = l * (z_var / 10)**epsilon
    # print( "Lz:", Lz )
    N1 = n1 * Lz / Vz
    # print( "N1:", N1 )
    Rn = (7.47 * N1) / (1+10.3 * N1)**(5/3)
    # print("Rn:", Rn)

    # 4. Compute Resonant Response Factor (R)
    R = math.sqrt((1/B_damping_percentage)*Rn*Rh*Rb*(0.53+0.47*Rl))
    # print("R:", R)

    # 5. Compute gR
    gR = np.sqrt(2*np.log(3600*n1)) + 0.577/np.sqrt(2*np.log(3600*n1))
    # print("gR:", gR)
    gQ = 3.4
    gv = 3.4

    # 6. Compute I and Q:
    Iz = c*(10/z)**(1/6)
    # print( "Iz:", Iz )
    Q = np.sqrt(1/(1+0.63*((B+h)/Lz)**0.63))
    # print("Q:", Q)

    # 6. Gust Factor, Gf:
    if T > 1:
        Gf = 0.925 * ((1+1.7*Iz*np.sqrt(gQ**2+gR**2*R**2))/(1+1.7*gv*Iz))
        # print( "Gust Factor, Gf:", Gf )
    # Return computed values
        return Gf
    else:
        asr = 1 + 1.7 * gQ * Iz * Q
        bsr = 1 + 1.7 * gv * Iz
        Gf = np.maximum(0.925 * (asr / bsr),0.85)
        # print( "Gust Factor, Gf:", Gf )
        # Return computed values
        return Gf


# Example usage:



def calculate_wall_pressure_coefficient(surface, L_over_B):
    """
    Calculate the wall pressure coefficient (Cp) based on surface type and dimensions.

    Parameters:
        surface (str): Type of wall surface (e.g., 'Windward Wall', 'Leeward Wall', 'Side Wall')
        L_over_B (float): Ratio of length to breadth
        q (float): Velocity pressure (q)

    Returns:
        float: Wall pressure coefficient (Cp)
    """
    
    if surface == 'Windward Wall':
        Cp = 0.8
    elif surface == 'Leeward Wall':
        if L_over_B <= 1:
            Cp = -0.5
        elif L_over_B <= 2:
            Cp = -0.3
        else:
            # Perform linear interpolation for L_over_B between 2 and infinity
            Cp = -0.3 + (-0.2 + (-0.3)) * (L_over_B - 2) / (L_over_B - 2 + 1)

    elif surface == 'Side Wall':
        Cp = -0.7
    else:
        Cp = None  # Unknown surface type

    return Cp

# Table 6.2.20: Values for Coefficients to Estimate Approximate Period
def calculate_base_shear_and_overturning_moment(type):
    """
    Calculate design base shear and overturning moment for a building.

    Table 6.2.20: Values for Coefficients to Estimate Approximate Period
    Structure type       Ct       m
    Concrete moment-resisting frames    0.0466    0.9
    Steel moment-resisting frames       0.0724    0.8
    Eccentrically braced steel frame    0.0731    0.75
    All other structural systems       0.0488    0.75

    Args:
    - num_stories (int): Number of stories in the building.
    - floor_height (float): Height of each floor in meters.

    Returns:
    - base_shear (float): Design base shear in kN.
    - overturning_moment (float): Overturning moment in kNm.
    """

    # Table 6.2.20: Values for Coefficients
    Ct_values = {
        "Concrete moment-resisting frames": 0.0466,
        "Steel moment-resisting frames": 0.0724,
        "Eccentrically braced steel frame": 0.0731,
        "All other structural systems": 0.0488
    }
    m_values = {
        "Concrete moment-resisting frames": 0.9,
        "Steel moment-resisting frames": 0.8,
        "Eccentrically braced steel frame": 0.75,
        "All other structural systems": 0.75
    }

    # Using Concrete moment-resisting frames values
    Ct = Ct_values[type]
    m = m_values[type]


    return Ct, m

# 2.5.7.2 Building period
def calculate_building_period(height, m, Ct, structural_type="concrete", AB=None):
    """
    Calculate the fundamental period of a building based on the given guidelines.

    Args:
    - height (float): Height of building in meters from foundation or top of rigid basement.
    - m (float): Parameter obtained from Table 6.2.20.
    - x (int): Number of shear walls in the building.
    - structural_type (str): Type of structure (default is "concrete").

    Returns:
    - T (float): Fundamental period of the building in seconds.
    """
    T = Ct* (height) ** m

    return T



def analyze_wind_direction(building_dimensions, wind_dir):
    """Determine length and width based on wind direction"""
    if wind_dir == "X":
        return building_dimensions["X"], building_dimensions["Y"]
    elif wind_dir == "Y":
        return building_dimensions["Y"], building_dimensions["X"]
    else:
        raise ValueError("Wind direction must be 'X' or 'Y'")


def calculate_wind_loads(wind_input_data):
    """
    Calculate wind loads for a building based on input parameters.
    
    Args:
        wind_input_data (dict): Dictionary containing all necessary input parameters:
            - num_stories: Number of stories in the building
            - story_height: Height of each story (meters)
            - building_dimensions: Dict with "X" and "Y" dimensions (meters)
            - wind_direction: "X" or "Y" indicating wind direction
            - structural_type: Structural system type
            - location: Building location for wind speed
            - exposure_category: Terrain exposure category
            - nature_of_occupancy: Description of building occupancy
            - structure_type: Type of structure for directionality factor
            - topographic_params: Dict with k1, k2, k3 for topographic factor
    """
    try:
        # Extract all input parameters
        num_stories = wind_input_data["num_stories"]
        story_height = wind_input_data["story_height"]
        building_dimensions = wind_input_data["building_dimensions"]
        wind_direction = wind_input_data["wind_direction"]
        structural_type = wind_input_data["structural_type"]
        location = wind_input_data["location"]
        exposure_category = wind_input_data["exposure_category"]
        nature_of_occupancy = wind_input_data["nature_of_occupancy"]
        structure_type = wind_input_data["structure_type"]
        topographic_params = wind_input_data.get("topographic_params", {"k1": 0, "k2": 0, "k3": 0})
        
        # Determine occupancy category
        occupancy_category = get_occupancy_category(nature_of_occupancy)
        
        total_height = num_stories * story_height
        
        # Get length and width based on wind direction
        building_length, building_width = analyze_wind_direction(building_dimensions, wind_direction)
        L_over_B = building_length / building_width

        def validate_numeric(value, name):
            """Ensure a value is numeric and not None"""
            if value is None:
                raise ValueError(f"{name} cannot be None")
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {type(value)}")
            return value

        # Get basic parameters with validation
        V = validate_numeric(basic_wind_speed(location), "Wind speed (V)")
        I = validate_numeric(importance_factor(occupancy_category, V), "Importance factor (I)")
        Kd = validate_numeric(wind_directionality_factor(structure_type), "Directionality factor (Kd)")
        Kzt = validate_numeric(calculate_topographic_factor(
            topographic_params["k1"], topographic_params["k2"], topographic_params["k3"]), 
            "Topographic factor (Kzt)")
        
        exposure_constants = terrain_exposure_constants(exposure_category)
        if not isinstance(exposure_constants, dict):
            raise ValueError("Exposure constants should be a dictionary")
            
        # Validate exposure constants
        for const in ['zg', 'alpha', 'z_min', 'c', 'L', 'epsilon', 'a_var', 'b_var']:
            validate_numeric(exposure_constants.get(const), f"Exposure constant {const}")

        Ct, m = calculate_base_shear_and_overturning_moment(structural_type)
        T = validate_numeric(calculate_building_period(total_height, m, Ct), "Building period (T)")

        # Calculate maximum qz at building top for leeward pressures
        Kz_top = validate_numeric(
            velocity_pressure_coefficient(total_height, exposure_constants['zg'], 
                                       exposure_constants['alpha'], exposure_category),
            "Velocity pressure coefficient at top (Kz)"
        )
        qz_max = 0.000613 * Kz_top * Kzt * Kd * (V**2) * I

        # Initialize result lists
        wind_pressures_windward = []
        wind_pressures_leeward = []
        total_pressures = []
        wind_forces = []
        
        for i in range(1, num_stories + 1):
            z = i * story_height
            
            # Velocity pressure coefficient for current story
            Kz = validate_numeric(
                velocity_pressure_coefficient(z, exposure_constants['zg'], 
                                           exposure_constants['alpha'], exposure_category),
                "Velocity pressure coefficient (Kz)"
            )
            
            # Velocity pressure for windward (story-specific)
            qz = 0.000613 * Kz * Kzt * Kd * (V**2) * I
            
            # Pressure coefficients
            Cp_windward = validate_numeric(
                calculate_wall_pressure_coefficient("Windward Wall", L_over_B),
                "Windward Cp"
            )
            Cp_leeward = validate_numeric(
                calculate_wall_pressure_coefficient("Leeward Wall", L_over_B),
                "Leeward Cp"
            )
                
            # Gust factor
            Gf = validate_numeric(
                Compute_Gust_factor(T, z, exposure_constants['z_min'], exposure_constants['c'],
                                 exposure_constants['L'], exposure_constants['epsilon'],
                                 V, exposure_constants['a_var'], exposure_constants['b_var'],
                                 total_height, building_length, building_width),
                "Gust factor (Gf)"
            )
            
            # Pressures (using qz_max for leeward)
            p_windward = qz * Gf * Cp_windward
            p_leeward = qz_max * Gf * Cp_leeward  # Using maximum qz
            p_total = p_windward + abs(p_leeward)  # Total pressure
            
            wind_pressures_windward.append(p_windward)
            wind_pressures_leeward.append(p_leeward)
            total_pressures.append(p_total)
            
            # Total force (windward + leeward)
            story_area = story_height * building_width
            F_total = p_total * story_area
            wind_forces.append(F_total)
            
            print(f"Story {i} (Height {z:.1f}m):")
            print(f"  Windward: qz={qz:.2f} N/m², Cp={Cp_windward:.2f}, Pressure={p_windward:.2f} N/m²")
            print(f"  Leeward: qz_max={qz_max:.2f} N/m², Cp={Cp_leeward:.2f}, Pressure={p_leeward:.2f} N/m²")
            print(f"  Total Pressure={p_total:.2f} N/m²")
            print(f"  Total Wind Force={F_total:.2f} N ({F_total/1000:.2f} kN)\n")

        # Enhanced summary table with total pressure
        print("\nSummary of Wind Loads:")
        print(f"Wind Direction: {wind_direction}-axis (L={building_length}m, B={building_width}m)")
        print("Story | Height | Windward (N/m²) | Leeward (N/m²) | Total Pressure (N/m²) | Total Force (kN)")
        print("------|--------|-----------------|----------------|-----------------------|-----------------")
        for i in range(num_stories):
            print(f"{i+1:5d} | {story_height*(i+1):6.1f} | {wind_pressures_windward[i]:15.2f} | "
                  f"{wind_pressures_leeward[i]:14.2f} | {total_pressures[i]:21.2f} | {wind_forces[i]*0.2248:15.2f}")

        total_wind_force = sum(wind_forces)
        print(f"\nTotal Wind Force: {total_wind_force:.2f} N ({total_wind_force/1000:.2f} kN)")

    except ValueError as e:
        print(f"Calculation error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {str(e)}")




'''
"structural_type":

Concrete moment-resisting frames    
Steel moment-resisting frames      
Eccentrically braced steel frame    
All other structural systems      

'''

location_names = "Angarpota, Lalmonirhat, Bagerhat, Madaripur, Bandarban, Magura, Barguna, Manikganj, Barisal, Meherpur, Bhola, Maheshkhali, Bogra, Moulvibazar, Brahmanbaria, Munshiganj, Chandpur, Mymensingh, Chapai Nawabganj, Naogaon, Chittagong, Narail, Chuadanga, Narayanganj, Comilla, Narsinghdi, Cox’s Bazar, Natore, Dahagram, Netrokona, Dhaka, Nilphamari, Dinajpur, Noakhali, Faridpur, Pabna, Feni, Panchagarh, Gaibandha, Patuakhali, Gazipur, Pirojpur, Gopalganj, Rajbari, Habiganj, Rajshahi, Hatiya, Rangamati, Ishurdi, Rangpur, Joypurhat, Satkhira, Jamalpur, Shariatpur, Jessore, Sherpur, Jhalakati, Sirajganj, Jhenaidah, Srimangal, Khagrachhari, St. Martin’s Island, Khulna, Sunamganj, Kutubdia, Sylhet, Kishoreganj, Sandwip, Kurigram, Tangail, Kushtia, Teknaf, Lakshmipur, Thakurgaon"

'''
Exposure Category:

2.4.6 
Exposure 
For each wind direction considered, the upwind exposure category shall be based on 
ground surface roughness that is determined from natural topography, vegetation, 
and constructed facilities. 
2.4.6.1 Wind directions and sectors 
For each selected wind direction at which the wind loads are to be evaluated, the 
exposure of the building or structure shall be determined for the two upwind sectors 
extending 45o either side of the selected wind direction. 
The exposures in these two sectors shall be determined in accordance with Sections 
2.4.6.2 and 2.4.6.3 and the exposure resulting in the highest wind loads shall be used 
to represent the winds from that direction. 
2.4.6.2 Surface roughness categories 
A ground surface roughness within each 45o sector shall be determined for a distance 
upwind of the site as defined in Sec 2.4.6.3 from the categories defined in the 
following text, for the purpose of assigning an exposure category as defined in  
Sec 2.4.6.3.  
Surface Roughness A: Urban and suburban areas, wooded areas, or other terrain with 
numerous closely spaced obstructions having the size of single-family dwellings or 
larger. 
Surface Roughness B: Open terrain with scattered obstructions having heights 
generally less than 9.1 m. This category includes flat open country, grasslands, and 
all water surfaces in cyclone prone regions.  
Surface Roughness C: Flat, unobstructed areas and water surfaces outside cyclone 
prone regions. This category includes smooth mud flats and salt flats. 
2.4.6.3 Exposure categories 
Exposure A: Exposure A shall apply where the ground surface roughness condition, 
as defined by Surface Roughness A, prevails in the upwind direction for a distance of 
at least 792 m or 20 times the height of the building, whichever is greater. 
Exception: For buildings whose mean roof height is less than or equal to 9.1 m, the 
upwind distance may be reduced to 457 m. 

Exposure B: Exposure B shall apply for all cases where Exposures A or C do not 
apply. 
Exposure C: Exposure C shall apply where the ground surface roughness, as 
defined by Surface Roughness C, prevails in the upwind direction for a distance 
greater than 1,524 m or 20 times the building height, whichever is greater. 
Exposure C shall extend into downwind areas of Surface Roughness A or B for a 
distance of 200 m or 20 times the height of the building, whichever is greater. 
'''

'''
'''
"Agricultural facilities",
"Certain temporary facilities",
"Minor storage facilities"

"More than 300 people congregate in one area",
"Day care facilities with a capacity greater than 150",
"Elementary school or secondary school facilities with a capacity greater than 250",
"Capacity greater than 500 for colleges or adult education facilities",
"Healthcare facilities with a capacity of 50 or more resident patients",
"Jails and detention facilities"

"Power generating stations",
"Water treatment facilities",
"Sewage treatment facilities",
"Telecommunication centers",
"Manufacture, process, handle, store, use, or dispose of hazardous substances"

"Hospitals and other healthcare facilities having surgery or emergency treatment facilities",
"Fire, rescue, ambulance, and police stations and emergency vehicle garages",
"Designated earthquake, hurricane, or other emergency shelters",
"Designated emergency preparedness, communication, and operation centers",
"Power generating stations and other public utility facilities required in an emergency",
"Ancillary structures required for operation of Occupancy Category IV structures during an emergency",
"Aviation control towers, air traffic control centers, and emergency aircraft hangars",
"Community water storage facilities and pump structures required to maintain water pressure for fire suppression",
"Buildings and other structures having critical national defense functions",
"Facilities containing highly toxic substances"

'''

'''

'''
"structure_type":

# Buildings
"Main Wind Force Resisting System",
"Components and Cladding",
"Arched Roofs",

# Chimneys, Tanks, and Similar Structures
"Square",
"Hexagonal",
"Round",

# Solid Signs
"Solid Signs",

# Open Signs and Lattice Framework
"Open Signs and Lattice Framework",

# Trussed Towers
"Triangular, square, rectangular",
"All other cross section"

    
'''

# Example usage:
wind_input_data = {
    "num_stories": 10,
    "story_height": 3.2,
    "building_dimensions": {"X": 30, "Y": 20},
    "wind_direction": "X",
    "structural_type": "Concrete moment-resisting frames",
    "location": "Dhaka",
    "exposure_category": "B",
    "nature_of_occupancy": "Elementary school or secondary school facilities with a capacity greater than 250",
    "structure_type": "Main Wind Force Resisting System",
    "topographic_params": {"k1": 0, "k2": 0, "k3": 0}
}

# calculate_wind_loads(wind_input_data)