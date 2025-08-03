import numpy as np


def calculate_N(d, N):
    total = 0
    a = 0
    b = 0
    for i in range(len(d)):
        a += (d[i] / N[i])
        b += d[i]
        total = b/a
    return total


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
# Table 6.2.14: Description of Seismic Zones


# Table 6.2.15: Seismic Zone Coefficient Z for Some Important Towns of Bangladesh
def get_seismic_zone_coefficient_town(town):
    """
    Get the seismic zone coefficient for a given town.

    Table 6.2.15: Seismic Zone Coefficient Z for Some Important Towns of Bangladesh
    Town         Z     Town         Z     Town         Z     Town         Z
    Bagerhat   0.12   Gaibandha  0.28   Magura     0.12   Patuakhali 0.12
    Bandarban  0.28   Gazipur    0.20   Manikganj  0.20   Pirojpur    0.12
    Barguna    0.12   Gopalganj  0.12   Maulvibazar0.36   Rajbari     0.20
    Barisal    0.12   Habiganj   0.36   Meherpur   0.12   Rajshahi    0.12
    Bhola      0.12   Jaipurhat  0.20   Mongla     0.12   Rangamati   0.28
    Bogra      0.28   Jamalpur   0.36   Munshiganj 0.20   Rangpur     0.28
    Brahmanbaria0.28   Jessore    0.12   Mymensingh 0.36   Satkhira    0.12
    Chandpur   0.20   Jhalokati  0.12   Narail     0.12   Shariatpur  0.20
    Chapainababganj 0.12   Jhenaidah 0.12  Narayanganj0.20   Sherpur     0.36
    Chittagong 0.28   Khagrachari0.28   Narsingdi  0.28   Sirajganj   0.28
    Chuadanga  0.12   Khulna     0.12   Natore     0.20   Srimangal   0.36
    Comilla    0.20   Kishoreganj0.36   Naogaon    0.20   Sunamganj   0.36
    Cox's Bazar0.28   Kurigram   0.36   Netrakona  0.36   Sylhet      0.36
    Dhaka      0.20   Kushtia    0.20   Nilphamari 0.12   Tangail     0.28
    Dinajpur   0.20   Lakshmipur 0.20   Noakhali   0.20   Thakurgaon  0.20
    Faridpur   0.20   Lalmanirhat0.28   Pabna      0.20
    Feni       0.20   Madaripur  0.20   Panchagarh 0.20

    Args:
    - town (str): Town for which seismic zone coefficient is required.

    Returns:
    - zone_coefficient (float): Seismic zone coefficient (Z).
    """
    # Table 6.2.15: Seismic Zone Coefficient Z for Some Important Towns of Bangladesh
    town_coefficients = {
        "Bagerhat": 0.12, "Gaibandha": 0.28, "Magura": 0.12, "Patuakhali": 0.12,
        "Bandarban": 0.28, "Gazipur": 0.20, "Manikganj": 0.20, "Pirojpur": 0.12,
        "Barguna": 0.12, "Gopalganj": 0.12, "Maulvibazar": 0.36, "Rajbari": 0.20,
        "Barisal": 0.12, "Habiganj": 0.36, "Meherpur": 0.12, "Rajshahi": 0.12,
        "Bhola": 0.12, "Jaipurhat": 0.20, "Mongla": 0.12, "Rangamati": 0.28,
        "Bogra": 0.28, "Jamalpur": 0.36, "Munshiganj": 0.20, "Rangpur": 0.28,
        "Brahmanbaria": 0.28, "Jessore": 0.12, "Mymensingh": 0.36, "Satkhira": 0.12,
        "Chandpur": 0.20, "Jhalokati": 0.12, "Narail": 0.12, "Shariatpur": 0.20,
        "Chapainababganj": 0.12, "Jhenaidah": 0.12, "Narayanganj": 0.20, "Sherpur": 0.36,
        "Chittagong": 0.28, "Khagrachari": 0.28, "Narsingdi": 0.28, "Sirajganj": 0.28,
        "Chuadanga": 0.12, "Khulna": 0.12, "Natore": 0.20, "Srimangal": 0.36,
        "Comilla": 0.20, "Kishoreganj": 0.36, "Naogaon": 0.20, "Sunamganj": 0.36,
        "Cox's Bazar": 0.28, "Kurigram": 0.36, "Netrakona": 0.36, "Sylhet": 0.36,
        "Dhaka": 0.20, "Kushtia": 0.20, "Nilphamari": 0.12, "Tangail": 0.28,
        "Dinajpur": 0.20, "Lakshmipur": 0.20, "Noakhali": 0.20, "Thakurgaon": 0.20,
        "Faridpur": 0.20, "Lalmanirhat": 0.28, "Pabna": 0.20,
        "Feni": 0.20, "Madaripur": 0.20, "Panchagarh": 0.20
    }

    # Find the seismic zone coefficient for the given town
    if town in town_coefficients:
        return town_coefficients[town]
    else:
        return None  # Town not found in the table
# Table 6.2.17: Importance Factors for Buildings and Structures for Earthquake Design
def get_importance_factor(occupancy_category):
    """
    Get the importance factor for a given occupancy category.

    Table 6.2.17: Importance Factors for Buildings and Structures for Earthquake Design
    Occupancy Category   Importance Factor (I)
    I, II                1.00
    III                  1.25
    IV                   1.50

    Args:
    - occupancy_category (str): Occupancy category for which the importance factor is required.

    Returns:
    - importance_factor (float): Importance factor (I).
    """
    # Table 6.2.17: Importance Factors for Buildings and Structures for Earthquake Design
    importance_factors = {
        "I": 1.00,
        "II": 1.00,
        "III": 1.25,
        "IV": 1.50
    }

    # Find the importance factor for the given occupancy category
    for category, factor in importance_factors.items():
        if category == occupancy_category:
            return factor

    return None  # Occupancy category not found in the table
# Table 6.2.18: Seismic Design Category of Buildings
def get_seismic_design_category(site_class, occupancy_category, seismic_zone):
    """
    Get the seismic design category for a given site class, occupancy category, and seismic zone.

    Table 6.2.18: Seismic Design Category of Buildings
    Site Class      Occupancy Category I, II and III   Occupancy Category IV
                    Zone 1  Zone 2  Zone 3  Zone 4    Zone 1  Zone 2  Zone 3  Zone 4
    SA              B       C       C       D         C       D       D       D
    SB              B       C       D       D         C       D       D       D
    SC              B       C       D       D         C       D       D       D
    SD              C       D       D       D         D       D       D       D
    SE, S1, S2      D       D       D       D         D       D       D       D

    Args:
    - site_class (str): Site class (e.g., SA, SB, SC, SD, SE, S1, S2).
    - occupancy_category (str): Occupancy category (I, II, III, IV).
    - seismic_zone (int): Seismic zone (1, 2, 3, 4).

    Returns:
    - seismic_design_category (str): Seismic design category (A, B, C, D).
    """
    # Table 6.2.18: Seismic Design Category of Buildings
    seismic_design_categories = {
        "SA": {"I": {"1": "B", "2": "C", "3": "C", "4": "D"}, "IV": {"1": "C", "2": "D", "3": "D", "4": "D"}},
        "SB": {"I": {"1": "B", "2": "C", "3": "D", "4": "D"}, "IV": {"1": "C", "2": "D", "3": "D", "4": "D"}},
        "SC": {"I": {"1": "B", "2": "C", "3": "D", "4": "D"}, "IV": {"1": "C", "2": "D", "3": "D", "4": "D"}},
        "SD": {"I": {"1": "C", "2": "D", "3": "D", "4": "D"}, "IV": {"1": "D", "2": "D", "3": "D", "4": "D"}},
        "SE": {"I": {"1": "D", "2": "D", "3": "D", "4": "D"}, "IV": {"1": "D", "2": "D", "3": "D", "4": "D"}},
        "S1": {"I": {"1": "D", "2": "D", "3": "D", "4": "D"}, "IV": {"1": "D", "2": "D", "3": "D", "4": "D"}},
        "S2": {"I": {"1": "D", "2": "D", "3": "D", "4": "D"}, "IV": {"1": "D", "2": "D", "3": "D", "4": "D"}}
    }

    # Find the seismic design category for the given parameters
    try:
        return seismic_design_categories[site_class][occupancy_category][str(seismic_zone)]
    except KeyError:
        return None  # Invalid input combination

# Table 6.1.1: Occupancy Category of Buildings and other Structures for Flood, Surge, Wind and Earthquake Loads.

def get_occupancy_category(nature_of_occupancy):
    """
    Get the occupancy category based on the nature of occupancy.

    Table 6.1.1: Occupancy Category of Buildings and other Structures for Flood, Surge, Wind and Earthquake Loads.

    Args:
    - nature_of_occupancy (str): Nature of occupancy for which the occupancy category is required.

    Returns:
    - occupancy_category (str): Occupancy category (I, II, III, IV).
    """
    if "Agricultural facilities" in nature_of_occupancy \
            or "Certain temporary facilities" in nature_of_occupancy \
            or "Minor storage facilities" in nature_of_occupancy:
        return "I"
    elif "More than 300 people congregate in one area" in nature_of_occupancy \
            or "Day care facilities with a capacity greater than 150" in nature_of_occupancy \
            or "Elementary school or secondary school facilities with a capacity greater than 250" in nature_of_occupancy \
            or "Capacity greater than 500 for colleges or adult education facilities" in nature_of_occupancy \
            or "Healthcare facilities with a capacity of 50 or more resident patients" in nature_of_occupancy \
            or "Jails and detention facilities" in nature_of_occupancy:
        return "II"
    elif "Power generating stations" in nature_of_occupancy \
            or "Water treatment facilities" in nature_of_occupancy \
            or "Sewage treatment facilities" in nature_of_occupancy \
            or "Telecommunication centers" in nature_of_occupancy \
            or "Manufacture, process, handle, store, use, or dispose of hazardous substances" in nature_of_occupancy:
        return "III"
    elif "Hospitals and other healthcare facilities having surgery or emergency treatment facilities" in nature_of_occupancy \
            or "Fire, rescue, ambulance, and police stations and emergency vehicle garages" in nature_of_occupancy \
            or "Designated earthquake, hurricane, or other emergency shelters" in nature_of_occupancy \
            or "Designated emergency preparedness, communication, and operation centers" in nature_of_occupancy \
            or "Power generating stations and other public utility facilities required in an emergency" in nature_of_occupancy \
            or "Ancillary structures required for operation of Occupancy Category IV structures during an emergency" in nature_of_occupancy \
            or "Aviation control towers, air traffic control centers, and emergency aircraft hangars" in nature_of_occupancy \
            or "Community water storage facilities and pump structures required to maintain water pressure for fire suppression" in nature_of_occupancy \
            or "Buildings and other structures having critical national defense functions" in nature_of_occupancy \
            or "Facilities containing highly toxic substances" in nature_of_occupancy:
        return "IV"
    else:
        return None  # Nature of occupancy not found in the table

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
# Table 6.2.13: Site Classification Based on Soil Properties
def classify_spt_value(spt_value):
    if spt_value > 50:
        return "SB"
    elif 15 <= spt_value <= 50:
        return "SC"
    elif spt_value < 15:
        return "SD"
    elif 10 <= spt_value <= 20:
        return "S1"
    else:
        return "Unknown"
# Table 6.2.13: Site Classification Based on Soil Properties
def get_site_classification(shear_wave_velocity, spt_value, undrained_shear_strength):
    """
    Determine the site classification based on shear wave velocity (Vs), SPT value, and undrained shear strength.

    Table 6.2.13: Site Classification Based on Soil Properties

    Args:
    - shear_wave_velocity (float): Shear wave velocity (Vs) in m/s.
    - spt_value (int): SPT value (blows/30cm).
    - undrained_shear_strength (int): Undrained shear strength in kPa.

    Returns:
    - site_class (str): Site classification (SA, SB, SC, SD, SE, S1, S2).
    """
    if shear_wave_velocity > 800:
        return "SA"
    elif 360 <= shear_wave_velocity <= 800 or spt_value > 50 or undrained_shear_strength > 250:
        return "SB"
    elif 180 <= shear_wave_velocity < 360 or 15 <= spt_value <= 50 or 70 <= undrained_shear_strength <= 250:
        return "SC"
    elif shear_wave_velocity < 180 or spt_value < 15 or undrained_shear_strength < 70:
        return "SD"
    elif shear_wave_velocity < 100 or (spt_value >= 10 and spt_value <= 20):
        return "S1"
    else:
        return "S2"

# Table 6.2.18: Seismic Design Category of Buildings
def get_seismic_design_category(site_class, occupancy_category, seismic_zone):
    categories = {
        "SA": {"1": "B", "2": "C", "3": "C", "4": "D"},
        "SB": {"1": "B", "2": "C", "3": "D", "4": "D"},
        "SC": {"1": "B", "2": "C", "3": "D", "4": "D"},
        "SD": {"1": "C", "2": "D", "3": "D", "4": "D"},
        "SE": {"1": "D", "2": "D", "3": "D", "4": "D"},
        "S1": {"1": "D", "2": "D", "3": "D", "4": "D"},
        "S2": {"1": "D", "2": "D", "3": "D", "4": "D"}
    }

    if occupancy_category in ['I', 'II', 'III']:
        if seismic_zone == 1:
            return categories[site_class]["1"]
        elif seismic_zone == 2:
            return categories[site_class]["2"]
        elif seismic_zone == 3 or seismic_zone == 4:
            return categories[site_class]["3"]  # For seismic zones 3 and 4, use the same value as for zone 3 in the table
        else:
            return "Unknown Seismic Zone"
    elif occupancy_category == 'IV':
        if seismic_zone == 1:
            return categories[site_class]["C"]
        elif seismic_zone == 2:
            return categories[site_class]["D"]
        elif seismic_zone == 3 or seismic_zone == 4:
            return categories[site_class]["D"]
        else:
            return "Unknown Seismic Zone"
    else:
        return "Unknown Occupancy Category"

# Table 6.2.16: Site Dependent Soil Factor and Other Parameters Defining Elastic Response Spectrum
def get_soil_parameters(soil_type):
    """
    Determine the site-dependent soil factor and other parameters defining the elastic response spectrum.

    Table 6.2.16: Site Dependent Soil Factor and Other Parameters Defining Elastic Response Spectrum

    Args:
    - soil_type (str): Soil type (SA, SB, SC, SD, SE).

    Returns:
    - soil_factor (float): Site-dependent soil factor (S).
    - tb (float): TB parameter.
    - tc (float): TC parameter.
    - td (float): TD parameter.
    """
    soil_parameters = {
        "SA": (1.0, 0.15, 0.40, 2.0),
        "SB": (1.2, 0.15, 0.50, 2.0),
        "SC": (1.15, 0.20, 0.60, 2.0),
        "SD": (1.35, 0.20, 0.80, 2.0),
        "SE": (1.4, 0.15, 0.50, 2.0)
    }

    try:
        return soil_parameters[soil_type]
    except KeyError:
        return None

# Calculate Cs:
def calculate_normalized_acceleration_spectrum(S, T, TB, TC, TD):
    """
    Calculate the normalized acceleration response spectrum based on structure period and soil type.

    Args:
    - T (float): Structure (building) period in seconds.
    - soil_type (str): Soil type (SA, SB, SC, SD, SE).

    Returns:
    - Cs (float): Normalized acceleration response spectrum.
    η 5 Damping correction factor as a function of damping with a
    reference value of η51 for 5% viscous damping. It is given by the
    following expression:
      10 /(5   )  0.55
    """
    # Constants
    # η = Damping correction factor as a function of damping with a  reference
    #     value of η51 for 5 % viscous damping.It is given by the following expression:
    viscous_damping = 0.05
    Cs = 0
    eta = 0
    # Calculate eta
    eta = np.sqrt( 10 / (5 + viscous_damping) )
    # Ensure eta is not less than 0.55
    eta = max( eta, 0.55 )

    # Calculate Cs based on T
    if 0 <= T < TB:
        Cs = S * (1 + (T / TB) * (2.5 * eta - 1))
    elif TB <= T <= TC:
        Cs = 2.5 * S * eta
    elif TC <= T <= TD:
        Cs = 2.5 * S * eta * (TC / T)
    elif TD <= T <= 4:
        Cs = 2.5 * S * eta * (TC * TD / T ** 2)
    else:
        Cs = None  # Handle invalid range of T

    return Cs
# Table 6.2.19: Response Reduction Factor, Deflection Amplification Factor and Height Limitations for Different Structural Systems
def get_system_info(system_type, subtype):
    '''
    Returns the values of R, omega, Cd, B, C, D for a given structural system and subtype.
    
    Args:
        system_type (str): The main structural system category (e.g., "A. BEARING WALL SYSTEMS").
        subtype (str): The specific subtype (e.g., "1. Special reinforced concrete shear walls").
        
    Returns:
        dict: A dictionary containing R, omega, Cd, B, C, D. Returns None if not found.
    '''
    system_info = {
        "A. BEARING WALL SYSTEMS": {
            "1. Special reinforced concrete shear walls": {"R": 5, "omega": 2.5, "Cd": 5, "B": "NL", "C": "NL", "D": 50},
            "2. Ordinary reinforced concrete shear walls ": {"R": 4, "omega": 2.5, "Cd": 4, "B": "NL", "C": "NL", "D": "NP"},
            "3. Ordinary reinforced masonry shear walls": {"R": 2, "omega": 2.5, "Cd": 1.75, "B": "NL", "C": 50, "D": "NP"},
            "4. Ordinary plain masonry shear walls": {"R": 1.5, "omega": 2.5, "Cd": 1.25, "B": 18, "C": "NP", "D": "NP"}
        },
        "B. BUILDING FRAME SYSTEMS (with bracing or shear wall)": {
            "1. Steel eccentrically braced frames, moment resisting connections at columns away from links": {"R": 8, "omega": 2, "Cd": 4, "B": "NL", "C": "NL", "D": 50},
            "2. Steel eccentrically braced frames, non-moment-resisting, connections at columns away from links": {"R": 7, "omega": 2, "Cd": 4, "B": "NL", "C": "NL", "D": 50},
            "3. Special steel concentrically braced frames": {"R": 6, "omega": 2, "Cd": 5, "B": "NL", "C": "NL", "D": 50},
            "4. Ordinary steel concentrically braced frames": {"R": 3.25, "omega": 2, "Cd": 3.25, "B": "NL", "C": "NL", "D": 11},
            "5. Special reinforced concrete shear walls": {"R": 6, "omega": 2.5, "Cd": 5, "B": "NL", "C": "NL", "D": 50},
            "6. Ordinary reinforced concrete shear walls": {"R": 5, "omega": 2.5, "Cd": 4.25, "B": "NL", "C": "NL", "D": "NP"},
            "7. Ordinary reinforced masonry shear walls": {"R": 2, "omega": 2.5, "Cd": 2, "B": "NL", "C": 50, "D": "NP"},
            "8. Ordinary plain masonry shear walls": {"R": 1.5, "omega": 2.5, "Cd": 1.25, "B": 18, "C": "NP", "D": "NP"}
        },
        "C. MOMENT RESISTING FRAME SYSTEMS (no shear wall)": {
            "1. Special steel moment frames": {"R": 8, "omega": 3, "Cd": 5.5, "B": "NL", "C": "NL", "D": "NL"},
            "2. Intermediate steel moment frames": {"R": 4.5, "omega": 3, "Cd": 4, "B": "NL", "C": "NL", "D": 35},
            "3. Ordinary steel moment frames": {"R": 3.5, "omega": 3, "Cd": 3, "B": "NL", "C": "NL", "D": "NP"},
            "4. Special reinforced concrete moment frames": {"R": 8, "omega": 3, "Cd": 5.5, "B": "NL", "C": "NL", "D": "NL"},
            "5. Intermediate reinforced concrete moment frames": {"R": 5, "omega": 3, "Cd": 4.5, "B": "NL", "C": "NL", "D": "NP"},
            "6. Ordinary reinforced concrete moment frames": {"R": 3, "omega": 3, "Cd": 2.5, "B": "NL", "C": "NP", "D": "NP"}
        },
        "D. DUAL SYSTEMS: SPECIAL MOMENT FRAMES CAPABLE OF RESISTING AT LEAST 25% OF PRESCRIBED SEISMIC FORCES (with bracing or shear wall)": {
            "1. Steel eccentrically braced frames": {"R": 8, "omega": 2.5, "Cd": 4, "B": "NL", "C": "NL", "D": "NL"},
            "2. Special steel concentrically braced frames": {"R": 7, "omega": 2.5, "Cd": 5.5, "B": "NL", "C": "NL", "D": "NL"},
            "3. Special reinforced concrete shear walls": {"R": 7, "omega": 2.5, "Cd": 5.5, "B": "NL", "C": "NL", "D": "NL"},
            "4. Ordinary reinforced concrete shear walls": {"R": 6, "omega": 2.5, "Cd": 5, "B": "NL", "C": "NL", "D": "NP"}
        },
        "E. DUAL SYSTEMS: INTERMEDIATE MOMENT FRAMES CAPABLE OF RESISTING AT LEAST 25% OF PRESCRIBED SEISMIC FORCES (with bracing or shear wall)": {
            "1. Special steel concentrically braced frames": {"R": 6, "omega": 2.5, "Cd": 5, "B": "NL", "C": "NL", "D": 11},
            "2. Special reinforced concrete shear walls": {"R": 6.5, "omega": 2.5, "Cd": 5, "B": "NL", "C": "NL", "D": 50},
            "3. Ordinary reinforced masonry shear walls": {"R": 3, "omega": 3, "Cd": 3, "B": "NL", "C": 50, "D": "NP"},
            "4. Ordinary reinforced concrete shear walls": {"R": 5.5, "omega": 2.5, "Cd": 4.5, "B": "NL", "C": "NL", "D": "NP"}
        },
        "F. DUAL SHEAR WALL-FRAME SYSTEM: ORDINARY REINFORCED CONCRETE MOMENT FRAMES AND ORDINARY REINFORCED CONCRETE SHEAR WALLS": {
            "1. Ordinary reinforced concrete shear walls": {"R": 4.5, "omega": 2.5, "Cd": 4, "B": "NL", "C": "NP", "D": "NP"}
        },
        "G. STEEL SYSTEMS NOT SPECIFICALLY DETAILED FOR SEISMIC RESISTANCE": {
            "1. Steel systems not specifically detailed for seismic resistance": {"R": 3, "omega": 3, "Cd": 3, "B": "NL", "C": "NL", "D": "NP"}
        }
    }

    # Get the system category
    system_category = system_info.get(system_type, None)
    if not system_category:
        return None

    # Get the subtype values
    subtype_values = system_category.get(subtype, None)
    return subtype_values








import numpy as np

d = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
N = [5, 10, 5, 14, 14, 20, 22, 24, 26, 24, 20, 30, 35, 35, 34, 24, 24, 10, 20, 25]

d_float = [float(num) for num in d]
N_float = [float(num) for num in N]

print("d_float:", d_float)
print("N_float:", N_float)

corrected_SPT_value = calculate_N(d_float, N_float)
print("Result:", corrected_SPT_value)

# Building parameters
num_stories = 10
floor_heights = [3.0] * num_stories  # Each floor is 3m high
total_height = sum(floor_heights)
floor_weights = [5000.0] * num_stories  # Each floor weighs 500 kN
total_weight = sum(floor_weights)
heights = [sum(floor_heights[:i+1]) for i in range(num_stories)]  # Cumulative heights

# Structural system
"""
Calculate design base shear and overturning moment for a building.

Table 6.2.20: Values for Coefficients to Estimate Approximate Period
Structure type             
Concrete moment-resisting frames    
Steel moment-resisting frames       
Eccentrically braced steel frame    
All other structural systems      

"""
building_type = "Concrete moment-resisting frames"
'''
A. BEARING WALL SYSTEMS  
B. BUILDING FRAME SYSTEMS (with bracing or shear wall)  
C. MOMENT RESISTING FRAME SYSTEMS (no shear wall)  
D. DUAL SYSTEMS: SPECIAL MOMENT FRAMES CAPABLE OF RESISTING AT LEAST 25% OF PRESCRIBED SEISMIC FORCES (with bracing or shear wall)

'''


# Location and site parameters
"""
Get the seismic zone coefficient for a given town.

Table 6.2.15: Seismic Zone Coefficient Z for Some Important Towns of Bangladesh
Town         Z     Town         Z     Town         Z     Town         Z
Bagerhat   0.12   Gaibandha  0.28   Magura     0.12   Patuakhali 0.12
Bandarban  0.28   Gazipur    0.20   Manikganj  0.20   Pirojpur    0.12
Barguna    0.12   Gopalganj  0.12   Maulvibazar0.36   Rajbari     0.20
Barisal    0.12   Habiganj   0.36   Meherpur   0.12   Rajshahi    0.12
Bhola      0.12   Jaipurhat  0.20   Mongla     0.12   Rangamati   0.28
Bogra      0.28   Jamalpur   0.36   Munshiganj 0.20   Rangpur     0.28
Brahmanbaria0.28   Jessore    0.12   Mymensingh 0.36   Satkhira    0.12
Chandpur   0.20   Jhalokati  0.12   Narail     0.12   Shariatpur  0.20
Chapainababganj 0.12   Jhenaidah 0.12  Narayanganj0.20   Sherpur     0.36
Chittagong 0.28   Khagrachari0.28   Narsingdi  0.28   Sirajganj   0.28
Chuadanga  0.12   Khulna     0.12   Natore     0.20   Srimangal   0.36
Comilla    0.20   Kishoreganj0.36   Naogaon    0.20   Sunamganj   0.36
Cox's Bazar0.28   Kurigram   0.36   Netrakona  0.36   Sylhet      0.36
Dhaka      0.20   Kushtia    0.20   Nilphamari 0.12   Tangail     0.28
Dinajpur   0.20   Lakshmipur 0.20   Noakhali   0.20   Thakurgaon  0.20
Faridpur   0.20   Lalmanirhat0.28   Pabna      0.20
Feni       0.20   Madaripur  0.20   Panchagarh 0.20

Args:
- town (str): Town for which seismic zone coefficient is required.

Returns:
- zone_coefficient (float): Seismic zone coefficient (Z).
    """
location = "Dhaka"
spt_value = 15
soil_type  = classify_spt_value(spt_value)
print("soil_type10 :", soil_type )

# Calculate required parameters using existing functions
# 1. Get seismic zone coefficient

Z = get_seismic_zone_coefficient_town(location)

# 2. Get importance factor
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
nature_of_occupancy = "Elementary school or secondary school facilities with a capacity greater than 250"
occupancy_category = get_occupancy_category(nature_of_occupancy)
I = get_importance_factor(occupancy_category)
'''
# 1. BEARING WALL SYSTEMS
BEARING_WALL_SYSTEMS = {
    "system_type": "A. BEARING WALL SYSTEMS",
    "subtypes": [
        "1. Special reinforced concrete shear walls",
        "2. Ordinary reinforced concrete shear walls",
        "3. Ordinary reinforced masonry shear walls",
        "4. Ordinary plain masonry shear walls"
    ]
}

# 2. BUILDING FRAME SYSTEMS (with bracing or shear wall)
BUILDING_FRAME_SYSTEMS = {
    "system_type": "B. BUILDING FRAME SYSTEMS (with bracing or shear wall)",
    "subtypes": [
        "1. Steel eccentrically braced frames, moment resisting connections at columns away from links",
        "2. Steel eccentrically braced frames, non-moment-resisting, connections at columns away from links",
        "3. Special steel concentrically braced frames",
        "4. Ordinary steel concentrically braced frames",
        "5. Special reinforced concrete shear walls",
        "6. Ordinary reinforced concrete shear walls",
        "7. Ordinary reinforced masonry shear walls",
        "8. Ordinary plain masonry shear walls"
    ]
}

# 3. MOMENT RESISTING FRAME SYSTEMS (no shear wall)
MOMENT_RESISTING_FRAME_SYSTEMS = {
    "system_type": "C. MOMENT RESISTING FRAME SYSTEMS (no shear wall)",
    "subtypes": [
        "1. Special steel moment frames",
        "2. Intermediate steel moment frames",
        "3. Ordinary steel moment frames",
        "4. Special reinforced concrete moment frames",
        "5. Intermediate reinforced concrete moment frames",
        "6. Ordinary reinforced concrete moment frames"
    ]
}

# 4. DUAL SYSTEMS (Special Moment Frames + Bracing/Shear Wall)
DUAL_SYSTEMS_SPECIAL = {
    "system_type": "D. DUAL SYSTEMS: SPECIAL MOMENT FRAMES CAPABLE OF RESISTING AT LEAST 25% OF PRESCRIBED SEISMIC FORCES (with bracing or shear wall)",
    "subtypes": [
        "1. Steel eccentrically braced frames",
        "2. Special steel concentrically braced frames",
        "3. Special reinforced concrete shear walls",
        "4. Ordinary reinforced concrete shear walls"
    ]
}

# 5. DUAL SYSTEMS (Intermediate Moment Frames + Bracing/Shear Wall)
DUAL_SYSTEMS_INTERMEDIATE = {
    "system_type": "E. DUAL SYSTEMS: INTERMEDIATE MOMENT FRAMES CAPABLE OF RESISTING AT LEAST 25% OF PRESCRIBED SEISMIC FORCES (with bracing or shear wall)",
    "subtypes": [
        "1. Special steel concentrically braced frames",
        "2. Special reinforced concrete shear walls",
        "3. Ordinary reinforced masonry shear walls",
        "4. Ordinary reinforced concrete shear walls"
    ]
}

# 6. DUAL SHEAR WALL-FRAME SYSTEM (Ordinary Moment Frames + Ordinary Shear Walls)
DUAL_SHEAR_WALL_FRAME = {
    "system_type": "F. DUAL SHEAR WALL-FRAME SYSTEM: ORDINARY REINFORCED CONCRETE MOMENT FRAMES AND ORDINARY REINFORCED CONCRETE SHEAR WALLS",
    "subtypes": [
        "1. Ordinary reinforced concrete shear walls"
    ]
}

# 7. STEEL SYSTEMS NOT DETAILED FOR SEISMIC RESISTANCE
STEEL_SYSTEMS_NON_SEISMIC = {
    "system_type": "G. STEEL SYSTEMS NOT SPECIFICALLY DETAILED FOR SEISMIC RESISTANCE",
    "subtypes": [
        "1. Steel systems not specifically detailed for seismic resistance"
    ]
}
'''
system_type = "C. MOMENT RESISTING FRAME SYSTEMS (no shear wall)"
subtypes = "5. Intermediate reinforced concrete moment frames"


# 3. Get structural system parameters
system_info = get_system_info(system_type, subtypes )
R = system_info['R']
omega = system_info['omega']
Cd = system_info['Cd']

# 4. Calculate building period
Ct, m = calculate_base_shear_and_overturning_moment(building_type)
# (height, m, Ct, structural_type="concrete", AB=None)
T = calculate_building_period(total_height, m, Ct, structural_type="concrete")
print(f"Building Period (T100bb): {T}")
# 5. Get site classification and soil parameters


soil_factor, TB, TC, TD = get_soil_parameters(soil_type)
Cs = calculate_normalized_acceleration_spectrum(soil_factor, T, TB, TC, TD)

print(f"Soil Factor: {soil_factor}")
print(f"TB: {TB}, TC: {TC}, TD: {TD}")
print(f"Normalized Acceleration Spectrum (Cs): {Cs}")
# 6. Calculate normalized acceleration spectrum (Cs)
viscous_damping = 0.05  # 5% damping
eta = max(np.sqrt(10 / (5 + viscous_damping)), 0.55)

# I = 1.0
# Cs = 1.2 
# 7. Calculate base shear
Sa = (2/3) * Z * I * Cs / R
V = Sa * total_weight  # Total base shear

# 8. Calculate story forces (Fx)
k = 1.0 if T <= 0.5 else 2.0 if T >= 2.5 else 1.0 + (T - 0.5)/2.0

print(f"Sa = (2/3) * Z * I * Cs / R = {Sa}")
print(f"Total base shear V = Sa * total_weight = {V}")
print(f"k value based on T = {T} is {k}")

Cvx = []
for i in range(num_stories):
    numerator = floor_weights[i] * heights[i]**k
    denominator = sum(floor_weights[j] * heights[j]**k for j in range(num_stories))
    Cvx.append(numerator / denominator)

Fx = [V * c for c in Cvx]

# 9. Calculate story shears (Vx)
Vx = []
for i in range(num_stories):
    Vx.append(sum(Fx[i:num_stories]))

# Print results
print("SEISMIC ANALYSIS RESULTS")
print("=======================")
print(f"Building Type: {building_type}")
print(f"Number of Stories: {num_stories}")
print(f"Total Height: {total_height:.2f} m")
print(f"Total Weight: {total_weight:.2f} kN")
print(f"Occupancy Category: {occupancy_category} (I={I})")
# print(f"Site Class: {site_class}, Soil Type: {soil_type}")
print(f"Structural System: (R={R}, Ω={omega}, Cd={Cd})")
print(f"Building Period: {T:.3f} sec")
print(f"Normalized Acceleration Spectrum (Cs): {Cs:.4f}")
print(f"Base Shear (V): {V:.2f} kN\n")

print("STORY FORCES AND SHEARS")
print("Story  Height(m)  Weight(kN)   Force(kN)   Shear(kN)")
print("----------------------------------------------------")
for i in reversed(range(num_stories)):  # Print from top to bottom
    story_num = num_stories - i
    print(f"{story_num:4d}  {heights[i]:9.2f}  {floor_weights[i]:10.2f}  {Fx[i]*0.2248:10.2f}  {Vx[i]*0.2248:10.2f}")


