# ========================
# WIND ANALYSIS PARAMETERS
# ========================

# Building Parameters
num_stories = 10
story_height = 3.2  # meters
building_dimensions = {"X": 30, "Y": 20}  # meters (length x width)
wind_direction = "X"  # Wind direction: "X" or "Y"

# Structural Parameters
structural_type = "Concrete moment-resisting frames"

# Location and Site Parameters
location = "Dhaka"  # Select from Bangladesh locations
exposure_category = "B"  # A, B, or C
nature_of_occupancy = "Elementary school or secondary school facilities with a capacity greater than 250"

# Wind Load Parameters
structure_type = "Main Wind Force Resisting System"
topographic_params = {"k1": 0, "k2": 0, "k3": 0}  # Topographic factors

# Analysis Options
analysis_type = "wind_analysis"
include_dynamic_effects = True
include_gust_effects = True

# Output Options
generate_plots = True
output_format = "detailed"

# ========================
# AVAILABLE OPTIONS
# ========================

# Structural Types:
# - "Concrete moment-resisting frames"
# - "Steel moment-resisting frames" 
# - "Eccentrically braced steel frame"
# - "All other structural systems"

# Bangladesh Locations:
# "Angarpota", "Lalmonirhat", "Bagerhat", "Madaripur", "Bandarban", "Magura", 
# "Barguna", "Manikganj", "Barisal", "Meherpur", "Bhola", "Maheshkhali", 
# "Bogra", "Moulvibazar", "Brahmanbaria", "Munshiganj", "Chandpur", "Mymensingh", 
# "Chapai Nawabganj", "Naogaon", "Chittagong", "Narail", "Chuadanga", "Narayanganj", 
# "Comilla", "Narsinghdi", "Cox's Bazar", "Natore", "Dahagram", "Netrokona", 
# "Dhaka", "Nilphamari", "Dinajpur", "Noakhali", "Faridpur", "Pabna", "Feni", 
# "Panchagarh", "Gaibandha", "Patuakhali", "Gazipur", "Pirojpur", "Gopalganj", 
# "Rajbari", "Habiganj", "Rajshahi", "Hatiya", "Rangamati", "Ishurdi", "Rangpur", 
# "Joypurhat", "Satkhira", "Jamalpur", "Shariatpur", "Jessore", "Sherpur", 
# "Jhalakati", "Sirajganj", "Jhenaidah", "Srimangal", "Khagrachhari", 
# "St. Martin's Island", "Khulna", "Sunamganj", "Kutubdia", "Sylhet", 
# "Kishoreganj", "Sandwip", "Kurigram", "Tangail", "Kushtia", "Teknaf", 
# "Lakshmipur", "Thakurgaon"

# Exposure Categories:
# A: Urban and suburban areas, wooded areas
# B: Open terrain with scattered obstructions (default)
# C: Flat, unobstructed areas and water surfaces

# Structure Types for Wind Directionality:
# Buildings: "Main Wind Force Resisting System", "Components and Cladding", "Arched Roofs"
# Chimneys/Tanks: "Square", "Hexagonal", "Round"
# Signs: "Solid Signs", "Open Signs and Lattice Framework"
# Towers: "Triangular, square, rectangular", "All other cross section"

# Occupancy Categories:
# Category I: "Agricultural facilities", "Certain temporary facilities", "Minor storage facilities"
# Category II: Standard occupancy buildings
# Category III: "More than 300 people congregate in one area", "Day care facilities with a capacity greater than 150", 
#              "Elementary school or secondary school facilities with a capacity greater than 250"
# Category IV: Essential facilities like hospitals, emergency centers, etc.