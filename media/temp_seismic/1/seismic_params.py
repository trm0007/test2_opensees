# ========================
# SEISMIC ANALYSIS PARAMETERS
# ========================

# Building Parameters
num_stories = 10
floor_height = 3.0  # meters
floor_weight = 5000.0  # kN per floor
building_type = "Concrete moment-resisting frames"

# Location and Site Parameters
location = "Dhaka"
nature_of_occupancy = "Elementary school or secondary school facilities with a capacity greater than 250"

# Structural System Parameters
system_type = "C. MOMENT RESISTING FRAME SYSTEMS (no shear wall)"
subtypes = "5. Intermediate reinforced concrete moment frames"

# SPT Data (Standard Penetration Test)
spt_depths = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # meters
spt_values = [5, 10, 5, 14, 14, 20, 22, 24, 26, 24, 20, 30, 35, 35, 34, 24, 24, 10, 20, 25]  # N-values

# Analysis Options
analysis_type = "seismic_analysis"
include_dynamic_effects = True
include_soil_structure_interaction = False

# Output Options
generate_plots = True
output_format = "detailed"