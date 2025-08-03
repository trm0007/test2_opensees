import os
from matplotlib import pyplot as plt
import numpy as np
import openseespy.opensees as ops
import opsvis as opsv
from opsvis.model import get_Ew_data_from_ops_domain_3d
from opsvis.secforces import section_force_distribution_3d
import json
import math
from functions import *



# if __name__ == "__main__":
#     (node_loads, element_uniform_loads, shell_pressure_loads, 
#      section_properties, elastic_section, aggregator_section, 
#      beam_integrations, frame_elements) = create_structural_model()
#     gravity_results = gravity_analysis(
#         request=mock_request,
#         project_pk=project_pk,
#         task_pk=task_pk,
#         node_loads=node_loads,
#         element_uniform_loads=element_uniform_loads,
#         shell_pressure_loads=shell_pressure_loads,
#         section_properties=section_properties,
#         elastic_section=elastic_section,
#         aggregator_section=aggregator_section,
#         beam_integrations=beam_integrations,
#         frame_elements=frame_elements,
#         num_points=5,
#         load_combination="gravity"
#     )

#     # Run response spectrum analysis
#     print("\nRunning response spectrum analysis...")
#     rsa_results = response_spectrum_analysis(
#         request=mock_request,
#         project_pk=project_pk,
#         task_pk=task_pk,
#         section_properties=section_properties,
#         Tn=Tn,
#         Sa=Sa,
#         direction=1,
#         num_modes=7,
#         load_combo="RSA"
#     )
    
#     print("\nAnalysis completed successfully!")
#     print("Gravity results saved to:", gravity_results[1])  # json_path is second return value
#     print("RSA results saved to:", rsa_results['saved_files']['modal_properties'])



