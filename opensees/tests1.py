# import os

# def find_text_in_py_files(root_path, search_text):
#     py_files = []

#     # print(f"Searching in: {root_path}")

#     for dirpath, dirnames, filenames in os.walk(root_path):
#         # Remove __pycache__ from directories to prevent walking into them
#         if '__pycache__' in dirnames:
#             dirnames.remove('__pycache__')

#         # print(f"\nChecking directory: {dirpath}")
#         # print(f"Subdirectories: {dirnames}")
#         # print(f"Files: {filenames}")

#         for file in filenames:
#             if file.endswith(".py") and file != "__init__.py":
#                 full_path = os.path.join(dirpath, file)
#                 py_files.append(full_path)
#                 # print(f"Found .py file: {full_path}")

#     if not py_files:
#         print("No .py files found.")

#     matched_files = []

#     for file_path in py_files:
#         try:
#             with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
#                 content = file.read()
#                 if search_text in content:
#                     matched_files.append(file_path)
#                     # print(f"Matched in: {file_path}")
#         except Exception as e:
#             print(f"Error reading {file_path}: {e}")

#     if not matched_files:
#         print("No matches found.")

#     return matched_files

# # Example usage:
# # root_dir = r"C:\Users\User\Desktop\django_opensees\myenv\Lib\site-packages\sectionproperties"
# root_dir = r"C:\Users\User\Desktop\django_opensees\myproject\opensees"
# search_for = "Warning! ele_load_type:"

# matched = find_text_in_py_files(root_dir, search_for)

# print("\nMatched files:")
# for file in matched:
#     print(file)



def get_element_shapes(section_properties, beam_integrations, frame_elements  ):
    """Directly extracts element shapes from provided data structures"""

    section_shapes = {}
    for sec in section_properties:
        tag = sec[0]
        sec_type = sec[1].lower()  # Case-insensitive
        
        if sec_type == 'rectangular':
            section_shapes[tag] = ['rect', [sec[6], sec[7]]]  # B, H
            
        elif sec_type == 'circular':
            section_shapes[tag] = ['circ', [sec[6]]]  # D
            
        elif sec_type == 'i' or sec_type == 'wideflange':
            section_shapes[tag] = ['I', [
                sec[6],  # B (flange width)
                sec[7],  # H (total depth)
                sec[8],  # tf (flange thickness)
                sec[9]   # tw (web thickness)
            ]]
            
        elif sec_type == 'l' or sec_type == 'angle':
            section_shapes[tag] = ['L', [
                sec[6],  # H (leg length)
                sec[7],  # B (leg length)
                sec[8]   # t (thickness)
            ]]
            
        elif sec_type == 't':
            section_shapes[tag] = ['T', [
                sec[6],  # B (flange width)
                sec[7],  # H (total depth)
                sec[8],  # tf (flange thickness)
                sec[9]   # tw (stem thickness)
            ]]
            
        elif sec_type == 'c' or sec_type == 'channel':
            section_shapes[tag] = ['C', [
                sec[6],  # B (flange width)
                sec[7],  # H (depth)
                sec[8]   # t (thickness)
            ]]
            
        elif sec_type == 'tube' or sec_type == 'pipe':
            section_shapes[tag] = ['tube', [
                sec[6],  # D (outer diameter)
                sec[7]   # t (wall thickness)
            ]]
            
        elif sec_type == 'box':
            section_shapes[tag] = ['box', [
                sec[6],  # B (outer width)
                sec[7],  # H (outer depth)
                sec[8]   # t (wall thickness)
            ]]
            
        else:
            print(f"Warning: Unsupported section type '{sec[1]}' (tag: {tag})")
            continue  # Skip unsupported sections

    # Create mapping: {integration_tag: section_tag}
    integration_to_section = {integ[1]: integ[2] for integ in beam_integrations}

    # Process elements
    ele_shapes = {}
    for elem in frame_elements:
        ele_tag = elem[1]  # Element ID
        integ_tag = elem[5]  # Integration tag
        
        if integ_tag in integration_to_section:
            sec_tag = integration_to_section[integ_tag]
            if sec_tag in section_shapes:
                ele_shapes[ele_tag] = section_shapes[sec_tag]
    
    return ele_shapes
