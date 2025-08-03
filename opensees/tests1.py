import os

def find_text_in_py_files(root_path, search_text):
    py_files = []

    # print(f"Searching in: {root_path}")

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Remove __pycache__ from directories to prevent walking into them
        if '__pycache__' in dirnames:
            dirnames.remove('__pycache__')

        # print(f"\nChecking directory: {dirpath}")
        # print(f"Subdirectories: {dirnames}")
        # print(f"Files: {filenames}")

        for file in filenames:
            if file.endswith(".py") and file != "__init__.py":
                full_path = os.path.join(dirpath, file)
                py_files.append(full_path)
                # print(f"Found .py file: {full_path}")

    if not py_files:
        print("No .py files found.")

    matched_files = []

    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                if search_text in content:
                    matched_files.append(file_path)
                    # print(f"Matched in: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not matched_files:
        print("No matches found.")

    return matched_files

# Example usage:
# root_dir = r"C:\Users\User\Desktop\django_opensees\myenv\Lib\site-packages\sectionproperties"
root_dir = r"C:\Users\User\Desktop\django_opensees\myproject\opensees"
search_for = "Warning! ele_load_type:"

matched = find_text_in_py_files(root_dir, search_for)

print("\nMatched files:")
for file in matched:
    print(file)


