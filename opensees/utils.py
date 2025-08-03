# opensees/utils.py
import os
import json
from pathlib import Path
from django.conf import settings

def get_user_data_dir(user):
    """Get the user's data directory path"""
    return os.path.join(settings.OUTPUT_FOLDER, user.username)

def get_project_dir(user, project_id):
    """Get the project directory path"""
    return os.path.join(get_user_data_dir(user), str(project_id))

def get_task_dir(user, project_id, task_id):
    """Get the task directory path"""
    return os.path.join(get_project_dir(user, project_id), str(task_id))

def ensure_dir_exists(path):
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)

def get_json_file_path(user, project_id, task_id, model_name, pk=None):
    """Get path to JSON file for a model instance"""
    task_dir = get_task_dir(user, project_id, task_id)
    ensure_dir_exists(task_dir)
    
    if pk:
        return os.path.join(task_dir, f"{model_name}_{pk}.json")
    return os.path.join(task_dir, f"{model_name}.json")

def load_json_data(user, project_id, task_id, model_name):
    """Load all JSON data for a model type"""
    task_dir = get_task_dir(user, project_id, task_id)
    if not os.path.exists(task_dir):
        return []
    
    data = []
    prefix = f"{model_name}_"
    for filename in os.listdir(task_dir):
        if filename.startswith(prefix) and filename.endswith('.json'):
            with open(os.path.join(task_dir, filename), 'r') as f:
                try:
                    item = json.load(f)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    return sorted(data, key=lambda x: x.get('pk', 0))

def save_json_data(user, project_id, task_id, model_name, data, pk=None):
    """Save data to JSON file"""
    filepath = get_json_file_path(user, project_id, task_id, model_name, pk)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def delete_json_data(user, project_id, task_id, model_name, pk):
    """Delete JSON file"""
    filepath = get_json_file_path(user, project_id, task_id, model_name, pk)
    if os.path.exists(filepath):
        os.remove(filepath)

def get_next_pk(user, project_id, task_id, model_name):
    """Get next available primary key"""
    data = load_json_data(user, project_id, task_id, model_name)
    if not data:
        return 1
    return max(item.get('pk', 0) for item in data) + 1