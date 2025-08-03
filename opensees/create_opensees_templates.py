import os
from pathlib import Path

def create_opensees_templates(base_dir):
    """Create all OpenSees template files dynamically"""
    
    # Define the directory structure
    components = [
        'model',
        'timeseries',
        'material',
        'section',
        'node',
        'element',
        'constraint',
        'diaphragm'
    ]
    
    # Base template directory
    templates_dir = os.path.join(base_dir, 'templates', 'opensees')
    
    # Create base directories
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(os.path.join(templates_dir, 'includes'), exist_ok=True)
    
    # Create component directories
    for component in components:
        os.makedirs(os.path.join(templates_dir, component), exist_ok=True)
    
    # Common template content
    base_content = """{% extends 'myapp/base.html' %}
{% block title %}{% block title_content %}{% endblock %} - {{ task.title }}{% endblock %}

{% block content %}
<div class="container mt-4">
    <div class="row">
        <div class="col-md-3">
            {% include 'opensees/includes/opensees_sidebar.html' %}
        </div>
        <div class="col-md-9">
            {% block component_content %}{% endblock %}
        </div>
    </div>
</div>
{% endblock %}"""
    
    # Create base template
    with open(os.path.join(templates_dir, 'base.html'), 'w') as f:
        f.write(base_content)
    
    # Create sidebar include
    # Create sidebar include
    sidebar_content = """<div class="card">
        <div class="card-header">
            OpenSees Components
        </div>
        <div class="card-body">
            <ul class="list-group list-group-flush">
                <li class="list-group-item">
                    <a href="{% url 'opensees:model_list' project.pk task.pk %}">Model</a>
                </li>
                <li class="list-group-item">
                    <a href="{% url 'opensees:timeseries_list' project.pk task.pk %}">Time Series</a>
                </li>
                <li class="list-group-item">
                    <a href="{% url 'opensees:material_list' project.pk task.pk %}">Material</a>
                </li>
                <li class="list-group-item">
                    <a href="{% url 'opensees:section_list' project.pk task.pk %}">Section</a>
                </li>
                <li class="list-group-item">
                    <a href="{% url 'opensees:node_list' project.pk task.pk %}">Node</a>
                </li>
                <li class="list-group-item">
                    <a href="{% url 'opensees:element_list' project.pk task.pk %}">Element</a>
                </li>
                <li class="list-group-item">
                    <a href="{% url 'opensees:constraint_list' project.pk task.pk %}">Constraint</a>
                </li>
                <li class="list-group-item">
                    <a href="{% url 'opensees:diaphragm_list' project.pk task.pk %}">Diaphragm</a>
                </li>
            </ul>
        </div>
    </div>"""
    with open(os.path.join(templates_dir, 'includes', 'opensees_sidebar.html'), 'w') as f:
        f.write(sidebar_content)
    
    # Template patterns for each view type
    templates = {
        'list.html': """{% extends 'opensees/base.html' %}
{% block title_content %}{% block list_title %}{% endblock %}{% endblock %}

{% block component_content %}
<h2>{% block list_header %}{% endblock %}</h2>

<div class="mb-3">
    <a href="{% block create_url %}{% endblock %}" class="btn btn-success">
        <i class="bi bi-plus-circle"></i> Create New
    </a>
</div>

{% if object_list %}
<div class="list-group">
    {% for object in object_list %}
    <div class="list-group-item">
        <div class="d-flex justify-content-between align-items-center">
            <div>
                <h5 class="mb-1">{{ object }}</h5>
                <small class="text-muted">Created: {{ object.created_at|date:"M d, Y" }}</small>
            </div>
            <div class="btn-group btn-group-sm">
                <a href="{% block detail_url %}{% endblock %}" class="btn btn-primary">
                    <i class="bi bi-eye"></i> View
                </a>
                <a href="{% block update_url %}{% endblock %}" class="btn btn-secondary">
                    <i class="bi bi-pencil"></i> Edit
                </a>
                <a href="{% block delete_url %}{% endblock %}" class="btn btn-danger">
                    <i class="bi bi-trash"></i> Delete
                </a>
            </div>
        </div>
    </div>
    {% endfor %}
</div>
{% else %}
<div class="alert alert-info">
    No {% block no_items_text %}{% endblock %} found. 
    <a href="{% block no_items_create_url %}{% endblock %}">Create your first one</a>.
</div>
{% endif %}

<a href="{% url 'task_detail' project.pk task.pk %}" class="btn btn-primary mt-3">
    <i class="bi bi-arrow-left"></i> Back to Task
</a>
{% endblock %}""",
        
        'detail.html': """{% extends 'opensees/base.html' %}
{% block title_content %}{{ object }}{% endblock %}

{% block component_content %}
<h2>{{ object }}</h2>

<div class="card mb-4">
    <div class="card-header">
        Details
    </div>
    <div class="card-body">
        {% block detail_content %}{% endblock %}
    </div>
</div>

<div class="btn-group">
    <a href="{% block list_url %}{% endblock %}" class="btn btn-primary">
        <i class="bi bi-list-ul"></i> Back to List
    </a>
    <a href="{% block update_url %}{% endblock %}" class="btn btn-secondary">
        <i class="bi bi-pencil"></i> Edit
    </a>
    <a href="{% block delete_url %}{% endblock %}" class="btn btn-danger">
        <i class="bi bi-trash"></i> Delete
    </a>
</div>
{% endblock %}""",
        
        'form.html': """{% extends 'opensees/base.html' %}
{% block title_content %}{% if object %}Edit{% else %}Create{% endif %} {% block form_title %}{% endblock %}{% endblock %}

{% block component_content %}
<h2>{% if object %}Edit{% else %}Create New{% endif %} {% block form_header %}{% endblock %}</h2>

<form method="post">
    {% csrf_token %}
    
    {% if form.errors %}
    <div class="alert alert-danger">
        Please correct the errors below.
    </div>
    {% endif %}
    
    {{ form.as_p }}
    
    <div class="form-group mt-4">
        <button type="submit" class="btn btn-primary">
            <i class="bi bi-save"></i> Save
        </button>
        <a href="{% block cancel_url %}{% endblock %}" class="btn btn-secondary">
            <i class="bi bi-x-circle"></i> Cancel
        </a>
    </div>
</form>
{% endblock %}""",
        
        'confirm_delete.html': """{% extends 'opensees/base.html' %}
{% block title_content %}Delete {{ object }}{% endblock %}

{% block component_content %}
<h2>Confirm Deletion</h2>

<div class="alert alert-danger">
    <p>Are you sure you want to delete "{{ object }}"?</p>
    <p>This action cannot be undone.</p>
</div>

<form method="post">
    {% csrf_token %}
    <button type="submit" class="btn btn-danger">
        <i class="bi bi-trash"></i> Confirm Delete
    </button>
    <a href="{% block cancel_url %}{% endblock %}" class="btn btn-secondary">
        <i class="bi bi-x-circle"></i> Cancel
    </a>
</form>
{% endblock %}"""
    }
    
    # Create templates for each component
    for component in components:
        component_dir = os.path.join(templates_dir, component)
        
        # Create list template
        with open(os.path.join(component_dir, 'list.html'), 'w') as f:
            content = templates['list.html'].replace('object_list', f'{component}_list')
            content = content.replace('{% block list_title %}{% endblock %}', f'{component.capitalize()} List')
            content = content.replace('{% block list_header %}{% endblock %}', f'{component.capitalize()} List')
            content = content.replace('{% block create_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_create' project.pk task.pk %}}")
            content = content.replace('{% block detail_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_detail' project.pk task.pk object.pk %}}")
            content = content.replace('{% block update_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_update' project.pk task.pk object.pk %}}")
            content = content.replace('{% block delete_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_delete' project.pk task.pk object.pk %}}")
            content = content.replace('{% block no_items_text %}{% endblock %}', f'{component}s')
            content = content.replace('{% block no_items_create_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_create' project.pk task.pk %}}")
            f.write(content)
        
        # Create detail template
        with open(os.path.join(component_dir, 'detail.html'), 'w') as f:
            content = templates['detail.html']
            content = content.replace('{% block list_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_list' project.pk task.pk %}}")
            content = content.replace('{% block update_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_update' project.pk task.pk object.pk %}}")
            content = content.replace('{% block delete_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_delete' project.pk task.pk object.pk %}}")
            f.write(content)
        
        # Create form template
        with open(os.path.join(component_dir, 'form.html'), 'w') as f:
            content = templates['form.html']
            content = content.replace('{% block form_title %}{% endblock %}', component.capitalize())
            content = content.replace('{% block form_header %}{% endblock %}', component.capitalize())
            content = content.replace('{% block cancel_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_list' project.pk task.pk %}}")
            f.write(content)
        
        # Create confirm delete template
        with open(os.path.join(component_dir, 'confirm_delete.html'), 'w') as f:
            content = templates['confirm_delete.html']
            content = content.replace('{% block cancel_url %}{% endblock %}', 
                                    f"{{% url 'opensees:{component}_detail' project.pk task.pk object.pk %}}")
            f.write(content)
    
    print(f"Successfully created OpenSees templates in {templates_dir}")

# Example usage:
if __name__ == "__main__":
    # Point this to your Django project's base directory
    BASE_DIR = Path(__file__).resolve().parent.parent
    create_opensees_templates(BASE_DIR)