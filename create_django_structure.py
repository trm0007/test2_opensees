import os
from pathlib import Path

def create_django_structure(base_path):
    """Create the complete Django project file and folder structure"""
    
    # Define all directories to create
    dirs = [
        "templates/myapp/includes",
        "templates/myapp/pages",
        "templates/myapp/auth",
        "templates/myapp/projects",
        "templates/myapp/tasks",
        "static/css",
        "static/js",
        "static/images"
    ]
    
    # Define all files with their content
    files = {
        # Base template
        "templates/myapp/base.html": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}My Django Site{% endblock %}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/css/style.css">
    {% block extra_css %}{% endblock %}
</head>
<body>
    {% include 'myapp/includes/header.html' %}
    {% include 'myapp/includes/navbar.html' %}
    
    <div class="container mt-4">
        <div class="row">
            {% if not hide_sidebar %}
            <div class="col-md-3">
                {% include 'myapp/includes/sidebar.html' %}
            </div>
            {% endif %}
            <div class="{% if not hide_sidebar %}col-md-9{% else %}col-12{% endif %}">
                {% block content %}{% endblock %}
            </div>
        </div>
    </div>
    
    {% include 'myapp/includes/footer.html' %}
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/script.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>""",
        
        # Includes
        "templates/myapp/includes/header.html": """<header class="bg-primary text-white text-center py-3">
    <h1>My Django Website</h1>
</header>""",
        
        "templates/myapp/includes/navbar.html": """<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container-fluid">
        <a class="navbar-brand" href="{% url 'home' %}">Django App</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav me-auto">
                <li class="nav-item">
                    <a class="nav-link" href="{% url 'home' %}">Home</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="{% url 'about' %}">About</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="{% url 'contact' %}">Contact</a>
                </li>
                {% if user.is_authenticated %}
                <li class="nav-item">
                    <a class="nav-link" href="{% url 'project_list' %}">Projects</a>
                </li>
                {% endif %}
            </ul>
            <ul class="navbar-nav">
                {% if user.is_authenticated %}
                <li class="nav-item">
                    <span class="nav-link">Welcome, {{ user.username }}</span>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="{% url 'logout' %}">Logout</a>
                </li>
                {% else %}
                <li class="nav-item">
                    <a class="nav-link" href="{% url 'login' %}">Login</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="{% url 'register' %}">Register</a>
                </li>
                {% endif %}
            </ul>
        </div>
    </div>
</nav>""",
        
        "templates/myapp/includes/sidebar.html": """<div class="card mb-4">
    <div class="card-header">
        Quick Links
    </div>
    <div class="card-body">
        <ul class="list-group list-group-flush">
            <li class="list-group-item"><a href="{% url 'home' %}">Home</a></li>
            <li class="list-group-item"><a href="{% url 'about' %}">About</a></li>
            <li class="list-group-item"><a href="{% url 'contact' %}">Contact</a></li>
            {% if user.is_authenticated %}
            <li class="list-group-item"><a href="{% url 'project_list' %}">My Projects</a></li>
            {% endif %}
        </ul>
    </div>
</div>""",
        
        "templates/myapp/includes/footer.html": """<footer class="bg-dark text-white text-center py-3 mt-4">
    <div class="container">
        <p>&copy; 2023 My Django Website. All rights reserved.</p>
    </div>
</footer>""",
        
        # Pages
        "templates/myapp/pages/home.html": """{% extends 'myapp/base.html' %}

{% block title %}Home{% endblock %}

{% block content %}
<div class="hero-section bg-light p-5 rounded mb-4">
    <h1 class="display-4">Welcome to My Django Website!</h1>
    <p class="lead">A simple Django project with authentication and CRUD functionality.</p>
    <hr class="my-4">
    <p>Get started by creating projects and tasks.</p>
    {% if not user.is_authenticated %}
    <a class="btn btn-primary btn-lg" href="{% url 'register' %}" role="button">Register</a>
    <a class="btn btn-secondary btn-lg" href="{% url 'login' %}" role="button">Login</a>
    {% else %}
    <a class="btn btn-success btn-lg" href="{% url 'project_list' %}" role="button">View Projects</a>
    {% endif %}
</div>

<!-- Slider -->
<div id="carouselExample" class="carousel slide mb-4" data-bs-ride="carousel">
    <div class="carousel-inner">
        <div class="carousel-item active">
            <img src="https://via.placeholder.com/800x300?text=First+Slide" class="d-block w-100" alt="First slide">
        </div>
        <div class="carousel-item">
            <img src="https://via.placeholder.com/800x300?text=Second+Slide" class="d-block w-100" alt="Second slide">
        </div>
        <div class="carousel-item">
            <img src="https://via.placeholder.com/800x300?text=Third+Slide" class="d-block w-100" alt="Third slide">
        </div>
    </div>
    <button class="carousel-control-prev" type="button" data-bs-target="#carouselExample" data-bs-slide="prev">
        <span class="carousel-control-prev-icon" aria-hidden="true"></span>
        <span class="visually-hidden">Previous</span>
    </button>
    <button class="carousel-control-next" type="button" data-bs-target="#carouselExample" data-bs-slide="next">
        <span class="carousel-control-next-icon" aria-hidden="true"></span>
        <span class="visually-hidden">Next</span>
    </button>
</div>

<div class="row">
    <div class="col-md-4">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Projects</h5>
                <p class="card-text">Create and manage your projects.</p>
                <a href="{% url 'project_list' %}" class="btn btn-primary">Go to Projects</a>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Tasks</h5>
                <p class="card-text">Organize tasks within projects.</p>
                <a href="{% url 'project_list' %}" class="btn btn-primary">Go to Tasks</a>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Dashboard</h5>
                <p class="card-text">View your activity and progress.</p>
                <a href="#" class="btn btn-primary">Go to Dashboard</a>
            </div>
        </div>
    </div>
</div>
{% endblock %}""",
        
        "templates/myapp/pages/about.html": """{% extends 'myapp/base.html' %}

{% block title %}About Us{% endblock %}

{% block content %}
<h1>About Us</h1>
<p>This is a simple Django website created to demonstrate various features including:</p>
<ul>
    <li>User authentication (login, register, logout)</li>
    <li>CRUD operations for Projects and Tasks</li>
    <li>Responsive design with Bootstrap</li>
    <li>Template inheritance and includes</li>
    <li>Function-based views</li>
</ul>
{% endblock %}""",
        
        "templates/myapp/pages/contact.html": """{% extends 'myapp/base.html' %}

{% block title %}Contact Us{% endblock %}

{% block content %}
<h1>Contact Us</h1>
<form>
    <div class="mb-3">
        <label for="name" class="form-label">Name</label>
        <input type="text" class="form-control" id="name">
    </div>
    <div class="mb-3">
        <label for="email" class="form-label">Email</label>
        <input type="email" class="form-control" id="email">
    </div>
    <div class="mb-3">
        <label for="message" class="form-label">Message</label>
        <textarea class="form-control" id="message" rows="3"></textarea>
    </div>
    <button type="submit" class="btn btn-primary">Submit</button>
</form>
{% endblock %}""",
        
        # Auth
        "templates/myapp/auth/login.html": """{% extends 'myapp/base.html' %}
{% block title %}Login{% endblock %}
{% block content %}
<h1>Login</h1>
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit" class="btn btn-primary">Login</button>
</form>
<p class="mt-3">Don't have an account? <a href="{% url 'register' %}">Register here</a></p>
{% endblock %}""",
        
        "templates/myapp/auth/register.html": """{% extends 'myapp/base.html' %}
{% block title %}Register{% endblock %}
{% block content %}
<h1>Register</h1>
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit" class="btn btn-primary">Register</button>
</form>
<p class="mt-3">Already have an account? <a href="{% url 'login' %}">Login here</a></p>
{% endblock %}""",
        
        # Projects
        "templates/myapp/projects/list.html": """{% extends 'myapp/base.html' %}
{% block title %}My Projects{% endblock %}
{% block content %}
<h1>My Projects</h1>
<a href="{% url 'project_create' %}" class="btn btn-success mb-3">Create New Project</a>
<div class="list-group">
    {% for project in projects %}
    <div class="list-group-item">
        <div class="d-flex w-100 justify-content-between">
            <h5 class="mb-1">{{ project.title }}</h5>
            <small>{{ project.created_at|date:"M d, Y" }}</small>
        </div>
        <p class="mb-1">{{ project.description|truncatechars:100 }}</p>
        <div class="btn-group btn-group-sm">
            <a href="{% url 'project_detail' project.pk %}" class="btn btn-primary">View</a>
            <a href="{% url 'project_update' project.pk %}" class="btn btn-secondary">Edit</a>
            <a href="{% url 'project_delete' project.pk %}" class="btn btn-danger">Delete</a>
            <a href="{% url 'task_list' project.pk %}" class="btn btn-info">Tasks</a>
        </div>
    </div>
    {% empty %}
    <div class="list-group-item">
        <p>No projects yet. <a href="{% url 'project_create' %}">Create your first project</a></p>
    </div>
    {% endfor %}
</div>
{% endblock %}""",
        
        "templates/myapp/projects/detail.html": """{% extends 'myapp/base.html' %}
{% block title %}{{ project.title }}{% endblock %}
{% block content %}
<h1>{{ project.title }}</h1>
<p>Created: {{ project.created_at|date:"M d, Y" }}</p>
<p>Last updated: {{ project.updated_at|date:"M d, Y" }}</p>
<div class="card mb-4">
    <div class="card-body">
        <h5 class="card-title">Description</h5>
        <p class="card-text">{{ project.description }}</p>
    </div>
</div>
<a href="{% url 'project_update' project.pk %}" class="btn btn-secondary">Edit</a>
<a href="{% url 'project_list' %}" class="btn btn-primary">Back to Projects</a>
{% endblock %}""",
        
        "templates/myapp/projects/form.html": """{% extends 'myapp/base.html' %}
{% block title %}{% if form.instance.pk %}Edit{% else %}Create{% endif %} Project{% endblock %}
{% block content %}
<h1>{% if form.instance.pk %}Edit{% else %}Create{% endif %} Project</h1>
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit" class="btn btn-primary">Save</button>
    <a href="{% url 'project_list' %}" class="btn btn-secondary">Cancel</a>
</form>
{% endblock %}""",
        
        # Tasks
        "templates/myapp/tasks/list.html": """{% extends 'myapp/base.html' %}
{% block title %}Tasks for {{ project.title }}{% endblock %}
{% block content %}
<h1>Tasks for {{ project.title }}</h1>
<a href="{% url 'task_create' project.pk %}" class="btn btn-success mb-3">Create New Task</a>
<a href="{% url 'project_list' %}" class="btn btn-secondary mb-3">Back to Projects</a>
<div class="list-group">
    {% for task in tasks %}
    <div class="list-group-item {% if task.completed %}list-group-item-success{% endif %}">
        <div class="d-flex w-100 justify-content-between">
            <h5 class="mb-1">
                {% if task.completed %}<s>{% endif %}
                {{ task.title }}
                {% if task.completed %}</s>{% endif %}
            </h5>
            <small>{{ task.created_at|date:"M d, Y" }}</small>
        </div>
        <p class="mb-1">{{ task.description|truncatechars:100 }}</p>
        <div class="btn-group btn-group-sm">
            <a href="{% url 'task_detail' project.pk task.pk %}" class="btn btn-primary">View</a>
            <a href="{% url 'task_update' project.pk task.pk %}" class="btn btn-secondary">Edit</a>
            <a href="{% url 'task_delete' project.pk task.pk %}" class="btn btn-danger">Delete</a>
        </div>
    </div>
    {% empty %}
    <div class="list-group-item">
        <p>No tasks yet. <a href="{% url 'task_create' project.pk %}">Create your first task</a></p>
    </div>
    {% endfor %}
</div>
{% endblock %}""",
        
        "templates/myapp/tasks/detail.html": """{% extends 'myapp/base.html' %}
{% block title %}{{ task.title }}{% endblock %}
{% block content %}
<h1>{{ task.title }}</h1>
<p>Status: {% if task.completed %}Completed{% else %}Pending{% endif %}</p>
<p>Created: {{ task.created_at|date:"M d, Y" }}</p>
<p>Last updated: {{ task.updated_at|date:"M d, Y" }}</p>
<div class="card mb-4">
    <div class="card-body">
        <h5 class="card-title">Description</h5>
        <p class="card-text">{{ task.description }}</p>
    </div>
</div>
<a href="{% url 'task_update' project.pk task.pk %}" class="btn btn-secondary">Edit</a>
<a href="{% url 'task_list' project.pk %}" class="btn btn-primary">Back to Tasks</a>
{% endblock %}""",
        
        "templates/myapp/tasks/form.html": """{% extends 'myapp/base.html' %}
{% block title %}{% if form.instance.pk %}Edit{% else %}Create{% endif %} Task{% endblock %}
{% block content %}
<h1>{% if form.instance.pk %}Edit{% else %}Create{% endif %} Task</h1>
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit" class="btn btn-primary">Save</button>
    <a href="{% url 'task_list' project.pk %}" class="btn btn-secondary">Cancel</a>
</form>
{% endblock %}""",
        
        # Static files
        "static/css/style.css": """/* General Styles */
body {
    padding-top: 20px;
    font-family: Arial, sans-serif;
}

.hero-section {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Navbar Styles */
.navbar {
    margin-bottom: 20px;
}

/* Footer Styles */
footer {
    margin-top: 50px;
    padding: 20px 0;
}

/* Task List Styles */
.completed {
    text-decoration: line-through;
    color: #6c757d;
}

/* Form Styles */
form {
    max-width: 600px;
    margin: 0 auto;
}

/* Card Styles */
.card {
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Button Styles */
.btn {
    margin-right: 5px;
}

/* Responsive Adjustments */
@media (max-width: 768px) {
    .sidebar {
        display: none;
    }
}""",
        
        "static/js/script.js": """// Initialize carousel
document.addEventListener('DOMContentLoaded', function() {
    // Slider functionality
    const myCarousel = new bootstrap.Carousel(document.getElementById('carouselExample'), {
        interval: 3000,
        wrap: true
    });
    
    // Any other custom JS can go here
    console.log('Script loaded successfully');
    
    // Example: Add active class to current page in navbar
    const currentUrl = window.location.pathname;
    document.querySelectorAll('.navbar-nav a').forEach(link => {
        if (link.getAttribute('href') === currentUrl) {
            link.classList.add('active');
        }
    });
});"""
    }
    
    # Create all directories
    for directory in dirs:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create all files
    for file_path, content in files.items():
        full_path = os.path.join(base_path, file_path)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created file: {full_path}")
    
    print("\nDjango project structure created successfully!")

if __name__ == "__main__":
    # Get the current directory or specify a different base path
    base_path = os.getcwd()
    
    # If you want to create this in a specific directory, uncomment and modify:
    # base_path = "path/to/your/project"
    
    create_django_structure(base_path)