from django.db import models
from django.contrib.auth.models import User
from myapp.models import Project, Task

class OpenSeesModel(models.Model):
    name = models.CharField(max_length=100)
    ndm = models.IntegerField()
    ndf = models.IntegerField()
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class TimeSeries(models.Model):
    TYPE_CHOICES = [
        ('Path', 'Path'),
        ('Constant', 'Constant'),
        ('Linear', 'Linear'),
    ]
    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    tag = models.IntegerField()
    time_values = models.TextField(blank=True, null=True)
    data_values = models.TextField(blank=True, null=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Material(models.Model):
    TYPE_CHOICES = [
        ('Elastic', 'Elastic'),
        ('Steel01', 'Steel01'),
        ('Concrete01', 'Concrete01'),
    ]
    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    tag = models.IntegerField()
    parameters = models.TextField()  # JSON string of parameters
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Section(models.Model):
    TYPE_CHOICES = [
        ('Elastic', 'Elastic'),
        ('Fiber', 'Fiber'),
        ('WideFlange', 'WideFlange'),
    ]
    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    tag = models.IntegerField()
    parameters = models.TextField()  # JSON string of parameters
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Node(models.Model):
    tag = models.IntegerField()
    x = models.FloatField()
    y = models.FloatField()
    z = models.FloatField()
    mass_values = models.TextField(blank=True, null=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Element(models.Model):
    TYPE_CHOICES = [
        ('forceBeamColumn', 'forceBeamColumn'),
        ('truss', 'truss'),
        ('shell', 'shell'),
    ]
    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    tag = models.IntegerField()
    nodes = models.TextField()  # JSON string of node tags
    parameters = models.TextField()  # JSON string of parameters
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Constraint(models.Model):
    TYPE_CHOICES = [
        ('fix', 'fix'),
        ('equalDOF', 'equalDOF'),
    ]
    type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    node_tags = models.TextField()  # JSON string of node tags
    dofs = models.TextField()  # JSON string of DOFs
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Diaphragm(models.Model):
    perp_dirn = models.IntegerField()
    master_node = models.IntegerField()
    slave_nodes = models.TextField()  # JSON string of slave nodes
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)