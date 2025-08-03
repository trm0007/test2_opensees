from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from myapp.models import Project, Task
from .models import *
from .forms import *

# Model Views
@login_required
def model_list(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    models = OpenSeesModel.objects.filter(task=task)
    print(models)
    return render(request, 'opensees/model/list.html', {
        'project': project,
        'task': task,
        'model_list': models
    })

@login_required
def model_detail(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    models = OpenSeesModel.objects.filter(task=task)
    print(models)
    model = get_object_or_404(OpenSeesModel, pk=pk, task=task)
    return render(request, 'opensees/model/detail.html', {
        'project': project,
        'task': task,
        'model': model,
        'object': model,  # make sure this line exists
    })

@login_required
def model_create(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    if request.method == 'POST':
        form = OpenSeesModelForm(request.POST)
        if form.is_valid():
            print(10)
            model = form.save(commit=False)
            model.project = project
            model.task = task
            model.save()
            print(100)
            return redirect('opensees:model_list', project_pk=project.pk, task_pk=task.pk)
    else:
        form = OpenSeesModelForm()
    return render(request, 'opensees/model/form.html', {
        'form': form,
        'project': project,
        'task': task
    })

@login_required
def model_update(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    model = get_object_or_404(OpenSeesModel, pk=pk, task=task)
    if request.method == 'POST':
        form = OpenSeesModelForm(request.POST, instance=model)
        if form.is_valid():
            form.save()
            return redirect('opensees:model_detail', project_pk=project.pk, task_pk=task.pk, pk=model.pk)
    else:
        form = OpenSeesModelForm(instance=model)
    return render(request, 'opensees/model/form.html', {
        'form': form,
        'project': project,
        'task': task,
        'model': model
    })

@login_required
def model_delete(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    model = get_object_or_404(OpenSeesModel, pk=pk, task=task)
    if request.method == 'POST':
        model.delete()
        return redirect('opensees:model_list', project_pk=project.pk, task_pk=task.pk)
    return render(request, 'opensees/model/confirm_delete.html', {
        'project': project,
        'task': task,
        'model': model
    })

# TimeSeries Views
@login_required
def timeseries_list(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    timeseries_list = TimeSeries.objects.filter(task=task)
    return render(request, 'opensees/timeseries/list.html', {
        'project': project,
        'task': task,
        'timeseries_list': timeseries_list
    })

@login_required
def timeseries_detail(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    timeseries = get_object_or_404(TimeSeries, pk=pk, task=task)
    return render(request, 'opensees/timeseries/detail.html', {
        'project': project,
        'task': task,
        'timeseries': timeseries
    })

@login_required
def timeseries_create(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    if request.method == 'POST':
        form = TimeSeriesForm(request.POST)
        if form.is_valid():
            timeseries = form.save(commit=False)
            timeseries.project = project
            timeseries.task = task
            timeseries.save()
            return redirect('opensees:timeseries_list', project_pk=project.pk, task_pk=task.pk)
    else:
        form = TimeSeriesForm()
    return render(request, 'opensees/timeseries/form.html', {
        'form': form,
        'project': project,
        'task': task
    })

@login_required
def timeseries_update(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    timeseries = get_object_or_404(TimeSeries, pk=pk, task=task)
    if request.method == 'POST':
        form = TimeSeriesForm(request.POST, instance=timeseries)
        if form.is_valid():
            form.save()
            return redirect('opensees:timeseries_detail', project_pk=project.pk, task_pk=task.pk, pk=timeseries.pk)
    else:
        form = TimeSeriesForm(instance=timeseries)
    return render(request, 'opensees/timeseries/form.html', {
        'form': form,
        'project': project,
        'task': task,
        'timeseries': timeseries
    })

@login_required
def timeseries_delete(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    timeseries = get_object_or_404(TimeSeries, pk=pk, task=task)
    if request.method == 'POST':
        timeseries.delete()
        return redirect('opensees:timeseries_list', project_pk=project.pk, task_pk=task.pk)
    return render(request, 'opensees/timeseries/confirm_delete.html', {
        'project': project,
        'task': task,
        'timeseries': timeseries
    })

# Material Views
@login_required
def material_list(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    materials = Material.objects.filter(task=task)
    return render(request, 'opensees/material/list.html', {
        'project': project,
        'task': task,
        'materials': materials
    })

@login_required
def material_detail(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    material = get_object_or_404(Material, pk=pk, task=task)
    return render(request, 'opensees/material/detail.html', {
        'project': project,
        'task': task,
        'material': material
    })

@login_required
def material_create(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    if request.method == 'POST':
        form = MaterialForm(request.POST)
        if form.is_valid():
            material = form.save(commit=False)
            material.project = project
            material.task = task
            material.save()
            return redirect('opensees:material_list', project_pk=project.pk, task_pk=task.pk)
    else:
        form = MaterialForm()
    return render(request, 'opensees/material/form.html', {
        'form': form,
        'project': project,
        'task': task
    })

@login_required
def material_update(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    material = get_object_or_404(Material, pk=pk, task=task)
    if request.method == 'POST':
        form = MaterialForm(request.POST, instance=material)
        if form.is_valid():
            form.save()
            return redirect('opensees:material_detail', project_pk=project.pk, task_pk=task.pk, pk=material.pk)
    else:
        form = MaterialForm(instance=material)
    return render(request, 'opensees/material/form.html', {
        'form': form,
        'project': project,
        'task': task,
        'material': material
    })

@login_required
def material_delete(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    material = get_object_or_404(Material, pk=pk, task=task)
    if request.method == 'POST':
        material.delete()
        return redirect('opensees:material_list', project_pk=project.pk, task_pk=task.pk)
    return render(request, 'opensees/material/confirm_delete.html', {
        'project': project,
        'task': task,
        'material': material
    })

# Section Views
@login_required
def section_list(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    sections = Section.objects.filter(task=task)
    return render(request, 'opensees/section/list.html', {
        'project': project,
        'task': task,
        'sections': sections
    })

@login_required
def section_detail(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    section = get_object_or_404(Section, pk=pk, task=task)
    return render(request, 'opensees/section/detail.html', {
        'project': project,
        'task': task,
        'section': section
    })

@login_required
def section_create(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    if request.method == 'POST':
        form = SectionForm(request.POST)
        if form.is_valid():
            section = form.save(commit=False)
            section.project = project
            section.task = task
            section.save()
            return redirect('opensees:section_list', project_pk=project.pk, task_pk=task.pk)
    else:
        form = SectionForm()
    return render(request, 'opensees/section/form.html', {
        'form': form,
        'project': project,
        'task': task
    })

@login_required
def section_update(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    section = get_object_or_404(Section, pk=pk, task=task)
    if request.method == 'POST':
        form = SectionForm(request.POST, instance=section)
        if form.is_valid():
            form.save()
            return redirect('opensees:section_detail', project_pk=project.pk, task_pk=task.pk, pk=section.pk)
    else:
        form = SectionForm(instance=section)
    return render(request, 'opensees/section/form.html', {
        'form': form,
        'project': project,
        'task': task,
        'section': section
    })

@login_required
def section_delete(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    section = get_object_or_404(Section, pk=pk, task=task)
    if request.method == 'POST':
        section.delete()
        return redirect('opensees:section_list', project_pk=project.pk, task_pk=task.pk)
    return render(request, 'opensees/section/confirm_delete.html', {
        'project': project,
        'task': task,
        'section': section
    })

# Node Views
@login_required
def node_list(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    nodes = Node.objects.filter(task=task).order_by('tag')
    return render(request, 'opensees/node/list.html', {
        'project': project,
        'task': task,
        'nodes': nodes
    })

@login_required
def node_detail(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    node = get_object_or_404(Node, pk=pk, task=task)
    return render(request, 'opensees/node/detail.html', {
        'project': project,
        'task': task,
        'node': node
    })

@login_required
def node_create(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    if request.method == 'POST':
        form = NodeForm(request.POST)
        if form.is_valid():
            node = form.save(commit=False)
            node.project = project
            node.task = task
            node.save()
            return redirect('opensees:node_list', project_pk=project.pk, task_pk=task.pk)
    else:
        form = NodeForm()
    return render(request, 'opensees/node/form.html', {
        'form': form,
        'project': project,
        'task': task
    })

@login_required
def node_update(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    node = get_object_or_404(Node, pk=pk, task=task)
    if request.method == 'POST':
        form = NodeForm(request.POST, instance=node)
        if form.is_valid():
            form.save()
            return redirect('opensees:node_detail', project_pk=project.pk, task_pk=task.pk, pk=node.pk)
    else:
        form = NodeForm(instance=node)
    return render(request, 'opensees/node/form.html', {
        'form': form,
        'project': project,
        'task': task,
        'node': node
    })

@login_required
def node_delete(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    node = get_object_or_404(Node, pk=pk, task=task)
    if request.method == 'POST':
        node.delete()
        return redirect('opensees:node_list', project_pk=project.pk, task_pk=task.pk)
    return render(request, 'opensees/node/confirm_delete.html', {
        'project': project,
        'task': task,
        'node': node
    })

# Element Views
@login_required
def element_list(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    elements = Element.objects.filter(task=task).order_by('tag')
    return render(request, 'opensees/element/list.html', {
        'project': project,
        'task': task,
        'elements': elements
    })

@login_required
def element_detail(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    element = get_object_or_404(Element, pk=pk, task=task)
    return render(request, 'opensees/element/detail.html', {
        'project': project,
        'task': task,
        'element': element
    })

@login_required
def element_create(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    if request.method == 'POST':
        form = ElementForm(request.POST)
        if form.is_valid():
            element = form.save(commit=False)
            element.project = project
            element.task = task
            element.save()
            return redirect('opensees:element_list', project_pk=project.pk, task_pk=task.pk)
    else:
        form = ElementForm()
    return render(request, 'opensees/element/form.html', {
        'form': form,
        'project': project,
        'task': task
    })

@login_required
def element_update(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    element = get_object_or_404(Element, pk=pk, task=task)
    if request.method == 'POST':
        form = ElementForm(request.POST, instance=element)
        if form.is_valid():
            form.save()
            return redirect('opensees:element_detail', project_pk=project.pk, task_pk=task.pk, pk=element.pk)
    else:
        form = ElementForm(instance=element)
    return render(request, 'opensees/element/form.html', {
        'form': form,
        'project': project,
        'task': task,
        'element': element
    })

@login_required
def element_delete(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    element = get_object_or_404(Element, pk=pk, task=task)
    if request.method == 'POST':
        element.delete()
        return redirect('opensees:element_list', project_pk=project.pk, task_pk=task.pk)
    return render(request, 'opensees/element/confirm_delete.html', {
        'project': project,
        'task': task,
        'element': element
    })

# Constraint Views
@login_required
def constraint_list(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    constraints = Constraint.objects.filter(task=task)
    return render(request, 'opensees/constraint/list.html', {
        'project': project,
        'task': task,
        'constraints': constraints
    })

@login_required
def constraint_detail(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    constraint = get_object_or_404(Constraint, pk=pk, task=task)
    return render(request, 'opensees/constraint/detail.html', {
        'project': project,
        'task': task,
        'constraint': constraint
    })

@login_required
def constraint_create(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    if request.method == 'POST':
        form = ConstraintForm(request.POST)
        if form.is_valid():
            constraint = form.save(commit=False)
            constraint.project = project
            constraint.task = task
            constraint.save()
            return redirect('opensees:constraint_list', project_pk=project.pk, task_pk=task.pk)
    else:
        form = ConstraintForm()
    return render(request, 'opensees/constraint/form.html', {
        'form': form,
        'project': project,
        'task': task
    })

@login_required
def constraint_update(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    constraint = get_object_or_404(Constraint, pk=pk, task=task)
    if request.method == 'POST':
        form = ConstraintForm(request.POST, instance=constraint)
        if form.is_valid():
            form.save()
            return redirect('opensees:constraint_detail', project_pk=project.pk, task_pk=task.pk, pk=constraint.pk)
    else:
        form = ConstraintForm(instance=constraint)
    return render(request, 'opensees/constraint/form.html', {
        'form': form,
        'project': project,
        'task': task,
        'constraint': constraint
    })

@login_required
def constraint_delete(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    constraint = get_object_or_404(Constraint, pk=pk, task=task)
    if request.method == 'POST':
        constraint.delete()
        return redirect('opensees:constraint_list', project_pk=project.pk, task_pk=task.pk)
    return render(request, 'opensees/constraint/confirm_delete.html', {
        'project': project,
        'task': task,
        'constraint': constraint
    })

# Diaphragm Views
@login_required
def diaphragm_list(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    diaphragms = Diaphragm.objects.filter(task=task)
    return render(request, 'opensees/diaphragm/list.html', {
        'project': project,
        'task': task,
        'diaphragms': diaphragms
    })

@login_required
def diaphragm_detail(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    diaphragm = get_object_or_404(Diaphragm, pk=pk, task=task)
    return render(request, 'opensees/diaphragm/detail.html', {
        'project': project,
        'task': task,
        'diaphragm': diaphragm
    })

@login_required
def diaphragm_create(request, project_pk, task_pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    if request.method == 'POST':
        form = DiaphragmForm(request.POST)
        if form.is_valid():
            diaphragm = form.save(commit=False)
            diaphragm.project = project
            diaphragm.task = task
            diaphragm.save()
            return redirect('opensees:diaphragm_list', project_pk=project.pk, task_pk=task.pk)
    else:
        form = DiaphragmForm()
    return render(request, 'opensees/diaphragm/form.html', {
        'form': form,
        'project': project,
        'task': task
    })

@login_required
def diaphragm_update(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    diaphragm = get_object_or_404(Diaphragm, pk=pk, task=task)
    if request.method == 'POST':
        form = DiaphragmForm(request.POST, instance=diaphragm)
        if form.is_valid():
            form.save()
            return redirect('opensees:diaphragm_detail', project_pk=project.pk, task_pk=task.pk, pk=diaphragm.pk)
    else:
        form = DiaphragmForm(instance=diaphragm)
    return render(request, 'opensees/diaphragm/form.html', {
        'form': form,
        'project': project,
        'task': task,
        'diaphragm': diaphragm
    })

@login_required
def diaphragm_delete(request, project_pk, task_pk, pk):
    project = get_object_or_404(Project, pk=project_pk, user=request.user)
    task = get_object_or_404(Task, pk=task_pk, project=project)
    diaphragm = get_object_or_404(Diaphragm, pk=pk, task=task)
    if request.method == 'POST':
        diaphragm.delete()
        return redirect('opensees:diaphragm_list', project_pk=project.pk, task_pk=task.pk)
    return render(request, 'opensees/diaphragm/confirm_delete.html', {
        'project': project,
        'task': task,
        'diaphragm': diaphragm
    })