from django import forms
from .models import *

class OpenSeesModelForm(forms.ModelForm):
    class Meta:
        model = OpenSeesModel
        fields = ['name', 'ndm', 'ndf']

class TimeSeriesForm(forms.ModelForm):
    class Meta:
        model = TimeSeries
        fields = ['type', 'tag', 'time_values', 'data_values']

class MaterialForm(forms.ModelForm):
    class Meta:
        model = Material
        fields = ['type', 'tag', 'parameters']

class SectionForm(forms.ModelForm):
    class Meta:
        model = Section
        fields = ['type', 'tag', 'parameters']

class NodeForm(forms.ModelForm):
    class Meta:
        model = Node
        fields = ['tag', 'x', 'y', 'z', 'mass_values']

class ElementForm(forms.ModelForm):
    class Meta:
        model = Element
        fields = ['type', 'tag', 'nodes', 'parameters']

class ConstraintForm(forms.ModelForm):
    class Meta:
        model = Constraint
        fields = ['type', 'node_tags', 'dofs']

class DiaphragmForm(forms.ModelForm):
    class Meta:
        model = Diaphragm
        fields = ['perp_dirn', 'master_node', 'slave_nodes']