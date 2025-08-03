# opensees/forms.py
from django import forms

from django import forms

class OpenSeesModelForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    ndm = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    ndf = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )


class TimeSeriesForm(forms.Form):
    TYPE_CHOICES = [
        ('Path', 'Path'),
        ('Constant', 'Constant'),
        ('Linear', 'Linear'),
    ]
    type = forms.ChoiceField(
        choices=TYPE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}))
    tag = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    time_values = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter comma-separated time values'
        }))
    data_values = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter comma-separated data values'
        }))

class MaterialForm(forms.Form):
    TYPE_CHOICES = [
        ('Elastic', 'Elastic'),
        ('Steel01', 'Steel01'),
        ('Concrete01', 'Concrete01'),
    ]
    type = forms.ChoiceField(
        choices=TYPE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}))
    tag = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    parameters = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter parameters as JSON'
        }))

class SectionForm(forms.Form):
    TYPE_CHOICES = [
        ('Elastic', 'Elastic'),
        ('Fiber', 'Fiber'),
        ('WideFlange', 'WideFlange'),
    ]
    type = forms.ChoiceField(
        choices=TYPE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}))
    tag = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    parameters = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter parameters as JSON'
        }))

class NodeForm(forms.Form):
    tag = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    x = forms.FloatField(
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    y = forms.FloatField(
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    z = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    mass_values = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter mass values as JSON array'
        }))

class ElementForm(forms.Form):
    TYPE_CHOICES = [
        ('forceBeamColumn', 'forceBeamColumn'),
        ('truss', 'truss'),
        ('shell', 'shell'),
    ]
    type = forms.ChoiceField(
        choices=TYPE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}))
    tag = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    nodes = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter node tags as JSON array'
        }))
    parameters = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter parameters as JSON'
        }))

class ConstraintForm(forms.Form):
    TYPE_CHOICES = [
        ('fix', 'fix'),
        ('equalDOF', 'equalDOF'),
    ]
    type = forms.ChoiceField(
        choices=TYPE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}))
    node_tags = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter node tags as JSON array'
        }))
    dofs = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter DOFs as JSON array'
        }))

class DiaphragmForm(forms.Form):
    perp_dirn = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    master_node = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control'}))
    slave_nodes = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter slave nodes as JSON array'
        }))