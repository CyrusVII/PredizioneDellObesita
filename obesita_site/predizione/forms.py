from django import forms

class InputForm(forms.Form):
    Gender = forms.ChoiceField(choices=[('0', 'Female'), ('1', 'Male')])
    Age = forms.FloatField()
    Height = forms.FloatField()
    Weight = forms.FloatField()
    FCVC = forms.FloatField(label="Frequency of Vegetables")
    NCP = forms.FloatField(label="Number of Meals")
    CAEC = forms.ChoiceField(choices=[('0','No'),('1','Sometimes'),('2','Frequently'),('3','Always')])
    CH2O = forms.FloatField(label="Water Intake")
    FAF = forms.FloatField(label="Physical Activity")
    TUE = forms.FloatField(label="Technology Usage")
    CALC = forms.ChoiceField(choices=[('0','No'),('1','Sometimes'),('2','Frequently'),('3','Always')])
    MTRANS = forms.ChoiceField(choices=[('0','Automobile'),('1','Motorbike'),('2','Bike'),('3','Public Transportation'),('4','Walking')])
