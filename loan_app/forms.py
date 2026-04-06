from django import forms

class LoanForm(forms.Form):
    name = forms.CharField(max_length=100)

    no_of_dependents = forms.IntegerField()
    education = forms.ChoiceField(choices=[(0, 'Graduate'), (1, 'Not Graduate')])
    self_employed = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    income_annum = forms.FloatField()
    loan_amount = forms.FloatField()
    loan_term = forms.IntegerField()
    cibil_score = forms.IntegerField()
    residential_assets_value = forms.FloatField()
    commercial_assets_value = forms.FloatField()
    luxury_assets_value = forms.FloatField()
    bank_asset_value = forms.FloatField()