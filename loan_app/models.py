from django.db import models

class LoanApplication(models.Model):
    name = models.CharField(max_length=100)

    no_of_dependents = models.IntegerField()
    education = models.IntegerField()
    self_employed = models.IntegerField()

    income_annum = models.FloatField()
    loan_amount = models.FloatField()
    loan_term = models.IntegerField()
    cibil_score = models.IntegerField()

    residential_assets_value = models.FloatField()
    commercial_assets_value = models.FloatField()
    luxury_assets_value = models.FloatField()
    bank_asset_value = models.FloatField()

    # Output
    decision = models.CharField(max_length=50)
    probability = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name