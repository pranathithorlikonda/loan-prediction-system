from django.shortcuts import render
from .forms import LoanForm
from ml.predict import predict_loan
from .models import LoanApplication


def home(request):
    form = LoanForm()
    return render(request, 'index.html', {'form': form})


def predict_view(request):
    if request.method == 'POST':
        form = LoanForm(request.POST)

        if form.is_valid():
            print("FORM VALID ✅")

            try:
                # ---------------------------
                # GET INPUT
                # ---------------------------
                name = form.cleaned_data['name']

                data = [
                    form.cleaned_data['no_of_dependents'],
                    int(form.cleaned_data['education']),
                    int(form.cleaned_data['self_employed']),
                    form.cleaned_data['income_annum'],
                    form.cleaned_data['loan_amount'],
                    form.cleaned_data['loan_term'],
                    form.cleaned_data['cibil_score'],
                    form.cleaned_data['residential_assets_value'],
                    form.cleaned_data['commercial_assets_value'],
                    form.cleaned_data['luxury_assets_value'],
                    form.cleaned_data['bank_asset_value'],
                ]

                # ---------------------------
                # ML PREDICTION + SHAP
                # ---------------------------
                result = predict_loan(data, use_scaling=True)
                print("PREDICTION + SHAP DONE ✅")

                # ---------------------------
                # SAVE TO DATABASE
                # ---------------------------
                LoanApplication.objects.create(
                    name=name,
                    no_of_dependents=form.cleaned_data['no_of_dependents'],
                    education=int(form.cleaned_data['education']),
                    self_employed=int(form.cleaned_data['self_employed']),
                    income_annum=form.cleaned_data['income_annum'],
                    loan_amount=form.cleaned_data['loan_amount'],
                    loan_term=form.cleaned_data['loan_term'],
                    cibil_score=form.cleaned_data['cibil_score'],
                    residential_assets_value=form.cleaned_data['residential_assets_value'],
                    commercial_assets_value=form.cleaned_data['commercial_assets_value'],
                    luxury_assets_value=form.cleaned_data['luxury_assets_value'],
                    bank_asset_value=form.cleaned_data['bank_asset_value'],
                    decision=result['decision'],
                    probability=result['probability']
                )

                print("DATA SAVED TO DATABASE ✅")

                # ---------------------------
                # SHAP DATA FOR UI
                # ---------------------------
                shap_items = []

                for feature, value in result['shap_values'].items():
                    shap_items.append({
                        'feature': feature,
                        'value': value,
                        'impact': 'Positive' if value > 0 else 'Negative'
                    })

                # Sort by importance (optional but professional)
                shap_items = sorted(shap_items, key=lambda x: abs(x['value']), reverse=True)

                # ---------------------------
                # RENDER RESULT
                # ---------------------------
                return render(request, 'result.html', {
                    'result': result,
                    'name': name,
                    'shap_items': shap_items
                })

            except Exception as e:
                print("ERROR ❌:", e)
                return render(request, 'index.html', {
                    'form': form,
                    'error': str(e)
                })

        else:
            print("FORM ERRORS ❌:", form.errors)

    return render(request, 'index.html', {'form': LoanForm()})