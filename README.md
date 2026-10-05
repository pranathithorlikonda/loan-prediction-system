# AI-Based Loan Prediction System

An intelligent full-stack web application that predicts whether a loan application should be **Approved**, sent for **Manual Review**, or **Rejected**. The system uses XGBoost for prediction and SHAP to explain the factors influencing each loan decision.

## Features

- Loan outcome prediction using XGBoost
- Explainable AI insights using SHAP
- Three prediction categories: Approved, Manual Review, and Rejected
- Data preprocessing, model training, and model evaluation
- PostgreSQL database integration
- Interactive dashboard with Chart.js
- Responsive user interface built with Bootstrap
- User-friendly loan application form
- Transparent and interpretable loan decision insights

## Technologies Used

| Layer | Technologies |
|---|---|
| Backend | Python, Django |
| Machine Learning | XGBoost, Scikit-learn, Pandas, NumPy, SHAP |
| Frontend | HTML, CSS, Bootstrap, JavaScript, Chart.js |
| Database | PostgreSQL |
| Tools | Git, GitHub, VS Code |

## Project Objective

To build an intelligent and explainable loan prediction system that analyzes borrower information and provides transparent loan decision insights. The application helps users understand not only the predicted outcome but also the key factors affecting that decision.

## Project Structure

```text
loan-prediction-system/
│
├── loan_app/               # Django application
├── templates/              # HTML templates
├── static/                 # CSS, JavaScript, and images
├── model/                  # Trained ML model and preprocessing files
├── requirements.txt
└── manage.py
```

## Installation

### Prerequisites

- Python 3.10 or above
- PostgreSQL
- Git

### Setup Instructions

1. Clone the repository:

```bash
git clone [https://github.com/pranathithorlikonda/loan-prediction-system.git](https://github.com/pranathithorlikonda/loan-prediction-system.git)
cd loan-prediction-system
```

2. Create and activate a virtual environment:

For Windows:

```bash
python -m venv loan_env
loan_env\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv loan_env
source loan_env/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Configure PostgreSQL:

Create a new PostgreSQL database and update the database name, username, password, host, and port in the `DATABASES` section of `settings.py`.

For better security, use environment variables instead of adding database credentials directly to the code.

5. Apply database migrations:

```bash
python manage.py migrate
```

6. Start the Django development server:

```bash
python manage.py runserver
```

7. Open the application in your browser:

```text
http://127.0.0.1:8000/
```

## How It Works

1. The user enters borrower and loan-related details.
2. The Django backend preprocesses the input data.
3. The trained XGBoost model predicts the loan outcome.
4. SHAP analyzes the contribution of each feature to the prediction.
5. The prediction result and explanation are displayed in the dashboard.

## Model Evaluation

The machine learning model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. SHAP values provide feature-level explanations for individual predictions, making the loan approval process transparent and interpretable.

## Future Enhancements

- Add user authentication and role-based access
- Deploy the application to a cloud platform
- Add an automated model retraining pipeline
- Improve dashboard analytics and reporting
- Provide REST API support for external integrations

## Author

**Pranathi Thorlikonda**

- GitHub: [pranathithorlikonda](https://github.com/pranathithorlikonda)
- LinkedIn: [Pranathi Thorlikonda](https://www.linkedin.com/in/pranathi-thorlikonda-929804376/)
