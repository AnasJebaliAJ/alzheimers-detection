# Dementia Detection Using Machine Learning

A machine-learning project that classifies dementia status using clinical and MRI-derived measurements from the OASIS-2 longitudinal dataset.

The project explores data preprocessing, classification algorithms, patient-level cross-validation, hyperparameter tuning, model interpretability, and automated testing.

## Project Overview

This project investigates whether machine-learning models can distinguish between demented and nondemented observations using demographic, cognitive, and brain-volume measurements.

Three classification algorithms were evaluated:

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

Logistic Regression achieved the highest average cross-validation balanced accuracy among the three initial models.

**Disclaimer:** This is an educational machine-learning project, not a clinically validated Alzheimer's disease diagnostic tool.

## Dataset

**Source:** [OASIS MRI and Alzheimer's Dataset](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers)

The OASIS-2 longitudinal dataset contains 373 MRI visit records with 15 columns.

The model uses the following input features:

| Feature | Description |
|---|---|
| Age | Age of the participant |
| M/F | Sex |
| EDUC | Years of education |
| SES | Socioeconomic status |
| MMSE | Mini-Mental State Examination score |
| eTIV | Estimated total intracranial volume |
| nWBV | Normalized whole-brain volume |
| ASF | Atlas scaling factor |

The classification target is the `Group` column.

For the initial binary classification experiment, `Converted` observations were excluded. The `CDR` feature was also excluded because it directly reflects clinical dementia severity.

## Methodology

1. **Data exploration:** Examined dataset shape, class distributions, and missing values.
2. **Preprocessing:** Applied median imputation to numerical features, categorical encoding, and scaling where required.
3. **Patient-level splitting:** Used `GroupShuffleSplit` to prevent visits from the same participant appearing in both training and test sets.
4. **Model comparison:** Evaluated three classification algorithms using five-fold stratified grouped cross-validation.
5. **Hyperparameter tuning:** Optimized Logistic Regression using `GridSearchCV`.
6. **Interpretability:** Examined permutation feature importance.
7. **Prediction:** Saved the trained model and implemented a separate prediction script.
8. **Testing:** Added automated tests with Pytest.

## Model Performance

### Cross-Validation Results

| Model | Mean Balanced Accuracy |
|---|---|
| Random Forest | 74.5% |
| Logistic Regression | 81.0% |
| SVM | 72.5% |

### Held-Out Test Results

| Model | Accuracy |
|---|---|
| Random Forest | 85.07% |
| Logistic Regression (untuned) | 91.04% |

The tuned Logistic Regression achieved **82.3% mean cross-validation balanced accuracy** with the following hyperparameters:

- `C = 0.1`
- `class_weight = "balanced"`

The reported 91.04% test accuracy belongs to the original, untuned Logistic Regression model.

Results should be interpreted cautiously because the dataset is relatively small and contains repeated observations from participants.

## Feature Importance

Permutation importance indicated that MMSE was the most influential feature for the evaluated Logistic Regression model.

Other contributing variables included education, normalized whole-brain volume, age, and sex.

Feature importance measures model reliance, not medical causation. These results were explored on the test data and should not be treated as independent confirmation of feature selection.

## Installation and Usage

**1. Clone the repository**

Clone your GitHub repository and open the project directory.

**2. Create a Python virtual environment**

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```powershell
python -m pip install -r requirements.txt
```

**4. Download the dataset**

Download `oasis_longitudinal.csv` from the dataset source and place it inside the `data/` directory.

**5. Train and evaluate the models**

```powershell
python main.py
```

This runs model training, evaluation, hyperparameter tuning, and model saving.

**6. Make a prediction**

```powershell
python predict.py
```

The prediction script uses synthetic example measurements and loads the saved model from `models/logistic_regression.joblib`.

**7. Run automated tests**

```powershell
python -m pytest -v
```

Training must be completed before running the prediction script or the current integration tests, because they require the saved model.

## Testing

The project includes five passing test cases covering:

- Valid prediction output
- Missing input features
- Invalid categorical input
- Negative age
- Age exceeding the accepted range

## Limitations

- The dataset is relatively small.
- Some participants have multiple MRI visits.
- The model uses clinical cognitive scores alongside MRI-derived measurements.
- The prediction probabilities have not been clinically calibrated or validated.
- Model evaluation and feature-importance exploration have reused the held-out test data.
- The model has not been externally validated on an independent clinical dataset.
- Input validation currently covers only selected fields.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, and Pytest.

## References

- [GeeksforGeeks — Alzheimer's Detection Using Machine Learning](https://www.geeksforgeeks.org/machine-learning/alzheimers-detection-using-machine-learning/)
- [OASIS MRI and Alzheimer's Dataset](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers)

## Project Status

Educational prototype with model training, evaluation, reusable prediction functionality, and automated tests implemented.