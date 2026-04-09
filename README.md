# Adult Census Income Prediction
### Binary Classification using Logistic Regression and Random Forest

> Predicting whether an individual earns more than $50K/year using sociodemographic
> features from the UCI Adult Census Income dataset.

**Course:** Introduction to Artificial Intelligence | **Date:** November 2024
**Authors:** Rasheed Albel, Val Allen Eltagonde, Franz Andrei Layug


---

## Overview

In a data-driven world, understanding income distribution at a granular level is
both a sociological and a machine learning challenge. This project builds a
**binary classifier** to predict whether an individual's annual income exceeds
**$50,000**, based on features such as age, education, occupation, and marital
status.

Two models are compared — **Logistic Regression** and a **Random Forest
Classifier** — evaluated on accuracy, AUC, and feature importance to determine
which is better suited for this task.

---

## Dataset

- **Source:** [UCI Adult Census Income Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Size:** 32,537 records × 15 features (after deduplication)
- **Target:** `income` — binary (`<=50K` or `>50K`)
- **Class imbalance:** Majority class is `<=50K`

### Features

| Feature          | Type        | Description                              |
|------------------|-------------|------------------------------------------|
| `age`            | Numerical   | Individual's age                         |
| `workclass`      | Categorical | Employment type (Private, Self-emp, etc.)|
| `education_num`  | Numerical   | Years of education (numerical scale)     |
| `marital_status` | Categorical | Marital status                           |
| `occupation`     | Categorical | Job category                             |
| `relationship`   | Categorical | Relationship role (Husband, Wife, etc.)  |
| `race`           | Categorical | Race/ethnicity                           |
| `sex`            | Categorical | Gender                                   |
| `capital_gain`   | Numerical   | Capital gains from asset sales           |
| `capital_loss`   | Numerical   | Capital losses                           |
| `hours_per_week` | Numerical   | Weekly hours worked                      |
| `native_country` | Categorical | Country of origin                        |

---

## Exploratory Data Analysis

Key findings from EDA that informed preprocessing and modeling:

- **Age** is positively skewed — most workers are concentrated in younger age groups
- **Education level** shows a near-exponential positive relationship with `>50K`
  probability; a notable jump occurs at education level 12+
- **Hours per week** distribution is centered around 40 hours; >50K earners tend
  to work 40–60 hours
- **Workclass** is imbalanced — `Private` dominates the dataset, potentially
  biasing models
- **Race and Sex:** Male `Asian-Pac-Islander` and `White` groups have the highest
  proportion of >50K earners

---

## Methodology

### Data Preprocessing
- Removed duplicates and stripped whitespace from categorical columns
- **One-Hot Encoding** for all categorical variables (compatible with both models)
- **Standard Scaling** for skewed numerical features (`age`, `hours_per_week`, etc.)

### Feature Selection
- Applied **Recursive Feature Elimination with Cross-Validation (RFECV)** to
  reduce the expanded feature space from one-hot encoding, improving interpretability
  and reducing overfitting

### Class Imbalance Handling
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to address the
  underrepresentation of `>50K` earners and the `Private` workclass dominance,
  generating synthetic minority class samples via interpolation

### Hyperparameter Tuning
- **GridSearchCV** applied iteratively — broad search first (1, 10, 100, 1000),
  then refined to precise ranges around promising values

---

## Results

### Model Performance Comparison

| Metric                  | Logistic Regression | Random Forest |
|-------------------------|---------------------|---------------|
| Overall Accuracy        | **0.83**            | 0.81          |
| AUC Score               | 0.91                | **0.92**      |
| Macro-avg F1            | **0.79**            | 0.78          |
| Weighted-avg F1         | **0.83**            | 0.82          |

### Per-class Performance

| Class   | Model               | Precision | Recall | F1   |
|---------|---------------------|-----------|--------|------|
| `<=50K` | Logistic Regression | 0.94      | 0.81   | 0.87 |
| `<=50K` | Random Forest       | **0.95**  | 0.79   | 0.86 |
| `>50K`  | Logistic Regression | —         | **0.88**| 0.70|
| `>50K`  | Random Forest       | —         | 0.85   | 0.70 |

---

## Model Conclusion

Neither model is a clear winner — the choice depends on the use case:

- **Use Logistic Regression** if **recall for `>50K`** is critical (e.g., identifying
  high earners for a targeted program) — it achieves higher recall (0.88 vs. 0.85)
  and better overall accuracy (0.83)
- **Use Random Forest** if **class discrimination** matters most — it achieves a
  slightly higher AUC (0.92 vs. 0.91), capturing non-linear patterns such as the
  interaction between marital status, family roles, and income

Random Forest's edge in AUC comes from its ability to model non-linear relationships
(e.g., marital and familial patterns) that Logistic Regression — limited to linear
associations — cannot fully capture.

---

## Feature Importance

### Logistic Regression — Top Features
| Feature                    | Direction | Interpretation                          |
|----------------------------|-----------|-----------------------------------------|
| `capital_gain`             | Positive  | Strongest predictor of >50K income      |
| `occupation_Priv-house-serv`| Negative | Private house servants earn less        |
| `native_country_Colombia`  | Negative  | Model sees this as a negative contributor|
| `occupation_Farming-fishing`| Negative | Reflects poor economic conditions       |

> Note: `hours_per_week` and `education_num` — prominent in EDA — were not
> selected as significant by Logistic Regression.

### Random Forest — Top Features
| Feature                          | Rank | Interpretation                         |
|----------------------------------|------|----------------------------------------|
| `marital_status_Married-civ-spouse`| #1 | Strongest predictor overall            |
| `education_num`                  | #2   | Affirms EDA's positive education trend |
| `relationship_Husband`           | Top  | Familial roles linked to higher income |
| `capital_gain`                   | Top  | Present but less dominant than in LogReg|
| `occupation_Exec-managerial`     | Top  | High-salary occupation                 |
| `hours_per_week`                 | Top  | Affirms EDA's hours-income correlation |

---

## Limitations

- Logistic Regression shows potential **racial bias** — certain `native_country`
  features appear as significant negative predictors, suggesting the model may
  reflect historical disparities in the data rather than causal relationships
- SMOTE addresses class imbalance but generates synthetic data — results may
  not fully reflect real-world distributions
- Additional imbalance-handling techniques (e.g., class weighting, threshold
  tuning) could further improve minority class performance

---

## Individual Contributions

This was a collaborative project completed as part of MATH 103.1.

My primary contributions included:
- Presenting the modeling workflow and analytical approach used in the project
- Interpreting model outputs and evaluation results for technical presentation
- Communicating key findings and their implications for regional SDG progression
- Preparing presentation materials summarizing methodology and results
- Supporting technical documentation of project insights

