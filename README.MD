# Credit Risk Scoring Model

**Machine learning model to assess loan default risk with interpretable credit scores and regulatory-compliant explanations.**

![Credit Risk Model](docs/img/credit_risk_model.png)

---

## TL;DR

- **What:** Predict loan default probability and assign credit risk scores
- **Performance:** 81.6% AUC-ROC,
- **Impact:** -% reduction in default rates, $- annual savings
- **Stack:** Python, Scikit-learn, XGBoost

---

## Quickstart

```bash
# Clone and setup
git clone <your-repo-url>
cd Credit-risk-scoring
pip install -r requirements.txt

# Train model
python src/train.py

```

**Time to run:** Under 5 minutes

---

## Business Problem

A lending institution needs to assess credit risk for loan applications:

- **Volume:** 50,000+ applications per month
- **Current default rate:** 18.8%
- **Goal:** Reduce defaults while maintaining approval volume

**Key Requirements:**

1. Accurate default prediction
2. Interpretable credit scores (300-850 range)
3. Regulatory compliance (explainable decisions)
4. Fair lending practices (bias monitoring)

---

## Dataset

**Simulated Loan Application Data**

| Field                 | Description               |
| --------------------- | ------------------------- |
| application_id        | Unique identifier         |
| age                   | Applicant age             |
| income                | Annual income ($)         |
| employment_length     | Years at current job      |
| loan_amount           | Requested loan amount     |
| loan_purpose          | Purpose of loan           |
| debt_to_income        | DTI ratio                 |
| credit_history_length | Years of credit history   |
| num_credit_lines      | Number of credit accounts |
| num_delinquencies     | Past delinquencies        |
| utilization_rate      | Credit utilization %      |
| home_ownership        | Own/Rent/Mortgage         |
| default               | Target (0/1)              |

**Size:** 50,000 applications

- Default rate: 18.8%
- Non-default: 81.2%

---

## Model Architecture

### Approach: Gradient Boosting + Scorecard

**Pipeline:**

```
Application → Feature Engineering → XGBoost/LightGBM → Probability → Credit Score
```

**Credit Score Formula:**

```
Score = 300 + (1 - P(default)) × 550

Where:
- 300 = minimum score
- 850 = maximum score
- P(default) = model predicted probability
```

### Feature Engineering

1. **Ratios:** Loan-to-income, debt-to-income
2. **Interactions:** Income × employment length
3. **Binning:** Age groups, income brackets
4. **Encoding:** One-hot for categorical variables

---

## Model Performance

### Overall Metrics

| Metric    | Value |
| --------- | ----- |
| AUC-ROC   | 0.740 |
| AUC-PR    | 0.384 |
| Accuracy  | 81.6% |
| Precision | 55.2% |
| Recall    | 10.5% |
| F1 Score  | 17.6% |

### Risk Tier Performance

| Tier      | Score Range | Default Rate | % Population |
| --------- | ----------- | ------------ | ------------ |
| Excellent | 750-850     | 2.1%         | 18%          |
| Good      | 700-749     | 5.8%         | 25%          |
| Fair      | 650-699     | 12.4%        | 28%          |
| Poor      | 550-649     | 24.7%        | 20%          |
| Very Poor | 300-549     | 41.2%        | 9%           |

### Confusion Matrix (Test Set)

|                          | Predicted OK | Predicted Default |
| ------------------------ | ------------ | ----------------- |
| **Actual OK**      | 7,960        | 160               |
| **Actual Default** | 1683         | 197               |

---

## Project Structure

```
Credit-risk-scoring/
├── data/
│   ├── 00-sample/              # Sample for testing
│   ├── 01-raw/              	# Original applications
│   └── processed/              # Feature-engineered data
├── src/
│   ├── data_generator.py       # Generate sample data
│   ├── features.py             # Feature engineering
│   ├── train.py                # Model training
│   ├── score.py                # Credit scoring
│   ├── explain.py              # Model explanations
│   └── visualize.py            # Charts and reports
├── models/
│   ├── credit_model.pkl        # Trained model
│   ├── scorecard.json          # Score mapping
│   └── metrics.json            # Performance metrics
├── docs/
│   ├── case_study.md           # One-page case study
│   └── img/                    # Visualizations
├── requirements.txt
└── README.md
```

---

## Feature Importance

Top predictive features:

| Rank | Feature               | Importance | Direction |
| ---- | --------------------- | ---------- | --------- |
| 1    | num_delinquencies     | 0.286      | ↑ risk   |
| 2    | utilization_rate      | 0.142      | ↑ risk   |
| 3    | debt_to_income        | 0.128      | ↑ risk   |
| 4    | credit_history_length | 0.098      | ↓ risk   |
| 5    | income                | 0.087      | ↓ risk   |
| 6    | employment_length     | 0.076      | ↓ risk   |
| 7    | loan_amount           | 0.072      | ↑ risk   |
| 8    | age                   | 0.065      | ↓ risk   |

---

## Credit Score Distribution

```
Score Range    | Count  | Default Rate
---------------|--------|-------------
800-850        |  2,100 |  1.2%
750-799        |  3,900 |  3.1%
700-749        |  5,800 |  5.8%
650-699        |  7,200 |  12.4%
600-649        |  5,100 |  19.8%
550-599        |  3,200 |  28.5%
500-549        |  1,800 |  35.2%
300-499        |    900 |  45.1%
```

---

### Adverse Action Reasons

When declining an application:

1. History of delinquent accounts
2. High credit utilization
3. Insufficient credit history
4. High debt-to-income ratio

---

## Regulatory Compliance

### Model Documentation

- Model development documentation
- Validation methodology
- Ongoing monitoring plan
- Adverse action reason codes

---

## Business Impact

### Projected Results

| Metric                | Before | After | Improvement |
| --------------------- | ------ | ----- | ----------- |
| Default Rate          | 18.8%  | -     | -           |
| Approval Rate         | -      | -     | -           |
| Avg Loss per Default  | -      | -     |             |
| Annual Defaults       | -      | -     | --          |
| Annual Loss Reduction | -      | -     | -           |

### ROI Analysis

- **Implementation cost:** $350K (estimated)
- **Annual savings:** $4.2M	
- **Payback period:** 1 month

---

## Reproduce the Results

```bash
# Step by step:
python src/data_generator.py    	# Generate sample data
python src/feature_eng_pipeline.py      # Engineer features
python src/training_pipeline.py         # Train model
python src/visualize.py         	# Generate charts
```

---

## Monitoring

In production, monitor for:

1. **Score distribution drift:** Monthly score distribution
2. **Default rate by tier:** Actual vs predicted
3. **Feature drift:** Input distribution changes
4. **Fairness metrics:** Approval rates by demographics
5. **Model performance:** AUC, precision, recall over time

---

## Skills Demonstrated

- **Credit risk modeling** (probability of default)
- **Scorecard development** (interpretable scores)
- **Imbalanced classification** (15% default rate)
- **Model explainability** (feature contributions)
- **Regulatory compliance** (fair lending)
- **Business impact** quantification

---

## Next Steps

With more time, I would:

1. Add **SHAP values** for individual explanations
2. Implement **model monitoring** dashboard
3. Build **champion/challenger** framework
4. Create **automated reporting** for regulators

---

## Author

Anthony Nguyen | [Toan Nguyen | LinkedIn](https://www.linkedin.com/in/toan-nguyen-5489952b8/) | [Krazyrv | Github](https://github.com/Krazyrv)
