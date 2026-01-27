# Credit Risk Scoring Model

## Case Study

---

### TL;DR

Built a credit risk scoring model to predict loan defaults and assign interpretable credit scores (300-850 range). Achieved 0.74 AUC-ROC with explainable features, enabling risk-based pricing and regulatory-compliant lending decisions.

---

### Role & Timeline

**Role:** Data Scientist (Solo Project)
**Timeline:** 3 weeks
**Responsibilities:** Feature engineering, model training, scorecard development, explainability

---

### Business Context

A lending institution needed to assess credit risk for loan applications:

- **Volume:** 50,000+ applications per month
- **Current default rate:** 18.8%
- **Goal:** Reduce defaults while maintaining approval volume

**Requirements:**

- Accurate default prediction
- Interpretable 300-850 credit scores
- Regulatory compliance (explainable)
- Fair lending monitoring

---

### Technical Approach

**Pipeline:**

```
Application → Feature Engineering → Gradient Boosting → Probability → Credit Score
```

**Feature Engineering:**

- Ratio features: loan-to-income, payment-to-income
- Interaction features: income × employment
- Binned features: age groups, income brackets

**Credit Score Formula:**

```
Score = 300 + (1 - P(default)) × 550
```

---

### Results

| Metric    | Value |
| --------- | ----- |
| AUC-ROC   | 0.740 |
| Accuracy  | 81.6% |
| Precision | 55.2% |
| Recall    | 10.5% |

**Note:** The model is conservative (high precision, lower recall) to minimize false positives in lending decisions.

### Feature Importance

| Rank | Feature              | Importance |
| ---- | -------------------- | ---------- |
| 1    | Num Delinquencies    | 28.6%      |
| 2    | Income × Employment | 12.7%      |
| 3    | Age                  | 11.4%      |
| 4    | Income               | 7.6%       |
| 5    | Utilization Rate     | 6.7%       |

---

### Risk Tiers

| Tier      | Score Range | Expected Default |
| --------- | ----------- | ---------------- |
| Excellent | 750-850     | 2-5%             |
| Good      | 700-749     | 5-10%            |
| Fair      | 650-699     | 10-20%           |
| Poor      | 550-649     | 20-35%           |
| Very Poor | 300-549     | 35%+             |

---

### Model Explainability

For each decision, the model provides:

- Credit score (300-850)
- Risk tier classification
- Top contributing factors
- Adverse action reasons (for declines)

**Example:**

```
Score: 682 (Fair)
Factors: -45 pts (delinquencies), -32 pts (utilization), +28 pts (income)
```

---

### Regulatory Compliance

**Fair Lending Analysis:**

- Monitored approval rates by demographics
- No disparate impact detected
- Documented model development process

**Adverse Action Reasons:**

- History of delinquent accounts
- High credit utilization
- Insufficient credit history
- High debt-to-income ratio

---

### Skills Demonstrated

- **Credit risk modeling** (PD estimation)
- **Scorecard development** (interpretable scores)
- **Feature engineering** (ratios, interactions)
- **Model explainability** (feature contributions)
- **Regulatory compliance** (fair lending)

---

### Next Steps

With more time, I would:

1. Add SHAP values for individual explanations
2. Implement model monitoring dashboard
3. Add fairness constraints in training
4. Build champion/challenger framework
5. Create automated regulatory reporting

---


**Code:** [Github/Krazyrv](https://github.com/Krazyrv/Credit-risk-scoring)
**Contact:** [anthonynguyen1422@gmail.com](anthonynguyen1422@gmail.com)
