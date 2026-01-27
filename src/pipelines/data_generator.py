"""
Credit Risk Data Generator
Creates realistic loan application data for credit scoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

# Configuration
N_APPLICATIONS = 50000
DEFAULT_RATE = 0.15

LOAN_PURPOSES = ['debt_consolidation', 'home_improvement', 'major_purchase', 
                 'medical', 'car', 'vacation', 'wedding', 'other']
HOME_OWNERSHIP = ['RENT', 'OWN', 'MORTGAGE']


def generate_applicant_profile():
    """Generate a single applicant profile."""
    
    # Age (skewed towards 25-55)
    age = int(np.clip(np.random.normal(38, 12), 21, 75))
    
    # Income (log-normal, correlated with age)
    base_income = np.random.lognormal(mean=10.8, sigma=0.5)
    age_factor = 1 + (age - 25) * 0.01 if age < 50 else 1 + 0.25 - (age - 50) * 0.01
    income = int(np.clip(base_income * age_factor, 20000, 500000))
    
    # Employment length (correlated with age)
    max_emp = min(age - 18, 40)
    employment_length = np.random.randint(0, max(1, max_emp))
    
    # Credit history length (correlated with age)
    max_history = age - 18
    credit_history_length = np.random.randint(0, max(1, max_history))
    
    # Number of credit lines (correlated with history)
    num_credit_lines = int(np.clip(np.random.poisson(3 + credit_history_length * 0.3), 1, 20))
    
    # Utilization rate (beta distribution)
    utilization_rate = round(np.random.beta(2, 5) * 100, 1)
    
    # Loan amount (correlated with income)
    loan_amount = int(np.clip(
        np.random.lognormal(mean=np.log(income * 0.3), sigma=0.5),
        1000, income * 2
    ))
    
    # Debt to income (partially random, partially based on loan)
    existing_debt_ratio = np.random.beta(2, 5) * 0.5
    new_payment_ratio = (loan_amount / 60) / (income / 12)  # Assume 60-month loan
    debt_to_income = round(existing_debt_ratio + new_payment_ratio, 3)
    
    # Home ownership (correlated with age and income)
    if age < 30 or income < 40000:
        home_probs = [0.6, 0.1, 0.3]
    elif income > 100000:
        home_probs = [0.15, 0.25, 0.6]
    else:
        home_probs = [0.35, 0.2, 0.45]
    home_ownership = np.random.choice(HOME_OWNERSHIP, p=home_probs)
    
    # Loan purpose
    loan_purpose = np.random.choice(LOAN_PURPOSES)
    
    return {
        'age': age,
        'income': income,
        'employment_length': employment_length,
        'loan_amount': loan_amount,
        'loan_purpose': loan_purpose,
        'debt_to_income': debt_to_income,
        'credit_history_length': credit_history_length,
        'num_credit_lines': num_credit_lines,
        'utilization_rate': utilization_rate,
        'home_ownership': home_ownership
    }


def calculate_default_probability(profile: dict) -> float:
    """
    Calculate probability of default based on profile.
    This simulates the relationship between features and default.
    """
    prob = 0.10  # Base probability
    
    # Age effect (younger = higher risk)
    if profile['age'] < 25:
        prob += 0.08
    elif profile['age'] < 30:
        prob += 0.04
    elif profile['age'] > 55:
        prob += 0.02
    
    # Income effect (lower income = higher risk)
    if profile['income'] < 30000:
        prob += 0.10
    elif profile['income'] < 50000:
        prob += 0.05
    elif profile['income'] > 100000:
        prob -= 0.03
    
    # Employment effect
    if profile['employment_length'] < 1:
        prob += 0.08
    elif profile['employment_length'] < 3:
        prob += 0.03
    elif profile['employment_length'] > 7:
        prob -= 0.02
    
    # DTI effect
    if profile['debt_to_income'] > 0.5:
        prob += 0.12
    elif profile['debt_to_income'] > 0.4:
        prob += 0.06
    elif profile['debt_to_income'] < 0.2:
        prob -= 0.02
    
    # Credit history effect
    if profile['credit_history_length'] < 2:
        prob += 0.08
    elif profile['credit_history_length'] > 10:
        prob -= 0.03
    
    # Utilization effect
    if profile['utilization_rate'] > 80:
        prob += 0.10
    elif profile['utilization_rate'] > 50:
        prob += 0.04
    elif profile['utilization_rate'] < 20:
        prob -= 0.02
    
    # Loan amount to income ratio
    lti = profile['loan_amount'] / profile['income']
    if lti > 1:
        prob += 0.08
    elif lti > 0.5:
        prob += 0.03
    
    # Home ownership effect
    if profile['home_ownership'] == 'OWN':
        prob -= 0.03
    elif profile['home_ownership'] == 'RENT':
        prob += 0.02
    
    return np.clip(prob, 0.01, 0.90)


def generate_applications(n: int = N_APPLICATIONS):
    """Generate loan application dataset."""
    print(f"🔄 Generating {n:,} loan applications...")
    
    applications = []
    
    for i in range(n):
        profile = generate_applicant_profile()
        
        # Calculate default probability
        default_prob = calculate_default_probability(profile)
        
        # Generate number of delinquencies (correlated with default risk)
        if default_prob > 0.3:
            num_delinquencies = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        elif default_prob > 0.15:
            num_delinquencies = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
        else:
            num_delinquencies = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
        
        # Delinquencies increase default probability
        default_prob += num_delinquencies * 0.08
        default_prob = np.clip(default_prob, 0.01, 0.95)
        
        # Determine default
        default = int(np.random.random() < default_prob)
        
        application = {
            'application_id': f'APP{i+1:07d}',
            **profile,
            'num_delinquencies': num_delinquencies,
            'default': default
        }
        
        applications.append(application)
    
    df = pd.DataFrame(applications)
    
    # Adjust to target default rate
    actual_rate = df['default'].mean()
    print(f"   Initial default rate: {actual_rate*100:.1f}%")
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total applications: {len(df):,}")
    print(f"   Default rate: {df['default'].mean()*100:.1f}%")
    print(f"   Avg income: ${df['income'].mean():,.0f}")
    print(f"   Avg loan amount: ${df['loan_amount'].mean():,.0f}")
    print(f"   Avg DTI: {df['debt_to_income'].mean()*100:.1f}%")
    
    return df


def main():
    """Generate and save data."""
    df = generate_applications()
    
    # Save files
    project_root = Path(__file__).parent.parent
    
    for dir_name in ['raw', 'sample']:
        data_dir = project_root / 'data' / dir_name
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_dir / 'applications.csv', index=False)
    
    print(f"\n💾 Saved data to data/raw/ and data/sample/")
    print("\n✅ Data generation complete!")
    
    return df


if __name__ == '__main__':
    main()
