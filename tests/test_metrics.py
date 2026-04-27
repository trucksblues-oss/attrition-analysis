import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
        "attrition": ["Yes", "No", "Yes", "Yes", "No", "No"],
        "overtime": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "monthly_income": [4000, 6000, 3500, 5000, 7000, 8000],
        "job_satisfaction": [1, 3, 2, 1, 4, 3],
    })


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent(sample_df):
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_all_staying():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leaving():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_rounds_to_two_decimal_places():
    df = pd.DataFrame({"employee_id": [1, 2, 3], "attrition": ["Yes", "No", "No"]})
    assert attrition_rate(df) == 33.33


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = dict(zip(result["department"], result["attrition_rate"]))
    assert rates["HR"] == 100.0    # 2 of 2 HR employees left
    assert rates["Sales"] == 50.0  # 1 of 2 Sales employees left
    assert rates["IT"] == 0.0      # 0 of 2 IT employees left


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = dict(zip(result["overtime"], result["attrition_rate"]))
    assert rates["Yes"] == 66.67  # 2 of 3 overtime workers left
    assert rates["No"] == 33.33   # 1 of 3 non-overtime workers left


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    income = dict(zip(result["attrition"], result["avg_monthly_income"]))
    assert income["Yes"] == 4166.67  # (4000 + 3500 + 5000) / 3
    assert income["No"] == 7000.0    # (6000 + 7000 + 8000) / 3


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rates(sample_df):
    result = satisfaction_summary(sample_df)
    rates = dict(zip(result["job_satisfaction"], result["attrition_rate"]))
    assert rates[1] == 100.0  # both sat=1 employees left
    assert rates[2] == 100.0  # the one sat=2 employee left
    assert rates[3] == 0.0    # neither sat=3 employee left
    assert rates[4] == 0.0    # the one sat=4 employee stayed


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)
