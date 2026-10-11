import pytest

from src.tools import (
    policy_search,
    check_employee_eligibility,
    validate_trip,
    calculate_reimbursement,
)


# ============================================================
# 1. POLICY SEARCH TESTS
# ============================================================

def test_policy_search_india_airport_policy():
    """Policy search should retrieve relevant India airport policy."""

    result = policy_search.invoke({
        "query": "What is the airport reimbursement limit in India?"
    })

    assert result["success"] is True
    assert len(result["results"]) > 0

    # At least one retrieved document should be relevant
    sources = [
        item["source"]
        for item in result["results"]
    ]

    assert any(
        "airport_policy" in source.lower()
        for source in sources
    )


def test_policy_search_us_policy():
    """Policy search should retrieve relevant US travel policy."""

    result = policy_search.invoke({
        "query": "What is the airport reimbursement limit in the US?"
    })

    assert result["success"] is True
    assert len(result["results"]) > 0


def test_policy_search_returns_expected_structure():
    """Policy search should return the expected response structure."""

    result = policy_search.invoke({
        "query": "What is the travel policy?"
    })

    assert "success" in result
    assert "query" in result
    assert "results" in result

    assert isinstance(result["success"], bool)
    assert isinstance(result["query"], str)
    assert isinstance(result["results"], list)


# ============================================================
# 2. EMPLOYEE ELIGIBILITY TESTS
# ============================================================

def test_eligible_employee():
    """EMP001 should be eligible for travel."""

    result = check_employee_eligibility.invoke({
        "employee_id": "EMP001"
    })

    assert result["success"] is True
    assert result["found"] is True
    assert result["eligible"] is True
    assert result["employee_id"] == "EMP001"


def test_ineligible_employee():
    """EMP004 should not be eligible for travel."""

    result = check_employee_eligibility.invoke({
        "employee_id": "EMP004"
    })

    assert result["success"] is True
    assert result["found"] is True
    assert result["eligible"] is False
    assert result["employee_id"] == "EMP004"


def test_unknown_employee():
    """Unknown employee IDs should be handled gracefully."""

    result = check_employee_eligibility.invoke({
        "employee_id": "EMP9999"
    })

    assert result["success"] is True
    assert result["found"] is False
    assert result["eligible"] is False


def test_employee_id_case_normalization():
    """Employee IDs should work regardless of letter case."""

    result = check_employee_eligibility.invoke({
        "employee_id": "emp001"
    })

    assert result["success"] is True
    assert result["found"] is True
    assert result["eligible"] is True


# ============================================================
# 3. TRIP VALIDATION TESTS
# ============================================================

def test_trip_within_india_airport_limit():
    """An eligible employee claiming within the India limit should be approved."""

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": 1500,
        "time": "10:00"
    })

    assert result["success"] is True
    assert result["eligible"] is True
    assert result["decision"] == "Approved"
    assert result["amount"] == 1500
    assert result["policy_limit"] == 2000


def test_trip_above_india_airport_limit():
    """Amount above the India airport limit should require approval."""

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": 2500,
        "time": "10:00"
    })

    assert result["success"] is True
    assert result["eligible"] is True
    assert result["decision"] == "Needs Approval"
    assert result["amount"] == 2500
    assert result["policy_limit"] == 2000


def test_ineligible_employee_trip():
    """An ineligible employee should not be allowed to claim travel."""

    result = validate_trip.invoke({
        "employee_id": "EMP004",
        "trip_type": "airport",
        "amount": 1500,
        "time": "10:00"
    })

    assert result["success"] is True
    assert result["eligible"] is False
    assert result["decision"] == "Not Eligible"


def test_late_night_trip():
    """Late-night travel should be detected."""

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": 1500,
        "time": "22:30"
    })

    assert result["success"] is True
    assert result["late_night"] is True


def test_non_late_night_trip():
    """Normal daytime travel should not be marked as late-night."""

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": 1500,
        "time": "14:00"
    })

    assert result["success"] is True
    assert result["late_night"] is False


def test_us_airport_limit():
    """US airport travel should use the US policy limit."""

    result = validate_trip.invoke({
        "employee_id": "EMP003",
        "trip_type": "airport",
        "amount": 75,
        "time": "10:00"
    })

    assert result["success"] is True
    assert result["eligible"] is True
    assert result["policy_limit"] == 75
    assert result["decision"] == "Approved"


def test_unknown_employee_trip():
    """Trip validation should gracefully handle an unknown employee."""

    result = validate_trip.invoke({
        "employee_id": "EMP9999",
        "trip_type": "airport",
        "amount": 1500,
        "time": "10:00"
    })

    assert result["success"] is True
    assert result["eligible"] is False
    assert result["decision"] == "Not Eligible"


# ============================================================
# 4. REIMBURSEMENT CALCULATION TESTS
# ============================================================

def test_reimbursement_below_limit():
    """Full amount should be reimbursable when below the policy limit."""

    result = calculate_reimbursement.invoke({
        "trip_amount": 1500,
        "policy_limit": 2000
    })

    assert result["success"] is True
    assert result["reimbursable_amount"] == 1500
    assert result["review_amount"] == 0


def test_reimbursement_equal_to_limit():
    """Full amount should be reimbursable when exactly at the limit."""

    result = calculate_reimbursement.invoke({
        "trip_amount": 2000,
        "policy_limit": 2000
    })

    assert result["success"] is True
    assert result["reimbursable_amount"] == 2000
    assert result["review_amount"] == 0


def test_reimbursement_above_limit():
    """Amount above the limit should be split into reimbursable and review amounts."""

    result = calculate_reimbursement.invoke({
        "trip_amount": 2500,
        "policy_limit": 2000
    })

    assert result["success"] is True
    assert result["reimbursable_amount"] == 2000
    assert result["review_amount"] == 500


# ============================================================
# 5. ERROR / EDGE CASE TESTS
# ============================================================

def test_empty_employee_id():
    """Empty employee ID should not return an eligible employee."""

    result = check_employee_eligibility.invoke({
        "employee_id": ""
    })

    assert result["success"] is True
    assert result["found"] is False
    assert result["eligible"] is False


def test_zero_reimbursement():
    """Zero trip amount should be handled safely."""

    result = calculate_reimbursement.invoke({
        "trip_amount": 0,
        "policy_limit": 2000
    })

    assert result["success"] is True
    assert result["reimbursable_amount"] == 0
    assert result["review_amount"] == 0