"""
src/tools.py

Tool definitions for the Travel & Policy Assistant.

Tools:
1. policy_search
2. check_employee_eligibility
3. validate_trip
4. calculate_reimbursement

Important architecture rule
----------------------------
Tool selection is performed by the LLM agent.

This file does NOT perform keyword-based intent classification.

There is no logic such as:
    if "policy" in query:
        ...
    if "eligibility" in query:
        ...

Each tool contains detailed semantic descriptions so the LLM can
decide which tool is appropriate for the user's request.
"""

import os
import re
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from src.rag import search_policy


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

EMPLOYEE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "employees.csv"
)


# ============================================================
# LOAD EMPLOYEE DATA
# ============================================================

def load_employee_data() -> pd.DataFrame:
    """
    Load employee information from employees.csv.

    Required columns:
        employee_id
        country
        employee_type
        department
        eligibility_status

    Returns
    -------
    pd.DataFrame
        Employee dataset.

    Raises
    ------
    FileNotFoundError
        If employees.csv does not exist.

    ValueError
        If required columns are missing.
    """

    if not os.path.exists(EMPLOYEE_FILE):
        raise FileNotFoundError(
            f"Employee dataset not found: {EMPLOYEE_FILE}"
        )

    df = pd.read_csv(EMPLOYEE_FILE)

    required_columns = {
        "employee_id",
        "country",
        "employee_type",
        "department",
        "eligibility_status",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Employee dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return df


# ============================================================
# TOOL 1 — POLICY SEARCH
# ============================================================
@tool
def policy_search(query: str) -> dict:
    """
    Search company travel-policy documents using semantic retrieval.

    USE THIS TOOL WHEN:
    - The user is asking about company policy.
    - The user wants a policy rule, limit, requirement, eligibility rule,
      reimbursement rule, cancellation rule, approval rule, etc.
    - The answer should come from the company policy documents.

    DO NOT USE THIS TOOL FOR:
    - Checking whether a specific employee exists or is eligible.
    - Validating a specific employee's trip.
    - Calculating reimbursement amounts.
    """

    if not query or not query.strip():
        return {
            "success": False,
            "query": query or "",
            "results": [],
            "error": "Policy search query cannot be empty."
        }

    try:
        retrieved = search_policy(
            query=query.strip(),
            top_k=3
        )

        # search_policy() currently returns:
        # {
        #     "success": True,
        #     "results": [...]
        # }
        #
        # Extract the actual list so the tool contract remains:
        # {
        #     "success": True,
        #     "query": "...",
        #     "results": [...]
        # }

        if isinstance(retrieved, dict):
            results = retrieved.get("results", [])
            rag_success = retrieved.get("success", True)

            if not rag_success:
                return {
                    "success": False,
                    "query": query.strip(),
                    "results": results,
                    "error": retrieved.get(
                        "error",
                        "Policy retrieval failed."
                    )
                }

        elif isinstance(retrieved, list):
            # Defensive handling in case rag.py is later changed
            # to return the list directly.
            results = retrieved

        else:
            results = []

        return {
            "success": True,
            "query": query.strip(),
            "results": results
        }

    except Exception as e:
        return {
            "success": False,
            "query": query.strip(),
            "results": [],
            "error": str(e)
        }

# ============================================================
# TOOL 2 — EMPLOYEE ELIGIBILITY
# ============================================================

@tool
def check_employee_eligibility(employee_id: str) -> dict:
    """
    Check the travel eligibility of ONE specific employee.

    WHEN TO USE THIS TOOL
    ---------------------
    Use this tool when the user provides a specific employee ID and
    asks whether that employee is eligible for company travel or
    travel benefits.

    Examples:
        - Is EMP001 eligible for travel?
        - Can EMP004 travel?
        - Is employee EMP003 eligible?
        - Check the travel eligibility of EMP001.
        - Can this employee claim travel benefits? EMP001

    IMPORTANT TOOL-SELECTION RULE
    -----------------------------
    This tool checks the actual employee record.

    If the user is asking about the GENERAL COMPANY ELIGIBILITY
    POLICY, without asking about a specific employee, use
    policy_search instead.

    If the user asks to validate a specific trip, use validate_trip
    when the required trip information is available. Do not call
    this tool alone when the user is asking for a complete trip
    validation.

    Parameters
    ----------
    employee_id : str
        Specific employee ID, such as EMP001.

    Returns
    -------
    dict
        Employee details and eligibility status.
    """

    normalized_employee_id = None

    try:

        # --------------------------------------------------------
        # Validate employee ID
        # --------------------------------------------------------

        if employee_id is None:
            return {
                "success": True,
                "found": False,
                "employee_id": None,
                "eligible": False,
                "error": "Employee ID cannot be empty."
            }

        normalized_employee_id = (
            str(employee_id)
            .strip()
            .upper()
        )

        if not normalized_employee_id:
            return {
                "success": True,
                "found": False,
                "employee_id": "",
                "eligible": False,
                "error": "Employee ID cannot be empty."
            }

        # --------------------------------------------------------
        # Load employee data
        # --------------------------------------------------------

        df = load_employee_data()

        # --------------------------------------------------------
        # Find employee
        # --------------------------------------------------------

        employee = df[
            df["employee_id"]
            .astype(str)
            .str.strip()
            .str.upper()
            == normalized_employee_id
        ]

        # --------------------------------------------------------
        # Employee not found
        # --------------------------------------------------------

        if employee.empty:

            return {
                "success": True,
                "found": False,
                "employee_id": normalized_employee_id,
                "eligible": False,
                "error": (
                    f"Employee {normalized_employee_id} "
                    "was not found."
                )
            }

        # --------------------------------------------------------
        # Extract employee details
        # --------------------------------------------------------

        row = employee.iloc[0]

        eligibility_status = str(
            row["eligibility_status"]
        ).strip()

        eligible = (
            eligibility_status.lower() == "eligible"
        )

        return {
            "success": True,
            "found": True,
            "employee_id": normalized_employee_id,
            "country": str(
                row["country"]
            ).strip(),
            "employee_type": str(
                row["employee_type"]
            ).strip(),
            "department": str(
                row["department"]
            ).strip(),
            "eligibility_status": eligibility_status,
            "eligible": eligible,
        }

    except Exception as e:

        return {
            "success": False,
            "found": False,
            "employee_id": normalized_employee_id,
            "eligible": False,
            "error": (
                "Employee eligibility check failed: "
                f"{str(e)}"
            )
        }


# ============================================================
# HELPER — NORMALIZE TRIP TYPE
# ============================================================

def normalize_trip_type(trip_type: str) -> str:
    """
    Normalize common representations of supported trip types.

    This is NOT used for agent intent classification.

    It only normalizes the trip type after the agent has already
    selected validate_trip.
    """

    if trip_type is None:
        return ""

    trip_type = str(trip_type).strip().lower()

    if not trip_type:
        return ""

    aliases = {
        "airport trip": "airport",
        "airport ride": "airport",
        "airport travel": "airport",
        "airport": "airport",
    }

    return aliases.get(
        trip_type,
        trip_type
    )


# ============================================================
# HELPER — NORMALIZE TIME
# ============================================================

def normalize_time(
    time: Optional[str]
) -> Optional[str]:
    """
    Normalize a supplied time into HH:MM format.

    Supported examples:
        22:30
        10:30 PM
        10 PM
        12 AM
        12 PM

    Returns
    -------
    str or None
        Normalized time in HH:MM format.

        Returns None if:
            - no time is supplied
            - the format is invalid
    """

    if time is None:
        return None

    time = str(time).strip()

    if not time:
        return None

    # --------------------------------------------------------
    # Already HH:MM
    # --------------------------------------------------------

    match = re.fullmatch(
        r"([01]?\d|2[0-3]):([0-5]\d)",
        time
    )

    if match:

        hour = int(match.group(1))
        minute = int(match.group(2))

        return f"{hour:02d}:{minute:02d}"

    # --------------------------------------------------------
    # 10 PM / 10:30 PM
    # --------------------------------------------------------

    match = re.fullmatch(
        r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)",
        time,
        re.IGNORECASE
    )

    if match:

        hour = int(match.group(1))

        minute = int(
            match.group(2) or 0
        )

        period = match.group(3).upper()

        if hour < 1 or hour > 12:
            return None

        if minute < 0 or minute > 59:
            return None

        if period == "AM":

            hour = (
                0
                if hour == 12
                else hour
            )

        else:

            hour = (
                12
                if hour == 12
                else hour + 12
            )

        return f"{hour:02d}:{minute:02d}"

    return None


# ============================================================
# HELPER — CHECK LATE NIGHT
# ============================================================

def is_late_night(
    time: Optional[str]
) -> bool:
    """
    Determine whether a supplied time falls within the
    10:00 PM–6:00 AM late-night window.

    Returns False when no valid time is supplied.
    """

    normalized = normalize_time(time)

    if normalized is None:
        return False

    hour, minute = map(
        int,
        normalized.split(":")
    )

    total_minutes = (
        hour * 60 + minute
    )

    return (
        total_minutes >= 22 * 60
        or
        total_minutes < 6 * 60
    )


# ============================================================
# TOOL 3 — VALIDATE TRIP
# ============================================================

@tool
def validate_trip(
    employee_id: str,
    trip_type: str,
    amount: float,
    time: str = ""
) -> dict:
    """
    Validate ONE SPECIFIC employee's travel request.

    WHEN TO USE THIS TOOL
    ---------------------
    Use this tool only when the user is asking to validate a
    specific trip for a specific employee.

    The tool requires:
        1. employee_id
        2. trip_type
        3. amount

    The time parameter is optional.

    Examples:
        - Can EMP001 claim INR 2500 for an airport trip?
        - Is EMP001's INR 1500 airport trip approved?
        - Validate EMP003's USD 100 airport trip.
        - Can EMP001 take an airport trip for INR 1800?
        - Is EMP004 eligible for an airport trip costing INR 1500?

    IMPORTANT
    ---------
    DO NOT use this tool for GENERAL POLICY QUESTIONS.

    For example, do NOT call this tool for:
        - What information is required to validate a trip?
        - What is the airport policy?
        - What is the India airport limit?
        - What are the travel approval rules?
        - What are the company travel policies?

    Those are policy questions and should use policy_search.

    DO NOT call this tool when required information is missing.

    For example, if the user asks:
        "Can I claim an airport trip?"

    and no employee ID or amount is available from the conversation,
    do not invent those values and do not call this tool.

    Instead, ask the user for the missing information.

    FOLLOW-UP QUESTIONS
    -------------------
    Conversation context may supply previously mentioned values.

    Example:
        User: Can EMP001 claim INR 2500 for an airport trip?
        User: What if it is INR 1500?

    The second question can reuse:
        employee_id = EMP001
        trip_type = airport

    and only update the amount.

    TIME
    ----
    Time is optional.

    Only use the time value when:
        - the user provides it, or
        - it is already available from conversation context.

    Do not invent a time.

    Parameters
    ----------
    employee_id : str
        Specific employee ID.

    trip_type : str
        Specific trip type, such as "airport".

    amount : float
        Requested trip amount.

    time : str, optional
        Travel time such as "22:30" or "10:30 PM".

    Returns
    -------
    dict
        Validation result including eligibility, applicable limit,
        decision, excess amount, and late-night information.
    """

    normalized_employee_id = None

    try:

        # ========================================================
        # VALIDATE EMPLOYEE ID
        # ========================================================

        if employee_id is None:

            return {
                "success": False,
                "found": False,
                "error": "Employee ID cannot be empty."
            }

        normalized_employee_id = (
            str(employee_id)
            .strip()
            .upper()
        )

        if not normalized_employee_id:

            return {
                "success": False,
                "found": False,
                "employee_id": "",
                "error": "Employee ID cannot be empty."
            }

        # ========================================================
        # VALIDATE TRIP TYPE
        # ========================================================

        trip_type = normalize_trip_type(
            trip_type
        )

        if not trip_type:

            return {
                "success": False,
                "employee_id": normalized_employee_id,
                "error": (
                    "Trip type cannot be empty."
                )
            }

        # ========================================================
        # VALIDATE AMOUNT
        # ========================================================

        try:

            if amount is None:
                raise ValueError

            amount = float(amount)

        except (TypeError, ValueError):

            return {
                "success": False,
                "employee_id": normalized_employee_id,
                "trip_type": trip_type,
                "error": (
                    "Trip amount must be a valid number."
                )
            }

        if amount < 0:

            return {
                "success": False,
                "employee_id": normalized_employee_id,
                "trip_type": trip_type,
                "amount": amount,
                "error": (
                    "Trip amount cannot be negative."
                )
            }

        # ========================================================
        # CHECK EMPLOYEE ELIGIBILITY
        # ========================================================

        eligibility = (
            check_employee_eligibility.invoke({
                "employee_id": normalized_employee_id
            })
        )

        # --------------------------------------------------------
        # Employee lookup failed because of a real system/data
        # problem.
        # --------------------------------------------------------

        if not eligibility.get("success"):

            return {
                "success": False,
                "found": eligibility.get(
                    "found",
                    False
                ),
                "employee_id": normalized_employee_id,
                "trip_type": trip_type,
                "amount": amount,
                "error": eligibility.get(
                    "error",
                    "Unable to check employee eligibility."
                )
            }

        # --------------------------------------------------------
        # Employee does not exist
        # --------------------------------------------------------

        if not eligibility.get("found"):

            return {
                "success": True,
                "found": False,
                "employee_id": normalized_employee_id,
                "trip_type": trip_type,
                "amount": amount,
                "eligible": False,
                "decision": "Not Eligible",
                "reason": (
                    f"Employee {normalized_employee_id} "
                    "was not found in the employee records."
                )
            }
        # ========================================================
        # INELIGIBLE EMPLOYEE
        # ========================================================

        if not eligibility.get("eligible"):

            return {
                "success": True,
                "found": True,
                "employee_id": normalized_employee_id,
                "country": eligibility.get(
                    "country"
                ),
                "employee_type": eligibility.get(
                    "employee_type"
                ),
                "department": eligibility.get(
                    "department"
                ),
                "trip_type": trip_type,
                "amount": amount,
                "eligible": False,
                "decision": "Not Eligible",
                "reason": (
                    f"{normalized_employee_id} is not eligible "
                    "for company travel."
                )
            }

        # ========================================================
        # EMPLOYEE COUNTRY
        # ========================================================

        country = str(
            eligibility.get(
                "country",
                ""
            )
        ).strip().lower()

        # ========================================================
        # AIRPORT POLICY
        # ========================================================

        if trip_type == "airport":

            # ----------------------------------------------------
            # INDIA
            # ----------------------------------------------------

            if country == "india":

                policy_limit = 2000.0
                currency = "INR"

            # ----------------------------------------------------
            # UNITED STATES
            # ----------------------------------------------------

            elif country in {
                "united states",
                "usa",
                "us",
                "united states of america"
            }:

                policy_limit = 75.0
                currency = "USD"

            # ----------------------------------------------------
            # UNSUPPORTED COUNTRY
            # ----------------------------------------------------

            else:

                return {
                    "success": False,
                    "found": True,
                    "employee_id": normalized_employee_id,
                    "country": eligibility.get(
                        "country"
                    ),
                    "trip_type": trip_type,
                    "amount": amount,
                    "error": (
                        "No airport spending limit is "
                        "defined for this employee's "
                        "country."
                    )
                }

            # ====================================================
            # APPROVAL DECISION
            # ====================================================

            if amount <= policy_limit:

                decision = "Approved"

                review_amount = 0.0

                reason = (
                    f"Trip amount is within the standard "
                    f"{currency} "
                    f"{policy_limit:,.0f} limit."
                )

            else:

                decision = "Needs Approval"

                review_amount = (
                    amount - policy_limit
                )

                reason = (
                    f"Trip amount exceeds the standard "
                    f"{currency} "
                    f"{policy_limit:,.0f} limit by "
                    f"{currency} "
                    f"{review_amount:,.0f}. "
                    "Additional approval is required."
                )

            # ====================================================
            # TIME INFORMATION
            # ====================================================

            normalized_time = normalize_time(
                time
            )

            late_night = is_late_night(
                time
            )

            # ----------------------------------------------------
            # Late-night note
            # ----------------------------------------------------

            if late_night:

                late_night_note = (
                    "The trip falls within the "
                    "10:00 PM–6:00 AM late-night "
                    "window. Late-night airport "
                    "trips are allowed when connected "
                    "to approved business travel. "
                    "Normal spending and approval "
                    "limits still apply."
                )

            else:

                late_night_note = None

            # ====================================================
            # FINAL RESULT
            # ====================================================

            return {
                "success": True,
                "found": True,
                "employee_id": normalized_employee_id,
                "country": eligibility.get(
                    "country"
                ),
                "employee_type": eligibility.get(
                    "employee_type"
                ),
                "department": eligibility.get(
                    "department"
                ),
                "eligible": True,
                "trip_type": trip_type,
                "amount": amount,
                "currency": currency,
                "policy_limit": policy_limit,
                "excess_amount": max(
                    0.0,
                    amount - policy_limit
                ),
                "decision": decision,
                "time": normalized_time,
                "late_night": late_night,
                "late_night_note": (
                    late_night_note
                ),
                "reason": reason,
            }

        # ========================================================
        # UNSUPPORTED TRIP TYPE
        # ========================================================

        return {
            "success": False,
            "found": True,
            "employee_id": normalized_employee_id,
            "trip_type": trip_type,
            "amount": amount,
            "error": (
                f"Trip validation for "
                f"'{trip_type}' is not currently "
                "supported."
            )
        }

    except Exception as e:

        return {
            "success": False,
            "employee_id": normalized_employee_id,
            "error": (
                f"Trip validation failed: {str(e)}"
            )
        }


# ============================================================
# TOOL 4 — CALCULATE REIMBURSEMENT
# ============================================================

@tool
def calculate_reimbursement(
    trip_amount: float,
    policy_limit: float
) -> dict:
    """
    Calculate the reimbursable amount for a specific trip.

    WHEN TO USE THIS TOOL
    ---------------------
    Use this tool when the user wants a numerical reimbursement
    calculation AND both of the following are known:

        1. trip amount
        2. applicable policy limit

    Examples:
        - How much can I be reimbursed for INR 2500 when the limit
          is INR 2000?
        - Calculate reimbursement for 1500 with a limit of 2000.
        - The trip cost 2500 and the policy limit is 2000. Calculate
          the reimbursable amount.

    IMPORTANT TOOL-SELECTION RULE
    -----------------------------
    Do NOT invent the policy limit.

    If the user asks:
        "How much can I claim for an airport trip in India?"

    and the policy limit has not yet been retrieved or is not
    available from conversation context, first use policy_search
    to retrieve the applicable policy limit.

    Then use this tool for the arithmetic calculation if a
    calculation is actually required.

    Do NOT use this tool for general questions about reimbursement
    policy. Use policy_search for those.

    Do NOT use this tool for complete employee trip validation.
    Use validate_trip for that.

    Parameters
    ----------
    trip_amount : float
        Actual amount spent on the trip.

    policy_limit : float
        Applicable policy spending/reimbursement limit.

    Returns
    -------
    dict
        Reimbursement calculation containing:
            reimbursable_amount
            review_amount
            decision
    """

    try:

        # ========================================================
        # VALIDATE TRIP AMOUNT
        # ========================================================

        try:

            if trip_amount is None:
                raise ValueError

            trip_amount = float(
                trip_amount
            )

        except (TypeError, ValueError):

            return {
                "success": False,
                "error": (
                    "Trip amount must be a "
                    "valid number."
                )
            }

        if trip_amount < 0:

            return {
                "success": False,
                "error": (
                    "Trip amount cannot be "
                    "negative."
                )
            }

        # ========================================================
        # VALIDATE POLICY LIMIT
        # ========================================================

        try:

            if policy_limit is None:
                raise ValueError

            policy_limit = float(
                policy_limit
            )

        except (TypeError, ValueError):

            return {
                "success": False,
                "error": (
                    "Policy limit must be "
                    "a valid number."
                )
            }

        if policy_limit < 0:

            return {
                "success": False,
                "error": (
                    "Policy limit cannot be "
                    "negative."
                )
            }

        # ========================================================
        # CALCULATE REIMBURSEMENT
        # ========================================================

        reimbursable_amount = min(
            trip_amount,
            policy_limit
        )

        review_amount = max(
            0.0,
            trip_amount - policy_limit
        )

        if review_amount > 0:

            decision = "Needs Review"

        else:

            decision = "Within Policy"

        # ========================================================
        # RETURN RESULT
        # ========================================================

        return {
            "success": True,
            "trip_amount": trip_amount,
            "policy_limit": policy_limit,
            "reimbursable_amount": (
                reimbursable_amount
            ),
            "review_amount": review_amount,
            "decision": decision,
        }

    except Exception as e:

        return {
            "success": False,
            "error": (
                "Reimbursement calculation "
                f"failed: {str(e)}"
            )
        }


# ============================================================
# ALL TOOLS
# ============================================================

TOOLS = [
    policy_search,
    check_employee_eligibility,
    validate_trip,
    calculate_reimbursement,
]


# ============================================================
# MANUAL TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TRAVEL POLICY TOOLS TEST")
    print("=" * 70)

    # ========================================================
    # TEST 1 — POLICY SEARCH
    # ========================================================

    print("\n1. POLICY SEARCH")
    print("-" * 70)

    result = policy_search.invoke({
        "query": (
            "What is the airport reimbursement "
            "limit in India?"
        )
    })

    print(result)

    # ========================================================
    # TEST 2 — EMPLOYEE ELIGIBILITY
    # ========================================================

    print("\n2. EMPLOYEE ELIGIBILITY")
    print("-" * 70)

    result = check_employee_eligibility.invoke({
        "employee_id": "EMP001"
    })

    print(result)

    # ========================================================
    # TEST 3 — INDIA AIRPORT WITHIN LIMIT
    # ========================================================

    print("\n3. EMP001 — INDIA AIRPORT — INR 1500")
    print("-" * 70)

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": 1500,
    })

    print(result)

    # ========================================================
    # TEST 4 — INDIA AIRPORT ABOVE LIMIT
    # ========================================================

    print("\n4. EMP001 — INDIA AIRPORT — INR 2500")
    print("-" * 70)

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": 2500,
    })

    print(result)

    # ========================================================
    # TEST 5 — INELIGIBLE EMPLOYEE
    # ========================================================

    print("\n5. EMP004 — INELIGIBLE EMPLOYEE")
    print("-" * 70)

    result = validate_trip.invoke({
        "employee_id": "EMP004",
        "trip_type": "airport",
        "amount": 1500,
    })

    print(result)

    # ========================================================
    # TEST 6 — US AIRPORT
    # ========================================================

    print("\n6. EMP003 — US AIRPORT — USD 100")
    print("-" * 70)

    result = validate_trip.invoke({
        "employee_id": "EMP003",
        "trip_type": "airport",
        "amount": 100,
    })

    print(result)

    # ========================================================
    # TEST 7 — LATE-NIGHT AIRPORT
    # ========================================================

    print("\n7. EMP001 — LATE NIGHT AIRPORT — INR 1500")
    print("-" * 70)

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": 1500,
        "time": "22:30",
    })

    print(result)

    # ========================================================
    # TEST 8 — REIMBURSEMENT
    # ========================================================

    print("\n8. REIMBURSEMENT — INR 2500 / INR 2000")
    print("-" * 70)

    result = calculate_reimbursement.invoke({
        "trip_amount": 2500,
        "policy_limit": 2000,
    })

    print(result)

    # ========================================================
    # TEST 9 — UNKNOWN EMPLOYEE
    # ========================================================

    print("\n9. UNKNOWN EMPLOYEE")
    print("-" * 70)

    result = check_employee_eligibility.invoke({
        "employee_id": "EMP9999"
    })

    print(result)

    # ========================================================
    # TEST 10 — EMPTY POLICY QUERY
    # ========================================================

    print("\n10. EMPTY POLICY QUERY")
    print("-" * 70)

    result = policy_search.invoke({
        "query": ""
    })

    print(result)

    # ========================================================
    # TEST 11 — INVALID TRIP AMOUNT
    # ========================================================

    print("\n11. INVALID TRIP AMOUNT")
    print("-" * 70)

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": "invalid",
    })

    print(result)

    # ========================================================
    # TEST 12 — NEGATIVE TRIP AMOUNT
    # ========================================================

    print("\n12. NEGATIVE TRIP AMOUNT")
    print("-" * 70)

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": -100,
    })

    print(result)

    # ========================================================
    # TEST 13 — INVALID TIME
    # ========================================================

    print("\n13. INVALID TIME")
    print("-" * 70)

    result = validate_trip.invoke({
        "employee_id": "EMP001",
        "trip_type": "airport",
        "amount": 1500,
        "time": "invalid",
    })

    print(result)

    # ========================================================
    # TEST 14 — REIMBURSEMENT WITHIN POLICY
    # ========================================================

    print("\n14. REIMBURSEMENT — INR 1500 / INR 2000")
    print("-" * 70)

    result = calculate_reimbursement.invoke({
        "trip_amount": 1500,
        "policy_limit": 2000,
    })

    print(result)

    # ========================================================
    # TEST 15 — INVALID REIMBURSEMENT INPUT
    # ========================================================

    print("\n15. INVALID REIMBURSEMENT INPUT")
    print("-" * 70)

    result = calculate_reimbursement.invoke({
        "trip_amount": "invalid",
        "policy_limit": 2000,
    })

    print(result)

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("TOOLS TEST COMPLETED")
    print("=" * 70)