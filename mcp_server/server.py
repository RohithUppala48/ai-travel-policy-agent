"""
MCP Server for Travel & Policy Assistant

Exposes company travel and employee-operation tools through MCP.

MCP Tools:
1. search_company_policy
2. get_employee_eligibility
3. validate_employee_trip
"""

from mcp.server import MCPServer

from src.tools import (
    policy_search,
    check_employee_eligibility,
    validate_trip,
)


# ============================================================
# Create MCP Server
# ============================================================

mcp = MCPServer("Travel Policy Assistant")


# ============================================================
# Tool 1: Search Company Policy
# ============================================================

@mcp.tool()
def search_company_policy(query: str) -> dict:
    """
    Search company travel, expense, airport, approval,
    and cancellation policies.

    Args:
        query: Policy-related question or search query.

    Returns:
        Dictionary containing relevant policy information
        and source metadata.
    """

    return policy_search.invoke({
        "query": query
    })


# ============================================================
# Tool 2: Check Employee Eligibility
# ============================================================

@mcp.tool()
def get_employee_eligibility(employee_id: str) -> dict:
    """
    Check whether an employee is eligible for company
    travel benefits.

    Args:
        employee_id: Employee ID such as EMP001.

    Returns:
        Employee eligibility information.
    """

    return check_employee_eligibility.invoke({
        "employee_id": employee_id
    })


# ============================================================
# Tool 3: Validate Employee Trip
# ============================================================

@mcp.tool()
def validate_employee_trip(
    employee_id: str,
    trip_type: str,
    amount: float,
    time: str = ""
) -> dict:
    """
    Validate an employee trip against company travel rules.

    Args:
        employee_id: Employee ID such as EMP001.
        trip_type: Type of trip, for example airport.
        amount: Trip amount.
        time: Optional trip time such as 22:30.

    Returns:
        Trip validation result including eligibility,
        policy limit, approval status, and any excess amount.
    """

    return validate_trip.invoke({
        "employee_id": employee_id,
        "trip_type": trip_type,
        "amount": amount,
        "time": time
    })


# ============================================================
# Server Entry Point
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Travel Policy MCP Server")
    print("=" * 60)

    print("Available MCP tools:")
    print("1. search_company_policy")
    print("2. get_employee_eligibility")
    print("3. validate_employee_trip")

    print("=" * 60)
    print("Starting MCP server...")
    print("=" * 60)

    mcp.run()