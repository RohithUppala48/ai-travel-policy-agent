"""
End-to-end MCP client test.

Starts the MCP server through stdio, discovers its tools,
and calls each tool to verify the MCP integration.
"""

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server.server"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize MCP connection
            await session.initialize()

            print("\n" + "=" * 60)
            print("MCP CONNECTION SUCCESSFUL")
            print("=" * 60)

            # ------------------------------------------------
            # Discover available tools
            # ------------------------------------------------

            tools_result = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools_result.tools:
                print(f"- {tool.name}")

            # ------------------------------------------------
            # Test Tool 1
            # ------------------------------------------------

            print("\n" + "=" * 60)
            print("TEST 1: SEARCH COMPANY POLICY")
            print("=" * 60)

            result = await session.call_tool(
                "search_company_policy",
                {
                    "query": "What is the airport travel limit in India?"
                },
            )

            print(result)

            # ------------------------------------------------
            # Test Tool 2
            # ------------------------------------------------

            print("\n" + "=" * 60)
            print("TEST 2: EMPLOYEE ELIGIBILITY")
            print("=" * 60)

            result = await session.call_tool(
                "get_employee_eligibility",
                {
                    "employee_id": "EMP001"
                },
            )

            print(result)

            # ------------------------------------------------
            # Test Tool 3
            # ------------------------------------------------

            print("\n" + "=" * 60)
            print("TEST 3: VALIDATE EMPLOYEE TRIP")
            print("=" * 60)

            result = await session.call_tool(
                "validate_employee_trip",
                {
                    "employee_id": "EMP001",
                    "trip_type": "airport",
                    "amount": 2500,
                    "time": "22:30",
                },
            )

            print(result)

            print("\n" + "=" * 60)
            print("ALL MCP TESTS COMPLETED")
            print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

