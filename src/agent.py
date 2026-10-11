"""
Travel Policy Agent
Agentic layer for the AI Travel & Policy Assistant.

Flow:

User Question
      ↓
Conversation Memory
      ↓
Domain / Scope Safety
      ↓
Context Safety Layer
      ↓
     LLM
      ↓
Semantic Tool Decision
      ↓
Tool Safety Validation
      ↓
Tool Execution
      ↓
Structured Tool Result
      ↓
Final Answer Generation
      ↓
Conversation Memory
      ↓
User

Important design principles:

1. The LLM decides which tool to use.
2. No hardcoded keyword-based tool routing is used.
3. Domain safety is separate from tool routing.
4. Context safety is separate from tool routing.
5. Previous context is inherited only when the current
   question clearly requires it.
6. Tool results are authoritative.
7. The final answer must not contradict a successful tool result.
8. Missing information must never be invented.
9. Intermediate planning/tool-selection messages are never
   returned directly to the user.
10. Current-turn information always has priority over old context.
11. Out-of-scope questions must not be answered using policy RAG.
"""

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_ollama import ChatOllama

from src.memory import ConversationMemory

from src.tools import (
    policy_search,
    check_employee_eligibility,
    validate_trip,
    calculate_reimbursement,
)

from src.prompts import AGENT_SYSTEM_PROMPT


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

MAX_MEMORY_MESSAGES = int(
    os.getenv(
        "MAX_MEMORY_MESSAGES",
        "20"
    )
)

DEFAULT_MAX_ITERATIONS = int(
    os.getenv(
        "MAX_AGENT_ITERATIONS",
        "5"
    )
)


# ============================================================
# TRAVEL POLICY AGENT
# ============================================================

class TravelPolicyAgent:
    """
    Agentic layer responsible for:

    1. Receiving user questions
    2. Maintaining conversation memory
    3. Checking domain scope
    4. Resolving safe conversation context
    5. Allowing the LLM to select tools
    6. Validating tool arguments
    7. Executing selected tools
    8. Passing structured results back to the model
    9. Generating final answers
    10. Tracking tools used
    11. Tracking policy sources
    12. Handling errors safely

    The agent does NOT use keyword-based tool routing.

    Context safety checks exist only to prevent:

        - stale employee IDs
        - stale trip amounts
        - stale trip information
        - accidental context leakage
        - hallucinated tool arguments
        - unrelated questions being answered through RAG

    The LLM remains responsible for deciding which tool
    should be used.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        print("=" * 70)
        print("INITIALIZING TRAVEL POLICY AGENT")
        print("=" * 70)

        # ----------------------------------------------------
        # Initialize Ollama
        # ----------------------------------------------------

        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            temperature=0
        )

        # ----------------------------------------------------
        # Register tools
        # ----------------------------------------------------

        self.tools = [
            policy_search,
            check_employee_eligibility,
            validate_trip,
            calculate_reimbursement,
        ]

        # ----------------------------------------------------
        # Tool lookup
        # ----------------------------------------------------

        self.tool_map = {
            tool.name: tool
            for tool in self.tools
        }

        # ----------------------------------------------------
        # Bind tools to LLM
        # ----------------------------------------------------

        self.llm_with_tools = self.llm.bind_tools(
            self.tools
        )

        # ----------------------------------------------------
        # Conversation memory
        # ----------------------------------------------------

        self.memory = ConversationMemory(
            max_messages=MAX_MEMORY_MESSAGES
        )

        # ----------------------------------------------------
        # Configuration output
        # ----------------------------------------------------

        print(f"Model: {OLLAMA_MODEL}")
        print(f"Ollama URL: {OLLAMA_URL}")
        print(f"Tools loaded: {len(self.tools)}")

        print("\nAvailable tools:")

        for tool in self.tools:
            print(f" - {tool.name}")

        print("\nConversation memory initialized.")

        print("\nAgent initialized successfully.")
        print("=" * 70)

    # ========================================================
    # GET TOOL
    # ========================================================

    def get_tool(
        self,
        tool_name: str
    ):
        """
        Return registered tool by name.
        """

        if not tool_name:
            return None

        return self.tool_map.get(
            tool_name
        )

    # ========================================================
    # EXTRACT CONTENT
    # ========================================================

    def extract_content(
        self,
        content: Any
    ) -> str:
        """
        Convert LangChain message content into plain text.
        """

        if content is None:
            return ""

        # ----------------------------------------------------
        # String
        # ----------------------------------------------------

        if isinstance(content, str):
            return content

        # ----------------------------------------------------
        # List
        # ----------------------------------------------------

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, str):

                    text_parts.append(
                        item
                    )

                elif isinstance(item, dict):

                    if item.get("type") == "text":

                        text_parts.append(
                            str(
                                item.get(
                                    "text",
                                    ""
                                )
                            )
                        )

            return "\n".join(
                part
                for part in text_parts
                if part
            )

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(content, dict):

            if "text" in content:

                return str(
                    content.get(
                        "text",
                        ""
                    )
                )

            if "content" in content:

                return self.extract_content(
                    content.get("content")
                )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        return str(content)

    # ========================================================
    # FORMAT HISTORY
    # ========================================================

    def format_history(
        self,
        history: Any
    ) -> str:
        """
        Convert conversation memory into textual context.
        """

        if not history:
            return ""

        # ----------------------------------------------------
        # String
        # ----------------------------------------------------

        if isinstance(history, str):
            return history.strip()

        # ----------------------------------------------------
        # List
        # ----------------------------------------------------

        if isinstance(history, list):

            formatted_parts = []

            for item in history:

                if isinstance(item, dict):

                    role = item.get(
                        "role",
                        "message"
                    )

                    content = item.get(
                        "content",
                        ""
                    )

                    content = self.extract_content(
                        content
                    )

                    if content:

                        formatted_parts.append(
                            f"{str(role).upper()}: {content}"
                        )

                else:

                    role = getattr(
                        item,
                        "type",
                        "message"
                    )

                    content = getattr(
                        item,
                        "content",
                        item
                    )

                    content = self.extract_content(
                        content
                    )

                    if content:

                        formatted_parts.append(
                            f"{str(role).upper()}: {content}"
                        )

            return "\n\n".join(
                formatted_parts
            )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        return self.extract_content(
            history
        )

    # ========================================================
    # BUILD CONVERSATION CONTEXT
    # ========================================================

    def build_conversation_context(
        self,
        conversation_id: str
    ) -> str:
        """
        Retrieve previous conversation context.
        """

        try:

            history = self.memory.build_context(
                conversation_id
            )

            return self.format_history(
                history
            )

        except Exception as error:

            print(
                f"Memory context error: {error}"
            )

            return ""

    # ========================================================
    # DOMAIN / SCOPE SAFETY
    # ========================================================

    def classify_question_scope(
        self,
        question: str
    ) -> str:
        """
        Determine whether the current question belongs to
        the supported Travel Policy / Employee Operations
        domain.

        This is a DOMAIN SAFETY CHECK.

        It does NOT decide which tool to use.

        Supported areas include:

            - company travel policies
            - employee travel eligibility
            - trip validation
            - reimbursement
            - airport travel
            - local/intercity travel
            - domestic/international travel
            - travel expenses
            - approval requirements
            - cancellation rules
            - travel-related policy questions

        Unsupported questions should be classified as:

            OUT_OF_SCOPE

        Expected result:

            IN_SCOPE
            OUT_OF_SCOPE
        """

        if not question or not question.strip():

            return "OUT_OF_SCOPE"

        scope_prompt = """
You are a strict domain classifier for a Corporate Travel
and Employee Operations Assistant.

Classify the user's question into exactly ONE category:

IN_SCOPE
or
OUT_OF_SCOPE

IN_SCOPE means the question is about topics such as:

- company travel policy
- employee travel eligibility
- travel requests
- trip validation
- airport travel
- local travel
- intercity travel
- domestic travel
- international travel
- travel reimbursement
- travel expenses
- approval requirements
- cancellation rules
- travel limits
- employee travel-related rules
- company travel procedures

OUT_OF_SCOPE means the question is unrelated to the
corporate travel and employee travel-policy domain.

Examples:

"What is the airport reimbursement limit in India?"
IN_SCOPE

"Is EMP001 eligible for travel?"
IN_SCOPE

"Can EMP001 take an airport trip costing INR 1500?"
IN_SCOPE

"What is the cancellation policy?"
IN_SCOPE

"How do I claim a travel expense?"
IN_SCOPE

"Who won the IPL?"
OUT_OF_SCOPE

"What is the weather today?"
OUT_OF_SCOPE

"Write me a Python game."
OUT_OF_SCOPE

"Tell me a joke."
OUT_OF_SCOPE

"Who is the Prime Minister of India?"
OUT_OF_SCOPE

Important:

- Do not answer the question.
- Do not select a tool.
- Do not retrieve policy information.
- Only classify the domain.
- Return exactly one label.

Return exactly:

IN_SCOPE

or

OUT_OF_SCOPE
"""

        try:

            response = self.llm.invoke(
                [
                    SystemMessage(
                        content=scope_prompt
                    ),
                    HumanMessage(
                        content=question
                    )
                ]
            )

            result = self.extract_content(
                response.content
            ).strip().upper()

            if "OUT_OF_SCOPE" in result:

                return "OUT_OF_SCOPE"

            if "IN_SCOPE" in result:

                return "IN_SCOPE"

            # ------------------------------------------------
            # Conservative fallback
            # ------------------------------------------------

            print(
                "Scope classifier returned an unclear result. "
                "Blocking request for safety."
            )

            return "OUT_OF_SCOPE"

        except Exception as error:

            print(
                f"Scope classification error: {error}"
            )

            # ------------------------------------------------
            # Fail closed.
            # ------------------------------------------------

            return "OUT_OF_SCOPE"

    # --------------------------------------------------------

    def is_out_of_scope(
        self,
        question: str
    ) -> bool:
        """
        Return True when the question is outside the supported
        travel-policy domain.
        """

        return (
            self.classify_question_scope(
                question
            )
            == "OUT_OF_SCOPE"
        )

    # --------------------------------------------------------

    def build_out_of_scope_response(
        self
    ) -> str:
        """
        User-facing response for unsupported questions.
        """

        return (
            "I can help with company travel policies, "
            "employee travel eligibility, trip validation, "
            "reimbursements, approvals, expenses, and "
            "cancellations."
        )

    # ========================================================
    # CONTEXT SAFETY
    # ========================================================

    def extract_employee_ids(
        self,
        text: str
    ) -> List[str]:
        """
        Extract explicitly mentioned employee IDs.

        Example:
            EMP001
            emp002

        Returns:
            ["EMP001"]
        """

        if not text:
            return []

        return [
            match.upper()
            for match in re.findall(
                r"\bEMP\d+\b",
                text,
                flags=re.IGNORECASE
            )
        ]

    # --------------------------------------------------------

    def extract_explicit_amounts(
        self,
        text: str
    ) -> List[float]:
        """
        Extract explicit monetary amounts from text.

        Examples:

            INR 1500
            Rs 2,500
            ₹2500
            USD 75
            1500 INR
            costing 1500
            amount 1500
        """

        if not text:
            return []

        patterns = [

            r"(?:INR|Rs\.?|₹|USD|\$)\s*([\d,]+(?:\.\d+)?)",

            r"\b([\d,]+(?:\.\d+)?)\s*(?:INR|USD)\b",

            r"\bcosting\s+([\d,]+(?:\.\d+)?)",

            r"\bamount\s+(?:of\s+)?([\d,]+(?:\.\d+)?)",

            r"\bfor\s+([\d,]+(?:\.\d+)?)"
        ]

        amounts = []

        for pattern in patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for match in matches:

                try:

                    value = float(
                        str(match).replace(
                            ",",
                            ""
                        )
                    )

                    if value not in amounts:

                        amounts.append(
                            value
                        )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

        return amounts

    # --------------------------------------------------------

    def extract_trip_types(
        self,
        text: str
    ) -> List[str]:
        """
        Extract recognizable trip types.

        This is used only for argument validation and context
        safety.

        It does NOT decide which tool should be called.
        """

        if not text:
            return []

        text_lower = text.lower()

        trip_types = [
            "airport",
            "local",
            "intercity",
            "domestic",
            "international",
            "business trip"
        ]

        found = []

        for trip_type in trip_types:

            if trip_type in text_lower:

                found.append(
                    trip_type
                )

        return list(
            dict.fromkeys(
                found
            )
        )

    # --------------------------------------------------------

    def has_employee_pronoun_reference(
        self,
        text: str
    ) -> bool:
        """
        Detect clear references to a previously established
        employee.

        This is intentionally conservative.
        """

        if not text:
            return False

        text_lower = text.lower()

        pronoun_patterns = [

            r"\bhe\b",
            r"\bshe\b",
            r"\bthey\b",
            r"\bthem\b",
            r"\bhim\b",
            r"\bher\b",

            r"\bthat employee\b",
            r"\bthis employee\b",
            r"\bsame employee\b",
            r"\bthe employee\b"
        ]

        return any(
            re.search(
                pattern,
                text_lower
            )
            for pattern in pronoun_patterns
        )

    # --------------------------------------------------------

    def is_general_eligibility_question(
        self,
        question: str
    ) -> bool:
        """
        Determine whether the user is asking about employee
        eligibility generally.

        Example:

            "How do I check whether an employee is eligible?"

        This must NOT inherit EMP001 from a previous turn.
        """

        if not question:
            return False

        text = question.lower().strip()

        # Explicit employee ID means employee-specific intent.
        if self.extract_employee_ids(question):

            return False

        general_patterns = [

            r"how\s+do\s+i\s+check.*eligib",
            r"how\s+can\s+i\s+check.*eligib",
            r"how\s+to\s+check.*eligib",
            r"how.*check.*employee.*eligib",
            r"what.*required.*eligib",
            r"what.*information.*eligib",
            r"what.*requirements.*eligib",
            r"how.*employee.*eligib",
            r"what.*employee.*eligib",
            r"what.*process.*eligib",
            r"how.*determine.*eligib",
            r"how.*verify.*eligib",
            r"how.*know.*eligib",
            r"how.*find.*eligib"
        ]

        return any(
            re.search(
                pattern,
                text
            )
            for pattern in general_patterns
        )

    # --------------------------------------------------------

    def is_valid_employee_followup(
        self,
        question: str,
        history: str
    ) -> bool:
        """
        Allow employee inheritance only when:

        1. There is previous conversation history.
        2. The question is not a general eligibility question.
        3. The question clearly refers to an employee.
        4. Exactly one employee ID is established in history.
        """

        if not history:
            return False

        if self.is_general_eligibility_question(
            question
        ):
            return False

        if not self.has_employee_pronoun_reference(
            question
        ):
            return False

        previous_employee_ids = list(
            dict.fromkeys(
                self.extract_employee_ids(
                    history
                )
            )
        )

        return len(
            previous_employee_ids
        ) == 1

    # --------------------------------------------------------

    def get_context_mode(
        self,
        question: str,
        conversation_id: str
    ) -> str:
        """
        Classify how previous conversation may safely be used.

        Modes:

            GENERAL
                General question. Previous entity-specific
                information must not leak.

            EXPLICIT
                Current question explicitly contains an
                employee ID.

            FOLLOW_UP
                Current question clearly refers to previous
                conversation.

            NEW_REQUEST
                Independent request with no safe inheritance.

        This is a CONTEXT SAFETY classification.

        It does NOT choose a tool.
        """

        history = self.build_conversation_context(
            conversation_id
        )

        explicit_employee_ids = (
            self.extract_employee_ids(
                question
            )
        )

        # ----------------------------------------------------
        # General eligibility
        # ----------------------------------------------------

        if self.is_general_eligibility_question(
            question
        ):

            return "GENERAL"

        # ----------------------------------------------------
        # Explicit employee ID
        # ----------------------------------------------------

        if explicit_employee_ids:

            return "EXPLICIT"

        # ----------------------------------------------------
        # Pronoun-based employee follow-up
        # ----------------------------------------------------

        if self.is_valid_employee_followup(
            question,
            history
        ):

            return "FOLLOW_UP"

        # ----------------------------------------------------
        # Other conversational follow-ups
        # ----------------------------------------------------

        text = question.lower()

        followup_phrases = [
            "what about",
            "what if",
            "how about",
            "instead",
            "that case",
            "same trip",
            "same employee",
            "same request",
            "then",
            "in the us",
            "in usa",
            "in america"
        ]

        if any(
            phrase in text
            for phrase in followup_phrases
        ):

            if history:

                return "FOLLOW_UP"

        # ----------------------------------------------------
        # Independent request
        # ----------------------------------------------------

        return "NEW_REQUEST"

    # --------------------------------------------------------

    def current_question_has_amount(
        self,
        question: str
    ) -> bool:
        """
        Check whether the CURRENT question explicitly contains
        a monetary amount.
        """

        return bool(
            self.extract_explicit_amounts(
                question
            )
        )

    # --------------------------------------------------------

    def current_question_has_trip_type(
        self,
        question: str
    ) -> bool:
        """
        Check whether the CURRENT question provides a
        recognizable trip type.
        """

        return bool(
            self.extract_trip_types(
                question
            )
        )

    # --------------------------------------------------------

    def get_previous_employee_id(
        self,
        conversation_id: str
    ) -> Optional[str]:
        """
        Return exactly one previously established employee ID.

        Used only for valid contextual follow-ups.
        """

        history = self.build_conversation_context(
            conversation_id
        )

        employee_ids = list(
            dict.fromkeys(
                self.extract_employee_ids(
                    history
                )
            )
        )

        if len(employee_ids) == 1:

            return employee_ids[0]

        return None

    # --------------------------------------------------------

    def get_previous_amounts(
        self,
        conversation_id: str
    ) -> List[float]:
        """
        Extract previous amounts.

        Previous amounts are NEVER automatically accepted as
        current tool arguments.
        """

        history = self.build_conversation_context(
            conversation_id
        )

        return self.extract_explicit_amounts(
            history
        )

    # --------------------------------------------------------

    def get_previous_trip_types(
        self,
        conversation_id: str
    ) -> List[str]:
        """
        Extract previously mentioned trip types.

        Used only for legitimate same-trip follow-ups.
        """

        history = self.build_conversation_context(
            conversation_id
        )

        return self.extract_trip_types(
            history
        )

    # ========================================================
    # TOOL ARGUMENT SAFETY VALIDATION
    # ========================================================

    def validate_requested_tool_against_question(
        self,
        tool_call: Dict[str, Any],
        question: str,
        conversation_id: str
    ) -> Optional[str]:
        """
        Validate an LLM-selected tool call against the
        CURRENT user question.

        This does NOT choose a tool.

        It only prevents unsafe arguments such as:

            Current:
                "Can EMP001 take an airport trip?"

            Unsafe LLM call:
                validate_trip(
                    employee_id="EMP001",
                    trip_type="airport",
                    amount=2500
                )

        The amount is rejected because it was not provided
        in the current request.
        """

        tool_name = tool_call.get(
            "name"
        )

        if not tool_name:
            return None

        args = tool_call.get(
            "args",
            {}
        )

        if not isinstance(
            args,
            dict
        ):

            return (
                "Invalid tool arguments."
            )

        context_mode = self.get_context_mode(
            question,
            conversation_id
        )

        # ====================================================
        # GENERAL ELIGIBILITY
        # ====================================================

        if (
            tool_name
            == "check_employee_eligibility"
            and context_mode == "GENERAL"
        ):

            return (
                "This is a general employee eligibility "
                "question. Do not check a specific employee. "
                "Answer the question generally using the "
                "employee eligibility policy."
            )

        # ====================================================
        # EMPLOYEE ID SAFETY
        # ====================================================

        if tool_name in {
            "check_employee_eligibility",
            "validate_trip"
        }:

            explicit_ids = (
                self.extract_employee_ids(
                    question
                )
            )

            requested_id = args.get(
                "employee_id"
            )

            # ------------------------------------------------
            # Current question explicitly provides employee ID
            # ------------------------------------------------

            if explicit_ids:

                if not requested_id:

                    return (
                        "The current question explicitly "
                        "provides an employee ID, but the "
                        "tool call did not provide it."
                    )

                requested_id = str(
                    requested_id
                ).upper()

                if requested_id not in explicit_ids:

                    return (
                        "The employee ID in the tool call "
                        "does not match the employee ID "
                        "provided in the current question. "
                        "Use only the current employee ID."
                    )

            # ------------------------------------------------
            # No employee ID in current question
            # ------------------------------------------------

            else:

                # Only genuine follow-ups can inherit ID.
                if context_mode != "FOLLOW_UP":

                    if requested_id:

                        return (
                            "The current question does not "
                            "provide an employee ID. Do not "
                            "inherit an employee ID from "
                            "previous conversation."
                        )

                else:

                    # -----------------------------------------
                    # FOLLOW-UP
                    # -----------------------------------------

                    previous_employee_id = (
                        self.get_previous_employee_id(
                            conversation_id
                        )
                    )

                    if not previous_employee_id:

                        return (
                            "The employee reference is "
                            "ambiguous. Ask the user to "
                            "provide the employee ID."
                        )

                    if not requested_id:

                        return (
                            "The follow-up refers to a previous "
                            "employee, but the tool call did "
                            "not provide the established "
                            "employee ID."
                        )

                    requested_id = str(
                        requested_id
                    ).upper()

                    if requested_id != (
                        previous_employee_id
                    ):

                        return (
                            "The employee ID in the tool call "
                            "does not match the employee "
                            "established by the follow-up. "
                            "Use only the previously established "
                            "employee."
                        )

        # ====================================================
        # TRIP VALIDATION
        # ====================================================

        if tool_name == "validate_trip":

            explicit_ids = (
                self.extract_employee_ids(
                    question
                )
            )

            # ------------------------------------------------
            # Employee ID
            # ------------------------------------------------

            if not explicit_ids:

                if context_mode != "FOLLOW_UP":

                    return (
                        "The current trip request does not "
                        "contain an employee ID. Ask the user "
                        "for the employee ID."
                    )

            # ------------------------------------------------
            # Amount
            #
            # Amount is NEVER inherited automatically.
            # ------------------------------------------------

            current_amounts = (
                self.extract_explicit_amounts(
                    question
                )
            )

            requested_amount = args.get(
                "amount"
            )

            if not current_amounts:

                return (
                    "The current trip request does not "
                    "contain a travel amount. Do not reuse "
                    "an amount from previous conversation. "
                    "Ask the user for the amount."
                )

            if requested_amount is None:

                return (
                    "The current trip request does not "
                    "contain a valid amount in the tool call. "
                    "Ask the user for the amount."
                )

            # ------------------------------------------------
            # Validate amount
            # ------------------------------------------------

            try:

                requested_amount_float = float(
                    str(
                        requested_amount
                    ).replace(
                        ",",
                        ""
                    )
                )

            except (
                ValueError,
                TypeError
            ):

                return (
                    "The travel amount in the tool call "
                    "is invalid."
                )

            if not any(
                abs(
                    requested_amount_float
                    - current_amount
                ) < 0.01
                for current_amount
                in current_amounts
            ):

                return (
                    "The travel amount in the tool call "
                    "does not match the amount provided "
                    "in the current question. Do not reuse "
                    "an amount from previous conversation."
                )

            # ------------------------------------------------
            # Trip type
            # ------------------------------------------------

            current_trip_types = (
                self.extract_trip_types(
                    question
                )
            )

            requested_trip_type = args.get(
                "trip_type"
            )

            if current_trip_types:

                if not requested_trip_type:

                    return (
                        "The current question provides a "
                        "trip type, but the tool call did "
                        "not provide it."
                    )

                requested_trip_type = str(
                    requested_trip_type
                ).lower()

                # Current trip type should be represented
                # by one of the recognized current types.
                if not any(
                    current_trip_type
                    in requested_trip_type
                    or requested_trip_type
                    in current_trip_type
                    for current_trip_type
                    in current_trip_types
                ):

                    return (
                        "The trip type in the tool call "
                        "does not match the current question."
                    )

            else:

                # ------------------------------------------------
                # A trip type may only be inherited when the
                # current question clearly refers to the same
                # previous trip.
                # ------------------------------------------------

                if context_mode != "FOLLOW_UP":

                    return (
                        "The current trip request does not "
                        "contain a trip type. Ask the user "
                        "for the trip type."
                    )

                previous_trip_types = (
                    self.get_previous_trip_types(
                        conversation_id
                    )
                )

                if not previous_trip_types:

                    return (
                        "The current follow-up does not "
                        "provide a trip type and no previous "
                        "trip type can be safely established."
                    )

                if not requested_trip_type:

                    return (
                        "The current follow-up refers to the "
                        "previous trip, but the tool call did "
                        "not provide the established trip type."
                    )

                requested_trip_type = str(
                    requested_trip_type
                ).lower()

                if not any(
                    previous_trip_type
                    in requested_trip_type
                    or requested_trip_type
                    in previous_trip_type
                    for previous_trip_type
                    in previous_trip_types
                ):

                    return (
                        "The trip type in the tool call "
                        "does not match the previously "
                        "established trip."
                    )

        return None

    # ========================================================
    # DETECT FAKE TOOL CALL TEXT
    # ========================================================

    def contains_fake_tool_call_text(
        self,
        text: str
    ) -> bool:
        """
        Detect cases where the model prints a tool call as
        ordinary text instead of using LangChain tool_calls.
        """

        if not text:
            return False

        text_lower = text.lower()

        tool_names = [
            "policy_search",
            "check_employee_eligibility",
            "validate_trip",
            "calculate_reimbursement"
        ]

        has_tool_name = any(
            tool_name in text_lower
            for tool_name in tool_names
        )

        json_markers = [
            '"name"',
            '"parameters"',
            '"arguments"',
            '"args"',
            "{"
        ]

        has_json_marker = any(
            marker in text_lower
            for marker in json_markers
        )

        return (
            has_tool_name
            and has_json_marker
        )

    # ========================================================
    # BUILD SAFE CLARIFICATION
    # ========================================================

    def build_missing_information_response(
        self,
        question: str
    ) -> Optional[str]:
        """
        Generate deterministic clarification for obvious
        missing trip information.

        This is a safety guard, not tool routing.
        """

        has_trip_type = (
            self.current_question_has_trip_type(
                question
            )
        )

        has_employee = bool(
            self.extract_employee_ids(
                question
            )
        )

        has_amount = bool(
            self.extract_explicit_amounts(
                question
            )
        )

        missing = []

        if not has_employee:

            missing.append(
                "employee ID"
            )

        if not has_trip_type:

            missing.append(
                "trip type"
            )

        if not has_amount:

            missing.append(
                "travel amount"
            )

        if not missing:

            return None

        if len(missing) == 1:

            return (
                f"Please provide the {missing[0]} "
                f"so I can validate the trip."
            )

        if len(missing) == 2:

            return (
                "Please provide the "
                + " and ".join(
                    missing
                )
                + " so I can validate the trip."
            )

        return (
            "Please provide the employee ID, trip type, "
            "and travel amount so I can validate the trip."
        )

    # ========================================================
    # TOOL ARGUMENT NORMALIZATION
    # ========================================================

    def normalize_tool_args(
        self,
        tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert common LLM placeholders into genuine
        missing values.
        """

        normalized = dict(
            tool_args or {}
        )

        missing_tokens = {
            "",
            "none",
            "null",
            "n/a",
            "na",
            "unknown",
            "not provided",
            "not available",
            "missing"
        }

        for key, value in list(
            normalized.items()
        ):

            if isinstance(
                value,
                str
            ):

                value = value.strip()

                normalized[key] = (
                    None
                    if value.lower()
                    in missing_tokens
                    else value
                )

        return normalized

    # --------------------------------------------------------

    def missing_tool_arguments(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> List[str]:
        """
        Check required arguments for each tool.
        """

        required = {

            "policy_search": [
                "query"
            ],

            "check_employee_eligibility": [
                "employee_id"
            ],

            "validate_trip": [
                "employee_id",
                "trip_type",
                "amount"
            ],

            "calculate_reimbursement": [
                "trip_amount",
                "policy_limit"
            ],
        }.get(
            tool_name,
            []
        )

        return [
            key
            for key in required
            if tool_args.get(key) is None
        ]

    # ========================================================
    # EXECUTE TOOL CALL
    # ========================================================

    def execute_tool_call(
        self,
        tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Safely execute an LLM-selected tool.
        """

        tool_name = tool_call.get(
            "name"
        )

        tool_args = tool_call.get(
            "args",
            {}
        )

        # ----------------------------------------------------
        # Validate tool name
        # ----------------------------------------------------

        if not tool_name:

            return {
                "success": False,
                "error": "Tool name was not provided."
            }

        # ----------------------------------------------------
        # Validate arguments
        # ----------------------------------------------------

        if tool_args is None:

            tool_args = {}

        if not isinstance(
            tool_args,
            dict
        ):

            return {
                "success": False,
                "error": (
                    f"Invalid arguments supplied to "
                    f"tool '{tool_name}'."
                )
            }

        tool_args = self.normalize_tool_args(
            tool_args
        )

        missing_args = (
            self.missing_tool_arguments(
                tool_name,
                tool_args
            )
        )

        if missing_args:

            return {
                "success": False,
                "blocked": True,
                "missing_arguments": missing_args,
                "error": (
                    "Missing required information: "
                    + ", ".join(
                        missing_args
                    )
                    + ". Ask the user for the missing "
                    "information instead of guessing it."
                )
            }

        # ----------------------------------------------------
        # Find tool
        # ----------------------------------------------------

        tool = self.get_tool(
            tool_name
        )

        if tool is None:

            return {
                "success": False,
                "error": (
                    f"Unknown tool: {tool_name}"
                )
            }

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        try:

            print(
                f"\nExecuting tool: {tool_name}"
            )

            print(
                f"Arguments: {tool_args}"
            )

            result = tool.invoke(
                tool_args
            )

            print(
                f"Tool completed: {tool_name}"
            )

            return result

        except Exception as error:

            print(
                f"Tool error [{tool_name}]: {error}"
            )

            return {
                "success": False,
                "error": (
                    f"Tool '{tool_name}' failed."
                ),
                "details": str(error)
            }

    # ========================================================
    # EXTRACT POLICY SOURCES
    # ========================================================

    def extract_sources(
        self,
        tool_name: str,
        tool_result: Any
    ) -> List[Dict[str, Any]]:
        """
        Extract source information from policy_search.
        """

        sources = []

        if tool_name != "policy_search":
            return sources

        if not isinstance(
            tool_result,
            dict
        ):
            return sources

        if not tool_result.get(
            "success",
            False
        ):
            return sources

        results_data = tool_result.get(
            "results",
            []
        )

        if isinstance(
            results_data,
            list
        ):

            results = results_data

        elif isinstance(
            results_data,
            dict
        ):

            results = results_data.get(
                "results",
                []
            )

        else:

            results = []

        if not isinstance(
            results,
            list
        ):
            return sources

        for result in results:

            if not isinstance(
                result,
                dict
            ):
                continue

            source = result.get(
                "source"
            )

            relevance_score = result.get(
                "relevance_score"
            )

            if not source:

                metadata = result.get(
                    "metadata",
                    {}
                )

                if isinstance(
                    metadata,
                    dict
                ):

                    source = metadata.get(
                        "source"
                    )

            if source:

                sources.append({
                    "source": source,
                    "relevance_score": relevance_score
                })

        return sources

    # ========================================================
    # DEDUPLICATE SOURCES
    # ========================================================

    def deduplicate_sources(
        self,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate policy sources.

        If the same source occurs multiple times,
        retain the highest relevance score.
        """

        best_sources = {}

        for source in sources:

            if not isinstance(
                source,
                dict
            ):
                continue

            source_name = source.get(
                "source"
            )

            if not source_name:
                continue

            score = source.get(
                "relevance_score"
            )

            if source_name not in best_sources:

                best_sources[
                    source_name
                ] = source

                continue

            existing = best_sources[
                source_name
            ]

            existing_score = existing.get(
                "relevance_score"
            )

            if (
                isinstance(
                    score,
                    (int, float)
                )
                and isinstance(
                    existing_score,
                    (int, float)
                )
                and score > existing_score
            ):

                best_sources[
                    source_name
                ] = source

        return list(
            best_sources.values()
        )

    # ========================================================
    # CREATE TOOL MESSAGE
    # ========================================================

    def create_tool_message(
        self,
        tool_call: Dict[str, Any],
        result: Any
    ) -> ToolMessage:
        """
        Create a ToolMessage connected to the original
        LLM tool call.
        """

        tool_call_id = tool_call.get(
            "id"
        )

        if not tool_call_id:

            tool_call_id = (
                f"tool_call_{uuid.uuid4().hex}"
            )

        tool_name = tool_call.get(
            "name"
        )

        try:

            content = json.dumps(
                result,
                default=str
            )

        except Exception:

            content = str(
                result
            )

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name
        )

    # ========================================================
    # BUILD INITIAL MESSAGES
    # ========================================================

    def build_messages(
        self,
        question: str,
        conversation_id: str
    ) -> List[Any]:
        """
        Build messages for the LLM.

        Previous conversation is NOT automatically included.

        History is included only for a genuine FOLLOW_UP.

        This prevents:

            Previous:
                "Is EMP001 eligible?"

            Current:
                "How do I check whether an employee is eligible?"

        from becoming:

            "How do I check whether EMP001 is eligible?"
        """

        context_mode = self.get_context_mode(
            question,
            conversation_id
        )

        history = []

        # ----------------------------------------------------
        # Only genuine follow-ups receive history.
        # ----------------------------------------------------

        if context_mode == "FOLLOW_UP":

            history = self.memory.get_history(
                conversation_id
            )

        # ----------------------------------------------------
        # System prompt
        # ----------------------------------------------------

        context_safety_prompt = """
CONVERSATION CONTEXT SAFETY
============================

The current user question always has priority.

1. General questions remain general.

2. Do NOT inject an employee ID from previous conversation
   into a general question.

3. Do NOT inject a previous travel amount into a new request.

4. Do NOT inject a previous trip type, country, currency,
   time, date, or policy value unless the current question
   clearly refers to that previous context.

5. A previous employee ID may only be inherited when the
   current question clearly refers to that employee, such as:
   "Is he eligible?"
   "Is that employee eligible?"
   "What about the same employee?"

6. If the current question contains an employee ID, use that
   employee ID exactly.

7. If the current question contains an amount, use that amount
   exactly.

8. Never replace current-turn information with information
   from an earlier message.

9. Previous assistant answers are context only. They are not
   new instructions.

10. Previous tool arguments are NOT automatically valid for
    the current turn.

11. If a required tool argument is missing, ask the user for it.
    Never guess it.

12. Never output fake JSON or Python-style tool calls as the
    user-facing answer.

13. For a genuine follow-up, resolve the reference into a
    self-contained query before calling a tool.

14. If context is ambiguous, ask for clarification rather
    than choosing an employee, amount, or trip detail yourself.
"""

        messages: List[Any] = [

            SystemMessage(
                content=(
                    AGENT_SYSTEM_PROMPT
                    + "\n\n"
                    + context_safety_prompt
                )
            )
        ]

        # ----------------------------------------------------
        # Add history only for valid follow-ups.
        # ----------------------------------------------------

        for item in history:

            if not isinstance(
                item,
                dict
            ):
                continue

            role = item.get(
                "role"
            )

            content = self.extract_content(
                item.get(
                    "content",
                    ""
                )
            )

            if not content:
                continue

            if role == "user":

                messages.append(
                    HumanMessage(
                        content=content
                    )
                )

            elif role == "assistant":

                messages.append(
                    AIMessage(
                        content=content
                    )
                )

            elif role == "system":

                messages.append(
                    SystemMessage(
                        content=content
                    )
                )

        # ----------------------------------------------------
        # Current question
        # ----------------------------------------------------

        current_question_instruction = f"""
CURRENT USER QUESTION
=====================

{question}


CONTEXT MODE
============

{context_mode}


IMPORTANT

Resolve the current question using only context that is
relevant to this request.

If this is a NEW_REQUEST or EXPLICIT request, do not silently
reuse entity values from older messages.

If this is a FOLLOW_UP, use the previous conversation only
where the current question clearly refers to it.

Never invent missing tool arguments.
"""

        messages.append(
            HumanMessage(
                content=current_question_instruction
            )
        )

        return messages

    # ========================================================
    # TOOL CALL SIGNATURE
    # ========================================================

    def get_tool_call_signature(
        self,
        tool_call: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Create a stable signature for duplicate-call detection.
        """

        tool_name = tool_call.get(
            "name",
            ""
        )

        args = tool_call.get(
            "args",
            {}
        )

        try:

            args_string = json.dumps(
                args,
                sort_keys=True,
                default=str
            )

        except Exception:

            args_string = str(
                args
            )

        return (
            tool_name,
            args_string
        )

    # ========================================================
    # FORMAT AUTHORITATIVE TOOL RESULTS
    # ========================================================

    def format_tool_results(
        self,
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """
        Convert structured tool results into explicit
        authoritative evidence for the final answer model.
        """

        if not tool_results:

            return (
                "No tool results were produced."
            )

        sections = []

        for index, item in enumerate(
            tool_results,
            start=1
        ):

            tool_name = item.get(
                "tool",
                "unknown"
            )

            args = item.get(
                "args",
                {}
            )

            result = item.get(
                "result",
                {}
            )

            try:

                serialized_result = json.dumps(
                    result,
                    indent=2,
                    default=str
                )

            except Exception:

                serialized_result = str(
                    result
                )

            try:

                serialized_args = json.dumps(
                    args,
                    default=str
                )

            except Exception:

                serialized_args = str(
                    args
                )

            sections.append(
                f"""
TOOL RESULT {index}
------------------
Tool: {tool_name}
Arguments: {serialized_args}

Result:
{serialized_result}
"""
            )

        return "\n".join(
            sections
        )

    # ========================================================
    # DETERMINISTIC TOOL ANSWER
    # ========================================================

    def build_deterministic_tool_answer(
        self,
        tool_results: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Build a deterministic answer when a structured
        business tool has already returned a successful result.

        This prevents the final LLM from contradicting
        authoritative business decisions.
        """

        if not tool_results:
            return None

        for item in reversed(
            tool_results
        ):

            tool_name = item.get(
                "tool"
            )

            result = item.get(
                "result"
            )

            if not isinstance(
                result,
                dict
            ):
                continue

            if not result.get(
                "success",
                False
            ):
                continue

            # =================================================
            # EMPLOYEE ELIGIBILITY
            # =================================================

            if tool_name == (
                "check_employee_eligibility"
            ):

                employee_id = result.get(
                    "employee_id"
                )

                found = result.get(
                    "found",
                    True
                )

                eligible = result.get(
                    "eligible"
                )

                if not found:

                    return (
                        f"Employee {employee_id} "
                        f"was not found in the employee records."
                    )

                if eligible is True:

                    return (
                        f"{employee_id} is eligible for travel."
                    )

                if eligible is False:

                    return (
                        f"{employee_id} is not eligible for travel."
                    )

            # =================================================
            # TRIP VALIDATION
            # =================================================

            if tool_name == "validate_trip":

                employee_id = result.get(
                    "employee_id"
                )

                found = result.get(
                    "found",
                    True
                )

                eligible = result.get(
                    "eligible"
                )

                decision = result.get(
                    "decision"
                )

                trip_type = result.get(
                    "trip_type"
                )

                amount = result.get(
                    "amount"
                )

                currency = result.get(
                    "currency"
                )

                policy_limit = result.get(
                    "policy_limit"
                )

                excess_amount = result.get(
                    "excess_amount"
                )

                late_night = result.get(
                    "late_night",
                    False
                )

                late_night_note = result.get(
                    "late_night_note"
                )

                reason = result.get(
                    "reason"
                )

                if not found:

                    return (
                        f"Employee {employee_id} was not found "
                        f"in the employee records."
                    )

                if eligible is False:

                    if reason:

                        return (
                            f"{employee_id} is not eligible "
                            f"for this trip. {reason}"
                        )

                    return (
                        f"{employee_id} is not eligible "
                        f"for this trip."
                    )

                if decision == "Approved":

                    answer = (
                        f"Yes. {employee_id} is eligible, "
                        f"and the {trip_type} trip costing "
                        f"{currency} {amount:,.0f} is within "
                        f"the standard {currency} "
                        f"{policy_limit:,.0f} limit. "
                        f"The trip is approved."
                    )

                    if (
                        late_night
                        and late_night_note
                    ):

                        answer += (
                            f" {late_night_note}"
                        )

                    return answer

                if decision == "Needs Approval":

                    answer = (
                        f"{employee_id} is eligible, but the "
                        f"{currency} {amount:,.0f} "
                        f"{trip_type} trip exceeds the standard "
                        f"{currency} {policy_limit:,.0f} limit"
                    )

                    if excess_amount is not None:

                        answer += (
                            f" by {currency} "
                            f"{excess_amount:,.0f}"
                        )

                    answer += (
                        ". Additional approval is required."
                    )

                    if (
                        late_night
                        and late_night_note
                    ):

                        answer += (
                            f" {late_night_note}"
                        )

                    return answer

                if decision == "Not Eligible":

                    if reason:

                        return (
                            f"{employee_id} is not eligible "
                            f"for this trip. {reason}"
                        )

                    return (
                        f"{employee_id} is not eligible "
                        f"for this trip."
                    )

                if decision:

                    if reason:

                        return (
                            f"The trip decision is "
                            f"{decision}. {reason}"
                        )

                    return (
                        f"The trip decision is "
                        f"{decision}."
                    )

            # =================================================
            # REIMBURSEMENT CALCULATION
            # =================================================

            if tool_name == (
                "calculate_reimbursement"
            ):

                decision = result.get(
                    "decision"
                )

                reimbursable_amount = result.get(
                    "reimbursable_amount"
                )

                review_amount = result.get(
                    "review_amount"
                )

                currency = result.get(
                    "currency",
                    ""
                )

                if decision:

                    if (
                        reimbursable_amount
                        is not None
                    ):

                        answer = (
                            f"Reimbursable amount: "
                            f"{currency} "
                            f"{float(reimbursable_amount):,.2f}."
                        )

                        if review_amount is not None:

                            answer += (
                                f" Amount requiring review: "
                                f"{currency} "
                                f"{float(review_amount):,.2f}."
                            )

                        return answer

                    return (
                        f"Reimbursement decision: "
                        f"{decision}."
                    )

        return None

    # ========================================================
    # GENERATE FINAL USER-FACING ANSWER
    # ========================================================

    def generate_final_answer(
        self,
        question: str,
        messages: List[Any],
        tool_results: Optional[
            List[Dict[str, Any]]
        ] = None
    ) -> str:
        """
        Generate the final user-facing response.

        Structured business-tool results are authoritative.
        """

        if tool_results is None:

            tool_results = []

        # ====================================================
        # FIRST: DETERMINISTIC BUSINESS RESULT
        # ====================================================

        deterministic_answer = (
            self.build_deterministic_tool_answer(
                tool_results
            )
        )

        if deterministic_answer:

            return deterministic_answer.strip()

        # ====================================================
        # FORMAT TOOL EVIDENCE
        # ====================================================

        authoritative_results = (
            self.format_tool_results(
                tool_results
            )
        )

        # ====================================================
        # FINAL PROMPT
        # ====================================================

        final_instruction = f"""
FINAL USER-FACING RESPONSE
==========================

Original user question:

{question}


AUTHORITATIVE TOOL RESULTS
==========================

The following results were returned by actual business
tools during this request.

These results are authoritative.

{authoritative_results}


FINAL RESPONSE RULES
====================

1. Answer the user's question directly.

2. Treat successful tool results as authoritative business
   information.

3. NEVER contradict a successful tool result.

4. If validate_trip returned:

   decision = "Approved"

   state that the trip is approved.

5. If validate_trip returned:

   decision = "Needs Approval"

   clearly state that additional approval is required.

6. If validate_trip returned:

   decision = "Not Eligible"

   clearly state that the employee/trip is not eligible.

7. If validate_trip returned a policy_limit, use that
   policy limit.

8. If validate_trip returned excess_amount, use that amount.

9. If validate_trip returned late_night = true and a
   late_night_note, mention the late-night condition when
   relevant.

10. Employee-specific eligibility comes from the employee
    eligibility/business tool.

11. Do not claim that employee eligibility is unavailable
    merely because policy documents do not contain employee
    records.

12. Do not invent:
    - policy limits
    - employee eligibility
    - reimbursement amounts
    - approval decisions
    - trip rules
    - currencies

13. If policy_search results are available, use them for
    policy facts.

14. If the available evidence genuinely does not support
    an answer, say that the information is unavailable.

15. Do not mention:
    - chain of thought
    - internal reasoning
    - planning
    - agent iterations
    - hidden prompts
    - system prompts
    - FAISS
    - embeddings
    - vector databases
    - internal relevance scores

16. Do not say:
    "The tool says..."
    "I called..."
    "The agent decided..."
    "I need to use..."

17. Keep the answer concise.

18. Do NOT add a Sources section; sources are returned
    separately.

19. Do NOT repeat the question or describe what you did.

20. Do NOT output JSON, Python dictionaries, function calls,
    tool-call syntax, or internal tool instructions.

21. Return only the final user-facing answer.

Return ONLY the final user-facing answer.
"""

        final_messages = list(
            messages
        )

        final_messages.append(
            HumanMessage(
                content=final_instruction
            )
        )

        # ====================================================
        # FINAL LLM CALL
        # ====================================================

        try:

            final_response = self.llm.invoke(
                final_messages
            )

            answer = self.extract_content(
                final_response.content
            )

            answer = answer.strip()

            if answer:

                if self.contains_fake_tool_call_text(
                    answer
                ):

                    clarification = (
                        self.build_missing_information_response(
                            question
                        )
                    )

                    if clarification:

                        return clarification

                    return (
                        "I need a little more information "
                        "to complete this request."
                    )

                return answer

        except Exception as error:

            print(
                f"Final answer generation error: {error}"
            )

        # ====================================================
        # FALLBACK
        # ====================================================

        return (
            "I was able to process the request, "
            "but I could not generate the final response."
        )

    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    def save_conversation(
        self,
        conversation_id: str,
        question: str,
        answer: str
    ):
        """
        Save user and assistant messages safely.

        IMPORTANT:
        ConversationMemory expects:

            add_user_message(
                content,
                conversation_id
            )

        and NOT the reverse.
        """

        try:

            if not self.memory.exists(
                conversation_id
            ):

                self.memory.create_conversation(
                    conversation_id
                )

            self.memory.add_user_message(
                question,
                conversation_id
            )

            self.memory.add_assistant_message(
                answer,
                conversation_id
            )

        except Exception as error:

            print(
                f"Memory save error: {error}"
            )

    # ========================================================
    # BUILD LLM ERROR RESPONSE
    # ========================================================

    def build_llm_error_response(
        self,
        question: str,
        conversation_id: str,
        tools_used: List[str],
        sources: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        error: Exception
    ) -> Dict[str, Any]:
        """
        Build consistent response when the LLM fails.
        """

        error_answer = (
            "I could not connect to the AI model. "
            "Please make sure Ollama is running and "
            "the configured model is available."
        )

        self.save_conversation(
            conversation_id,
            question,
            error_answer
        )

        return {
            "answer": error_answer,
            "tools_used": list(
                dict.fromkeys(
                    tools_used
                )
            ),
            "sources": self.deduplicate_sources(
                sources
            ),
            "tool_results": tool_results,
            "error": str(error)
        }

    # ========================================================
    # HANDLE UNSAFE TOOL CALL
    # ========================================================

    def handle_unsafe_tool_call(
        self,
        question: str,
        conversation_id: str,
        validation_error: str,
        tools_used: List[str],
        sources: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Safely terminate a request when the LLM attempts to
        use stale or invented tool arguments.
        """

        clarification = (
            self.build_missing_information_response(
                question
            )
        )

        if clarification:

            answer = clarification

        elif (
            self.is_general_eligibility_question(
                question
            )
        ):

            answer = (
                "To check an employee's eligibility, use the "
                "employee's employee ID and verify their "
                "eligibility status against the employee "
                "eligibility policy."
            )

        else:

            answer = (
                "I need the required information for this "
                "request before I can validate it."
            )

        print(
            f"\nUnsafe tool call blocked: "
            f"{validation_error}"
        )

        self.save_conversation(
            conversation_id,
            question,
            answer
        )

        return {
            "answer": answer,
            "tools_used": list(
                dict.fromkeys(
                    tools_used
                )
            ),
            "sources": self.deduplicate_sources(
                sources
            ),
            "tool_results": tool_results,
            "blocked": True
        }

    # ========================================================
    # MAIN AGENT LOOP
    # ========================================================

    def run_agent(
        self,
        question: str,
        conversation_id: str = "default",
        max_iterations: int = DEFAULT_MAX_ITERATIONS
    ) -> Dict[str, Any]:
        """
        Run the complete agentic loop.

        Steps:

        1. Validate question
        2. Check domain scope
        3. Build safe contextual messages
        4. Ask the LLM
        5. Validate requested tool arguments
        6. Execute safe tools
        7. Store structured results
        8. Send results back to the LLM
        9. Generate final answer
        10. Save conversation
        """

        # ====================================================
        # VALIDATE QUESTION
        # ====================================================

        if not question or not question.strip():

            return {
                "answer": "Please enter a question.",
                "tools_used": [],
                "sources": [],
                "tool_results": []
            }

        question = question.strip()

        # ====================================================
        # DOMAIN / SCOPE SAFETY
        #
        # IMPORTANT:
        #
        # This happens BEFORE:
        #
        #     build_messages()
        #     LLM tool selection
        #     RAG
        #     employee tools
        #
        # This prevents unrelated questions from reaching
        # the policy retrieval layer.
        # ====================================================

        scope = self.classify_question_scope(
            question
        )

        if scope == "OUT_OF_SCOPE":

            answer = (
                self.build_out_of_scope_response()
            )

            self.save_conversation(
                conversation_id,
                question,
                answer
            )

            return {
                "answer": answer,
                "tools_used": [],
                "sources": [],
                "tool_results": [],
                "out_of_scope": True,
                "scope": scope
            }

        # ====================================================
        # ITERATION LIMIT
        # ====================================================

        if max_iterations < 1:

            max_iterations = 1

        # ====================================================
        # CREATE CONVERSATION
        # ====================================================

        if not self.memory.exists(
            conversation_id
        ):

            self.memory.create_conversation(
                conversation_id
            )

        # ====================================================
        # BUILD INITIAL MESSAGES
        # ====================================================

        messages = self.build_messages(
            question=question,
            conversation_id=conversation_id
        )

        # ====================================================
        # TRACKING
        # ====================================================

        tools_used: List[str] = []

        sources: List[
            Dict[str, Any]
        ] = []

        tool_results: List[
            Dict[str, Any]
        ] = []

        executed_tool_signatures = set()

        # ====================================================
        # AGENT LOOP
        # ====================================================

        for iteration in range(
            max_iterations
        ):

            print(
                f"\nAgent iteration "
                f"{iteration + 1}/{max_iterations}"
            )

            # ------------------------------------------------
            # ASK LLM
            # ------------------------------------------------

            try:

                response = self.llm_with_tools.invoke(
                    messages
                )

            except Exception as error:

                print(
                    f"LLM error on iteration "
                    f"{iteration + 1}: {error}"
                )

                return self.build_llm_error_response(
                    question=question,
                    conversation_id=conversation_id,
                    tools_used=tools_used,
                    sources=sources,
                    tool_results=tool_results,
                    error=error
                )

            # ------------------------------------------------
            # GET TOOL CALLS
            # ------------------------------------------------

            tool_calls = getattr(
                response,
                "tool_calls",
                []
            )

            if tool_calls is None:

                tool_calls = []

            # =================================================
            # NO TOOL CALL
            # =================================================

            if not tool_calls:

                print(
                    "\nNo additional tool calls requested."
                )

                response_text = self.extract_content(
                    response.content
                )

                # ------------------------------------------------
                # Prevent fake tool-call output.
                # ------------------------------------------------

                if self.contains_fake_tool_call_text(
                    response_text
                ):

                    clarification = (
                        self.build_missing_information_response(
                            question
                        )
                    )

                    if clarification:

                        self.save_conversation(
                            conversation_id,
                            question,
                            clarification
                        )

                        return {
                            "answer": clarification,
                            "tools_used": list(
                                dict.fromkeys(
                                    tools_used
                                )
                            ),
                            "sources": self.deduplicate_sources(
                                sources
                            ),
                            "tool_results": tool_results,
                            "blocked": True
                        }

                    answer = (
                        "I need a little more information "
                        "to complete this request."
                    )

                    self.save_conversation(
                        conversation_id,
                        question,
                        answer
                    )

                    return {
                        "answer": answer,
                        "tools_used": list(
                            dict.fromkeys(
                                tools_used
                            )
                        ),
                        "sources": self.deduplicate_sources(
                            sources
                        ),
                        "tool_results": tool_results,
                        "blocked": True
                    }

                # ------------------------------------------------
                # Store model response internally.
                # ------------------------------------------------

                messages.append(
                    response
                )

                # ------------------------------------------------
                # Generate final answer.
                # ------------------------------------------------

                answer = self.generate_final_answer(
                    question=question,
                    messages=messages,
                    tool_results=tool_results
                )

                # ------------------------------------------------
                # Save conversation.
                # ------------------------------------------------

                self.save_conversation(
                    conversation_id,
                    question,
                    answer
                )

                return {
                    "answer": answer,
                    "tools_used": list(
                        dict.fromkeys(
                            tools_used
                        )
                    ),
                    "sources": self.deduplicate_sources(
                        sources
                    ),
                    "tool_results": tool_results,
                    "scope": scope
                }

            # =================================================
            # TOOL CALLS REQUESTED
            # =================================================

            print(
                f"LLM requested "
                f"{len(tool_calls)} tool call(s)"
            )

            # ------------------------------------------------
            # Add assistant response containing tool calls.
            # ------------------------------------------------

            messages.append(
                response
            )

            # ------------------------------------------------
            # Execute tools.
            # ------------------------------------------------

            for tool_call in tool_calls:

                tool_name = tool_call.get(
                    "name"
                )

                tool_args = tool_call.get(
                    "args",
                    {}
                )

                if tool_name:

                    tools_used.append(
                        tool_name
                    )

                # ---------------------------------------------
                # CONTEXT SAFETY VALIDATION
                #
                # This MUST happen before execution.
                # ---------------------------------------------

                validation_error = (
                    self.validate_requested_tool_against_question(
                        tool_call=tool_call,
                        question=question,
                        conversation_id=conversation_id
                    )
                )

                if validation_error:

                    return self.handle_unsafe_tool_call(
                        question=question,
                        conversation_id=conversation_id,
                        validation_error=validation_error,
                        tools_used=tools_used,
                        sources=sources,
                        tool_results=tool_results
                    )

                # ---------------------------------------------
                # Detect duplicate tool call
                # ---------------------------------------------

                signature = (
                    self.get_tool_call_signature(
                        tool_call
                    )
                )

                if signature in executed_tool_signatures:

                    print(
                        f"Repeated tool call detected: "
                        f"{tool_name}"
                    )

                    result = {
                        "success": False,
                        "error": (
                            "The same tool call was already "
                            "executed with the same arguments. "
                            "Use the existing result instead "
                            "of repeating the call."
                        )
                    }

                else:

                    executed_tool_signatures.add(
                        signature
                    )

                    # -----------------------------------------
                    # Execute
                    # -----------------------------------------

                    result = self.execute_tool_call(
                        tool_call
                    )

                # ------------------------------------------------
                # Store structured result
                # ------------------------------------------------

                tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result
                })

                # ------------------------------------------------
                # Extract policy sources
                # ------------------------------------------------

                tool_sources = self.extract_sources(
                    tool_name,
                    result
                )

                sources.extend(
                    tool_sources
                )

                # ------------------------------------------------
                # Add ToolMessage
                # ------------------------------------------------

                messages.append(
                    self.create_tool_message(
                        tool_call,
                        result
                    )
                )

        # ====================================================
        # MAXIMUM ITERATIONS REACHED
        # ====================================================

        deterministic_answer = (
            self.build_deterministic_tool_answer(
                tool_results
            )
        )

        if deterministic_answer:

            self.save_conversation(
                conversation_id,
                question,
                deterministic_answer
            )

            return {
                "answer": deterministic_answer,
                "tools_used": list(
                    dict.fromkeys(
                        tools_used
                    )
                ),
                "sources": self.deduplicate_sources(
                    sources
                ),
                "tool_results": tool_results,
                "scope": scope
            }

        # ----------------------------------------------------
        # Try LLM final response as last fallback
        # ----------------------------------------------------

        try:

            answer = self.generate_final_answer(
                question=question,
                messages=messages,
                tool_results=tool_results
            )

            self.save_conversation(
                conversation_id,
                question,
                answer
            )

            return {
                "answer": answer,
                "tools_used": list(
                    dict.fromkeys(
                        tools_used
                    )
                ),
                "sources": self.deduplicate_sources(
                    sources
                ),
                "tool_results": tool_results,
                "scope": scope
            }

        except Exception:

            max_iteration_answer = (
                "I was unable to complete the request "
                "within the allowed processing steps. "
                "Please try asking the question again."
            )

            self.save_conversation(
                conversation_id,
                question,
                max_iteration_answer
            )

            return {
                "answer": max_iteration_answer,
                "tools_used": list(
                    dict.fromkeys(
                        tools_used
                    )
                ),
                "sources": self.deduplicate_sources(
                    sources
                ),
                "tool_results": tool_results,
                "scope": scope
            }

    # ========================================================
    # PUBLIC ASK METHOD
    # ========================================================

    def ask(
        self,
        question: str,
        conversation_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Main public method used by:

        - Flask
        - APIs
        - notebooks
        - tests
        - other application components
        """

        if not question or not question.strip():

            return {
                "answer": "Please enter a question.",
                "tools_used": [],
                "sources": [],
                "tool_results": []
            }

        try:

            result = self.run_agent(
                question=question,
                conversation_id=conversation_id
            )

            return result

        except Exception as error:

            print(
                f"Agent error: {error}"
            )

            return {
                "answer": (
                    "I was unable to process your request. "
                    "Please try again."
                ),
                "tools_used": [],
                "sources": [],
                "tool_results": [],
                "error": str(error)
            }

    # ========================================================
    # MEMORY METHODS
    # ========================================================

    def get_conversation_history(
        self,
        conversation_id: str = "default"
    ):
        """
        Return conversation history.
        """

        return self.memory.get_history(
            conversation_id
        )

    # --------------------------------------------------------

    def clear_conversation(
        self,
        conversation_id: str = "default"
    ):
        """
        Clear one conversation.
        """

        self.memory.clear(
            conversation_id
        )

    # --------------------------------------------------------

    def clear_all_conversations(self):
        """
        Clear all conversations.
        """

        self.memory.clear_all()


# ============================================================
# SHARED AGENT INSTANCE
# ============================================================

_agent_instance: Optional[
    TravelPolicyAgent
] = None


# ============================================================
# GET SHARED AGENT
# ============================================================

def get_agent() -> TravelPolicyAgent:
    """
    Return shared TravelPolicyAgent instance.

    A single instance is used so that conversation memory
    remains available across Flask/API requests.
    """

    global _agent_instance

    if _agent_instance is None:

        _agent_instance = TravelPolicyAgent()

    return _agent_instance


# ============================================================
# MODULE-LEVEL ASK WRAPPER
# ============================================================

def ask(
    question: str,
    conversation_id: str = "default"
):
    """
    Module-level wrapper used by Flask and APIs.
    """

    agent = get_agent()

    return agent.ask(
        question=question,
        conversation_id=conversation_id
    )


# ============================================================
# CLEAR ONE CONVERSATION
# ============================================================

def clear_memory(
    conversation_id: str = "default"
):
    """
    Clear one conversation.
    """

    agent = get_agent()

    agent.clear_conversation(
        conversation_id=conversation_id
    )


# ============================================================
# CLEAR ALL CONVERSATIONS
# ============================================================

def clear_all_memory():
    """
    Clear all conversations.
    """

    agent = get_agent()

    agent.clear_all_conversations()


# ============================================================
# TEST HELPERS
# ============================================================

def print_result(
    result: Dict[str, Any],
    show_tool_results: bool = False
):
    """
    Pretty-print agent result for local testing.
    """

    print("\nANSWER:")

    print(
        result.get(
            "answer"
        )
    )

    print("\nTOOLS USED:")

    print(
        result.get(
            "tools_used"
        )
    )

    print("\nSOURCES:")

    print(
        result.get(
            "sources"
        )
    )

    if result.get(
        "out_of_scope"
    ):

        print("\nSCOPE:")

        print(
            result.get(
                "scope"
            )
        )

    if show_tool_results:

        print("\nTOOL RESULTS:")

        print(
            json.dumps(
                result.get(
                    "tool_results",
                    []
                ),
                indent=2,
                default=str
            )
        )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("TRAVEL POLICY AGENT + MEMORY TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Initialize agent
    # --------------------------------------------------------

    agent = TravelPolicyAgent()

    # ========================================================
    # TEST 1 — POLICY SEARCH
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 1 — POLICY SEARCH")
    print("-" * 70)

    question = (
        "What is the airport reimbursement limit in India?"
    )

    print("\nQUESTION:")
    print(question)

    result = agent.ask(
        question,
        conversation_id="test_policy"
    )

    print_result(
        result
    )

    # ========================================================
    # TEST 2 — EMPLOYEE ELIGIBILITY
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 2 — EMPLOYEE ELIGIBILITY")
    print("-" * 70)

    question = (
        "Is EMP001 eligible for travel?"
    )

    print("\nQUESTION:")
    print(question)

    result = agent.ask(
        question,
        conversation_id="test_eligibility"
    )

    print_result(
        result
    )

    # ========================================================
    # TEST 3 — APPROVED TRIP
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 3 — APPROVED TRIP")
    print("-" * 70)

    question = (
        "Can EMP001 take an airport trip costing INR 1500?"
    )

    print("\nQUESTION:")
    print(question)

    result = agent.ask(
        question,
        conversation_id="test_validation"
    )

    print_result(
        result,
        show_tool_results=True
    )

    # ========================================================
    # TEST 4 — NEEDS APPROVAL
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 4 — COMPLEX TRIP VALIDATION")
    print("-" * 70)

    question = (
        "Can EMP001 claim INR 2500 for an airport trip at 22:30?"
    )

    print("\nQUESTION:")
    print(question)

    result = agent.ask(
        question,
        conversation_id="test_complex_validation"
    )

    print_result(
        result,
        show_tool_results=True
    )

    # ========================================================
    # TEST 5 — CONVERSATIONAL MEMORY
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 5 — CONVERSATIONAL MEMORY")
    print("-" * 70)

    conversation_id = "employee_001"

    question_1 = (
        "What is the airport reimbursement limit in India?"
    )

    print("\nUSER:")
    print(question_1)

    result_1 = agent.ask(
        question_1,
        conversation_id=conversation_id
    )

    print("\nASSISTANT:")

    print(
        result_1.get(
            "answer"
        )
    )

    print("\nTOOLS USED:")

    print(
        result_1.get(
            "tools_used"
        )
    )

    question_2 = (
        "What about the US?"
    )

    print("\nUSER:")
    print(question_2)

    result_2 = agent.ask(
        question_2,
        conversation_id=conversation_id
    )

    print("\nASSISTANT:")

    print(
        result_2.get(
            "answer"
        )
    )

    print("\nTOOLS USED:")

    print(
        result_2.get(
            "tools_used"
        )
    )

    print("\nSOURCES:")

    print(
        result_2.get(
            "sources"
        )
    )

    # ========================================================
    # TEST 6 — MISSING INFORMATION
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 6 — MISSING INFORMATION")
    print("-" * 70)

    question = (
        "What information is required to validate an employee trip?"
    )

    print("\nQUESTION:")
    print(question)

    result = agent.ask(
        question,
        conversation_id="test_missing_information"
    )

    print_result(
        result
    )

    # ========================================================
    # TEST 7 — CONTEXT SAFETY
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 7 — CONTEXT SAFETY")
    print("-" * 70)

    context_test_id = "context_safety"

    print("\n7A — Employee-specific question")

    question = (
        "Is EMP001 eligible for travel?"
    )

    result = agent.ask(
        question,
        context_test_id
    )

    print("ANSWER:")

    print(
        result["answer"]
    )

    print("\n7B — General question after employee-specific question")

    question = (
        "How do I check whether an employee is eligible?"
    )

    result = agent.ask(
        question,
        context_test_id
    )

    print("ANSWER:")

    print(
        result["answer"]
    )

    # ========================================================
    # TEST 8 — MISSING REQUIRED INFORMATION
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 8 — MISSING REQUIRED INFORMATION")
    print("-" * 70)

    missing_test_id = "missing_information"

    print("\n8A — Missing amount")

    question = (
        "Can EMP001 take an airport trip?"
    )

    result = agent.ask(
        question,
        missing_test_id
    )

    print("ANSWER:")

    print(
        result["answer"]
    )

    print("\n8B — Missing employee ID")

    question = (
        "Can I take an airport trip costing INR 1500?"
    )

    result = agent.ask(
        question,
        missing_test_id
    )

    print("ANSWER:")

    print(
        result["answer"]
    )

    # ========================================================
    # TEST 9 — PRONOUN / FOLLOW-UP CONTEXT
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 9 — FOLLOW-UP CONTEXT")
    print("-" * 70)

    followup_test_id = "followup_context"

    print("\n9A — Establish employee context")

    question = (
        "Tell me whether EMP001 is eligible for travel."
    )

    result = agent.ask(
        question,
        followup_test_id
    )

    print("ANSWER:")

    print(
        result["answer"]
    )

    print("\n9B — Pronoun follow-up")

    question = (
        "Is he eligible?"
    )

    result = agent.ask(
        question,
        followup_test_id
    )

    print("ANSWER:")

    print(
        result["answer"]
    )

    # ========================================================
    # TEST 10 — NEW REQUEST MUST NOT INHERIT OLD AMOUNT
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 10 — STALE AMOUNT PROTECTION")
    print("-" * 70)

    stale_amount_id = "stale_amount"

    print("\n10A — Establish previous amount")

    question = (
        "Can EMP001 take an airport trip costing INR 2500?"
    )

    result = agent.ask(
        question,
        stale_amount_id
    )

    print("ANSWER:")

    print(
        result["answer"]
    )

    print("\n10B — New request without amount")

    question = (
        "Can EMP001 take an airport trip?"
    )

    result = agent.ask(
        question,
        stale_amount_id
    )

    print("ANSWER:")

    print(
        result["answer"]
    )

    # ========================================================
    # TEST 11 — OUT OF SCOPE
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 11 — OUT OF SCOPE")
    print("-" * 70)

    out_of_scope_test_id = "out_of_scope"

    questions = [
        "Who won the IPL?",
        "What is the weather today?",
        "Tell me a joke.",
        "Write a Python program to sort a list."
    ]

    for question in questions:

        print("\nQUESTION:")
        print(question)

        result = agent.ask(
            question,
            out_of_scope_test_id
        )

        print("\nANSWER:")
        print(
            result.get(
                "answer"
            )
        )

        print("\nTOOLS USED:")
        print(
            result.get(
                "tools_used"
            )
        )

        print("\nOUT OF SCOPE:")
        print(
            result.get(
                "out_of_scope",
                False
            )
        )

    # ========================================================
    # TEST 12 — CURRENT AMOUNT OVERRIDES OLD AMOUNT
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 12 — CURRENT AMOUNT PRIORITY")
    print("-" * 70)

    amount_priority_id = "amount_priority"

    print("\n12A — First request")

    question = (
        "Can EMP001 take an airport trip costing INR 1500?"
    )

    result = agent.ask(
        question,
        amount_priority_id
    )

    print("ANSWER:")

    print(
        result.get(
            "answer"
        )
    )

    print("\n12B — New request with different amount")

    question = (
        "Can EMP001 take an airport trip costing INR 2500?"
    )

    result = agent.ask(
        question,
        amount_priority_id
    )

    print("ANSWER:")

    print(
        result.get(
            "answer"
        )
    )

    print("\nTOOL RESULTS:")

    print(
        json.dumps(
            result.get(
                "tool_results",
                []
            ),
            indent=2,
            default=str
        )
    )

    # ========================================================
    # TEST 13 — EMPLOYEE CONTEXT SAFETY
    # ========================================================

    print("\n")
    print("-" * 70)
    print("TEST 13 — EMPLOYEE CONTEXT SAFETY")
    print("-" * 70)

    employee_context_id = "employee_context_safety"

    print("\n13A — Establish EMP001")

    question = (
        "Is EMP001 eligible for travel?"
    )

    result = agent.ask(
        question,
        employee_context_id
    )

    print("ANSWER:")

    print(
        result.get(
            "answer"
        )
    )

    print("\n13B — New independent request")

    question = (
        "Is the employee eligible for travel?"
    )

    result = agent.ask(
        question,
        employee_context_id
    )

    print("ANSWER:")

    print(
        result.get(
            "answer"
        )
    )

    print("\nTOOLS USED:")

    print(
        result.get(
            "tools_used"
        )
    )

    # ========================================================
    # FINAL MEMORY VERIFICATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL MEMORY VERIFICATION")
    print("=" * 70)

    print("\nContext safety memory:")

    agent.memory.print_history(
        context_test_id
    )

    print("\nFollow-up memory:")

    agent.memory.print_history(
        followup_test_id
    )

    print("\nClearing test conversations...")

    test_conversations = [
        context_test_id,
        missing_test_id,
        followup_test_id,
        stale_amount_id,
        out_of_scope_test_id,
        amount_priority_id,
        employee_context_id,
        "test_policy",
        "test_eligibility",
        "test_validation",
        "test_complex_validation",
        "test_missing_information",
        "employee_001"
    ]

    for conversation_id in test_conversations:

        agent.memory.clear(
            conversation_id
        )

    print("\nMemory after clear:")

    for conversation_id in test_conversations:

        print(
            conversation_id,
            ":",
            agent.memory.get_history(
                conversation_id
            )
        )

    print("\n")
    print("=" * 70)
    print("ALL AGENT + MEMORY + SCOPE TESTS COMPLETED")
    print("=" * 70)