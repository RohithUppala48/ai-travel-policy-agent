"""
Conversation Memory
===================

Memory layer for the AI Travel & Policy Assistant.

Responsibilities:
1. Store conversation history
2. Retrieve previous messages
3. Build conversation context
4. Support follow-up questions
5. Clear conversation history

Example:

User:
    Can I take an airport trip?

Assistant:
    Yes, airport travel is covered by the policy.

User:
    What if it costs INR 2500?

The memory layer allows the system to understand that
"it" refers to the airport trip.
"""

from typing import List, Dict, Optional


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Simple in-memory conversation history.

    Each conversation is stored using a conversation_id.

    Example structure:

    {
        "default": [
            {
                "role": "user",
                "content": "Can I take an airport trip?"
            },
            {
                "role": "assistant",
                "content": "Yes..."
            }
        ]
    }
    """

    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation memory.

        Args:
            max_messages:
                Maximum number of messages stored for
                each conversation.
        """

        self.max_messages = max_messages

        self.conversations: Dict[
            str,
            List[Dict[str, str]]
        ] = {}

    # ========================================================
    # CREATE CONVERSATION
    # ========================================================

    def create_conversation(
        self,
        conversation_id: str = "default"
    ):
        """
        Create a conversation if it does not already exist.
        """

        if conversation_id not in self.conversations:

            self.conversations[
                conversation_id
            ] = []

    # ========================================================
    # ADD MESSAGE
    # ========================================================

    def add_message(
        self,
        role: str,
        content: str,
        conversation_id: str = "default"
    ):
        """
        Add a message to conversation history.

        Args:
            role:
                user or assistant

            content:
                Message text

            conversation_id:
                Unique conversation identifier
        """

        # ----------------------------------------------------
        # Validate role
        # ----------------------------------------------------

        if role not in {
            "user",
            "assistant",
            "system"
        }:

            raise ValueError(
                "Role must be 'user', 'assistant', "
                "or 'system'."
            )

        # ----------------------------------------------------
        # Validate content
        # ----------------------------------------------------

        if content is None:

            return

        content = str(content).strip()

        if not content:

            return

        # ----------------------------------------------------
        # Create conversation if required
        # ----------------------------------------------------

        self.create_conversation(
            conversation_id
        )

        # ----------------------------------------------------
        # Add message
        # ----------------------------------------------------

        self.conversations[
            conversation_id
        ].append({
            "role": role,
            "content": content
        })

        # ----------------------------------------------------
        # Limit conversation size
        # ----------------------------------------------------

        if len(
            self.conversations[conversation_id]
        ) > self.max_messages:

            self.conversations[
                conversation_id
            ] = self.conversations[
                conversation_id
            ][-self.max_messages:]

    # ========================================================
    # ADD USER MESSAGE
    # ========================================================

    def add_user_message(
        self,
        content: str,
        conversation_id: str = "default"
    ):
        """
        Convenience method for user messages.
        """

        self.add_message(
            role="user",
            content=content,
            conversation_id=conversation_id
        )

    # ========================================================
    # ADD ASSISTANT MESSAGE
    # ========================================================

    def add_assistant_message(
        self,
        content: str,
        conversation_id: str = "default"
    ):
        """
        Convenience method for assistant messages.
        """

        self.add_message(
            role="assistant",
            content=content,
            conversation_id=conversation_id
        )

    # ========================================================
    # GET HISTORY
    # ========================================================

    def get_history(
        self,
        conversation_id: str = "default"
    ) -> List[Dict[str, str]]:
        """
        Return conversation history.

        Returns a copy so external code cannot accidentally
        modify the internal memory.
        """

        if conversation_id not in self.conversations:

            return []

        return list(
            self.conversations[
                conversation_id
            ]
        )

    # ========================================================
    # GET LAST MESSAGE
    # ========================================================

    def get_last_message(
        self,
        conversation_id: str = "default"
    ) -> Optional[Dict[str, str]]:
        """
        Return the most recent message.
        """

        history = self.get_history(
            conversation_id
        )

        if not history:

            return None

        return history[-1]

    # ========================================================
    # GET LAST USER MESSAGE
    # ========================================================

    def get_last_user_message(
        self,
        conversation_id: str = "default"
    ) -> Optional[str]:
        """
        Return the most recent user message.
        """

        history = self.get_history(
            conversation_id
        )

        for message in reversed(history):

            if message.get("role") == "user":

                return message.get(
                    "content",
                    ""
                )

        return None

    # ========================================================
    # GET LAST ASSISTANT MESSAGE
    # ========================================================

    def get_last_assistant_message(
        self,
        conversation_id: str = "default"
    ) -> Optional[str]:
        """
        Return the most recent assistant message.
        """

        history = self.get_history(
            conversation_id
        )

        for message in reversed(history):

            if message.get("role") == "assistant":

                return message.get(
                    "content",
                    ""
                )

        return None

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(
        self,
        conversation_id: str = "default"
    ) -> str:
        """
        Convert conversation history into readable
        text context for the LLM.
        """

        history = self.get_history(
            conversation_id
        )

        if not history:

            return ""

        context_lines = []

        for message in history:

            role = message.get(
                "role",
                "unknown"
            )

            content = message.get(
                "content",
                ""
            )

            if role == "user":

                context_lines.append(
                    f"User: {content}"
                )

            elif role == "assistant":

                context_lines.append(
                    f"Assistant: {content}"
                )

            elif role == "system":

                context_lines.append(
                    f"System: {content}"
                )

        return "\n".join(
            context_lines
        )

    # ========================================================
    # GET MESSAGE COUNT
    # ========================================================

    def get_message_count(
        self,
        conversation_id: str = "default"
    ) -> int:
        """
        Return the number of stored messages.
        """

        return len(
            self.get_history(
                conversation_id
            )
        )

    # ========================================================
    # CLEAR CONVERSATION
    # ========================================================

    def clear(
        self,
        conversation_id: str = "default"
    ):
        """
        Clear a specific conversation.
        """

        self.conversations[
            conversation_id
        ] = []

    # ========================================================
    # CLEAR ALL CONVERSATIONS
    # ========================================================

    def clear_all(self):
        """
        Clear all conversations.
        """

        self.conversations = {}

    # ========================================================
    # CHECK WHETHER CONVERSATION EXISTS
    # ========================================================

    def exists(
        self,
        conversation_id: str = "default"
    ) -> bool:
        """
        Check whether a conversation exists.
        """

        return (
            conversation_id in self.conversations
            and len(
                self.conversations[
                    conversation_id
                ]
            ) > 0
        )

    # ========================================================
    # PRINT HISTORY
    # ========================================================

    def print_history(
        self,
        conversation_id: str = "default"
    ):
        """
        Print conversation history.
        """

        history = self.get_history(
            conversation_id
        )

        print("\n")
        print("=" * 70)
        print(
            f"CONVERSATION: {conversation_id}"
        )
        print("=" * 70)

        if not history:

            print("No conversation history.")

            return

        for index, message in enumerate(
            history,
            start=1
        ):

            role = message.get(
                "role",
                "unknown"
            )

            content = message.get(
                "content",
                ""
            )

            print(
                f"{index}. {role.upper()}: "
                f"{content}"
            )

        print("=" * 70)


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("CONVERSATION MEMORY TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Create memory
    # --------------------------------------------------------

    memory = ConversationMemory(
        max_messages=10
    )

    conversation_id = "employee_001"

    # --------------------------------------------------------
    # Add first conversation
    # --------------------------------------------------------

    memory.add_user_message(
        "Can I take an airport trip?",
        conversation_id
    )

    memory.add_assistant_message(
        "Yes, airport travel is covered under the "
        "company travel policy.",
        conversation_id
    )

    # --------------------------------------------------------
    # Add follow-up
    # --------------------------------------------------------

    memory.add_user_message(
        "What if it costs INR 2500?",
        conversation_id
    )

    memory.add_assistant_message(
        "The India airport reimbursement limit is "
        "INR 2000, so INR 500 requires review.",
        conversation_id
    )

    # --------------------------------------------------------
    # Print history
    # --------------------------------------------------------

    memory.print_history(
        conversation_id
    )

    # --------------------------------------------------------
    # Test history
    # --------------------------------------------------------

    print("\nMESSAGE COUNT:")

    print(
        memory.get_message_count(
            conversation_id
        )
    )

    # --------------------------------------------------------
    # Test last user message
    # --------------------------------------------------------

    print("\nLAST USER MESSAGE:")

    print(
        memory.get_last_user_message(
            conversation_id
        )
    )

    # --------------------------------------------------------
    # Test last assistant message
    # --------------------------------------------------------

    print("\nLAST ASSISTANT MESSAGE:")

    print(
        memory.get_last_assistant_message(
            conversation_id
        )
    )

    # --------------------------------------------------------
    # Test context
    # --------------------------------------------------------

    print("\nCONVERSATION CONTEXT:")

    print(
        memory.build_context(
            conversation_id
        )
    )

    # --------------------------------------------------------
    # Test existence
    # --------------------------------------------------------

    print("\nCONVERSATION EXISTS:")

    print(
        memory.exists(
            conversation_id
        )
    )

    # --------------------------------------------------------
    # Test clear
    # --------------------------------------------------------

    memory.clear(
        conversation_id
    )

    print("\nAFTER CLEAR:")

    print(
        memory.get_history(
            conversation_id
        )
    )

    print("\n")
    print("=" * 70)
    print("MEMORY TEST COMPLETED")
    print("=" * 70)
