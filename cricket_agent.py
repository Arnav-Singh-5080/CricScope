import os
import streamlit as st

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


SYSTEM_PROMPT = """You are CricScope's AI Cricket Assistant — a knowledgeable, 
passionate, and articulate expert on all things cricket.

YOUR EXPERTISE COVERS:
- Match formats: Test, ODI, T20I, IPL, domestic leagues (BBL, PSL, CPL, etc.)
- Rules & regulations: LBW, DRS, DLS method, powerplay rules, super over, etc.
- Player profiles & career analysis: batting/bowling styles, strengths, records
- Team strategies: field placements, batting orders, death bowling, etc.
- Tournament history: World Cups, IPL seasons, Ashes, Border-Gavaskar Trophy
- Statistical analysis & comparisons between players, teams, and eras
- Cricket terminology: yorker, googly, reverse sweep, doosra, carrom ball, etc.
- Current cricket landscape: franchises, rankings, recent form (up to your knowledge cutoff)

YOUR PERSONALITY:
- Enthusiastic but precise — you love cricket and it shows
- Use cricket terminology naturally, but explain jargon when needed
- Give opinions when asked (e.g. GOAT debates) but back them with stats/reasoning
- Keep responses conversational and engaging, not dry like a Wikipedia article
- Use formatting (bold names, line breaks) to make stats and comparisons readable

YOUR LIMITATIONS (be honest about these):
- You don't have access to real-time live scores or today's match results
- Your knowledge has a cutoff date — for very recent matches, tell the user to check live sources
- When unsure about a specific stat, say so rather than guess

RESPONSE STYLE:
- Short questions → concise answers (2-4 sentences)
- Analysis questions → structured with clear sections
- Stat comparisons → use clean formatting with line breaks
- Never use bullet points for simple conversational replies
- Always stay on topic — cricket only. If asked about unrelated topics, 
  politely redirect back to cricket."""


@st.cache_resource
def get_chain():
    """
    Build and cache the LCEL chain:
        prompt | llm | output_parser

    Cached so the LLM isn't re-instantiated on every Streamlit rerun.
    """

    if ChatGroq is None:
        raise RuntimeError(
            "langchain-groq is not installed.\n" "Run:\n" "pip install langchain-groq"
        )
    try:
        api_key = st.secrets["groq"]["key"]
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        st.error(" Groq API key not found.")

        st.info(
            "To enable the chatbot, either:\n\n"
            "**Option A —** create a `.streamlit/secrets.toml` file in the project root with:\n"
            '```toml\n[groq]\nkey = "YOUR_GROQ_API_KEY"\n```\n\n'
            "**Option B —** create a `.env` file in the project root with:\n"
            "```\nGROQ_API_KEY=YOUR_GROQ_API_KEY\n```\n\n"
            "Get your free API key at https://console.groq.com"
        )

        st.stop()

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain


def _build_history(chat_history: list[dict], window: int = 6) -> list:
    """
    Convert CricScope's session_state format:
        [{"role": "user"|"assistant", "content": "..."}]
    → LangChain HumanMessage / AIMessage objects.

    Only the last `window` turns are kept to avoid token overflow.
    """
    # Take the last N messages (window * 2 covers N full turns)
    recent = chat_history[-(window * 2) :]

    lc_messages = []
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    return lc_messages


def run_agent(user_message: str, chat_history: list[dict]) -> str:
    """
    Run the conversational chain for a single user turn.

    Args:
        user_message:  The latest message from the user.
        chat_history:  Full st.session_state.chat_messages list
                       (excluding the current user message — passed separately).

    Returns:
        The assistant's response as a plain string.
    """
    chain = get_chain()

    history = _build_history(chat_history)

    try:
        response = chain.invoke(
            {
                "history": history,
                "question": user_message,
            }
        )
        return response

    except Exception as e:
        err = str(e).lower()

        # Friendly error messages for common failure modes
        if "api_key" in err or "authentication" in err or "401" in err:
            return "⚠️ Invalid Groq API key."
        elif "rate" in err or "429" in err:
            return "⚠️ Groq rate limit hit. Wait a moment and try again."
        elif "timeout" in err:
            return "⚠️ Request timed out. Please try again."
        else:
            return f"⚠️ Something went wrong: {str(e)}"
