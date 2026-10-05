from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_hybrid_agent(
    retriever,
    skill_instructions: str,
    anthropic_api_key: str,
    anthropic_workspace_id: str = None,
    tavily_api_key: str = None
):

    @tool
    def search_my_documents(query: str) -> str:
        """
        חובה להשתמש בכלי זה ראשון!
        מחפש מידע בתוך המסמכים שהמשתמש העלה למערכת.
        """
        results = retriever.invoke(query)

        if not results:
            return "לא נמצא שום מידע רלוונטי במסמכים שהועלו."

        # PyPDFLoader שם metadata['page'] (0-indexed) על כל קטע,
        # כך שהסוכן יכול לצטט מספר עמוד מדויק בתשובה שלו.
        formatted = []

        for doc in results:
            page = doc.metadata.get("page")
            page_label = f"עמוד {page + 1}" if page is not None else "עמוד לא ידוע"
            file_name = doc.metadata.get("file_name")
            source_label = f"{file_name}, {page_label}" if file_name else page_label
            formatted.append(f"[{source_label}]:\n{doc.page_content}")

        return "\n\n".join(formatted)

    tools = [search_my_documents]

    # Tavily רק אם קיים API Key
    if tavily_api_key:
        tools.append(
            TavilySearchResults(
                max_results=3,
                tavily_api_key=tavily_api_key
            )
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", skill_instructions),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # ==========================
    # Claude
    # ==========================

    llm_kwargs = {
        # "claude-sonnet-4-6" לא קיים - זה גרם לשגיאות API בלתי צפויות.
        # claude-sonnet-5-5 הוא המודל הנוכחי המומלץ (מהירות + איכות).
        "model": "claude-sonnet-4-6",
        "temperature": 0,
        "anthropic_api_key": anthropic_api_key,
    }

    # Workspace רק אם קיים
    if anthropic_workspace_id:
        llm_kwargs["default_headers"] = {
            "anthropic-workspace-id": anthropic_workspace_id
        }

    llm = ChatAnthropic(**llm_kwargs)

    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )