import os
from urllib.parse import urlsplit

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
STATUS_MESSAGES = {
    "not_configured": (
        "FAQ-only mode",
        "Add SERPAPIKEY to the API's .env file to enable live search.",
        "warning",
    ),
    "configured_not_tested": (
        "Search ready to test",
        "SerpAPI key is configured. Live access is checked when you ask a question.",
        "info",
    ),
    "searching": ("Searching trusted sources", "SerpAPI request in progress.", "info"),
    "connected": (
        "Live search connected",
        "Current results are being retrieved from selected health publishers.",
        "success",
    ),
    "certificate_error": (
        "Network is blocking live search",
        "The SerpAPI TLS certificate could not be verified. A network proxy or "
        "certificate policy may be blocking it; do not disable certificate checks.",
        "error",
    ),
    "network_error": (
        "SerpAPI cannot be reached",
        "Check network access to serpapi.com and retry.",
        "error",
    ),
    "http_401": (
        "SerpAPI rejected the key",
        "Check the SERPAPIKEY value in the API's .env file. Azure keys are not SerpAPI keys.",
        "error",
    ),
    "http_403": (
        "SerpAPI access was denied",
        "Check SerpAPI account access and network policy.",
        "error",
    ),
    "http_429": (
        "SerpAPI quota or rate limit reached",
        "Check the SerpAPI account quota and retry later.",
        "warning",
    ),
    "provider_error": (
        "SerpAPI returned an error",
        "Check SerpAPI account access and quota.",
        "error",
    ),
    "invalid_response": (
        "SerpAPI returned an invalid response",
        "Retry shortly. If the issue continues, check the API service.",
        "error",
    ),
}

st.set_page_config(
    page_title="CareGuide | Health Knowledge",
    page_icon="✚",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    :root {
        --navy: #14324a;
        --teal: #087e83;
        --teal-light: #e6f5f3;
        --ink: #243746;
        --muted: #627586;
        --paper: #f3f7fa;
        --line: #dce6ed;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 7rem; }
    h1, h2, h3 { color: var(--navy); letter-spacing: -0.025em; }
    .care-hero {
        padding: 1.5rem 1.7rem;
        margin: 0 0 1rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: linear-gradient(120deg, #ffffff 0%, #eaf6f5 100%);
        box-shadow: 0 8px 24px rgba(20, 50, 74, 0.06);
    }
    .care-eyebrow {
        color: var(--teal);
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .care-hero h1 { margin: 0.35rem 0 0.4rem; font-size: 2rem; }
    .care-hero p { color: var(--muted); margin: 0; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    div[data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 4px 14px rgba(20, 50, 74, 0.04);
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stChatMessage"] p { color: var(--ink); line-height: 1.65; }
    div[data-testid="stChatInput"] textarea {
        border: 1px solid #c9d9e2;
        border-radius: 14px;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: var(--teal);
        box-shadow: 0 0 0 2px rgba(8, 126, 131, 0.12);
    }
    .stLinkButton a, .stButton button {
        border-radius: 10px;
        border-color: #cbdde3;
    }
    small, .stCaption { color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="care-hero">
        <div class="care-eyebrow">Trusted health information</div>
        <h1>CareGuide</h1>
        <p>Clear, source-linked answers for general health questions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    "Educational information only—not a diagnosis or treatment plan. "
    "Do not enter identifying or highly sensitive details. For personal care, "
    "talk with a qualified clinician; contact local emergency services for emergencies."
)


def show_search_status():
    try:
        response = requests.get(f"{API}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        status = health.get(
            "web_search_status",
            "configured_not_tested"
            if health.get("web_search_configured")
            else "not_configured",
        )
    except (requests.RequestException, ValueError):
        st.warning(
            "Can't reach the API. Start it with "
            "`uvicorn api.main:app --reload` in the project environment."
        )
        return

    title, detail, level = STATUS_MESSAGES.get(
        status,
        ("Search status unknown", "Check the API logs and try again.", "warning"),
    )
    message = f"**{title}.** {detail}"
    if level == "success":
        st.success(message)
    elif level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.caption(message)


def show_sources(sources):
    unique_sources = []
    seen_urls = set()
    for source in sources:
        citation, separator, url = source.partition(": ")
        if not separator:
            continue
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or url in seen_urls:
            continue
        seen_urls.add(url)
        unique_sources.append((f"{citation} · {parsed.hostname}", url))

    if unique_sources:
        st.caption("Sources")
        columns = st.columns(min(2, len(unique_sources)))
        for index, (label, url) in enumerate(unique_sources):
            with columns[index % len(columns)]:
                st.link_button(
                    label,
                    url,
                    use_container_width=True,
                )


status_slot = st.empty()
with status_slot.container():
    show_search_status()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])
        if message.get("context"):
            with st.expander("View retrieved evidence"):
                for item in message["context"]:
                    st.write(item)

question = st.chat_input("Ask a general health question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching trusted sources and your FAQ library..."):
            try:
                response = requests.post(
                    f"{API}/query",
                    json={
                        "query": question,
                        "session_id": st.session_state.get("session_id"),
                    },
                    timeout=90,
                )
                response.raise_for_status()
                data = response.json()
                st.session_state.session_id = data["session_id"]
                st.markdown(data["answer"])
                show_sources(data["sources"])
                with st.expander("View retrieved evidence"):
                    for item in data["retrieved_context"]:
                        st.write(item)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"],
                        "context": data["retrieved_context"],
                    }
                )

                first_column, second_column = st.columns(2)
                for label, rating, column in [
                    ("Helpful", 1, first_column),
                    ("Needs improvement", -1, second_column),
                ]:
                    with column:
                        if st.button(
                            label, key=f"{data['session_id']}-feedback-{rating}"
                        ):
                            feedback_response = requests.post(
                                f"{API}/feedback",
                                json={
                                    "session_id": data["session_id"],
                                    "query": question,
                                    "rating": rating,
                                },
                                timeout=15,
                            )
                            feedback_response.raise_for_status()
                            st.toast("Thanks—feedback recorded.")
            except requests.HTTPError as exc:
                detail = "The API could not answer this question."
                try:
                    detail = exc.response.json().get("detail", detail)
                except (AttributeError, ValueError):
                    pass
                st.error(f"Request failed: {detail}")
            except requests.RequestException:
                st.error(
                    "Can't connect to the healthcare API. Confirm it is running "
                    "at the configured API_URL, then retry."
                )

    with status_slot.container():
        show_search_status()
