"""
Draftwise — LinkedIn post writer + reviewer loop (LangGraph + Streamlit)

Run:   streamlit run app.py
Keys:  entered in the UI, verified on submit, kept only in the session.
Needs: streamlit>=1.40 (>=1.46 lets the app open in your system theme), requests, reportlab,
       langgraph, langchain-core, langchain-openai, langchain-tavily
       pip install -U streamlit requests reportlab langgraph langchain-core langchain-openai langchain-tavily

Every provider below is reached through its OpenAI-compatible endpoint, so no extra SDKs are needed.
"""

import html
import io
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Annotated, TypedDict
from xml.sax.saxutils import escape as xml_escape

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Draftwise · LinkedIn post studio",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════════
#  PROVIDER CATALOG  (all OpenAI-compatible; model IDs rotate — use "Custom" for anything else)
# ════════════════════════════════════════════════════════════════════════

CUSTOM = "Custom model ID…"

PROVIDERS = {
    "openai": dict(
        label="OpenAI", tag="Paid", free=False, base=None, ph="sk-…",
        url="https://platform.openai.com/api-keys",
        blurb="Most dependable tool-calling. Pay as you go.",
        models=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"],
    ),
    "gemini": dict(
        label="Google Gemini", tag="Free tier", free=True,
        base="https://generativelanguage.googleapis.com/v1beta/openai/", ph="AIza…",
        url="https://aistudio.google.com/apikey",
        blurb="Generous free quota through Google AI Studio.",
        models=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    ),
    "groq": dict(
        label="Groq", tag="Free tier", free=True, base="https://api.groq.com/openai/v1", ph="gsk_…",
        url="https://console.groq.com/keys",
        blurb="Very fast inference. Free tier with rate limits.",
        models=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    ),
    "openrouter": dict(
        label="OpenRouter", tag="Free models", free=True, base="https://openrouter.ai/api/v1", ph="sk-or-…",
        url="https://openrouter.ai/keys",
        blurb="One key, many models. Pick IDs ending in :free.",
        models=["openai/gpt-oss-20b:free", "meta-llama/llama-3.3-70b-instruct:free"],
    ),
    "mistral": dict(
        label="Mistral", tag="Free tier", free=True, base="https://api.mistral.ai/v1", ph="Paste key",
        url="https://console.mistral.ai/api-keys",
        blurb="Free Experiment plan for prototyping.",
        models=["mistral-small-latest", "mistral-large-latest", "open-mistral-nemo"],
    ),
    "cerebras": dict(
        label="Cerebras", tag="Free tier", free=True, base="https://api.cerebras.ai/v1", ph="csk-…",
        url="https://cloud.cerebras.ai",
        blurb="Extremely fast open models. Free tier available.",
        models=["gpt-oss-120b", "llama-3.3-70b", "qwen-3-32b"],
    ),
}
TAVILY = dict(label="Tavily", ph="tvly-…", url="https://app.tavily.com")
DEFAULT_PROV = {"w": "openai", "r": "groq"}


def key_label(kid: str) -> str:
    return f"{(TAVILY if kid == 'tavily' else PROVIDERS[kid])['label']} API key"


def key_ph(kid: str) -> str:
    return (TAVILY if kid == "tavily" else PROVIDERS[kid])["ph"]


def key_url(kid: str) -> str:
    return (TAVILY if kid == "tavily" else PROVIDERS[kid])["url"]


def resolve(role: str):
    """(provider id, model id) currently selected for role 'w' (writer) or 'r' (reviewer)."""
    ss = st.session_state
    pid = ss.get(f"{role}_prov", DEFAULT_PROV[role])
    if pid not in PROVIDERS:
        pid = DEFAULT_PROV[role]
    m = ss.get(f"{role}_model_{pid}", PROVIDERS[pid]["models"][0])
    if m == CUSTOM:
        m = (ss.get(f"{role}_custom_{pid}") or "").strip()
    return pid, m


def research_on() -> bool:
    return st.session_state.get("research") != "Off"


def max_attempts() -> int:
    return st.session_state.get("max_attempts") or 3


def needed_ids() -> list:
    ids = []
    for r in ("w", "r"):
        p = resolve(r)[0]
        if p not in ids:
            ids.append(p)
    if research_on():
        ids.append("tavily")
    return ids


# ════════════════════════════════════════════════════════════════════════
#  BACKEND — LangGraph pipeline (built per session, keys never cached)
# ════════════════════════════════════════════════════════════════════════

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "{search}"
    "If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags. Output only the post text."
)
SEARCH_LINE = (
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. "
)

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)


def _text(content) -> str:
    """Normalise LLM message content (str or list of parts) to plain text."""
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


def make_llm(pid: str, model: str, key: str, temperature: float):
    from langchain_openai import ChatOpenAI

    kw = dict(model=model, api_key=key, temperature=temperature, timeout=90, max_retries=2)
    if PROVIDERS[pid]["base"]:
        kw["base_url"] = PROVIDERS[pid]["base"]
    return ChatOpenAI(**kw)


def build_graph(wp, wm, wkey, rp, rm, rkey, tavily_key=None):
    from langchain_core.messages import HumanMessage, ToolMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    tools = []
    if tavily_key:
        from langchain_tavily import TavilySearch

        tools = [TavilySearch(max_results=5, tavily_api_key=tavily_key)]

    writer_llm = make_llm(wp, wm, wkey, 0.75)
    writer = writer_llm.bind_tools(tools) if tools else writer_llm
    reviewer_llm = make_llm(rp, rm, rkey, 0.25)
    system_prompt = WRITER_SYSTEM_PROMPT.replace("{search}", SEARCH_LINE if tools else "")

    class State(TypedDict):
        topic: str
        tone: str
        messages: Annotated[list, add_messages]
        draft: str
        review_feedback: str
        is_approved: bool
        attempt: int
        max_attempts: int

    def writer_node(state: State) -> dict:
        msgs = state["messages"]

        # Came back from the search tool → finish the SAME draft with the results.
        if msgs and isinstance(msgs[-1], ToolMessage):
            last_human = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
            response = writer.invoke([("system", system_prompt)] + msgs[last_human:])
            return {"messages": [response]}

        attempt = state.get("attempt", 0) + 1
        topic = state["topic"]
        tone = state.get("tone") or "Thoughtful"
        search_hint = "If you need current info, search the web first." if tools else ""

        if attempt == 1:
            user_message = f"Write a LinkedIn post on this topic: {topic}. Tone: {tone}. {search_hint}"
        else:
            user_message = (
                f"Your previous draft on '{topic}' was rejected. "
                f"Here is the reviewer's feedback:\n\n{state['review_feedback']}\n\n"
                f"Tone: {tone}. Write a new, improved draft that fixes every issue "
                f"mentioned. Do not repeat the same mistake."
            )

        response = writer.invoke([("system", system_prompt), ("human", user_message)])
        return {"messages": [("human", user_message), response], "attempt": attempt}

    def extract_draft_node(state: State) -> dict:
        return {"draft": _text(state["messages"][-1].content)}

    def reviewer_node(state: State) -> dict:
        prompt = f"Review this LinkedIn post draft:\n{state['draft']}\nGive your review."
        response = reviewer_llm.invoke([("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)])
        review_text = _text(response.content).strip()
        verdict_part = review_text.upper().split("FEEDBACK")[0]
        is_approved = "APPROVED" in verdict_part and "REJECTED" not in verdict_part
        feedback = (
            review_text.split("FEEDBACK:", 1)[1].strip()
            if "FEEDBACK:" in review_text
            else review_text
        )
        return {"review_feedback": feedback, "is_approved": is_approved}

    def should_use_tool(state: State):
        last = state["messages"][-1]
        return "tools" if tools and getattr(last, "tool_calls", None) else "extract_draft"

    def should_stop_looping(state: State):
        if state["is_approved"] or state["attempt"] >= state.get("max_attempts", 3):
            return END
        return "writer"

    g = StateGraph(State)
    g.add_node("writer", writer_node)
    g.add_node("extract_draft", extract_draft_node)
    g.add_node("reviewer", reviewer_node)
    route = {"extract_draft": "extract_draft"}
    if tools:
        g.add_node("tools", ToolNode(tools))
        g.add_edge("tools", "writer")  # after searching, the writer actually writes the post
        route["tools"] = "tools"

    g.add_edge(START, "writer")
    g.add_conditional_edges("writer", should_use_tool, route)
    g.add_edge("extract_draft", "reviewer")
    g.add_conditional_edges("reviewer", should_stop_looping, {END: END, "writer": "writer"})
    return g.compile()


def get_graph():
    """Build the graph for THIS session only; rebuild if models or keys change."""
    ss = st.session_state
    creds = ss["creds"]
    (wp, wm), (rp, rm) = resolve("w"), resolve("r")
    tav = creds.get("tavily") if research_on() else None
    sig = (wp, wm, creds[wp], rp, rm, creds[rp], tav)
    if ss.get("_graph_sig") != sig:
        ss["_graph"] = build_graph(wp, wm, creds[wp], rp, rm, creds[rp], tav)
        ss["_graph_sig"] = sig
    return ss["_graph"]


def _clear_keys():
    for k in ("creds", "_graph", "_graph_sig"):
        st.session_state.pop(k, None)
    for k in [*PROVIDERS, "tavily"]:
        st.session_state.pop(f"in_{k}", None)


# ── key validation ──────────────────────────────────────────────────────

def _status(url: str, headers: dict, method: str = "GET", **kw):
    """HTTP status code, or None if the service couldn't be reached."""
    try:
        return requests.request(method, url, headers=headers, timeout=12, **kw).status_code
    except requests.RequestException:
        return None


def _verdict(code):
    if code is None:
        return False, "Couldn’t reach the service"
    if code == 200:
        return True, ""
    if code in (400, 401, 403):
        return False, "Key was rejected"
    return False, f"Unexpected response ({code})"


def _check(kid: str, key: str):
    if not key:
        return False, "Missing"
    bearer = {"Authorization": f"Bearer {key}"}
    if kid == "tavily":
        code = _status("https://api.tavily.com/usage", bearer)
        if code not in (200, 401, 403):  # endpoint unavailable → fall back to a 1-result search
            code = _status("https://api.tavily.com/search", bearer, "POST",
                           json={"query": "ping", "max_results": 1})
        return _verdict(code)
    if kid == "gemini":
        return _verdict(_status("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
                                {"x-goog-api-key": key}))
    base = (PROVIDERS[kid]["base"] or "https://api.openai.com/v1").rstrip("/")
    return _verdict(_status(base + ("/key" if kid == "openrouter" else "/models"), bearer))


def validate_keys(keys: dict) -> dict:
    """Returns {id: (ok, message)}. Uses cheap, read-only endpoints, checked in parallel."""
    with ThreadPoolExecutor(max_workers=len(keys) or 1) as ex:
        futs = {k: ex.submit(_check, k, v) for k, v in keys.items()}
        return {k: f.result() for k, f in futs.items()}


# ════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
#  Light  paper #F3F5F9 · surface #FFF · ink #101828 · cobalt #3B5BFF
#  Dark   paper #0A0E17 · surface #111826 · ink #EDF1F9 · cobalt #7089FF
#  Status mint (approved) · rose (sent back) · amber (needs a light edit)
#  Type   Bricolage Grotesque (display) + Figtree (interface)
#  Motion one orchestrated entrance, a slow ambient drift, motion that answers actions
# ════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600;700&display=swap');

:root{
  --paper:#F3F5F9; --surface:#fff; --surface-2:#FAFBFD; --ink:#101828; --ink-2:#475467; --ink-3:#98A2B3;
  --line:#E4E8F0; --dot:#D5DAE5; --cobalt:#3B5BFF; --cobalt-soft:#E8ECFF;
  --mint:#12B76A; --mint-ink:#067647; --mint-soft:#E6F7EE; --mint-line:#B7E8CF;
  --rose:#F04438; --rose-ink:#B42318; --rose-soft:#FEECEB; --rose-line:#FECDCA;
  --amber:#F79009; --amber-ink:#B54708; --amber-soft:#FEF3E2;
  --solid:#101828; --on-solid:#fff; --body:#1D2433; --draft:#344054;
  --skel-a:#EEF1F6; --skel-b:#F8F9FC;
  --glow1:rgba(59,91,255,.15); --glow2:rgba(18,183,106,.09);
  --shadow-1:0 1px 2px rgba(16,24,40,.04);
  --shadow-2:0 1px 2px rgba(16,24,40,.04),0 24px 48px -24px rgba(16,24,40,.16);
  --display:'Bricolage Grotesque',system-ui,sans-serif; --ui:'Figtree',system-ui,sans-serif;
  --ease:cubic-bezier(.22,.8,.3,1);
}

/* ── page chrome ───────────────────────────────── */
.stApp[data-testid="stApp"]{
  background-color:var(--paper)!important;
  background-image:radial-gradient(var(--dot) 1px,transparent 1px);
  background-size:26px 26px; font-family:var(--ui); color:var(--ink);
}
.stApp::before{ /* slow ambient light behind everything */
  content:"";position:fixed;inset:-20%;z-index:0;pointer-events:none;
  background:radial-gradient(38% 32% at 14% 8%,var(--glow1),transparent 70%),
             radial-gradient(34% 30% at 88% 22%,var(--glow2),transparent 70%),
             radial-gradient(40% 34% at 55% 98%,var(--glow1),transparent 70%);
  animation:drift 26s ease-in-out infinite alternate;
}
[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"]{background:transparent!important}
[data-testid="stAppViewContainer"]{position:relative;z-index:1}
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stSidebar"]{display:none!important}
.block-container{max-width:1120px!important;padding:1.6rem 1.5rem 4rem!important}

.stApp :where(p,label,div,span,button,input,textarea,li){font-family:var(--ui)}
/* Streamlit's icons are font ligatures — never override their font */
.stApp [data-testid="stIconMaterial"],.stApp .material-symbols-rounded{font-family:"Material Symbols Rounded"!important}
.stApp .brand,.stApp .notes-h,.stApp .avatar,.stApp .step .n{font-family:var(--display)}
.stApp [data-testid="stWidgetLabel"],.stApp [data-testid="stWidgetLabel"] *,
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:var(--ink)!important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--ink-2)!important}
[data-testid="stToast"]{background:var(--surface)!important;color:var(--ink)!important;border:1px solid var(--line)}
[data-testid="stToast"] *{color:var(--ink)!important}
::selection{background:var(--cobalt);color:#fff}

/* theme change glides instead of snapping */
.post,.note,.pipe,.chip,.st-key-composer,[data-testid="stForm"],.empty-note,.prov,.st-key-mp_w,.st-key-mp_r{
  transition:background-color .45s var(--ease),border-color .45s var(--ease),color .45s var(--ease),box-shadow .45s var(--ease)}

/* ── top bar ───────────────────────────────────── */
.topbar{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:21px;letter-spacing:-.02em;color:var(--ink)}
.brand .mark{width:34px;height:34px;border-radius:10px;background:var(--solid);color:var(--on-solid);display:grid;place-items:center}
.brand .mark svg{width:18px;height:18px;animation:spin 14s linear infinite}
.models{display:flex;gap:8px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:var(--surface);font-size:12.5px;color:var(--ink-2);font-weight:500}
.chip b{font-weight:700;color:var(--ink)}
.chip i{width:7px;height:7px;border-radius:50%;background:var(--cobalt);animation:breathe-dot 2.4s ease-in-out infinite}
.chip i.g{background:var(--mint);animation-delay:.6s}
.st-key-theme_btn{display:flex;justify-content:flex-end}
.st-key-theme_btn button{min-height:0!important;padding:.4rem .95rem!important;gap:.4rem}
.st-key-theme_btn [data-testid="stIconMaterial"]{transition:transform .5s var(--ease)}
.st-key-theme_btn button:hover [data-testid="stIconMaterial"]{transform:rotate(60deg) scale(1.15)}

/* ── hero: the one orchestrated entrance ───────── */
.hero{padding:44px 0 28px;max-width:780px}
.hero h1{font-family:var(--display);font-weight:700;font-size:clamp(38px,6vw,64px);line-height:1.02;letter-spacing:-.035em;margin:0 0 18px;color:var(--ink)}
.hero h1 span{display:block;overflow:hidden;padding-bottom:.06em}
.hero h1 span b{display:block;font-weight:700;animation:lift .9s var(--ease) both}
.hero h1 span:nth-child(2) b{animation-delay:.12s}
.hero p{font-size:18px;line-height:1.55;color:var(--ink-2);margin:0;max-width:600px;animation:fade 1s .45s var(--ease) both}
.hero .rule{height:3px;width:72px;border-radius:3px;background:var(--cobalt);margin:0 0 22px;transform-origin:left;animation:grow-x .9s .3s var(--ease) both}

/* ── composer ──────────────────────────────────── */
.st-key-composer{background:var(--surface);border:1px solid var(--line);border-radius:22px;padding:24px 26px 16px;box-shadow:var(--shadow-2);animation:fade .9s .55s var(--ease) both}
.step{display:flex;gap:12px;align-items:center;margin:0 0 14px}
.step .n{width:26px;height:26px;border-radius:50%;background:var(--solid);color:var(--on-solid);display:grid;place-items:center;font-weight:700;font-size:12.5px;flex:none}
.step b{display:block;font-size:15.5px;font-weight:700;color:var(--ink)}
.step span.d{display:block;font-size:13px;color:var(--ink-2);line-height:1.4}
.sep{height:1px;background:var(--line);margin:22px -26px 20px}
.st-key-composer label p{font-weight:600;font-size:14px}

div[data-baseweb="textarea"]{border-radius:14px!important;border:1px solid var(--line)!important;background:var(--surface-2)!important;transition:border-color .2s,box-shadow .2s,background .2s}
div[data-baseweb="textarea"]:focus-within{border-color:var(--cobalt)!important;box-shadow:0 0 0 4px var(--cobalt-soft)}
div[data-baseweb="textarea"] textarea{background:transparent!important;font-size:16.5px!important;line-height:1.5!important;color:var(--ink)!important;-webkit-text-fill-color:var(--ink)}
div[data-baseweb="input"]{border-radius:12px!important;border:1px solid var(--line)!important;background:var(--surface-2)!important}
div[data-baseweb="input"]>div{background:transparent!important}
div[data-baseweb="input"]:focus-within{border-color:var(--cobalt)!important;box-shadow:0 0 0 4px var(--cobalt-soft)}
div[data-baseweb="input"] input{font-size:15px!important;letter-spacing:.02em;color:var(--ink)!important;-webkit-text-fill-color:var(--ink)}
div[data-baseweb="input"] button{background:transparent!important;color:var(--ink-2)!important}
textarea::placeholder,input::placeholder{color:var(--ink-3)!important;-webkit-text-fill-color:var(--ink-3)!important;opacity:1}
div[data-baseweb="select"]>div{background:var(--surface-2)!important;border:1px solid var(--line)!important;border-radius:12px!important;transition:border-color .2s,box-shadow .2s}
div[data-baseweb="select"]>div:hover,div[data-baseweb="select"]>div:focus-within{border-color:var(--cobalt)!important;box-shadow:0 0 0 4px var(--cobalt-soft)}
div[data-baseweb="select"] *{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)}
div[data-baseweb="select"] svg{fill:var(--ink-2)!important}
div[data-baseweb="popover"]>div,div[data-baseweb="popover"] ul,div[data-baseweb="popover"] [role="listbox"]{background:var(--surface)!important;border-radius:12px;border:1px solid var(--line)}
div[data-baseweb="popover"] li{color:var(--ink)!important;background:transparent!important}
div[data-baseweb="popover"] li:hover,div[data-baseweb="popover"] li[aria-selected="true"]{background:var(--cobalt-soft)!important}
div[data-baseweb="popover"] li *{color:var(--ink)!important}
[data-testid="stExpander"] details{border:1px solid var(--line)!important;border-radius:14px!important;background:var(--surface-2)!important;margin-bottom:6px}
[data-testid="stForm"]{border:1px solid var(--line)!important;border-radius:16px!important;background:var(--surface-2);padding:18px 20px 14px!important;margin-bottom:10px}
.form-title{font-weight:700;font-size:15px;margin:0 0 2px;color:var(--ink)}
.keynote{font-size:13px;color:var(--ink-2);line-height:1.55;margin:2px 0 12px}
.keynote a,.prov a{color:var(--cobalt);text-decoration:none;font-weight:600}
.keynote a:hover,.prov a:hover{text-decoration:underline}

/* model pickers */
.st-key-mp_w,.st-key-mp_r{background:var(--surface-2);border:1px solid var(--line);border-radius:16px;padding:16px 16px 12px}
.role{display:flex;gap:11px;align-items:center;margin-bottom:10px}
.role .ri{width:32px;height:32px;border-radius:10px;display:grid;place-items:center;background:var(--cobalt-soft);color:var(--cobalt);flex:none}
.role .ri.rv{background:var(--mint-soft);color:var(--mint)}
.role .ri svg{width:16px;height:16px}
.role b{display:block;font-size:14.5px;color:var(--ink)}
.role span.d{display:block;font-size:12.5px;color:var(--ink-2)}
.prov{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:2px 0 4px;font-size:12.5px;color:var(--ink-2);animation:fade .4s var(--ease) both}
.tag{font-weight:700;font-size:11.5px;padding:2px 9px;border-radius:999px}
.tag.free{background:var(--mint-soft);color:var(--mint-ink)}.tag.paid{background:var(--amber-soft);color:var(--amber-ink)}

/* buttons */
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{font-family:var(--ui);font-weight:600;border-radius:999px;transition:transform .18s var(--ease),box-shadow .2s,background .2s,border-color .2s,color .2s}
.stApp button p{color:inherit!important}
button[data-testid="stBaseButton-secondary"],button[data-testid="stBaseButton-secondaryFormSubmit"],.stDownloadButton button{background:var(--surface)!important;border:1px solid var(--line)!important;color:var(--ink-2)!important;font-size:13.5px;padding:.35rem .9rem;min-height:0}
button[data-testid="stBaseButton-secondary"]:hover,button[data-testid="stBaseButton-secondaryFormSubmit"]:hover,.stDownloadButton button:hover{border-color:var(--cobalt)!important;color:var(--cobalt)!important;background:var(--cobalt-soft)!important;transform:translateY(-1px)}
button[data-testid="stBaseButton-secondary"]:active,.stDownloadButton button:active{transform:scale(.97)}
button[data-testid="stBaseButton-secondary"]:focus-visible,button[data-testid="stBaseButton-primary"]:focus-visible{outline:3px solid var(--cobalt-soft);outline-offset:2px}
button[data-testid="stBaseButton-primary"]{position:relative;overflow:hidden;background:var(--solid)!important;border:none!important;color:var(--on-solid)!important;font-size:16px;padding:.8rem 1.6rem;border-radius:14px;box-shadow:0 8px 20px -8px rgba(16,24,40,.5)}
button[data-testid="stBaseButton-primary"]::after{content:"";position:absolute;top:0;bottom:0;width:40%;left:-60%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.28),transparent);transform:skewX(-18deg);animation:sheen 4.5s 1.5s ease-in-out infinite}
button[data-testid="stBaseButton-primary"]:hover{background:var(--cobalt)!important;color:#fff!important;transform:translateY(-2px);box-shadow:0 14px 28px -10px rgba(59,91,255,.6)}
button[data-testid="stBaseButton-primary"]:active{transform:scale(.98)}
button[data-testid="stBaseButton-primary"]:disabled{background:var(--line)!important;color:var(--ink-3)!important;box-shadow:none;transform:none}
button[data-testid="stBaseButton-primary"]:disabled::after{display:none}
/* pills & segmented controls */
button[data-testid^="stBaseButton-pills"],button[data-testid^="stBaseButton-segmented_control"]{background:var(--surface-2)!important;border:1px solid var(--line)!important;color:var(--ink-2)!important;border-radius:999px;font-weight:600;transition:all .2s var(--ease)}
button[data-testid^="stBaseButton-pills"]:hover,button[data-testid^="stBaseButton-segmented_control"]:hover{border-color:var(--cobalt)!important;color:var(--cobalt)!important}
button[data-testid$="Active"]{background:var(--cobalt-soft)!important;border-color:var(--cobalt)!important;color:var(--cobalt)!important}

/* ── keys verified ─────────────────────────────── */
.keys-done{display:flex;align-items:center;gap:14px;padding:16px 18px;border-radius:16px;background:var(--mint-soft);
  border:1px solid var(--mint-line);color:var(--mint-ink);overflow:hidden;animation:settle .8s var(--ease) both;margin-bottom:10px}
.keys-done .ic{width:38px;height:38px;border-radius:50%;background:var(--mint);color:#fff;display:grid;place-items:center;flex:none;animation:pop .6s .15s var(--ease) both}
.keys-done .ic svg{width:19px;height:19px}
.keys-done b{display:block;font-size:14.5px;margin-bottom:2px}
.keys-done span.t{display:block;font-size:13px;line-height:1.5;opacity:.92}
.keys-done .ic path{stroke-dasharray:24;stroke-dashoffset:24;animation:draw .5s .45s var(--ease) forwards}
.vchips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.vchip{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:999px;background:var(--surface);border:1px solid var(--mint-line);font-size:12px;font-weight:600;color:var(--mint-ink)}
.vchip svg{width:12px;height:12px}
.keys-wait{display:flex;align-items:center;gap:12px;padding:14px 18px;border-radius:14px;background:var(--cobalt-soft);color:var(--cobalt);font-size:14px;font-weight:600;margin-bottom:10px;animation:rise .3s var(--ease) both}
.keyres{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 10px;animation:rise .4s var(--ease) both}
.keyres span{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;font-size:12.5px;font-weight:600}
.keyres .ok{background:var(--mint-soft);color:var(--mint-ink)}.keyres .no{background:var(--rose-soft);color:var(--rose-ink)}

/* ── agent pipeline ────────────────────────────── */
.pipe{position:relative;overflow:hidden;display:flex;align-items:center;background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:18px 22px;
  box-shadow:var(--shadow-1);margin:30px 0 22px}
.pipe.running::before{content:"";position:absolute;left:0;top:0;height:2px;width:40%;background:linear-gradient(90deg,transparent,var(--cobalt),transparent);animation:sweep 1.5s ease-in-out infinite}
.node{display:flex;align-items:center;gap:12px;min-width:0}
.dot{position:relative;width:42px;height:42px;border-radius:13px;display:grid;place-items:center;border:1.5px solid var(--line);color:var(--ink-3);background:var(--surface);transition:all .35s var(--ease);flex:none}
.dot svg{width:20px;height:20px}
.txt b{display:block;font-size:14px;font-weight:600;color:var(--ink)}
.txt span{display:block;font-size:12.5px;color:var(--ink-3);white-space:nowrap}
.node.active .dot{border-color:var(--cobalt);color:var(--cobalt);background:var(--cobalt-soft);animation:pulse 1.7s ease-out infinite}
.node.active .dot::after{content:"";position:absolute;inset:-6px;border-radius:17px;border:1.5px dashed var(--cobalt);opacity:.45;animation:spin 7s linear infinite}
.node.active .txt span{color:var(--cobalt);font-weight:600}
.node.done .dot{background:var(--solid);border-color:var(--solid);color:var(--on-solid)}
.node.done .txt span{color:var(--ink-2)}
.node.good .dot{background:var(--mint);border-color:var(--mint);color:#fff;animation:pop .5s var(--ease)}
.node.good .txt span{color:var(--mint);font-weight:600}
.node.bad .dot{background:var(--amber);border-color:var(--amber);color:#fff}
.node.bad .txt span{color:var(--amber);font-weight:600}
.node.skip .dot{border-style:dashed;opacity:.7}
.link{flex:1;height:2px;margin:0 14px;min-width:16px;background:var(--line);border-radius:2px;position:relative;overflow:hidden}
.link.fill{background:var(--solid)}
.link.flow::after{content:"";position:absolute;top:0;bottom:0;width:45%;background:linear-gradient(90deg,transparent,var(--cobalt),transparent);animation:slide 1s linear infinite}
.tries{margin-left:18px;padding-left:20px;border-left:1px solid var(--line);flex:none}
.tries small{display:block;font-size:12.5px;color:var(--ink-2);font-weight:600;margin-bottom:6px;white-space:nowrap}
.tries div{display:flex;gap:5px}
.tries em{width:18px;height:6px;border-radius:4px;background:var(--line);transition:background .4s,transform .4s var(--ease)}
.tries em.on{background:var(--solid)}
.tries em.cur{background:var(--cobalt);animation:breathe 1.4s ease-in-out infinite}

/* ── post preview ──────────────────────────────── */
.post{background:var(--surface);border:1px solid var(--line);border-radius:20px;overflow:hidden;position:relative;box-shadow:var(--shadow-2)}
.post-head{display:flex;gap:12px;align-items:center;padding:20px 22px 4px}
.avatar{width:50px;height:50px;border-radius:50%;background:var(--solid);color:var(--on-solid);display:grid;place-items:center;font-weight:700;font-size:18px;flex:none}
.who b{display:block;font-size:15px;font-weight:600;color:var(--ink)}
.who span{display:block;font-size:12.5px;color:var(--ink-2);line-height:1.35}
.badge-draft{margin-left:auto;font-size:12px;font-weight:600;padding:4px 10px;border-radius:999px;background:var(--cobalt-soft);color:var(--cobalt);white-space:nowrap;animation:pop .45s var(--ease)}
.post-body{position:relative;padding:12px 22px 20px;font-size:15.5px;line-height:1.62;color:var(--body);min-height:210px;overflow:hidden}
.post-body p{margin:0 0 .95em;color:var(--body)}
.post-body p:first-child{font-weight:600;font-size:16.5px}
.reveal p{opacity:0;animation:rise .55s var(--ease) forwards}
.scan::after{content:"";position:absolute;left:0;right:0;top:0;height:90px;background:linear-gradient(180deg,transparent,var(--cobalt-soft),transparent);border-bottom:1.5px solid var(--cobalt);animation:scan 2.4s ease-in-out infinite}
.skel .ln{height:12px;border-radius:7px;margin:13px 0;background:linear-gradient(90deg,var(--skel-a) 25%,var(--skel-b) 50%,var(--skel-a) 75%);background-size:200% 100%;animation:shimmer 1.4s linear infinite}
.skel.still .ln{animation:none}
.skel .cap{font-size:13.5px;color:var(--ink-2);margin:0 0 14px;font-weight:500}
.post-meta{display:flex;align-items:center;gap:14px;padding:12px 22px;border-top:1px solid var(--line);font-size:12.5px;color:var(--ink-2)}
.post-meta b{color:var(--ink)}
.meter{flex:1;height:5px;background:var(--line);border-radius:4px;overflow:hidden;max-width:180px}
.meter i{display:block;height:100%;background:var(--mint);border-radius:4px;transform-origin:left;animation:grow-x 1s var(--ease) both}
.meter i.warn{background:var(--amber)}
.banner{display:flex;align-items:center;gap:12px;padding:14px 22px;font-size:13.5px;animation:rise .5s var(--ease) both}
.banner b{display:block;font-size:14px}
.banner span{color:var(--ink-2)}
.banner.ok{background:var(--mint-soft);color:var(--mint-ink)}
.banner.bad{background:var(--amber-soft);color:var(--amber-ink)}
.banner.live{background:var(--cobalt-soft);color:var(--cobalt)}
.banner .ic{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;color:#fff;flex:none}
.banner.ok .ic{background:var(--mint)} .banner.bad .ic{background:var(--amber)}
.banner svg{width:16px;height:16px}
.tick path{stroke-dasharray:24;stroke-dashoffset:24;animation:draw .5s .15s var(--ease) forwards}
.dots i{display:inline-block;width:5px;height:5px;border-radius:50%;background:currentColor;margin-left:3px;animation:blink 1.2s infinite}
.dots i:nth-child(2){animation-delay:.2s}.dots i:nth-child(3){animation-delay:.4s}
.confetti{position:absolute;inset:0;pointer-events:none;overflow:hidden;z-index:3}
.confetti i{position:absolute;top:-12px;width:7px;height:12px;border-radius:2px;opacity:0;animation:fall 1.9s var(--ease) forwards}

/* ── review notes ──────────────────────────────── */
.notes-h{font-weight:700;font-size:20px;letter-spacing:-.02em;margin:2px 0 14px;color:var(--ink)}
.note{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:16px;padding:16px 18px;margin-bottom:12px;animation:rise .5s var(--ease) both}
.note.ok{border-left-color:var(--mint)} .note.bad{border-left-color:var(--rose)} .note.live{border-left-color:var(--cobalt)}
.note-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.note-h b{font-size:14.5px;color:var(--ink)}
.pill{font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px}
.pill.ok{background:var(--mint-soft);color:var(--mint-ink)}.pill.bad{background:var(--rose-soft);color:var(--rose-ink)}.pill.live{background:var(--cobalt-soft);color:var(--cobalt)}
.note .fb{font-size:14px;line-height:1.55;color:var(--ink-2);margin:0 0 10px}
.note .meta{font-size:12.5px;color:var(--ink-3);margin-bottom:6px}
.note details summary{cursor:pointer;font-size:13px;font-weight:600;color:var(--cobalt);list-style:none;display:inline-flex;align-items:center;gap:6px;padding:2px 0;transition:gap .2s}
.note details summary::-webkit-details-marker{display:none}
.note details summary::before{content:"+";display:inline-grid;place-items:center;width:16px;height:16px;border-radius:50%;background:var(--cobalt-soft);font-size:13px;line-height:1;transition:transform .25s var(--ease)}
.note details[open] summary::before{transform:rotate(45deg)}
.note details .draftbox{margin-top:10px;padding:12px 14px;background:var(--surface-2);border:1px solid var(--line);border-radius:12px;font-size:13.5px;line-height:1.55;color:var(--draft);animation:rise .35s var(--ease) both}
.note details .draftbox p{margin:0 0 .7em;color:var(--draft)}
.empty-note{border:1.5px dashed var(--line);border-radius:16px;padding:22px;color:var(--ink-2);font-size:14px;line-height:1.55;background:var(--surface-2)}

.alert{background:var(--rose-soft);border:1px solid var(--rose-line);color:var(--rose-ink);border-radius:14px;padding:14px 18px;font-size:14px;margin-top:14px;animation:rise .4s var(--ease) both}
.alert code{background:var(--surface);padding:1px 6px;border-radius:6px}
.foot{margin-top:48px;text-align:center;font-size:12.5px;color:var(--ink-3)}

/* ── keyframes ─────────────────────────────────── */
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes lift{from{transform:translateY(105%)}to{transform:none}}
@keyframes grow-x{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes drift{from{transform:translate3d(0,0,0) scale(1)}to{transform:translate3d(3%,2.5%,0) scale(1.1)}}
@keyframes settle{from{max-height:380px;opacity:0;transform:scale(.97)}to{max-height:260px;opacity:1;transform:none}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(59,91,255,.4)}70%{box-shadow:0 0 0 12px rgba(59,91,255,0)}100%{box-shadow:0 0 0 0 rgba(59,91,255,0)}}
@keyframes pop{0%{transform:scale(.7)}60%{transform:scale(1.12)}100%{transform:scale(1)}}
@keyframes slide{from{left:-45%}to{left:100%}}
@keyframes sweep{from{left:-40%}to{left:100%}}
@keyframes sheen{0%,60%{left:-60%}100%{left:140%}}
@keyframes shimmer{from{background-position:200% 0}to{background-position:-200% 0}}
@keyframes scan{0%{top:-90px}100%{top:100%}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}
@keyframes breathe{0%,100%{transform:scaleX(1)}50%{transform:scaleX(1.25)}}
@keyframes breathe-dot{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.5);opacity:.55}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes fall{0%{opacity:1;transform:translate3d(0,0,0) rotate(0)}100%{opacity:0;transform:translate3d(var(--dx),420px,0) rotate(var(--r))}}

@media (max-width:760px){
  .txt span{display:none}.pipe{padding:14px}.tries{display:none}.hero{padding-top:28px}
  .st-key-composer{padding:18px 16px 12px}.sep{margin:20px -16px 18px}
}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;animation-delay:0s!important;transition:none!important}}
</style>
"""

DARK_VARS = """
<style>
:root{
  --paper:#0A0E17; --surface:#111826; --surface-2:#0D1320; --ink:#EDF1F9; --ink-2:#A3AEC4; --ink-3:#6C7791;
  --line:#222C40; --dot:#18202F; --cobalt:#7089FF; --cobalt-soft:rgba(112,137,255,.16);
  --mint:#2FD58B; --mint-ink:#5BE3A8; --mint-soft:rgba(47,213,139,.13); --mint-line:rgba(47,213,139,.32);
  --rose:#FF6B61; --rose-ink:#FF9C94; --rose-soft:rgba(255,107,97,.13); --rose-line:rgba(255,107,97,.32);
  --amber:#FDB022; --amber-ink:#FEC84B; --amber-soft:rgba(253,176,34,.13);
  --solid:#E7ECF7; --on-solid:#0A0E17; --body:#DDE3F0; --draft:#C3CCDD;
  --skel-a:#1A2236; --skel-b:#25304A;
  --glow1:rgba(112,137,255,.17); --glow2:rgba(47,213,139,.09);
  --shadow-1:0 1px 2px rgba(0,0,0,.4);
  --shadow-2:0 1px 2px rgba(0,0,0,.4),0 28px 56px -24px rgba(0,0,0,.75);
}
.stApp[data-testid="stApp"]{color-scheme:dark}
button[data-testid="stBaseButton-primary"]{box-shadow:0 8px 24px -10px rgba(112,137,255,.45)}
</style>
"""

ICONS = {
    "pen": '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "flag": '<path d="M4 22V4"/><path d="M4 4h13l-2 4 2 4H4"/>',
    "check": '<path d="M5 13l4 4L19 7"/>',
    "alert": '<path d="M12 8v5"/><path d="M12 17h.01"/>',
    "loop": '<path d="M4 12a8 8 0 0114-5.3"/><path d="M18 3v4h-4"/><path d="M20 12a8 8 0 01-14 5.3"/><path d="M6 21v-4h4"/>',
}


def svg(name: str, cls: str = "") -> str:
    return (
        f'<svg class="{cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )


# ════════════════════════════════════════════════════════════════════════
#  HTML BUILDERS
# ════════════════════════════════════════════════════════════════════════

def h(s: str) -> str:
    """Collapse to one line so Streamlit's markdown never treats it as a code block."""
    return "".join(line.strip() for line in s.strip().splitlines())


def esc(t: str) -> str:
    return html.escape(t).replace("$", "&#36;")


def short(t: str, n: int = 70) -> str:
    t = " ".join(t.split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def paras(text: str, animate: bool = False) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    out = []
    for i, b in enumerate(blocks):
        delay = f' style="animation-delay:{i * 0.14:.2f}s"' if animate else ""
        out.append(f"<p{delay}>{esc(b).replace(chr(10), '<br>')}</p>")
    return "".join(out)


def step_html(n: int, title: str, desc: str) -> str:
    return h(f'<div class="step"><span class="n">{n}</span><div><b>{title}</b><span class="d">{desc}</span></div></div>')


def topbar_html() -> str:
    (_, wm), (_, rm) = resolve("w"), resolve("r")
    return h(
        f"""<div class="topbar">
        <div class="brand"><div class="mark">{svg("loop")}</div>Draftwise</div>
        <div class="models">
          <span class="chip"><i></i><b>Writer</b>{esc(short(wm or "choose a model", 26))}</span>
          <span class="chip"><i class="g"></i><b>Reviewer</b>{esc(short(rm or "choose a model", 26))}</span>
        </div></div>"""
    )


HERO_HTML = h(
    """<div class="hero"><div class="rule"></div>
    <h1><span><b>Posts that survive</b></span><span><b>a second reader.</b></span></h1>
    <p>A writer agent drafts. A strict reviewer pushes back. The loop repeats until your LinkedIn post is ready to publish.</p></div>"""
)


def new_run(topic="", tone="Thoughtful", max_attempts=3, research=True) -> dict:
    return {
        "topic": topic, "tone": tone, "max_attempts": max_attempts, "research": research,
        "stage": "idle",          # idle | writing | searching | writing_post | reviewing | done
        "attempt": 1, "draft": "", "queries_now": [], "searched_now": False,
        "history": [], "approved": False, "fresh": False, "error": "",
    }


def pipeline_html(run: dict) -> str:
    s, att, mx = run["stage"], run["attempt"], run["max_attempts"]
    idle = s == "idle"
    off = not run.get("research", True)

    w = "idle" if idle else ("active" if s in ("writing", "writing_post") else "done")
    if off:
        r = "skip"
    elif s == "searching":
        r = "active"
    elif run["searched_now"] and s in ("writing_post", "reviewing", "done"):
        r = "done"
    elif s in ("reviewing", "done") and not run["searched_now"]:
        r = "skip"
    else:
        r = "idle"
    v = "active" if s == "reviewing" else ("done" if s == "done" else "idle")
    f = ("good" if run["approved"] else "bad") if s == "done" else "idle"

    nq = len(run["queries_now"])
    sub = {
        "w": {"idle": "Waiting", "active": f"Writing draft {att}", "done": f"Draft {att} written"}[w],
        "r": {"idle": "When needed", "active": "Searching the web", "skip": "Turned off" if off else "Not needed",
              "done": f"{nq} search{'es' if nq != 1 else ''}"}[r],
        "v": {"idle": "Strict critic", "active": f"Reading draft {att}", "done": "Verdict in"}[v],
        "f": {"idle": "Publish-ready", "good": "Approved", "bad": "Out of drafts"}[f],
    }
    icon = {"w": "pen", "r": "search", "v": "eye", "f": "flag"}

    def node(k, label, st_):
        ic = svg("check") if st_ in ("done", "good") else svg(icon[k])
        return (
            f'<div class="node {st_}"><div class="dot">{ic}</div>'
            f'<div class="txt"><b>{label}</b><span>{sub[k]}</span></div></div>'
        )

    def link(a, b):
        cls = "link"
        if a == "done":
            cls += " fill"
        if b == "active":
            cls += " flow"
        return f'<div class="{cls}"></div>'

    dots = "".join(
        f'<em class="{"cur" if (i + 1 == att and s not in ("idle", "done")) else ("on" if i < att and not idle else "")}"></em>'
        for i in range(mx)
    )
    tries = f"Draft {att} of {mx}" if not idle else f"Up to {mx} drafts"
    running = "pipe running" if s not in ("idle", "done") else "pipe"

    return h(
        f"""<div class="{running}">
        {node("w", "Writer", w)}{link(w, r)}
        {node("r", "Research", r)}{link(r if r != "skip" else w, v)}
        {node("v", "Reviewer", v)}{link(v, f)}
        {node("f", "Ready", f)}
        <div class="tries"><small>{tries}</small><div>{dots}</div></div>
        </div>"""
    )


def confetti_html() -> str:
    rnd = random.Random(11)
    cols = ["var(--cobalt)", "var(--mint)", "var(--amber)", "var(--rose)", "var(--solid)"]
    bits = "".join(
        f'<i style="left:{rnd.randint(4, 96)}%;background:{cols[i % 5]};--dx:{rnd.randint(-70, 70)}px;'
        f'--r:{rnd.randint(-420, 420)}deg;animation-delay:{rnd.uniform(0, .4):.2f}s"></i>'
        for i in range(28)
    )
    return f'<div class="confetti">{bits}</div>'


def post_html(run: dict, animate: bool) -> str:
    s, att = run["stage"], run["attempt"]
    name = st.session_state.get("name") or "Your Name"
    # one shared source: whatever is typed in the topic box also shows in the preview byline
    topic = (run.get("topic") if s != "idle" else "") or st.session_state.get("topic", "")
    headline = short(topic) if topic.strip() else "Your topic appears here"
    initials = "".join(p[0] for p in name.split()[:2]).upper() or "Y"

    badge = f'<span class="badge-draft" key="{att}">Draft {att}</span>' if s != "idle" else ""
    head = (
        f'<div class="post-head"><div class="avatar">{esc(initials)}</div>'
        f'<div class="who"><b>{esc(name)}</b><span>{esc(headline)}</span><span>Now · 🌐</span></div>{badge}</div>'
    )

    showing_draft = s in ("reviewing", "done") and run["draft"]
    if showing_draft:
        cls = "post-body " + ("scan" if s == "reviewing" else ("reveal" if animate else ""))
        body = f'<div class="{cls}">{paras(run["draft"], animate and s == "done")}</div>'
    elif s == "idle":
        body = (
            '<div class="post-body"><div class="skel still"><p class="cap">Your post will appear here.</p>'
            '<div class="ln" style="width:92%"></div><div class="ln" style="width:100%"></div>'
            '<div class="ln" style="width:78%"></div><div class="ln" style="width:0"></div>'
            '<div class="ln" style="width:96%"></div><div class="ln" style="width:64%"></div></div></div>'
        )
    else:
        cap = {
            "searching": "Searching for current facts…",
            "writing_post": "Writing the post with what it found…",
        }.get(s, "Writing the first draft…" if att == 1 else f"Rewriting draft {att} with the reviewer’s notes…")
        body = (
            f'<div class="post-body"><div class="skel"><p class="cap">{cap}</p>'
            '<div class="ln" style="width:90%"></div><div class="ln" style="width:100%"></div>'
            '<div class="ln" style="width:82%"></div><div class="ln" style="width:0"></div>'
            '<div class="ln" style="width:96%"></div><div class="ln" style="width:58%"></div></div></div>'
        )

    meta = banner = ""
    if showing_draft:
        words = len(run["draft"].split())
        ok = 150 <= words <= 200
        pct = min(words / 200, 1) * 100
        meta = (
            f'<div class="post-meta"><b>{words} words</b>'
            f'<div class="meter"><i class="{"" if ok else "warn"}" style="width:{pct:.0f}%"></i></div>'
            f'<span>{"In the 150–200 range" if ok else "Target 150–200"}</span></div>'
        )
    if s == "reviewing":
        banner = (
            '<div class="banner live"><div><b>Reviewer is reading<span class="dots">'
            "<i></i><i></i><i></i></span></b><span>Checking hook, takeaway, length and call to action.</span></div></div>"
        )
    elif s == "done" and run["approved"]:
        banner = (
            f'<div class="banner ok"><div class="ic">{svg("check", "tick")}</div>'
            "<div><b>Approved by the reviewer</b><span>This post is ready to publish.</span></div></div>"
        )
    elif s == "done":
        banner = (
            f'<div class="banner bad"><div class="ic">{svg("alert")}</div>'
            f'<div><b>Not fully approved</b><span>Reached the limit of {run["max_attempts"]} drafts. '
            "This is the latest version — worth a light edit.</span></div></div>"
        )
    party = confetti_html() if (animate and s == "done" and run["approved"]) else ""
    return h(f'<div class="post">{party}{head}{body}{meta}{banner}</div>')


def notes_html(run: dict) -> str:
    items = []
    if run["stage"] == "reviewing":
        items.append(
            f'<div class="note live"><div class="note-h"><b>Draft {run["attempt"]}</b>'
            '<span class="pill live">In review</span></div>'
            '<p class="fb">The reviewer is reading this draft<span class="dots"><i></i><i></i><i></i></span></p></div>'
        )
    for e in reversed(run["history"]):
        ok = e["approved"]
        words = len(e["draft"].split())
        searched = f" · searched {len(e['queries'])}×" if e["queries"] else ""
        items.append(
            f'<div class="note {"ok" if ok else "bad"}"><div class="note-h"><b>Draft {e["attempt"]}</b>'
            f'<span class="pill {"ok" if ok else "bad"}">{"Approved" if ok else "Sent back"}</span></div>'
            f'<p class="fb">{esc(e["feedback"])}</p>'
            f'<div class="meta">{words} words{searched}</div>'
            f'<details><summary>View this draft</summary><div class="draftbox">{paras(e["draft"])}</div></details></div>'
        )
    if not items:
        items.append(
            '<div class="empty-note">After each draft, the reviewer’s verdict and feedback land here. '
            "If it’s sent back, the writer revises and tries again.</div>"
        )
    return h('<div class="notes-h">Review notes</div>' + "".join(items))


def render(ph: dict, run: dict, animate: bool = False):
    ph["pipe"].markdown(pipeline_html(run), unsafe_allow_html=True)
    ph["post"].markdown(post_html(run, animate), unsafe_allow_html=True)
    ph["notes"].markdown(notes_html(run), unsafe_allow_html=True)


def copy_button(text: str, dark: bool = False):
    payload = json.dumps(text).replace("</", "<\\/")
    c = (
        dict(bg="#111826", line="#222C40", ink="#A3AEC4", hov="#7089FF", hbg="rgba(112,137,255,.16)")
        if dark
        else dict(bg="#fff", line="#E4E8F0", ink="#475467", hov="#3B5BFF", hbg="#E8ECFF")
    )
    components.html(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@600&display=swap');
          body{{margin:0;background:transparent}}
          button{{width:100%;height:42px;border-radius:999px;border:1px solid {c['line']};background:{c['bg']};color:{c['ink']};
                 font:600 13.5px Figtree,system-ui,sans-serif;cursor:pointer;transition:all .2s cubic-bezier(.22,.8,.3,1)}}
          button:hover{{border-color:{c['hov']};color:{c['hov']};background:{c['hbg']};transform:translateY(-1px)}}
          button:active{{transform:scale(.97)}}
          button.ok{{background:#12B76A;border-color:#12B76A;color:#fff}}
        </style>
        <button id="b">Copy post</button>
        <script>
          const t = {payload}, b = document.getElementById('b');
          b.onclick = async () => {{
            try {{ await navigator.clipboard.writeText(t); }}
            catch (e) {{
              const a = document.createElement('textarea'); a.value = t; document.body.appendChild(a);
              a.select(); document.execCommand('copy'); a.remove();
            }}
            b.textContent = 'Copied'; b.classList.add('ok');
            setTimeout(() => {{ b.textContent = 'Copy post'; b.classList.remove('ok'); }}, 1800);
          }};
        </script>
        """,
        height=46,
    )


# ════════════════════════════════════════════════════════════════════════
#  PDF EXPORT (reportlab) — always print-friendly light styling
# ════════════════════════════════════════════════════════════════════════

def _pdf_clean(t: str) -> str:
    """Built-in PDF fonts are Latin-only: drop emoji/unsupported glyphs, keep curly quotes & dashes."""
    t = t.encode("cp1252", "ignore").decode("cp1252")
    t = xml_escape(t)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)


def build_pdf(run: dict, author: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    INK, INK2, INK3 = colors.HexColor("#101828"), colors.HexColor("#475467"), colors.HexColor("#98A2B3")
    COBALT, LINE = colors.HexColor("#3B5BFF"), colors.HexColor("#E4E8F0")
    MINT, AMBER = colors.HexColor("#067647"), colors.HexColor("#B54708")

    st_brand = ParagraphStyle("brand", fontName="Helvetica-Bold", fontSize=10, textColor=COBALT, leading=12)
    st_title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, textColor=INK, leading=27, spaceAfter=4)
    st_sub = ParagraphStyle("sub", fontName="Helvetica", fontSize=9.5, textColor=INK2, leading=13)
    st_hook = ParagraphStyle("hook", fontName="Helvetica-Bold", fontSize=13, textColor=INK, leading=19,
                             spaceAfter=10, alignment=TA_LEFT)
    st_body = ParagraphStyle("body", fontName="Helvetica", fontSize=11, textColor=colors.HexColor("#1D2433"),
                             leading=17.5, spaceAfter=10, alignment=TA_LEFT)
    st_meta = ParagraphStyle("meta", fontName="Helvetica", fontSize=9, textColor=INK2, leading=12)

    draft = run["draft"].strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", draft) if b.strip()]
    words = len(draft.split())
    approved = run["approved"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=24 * mm, bottomMargin=22 * mm,
        title="LinkedIn post", author=author or "Draftwise",
    )

    story = [
        Paragraph("DRAFTWISE  ·  LINKEDIN POST", st_brand),
        Spacer(1, 6),
        Paragraph("Your post, ready to publish" if approved else "Your post (latest draft)", st_title),
        Paragraph(
            f"{_pdf_clean(author or 'Author')}  ·  Topic: {_pdf_clean(short(run['topic'], 110))}  ·  "
            f"Tone: {_pdf_clean(run['tone'])}",
            st_sub,
        ),
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=0.8, color=LINE, spaceAfter=16),
    ]

    cell = []
    for i, b in enumerate(blocks):
        cell.append(Paragraph(_pdf_clean(b).replace("\n", "<br/>"), st_hook if i == 0 else st_body))
    card = Table([[cell]], colWidths=[doc.width])
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFBFD")),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [card, Spacer(1, 16)]

    status = Paragraph(
        f'<font color="{(MINT if approved else AMBER).hexval()}"><b>'
        f'{"Approved by reviewer" if approved else "Not fully approved"}</b></font>', st_meta)
    strip = Table(
        [[Paragraph(f"<b>{words}</b> words", st_meta),
          Paragraph(f"<b>{run['attempt']}</b> of {run['max_attempts']} drafts used", st_meta),
          status]],
        colWidths=[doc.width / 3] * 3,
    )
    strip.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, LINE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(strip)

    def chrome(canvas, d):
        canvas.saveState()
        w, hgt = A4
        canvas.setFillColor(COBALT)
        canvas.rect(0, hgt - 6 * mm, w, 6 * mm, stroke=0, fill=1)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(INK3)
        canvas.drawString(22 * mm, 12 * mm, f"Generated with Draftwise · {date.today():%d %b %Y}")
        canvas.drawRightString(w - 22 * mm, 12 * mm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════
#  PAGE
# ════════════════════════════════════════════════════════════════════════

ss = st.session_state


def _default_theme() -> str:
    """Open in the viewer's system theme when Streamlit can tell us (>=1.46); otherwise light."""
    try:
        t = st.context.theme.type
        return t if t in ("light", "dark") else "light"
    except Exception:  # noqa: BLE001
        return "light"


def _toggle_theme():
    ss["theme"] = "light" if ss.get("theme") == "dark" else "dark"


ss.setdefault("theme", _default_theme())
dark = ss["theme"] == "dark"
st.markdown(CSS + (DARK_VARS if dark else ""), unsafe_allow_html=True)

# defaults
ss.setdefault("topic", "")
ss.setdefault("name", "Your Name")
for _r in ("w", "r"):
    ss.setdefault(f"{_r}_prov", DEFAULT_PROV[_r])

with st.container(key="topbar"):
    tl, tr = st.columns([5, 1], vertical_alignment="center")
    tl.markdown(topbar_html(), unsafe_allow_html=True)
    tr.button(
        "Light" if dark else "Dark",
        icon=":material/light_mode:" if dark else ":material/dark_mode:",
        key="theme_btn", on_click=_toggle_theme, help="Switch between light and dark mode",
    )
st.markdown(HERO_HTML, unsafe_allow_html=True)

EXAMPLES = [
    "AI pilots that stall after the demo",
    "Lessons from my first year managing engineers",
    "Async work done right",
    "The case for boring technology",
]


def _use_example(t):
    ss.topic = t


def _reset():
    ss.pop("run", None)
    ss.topic = ""


def model_picker(role: str, title: str, sub: str, icon: str, cls: str):
    with st.container(key=f"mp_{role}"):
        st.markdown(
            h(f'<div class="role"><span class="ri {cls}">{svg(icon)}</span>'
              f'<div><b>{title}</b><span class="d">{sub}</span></div></div>'),
            unsafe_allow_html=True,
        )
        pid = st.selectbox(
            "Provider", list(PROVIDERS), key=f"{role}_prov",
            format_func=lambda p: f"{PROVIDERS[p]['label']}  ·  {PROVIDERS[p]['tag']}",
        )
        P = PROVIDERS[pid]
        m = st.selectbox("Model", P["models"] + [CUSTOM], key=f"{role}_model_{pid}")
        if m == CUSTOM:
            st.text_input("Custom model ID", key=f"{role}_custom_{pid}", placeholder="Paste the exact model ID")
        st.markdown(
            h(f'<div class="prov"><span class="tag {"free" if P["free"] else "paid"}">{P["tag"]}</span>'
              f'<span>{P["blurb"]}</span><a href="{P["url"]}" target="_blank" rel="noopener">Get a key</a></div>'),
            unsafe_allow_html=True,
        )


with st.container(key="composer"):
    # ── 1 · topic ───────────────────────────────────────────────
    st.markdown(step_html(1, "Topic and tone", "What is the post about, and how should it sound?"), unsafe_allow_html=True)
    st.text_area(
        "What should the post be about?",
        key="topic",
        height=96,
        placeholder="e.g. Why most AI pilots stall after the demo — and what the teams that ship do differently",
    )
    cols = st.columns(len(EXAMPLES))
    for c, ex in zip(cols, EXAMPLES):
        c.button(ex, key=f"ex_{ex}", on_click=_use_example, args=(ex,), use_container_width=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    tone = st.pills(
        "Tone", ["Thoughtful", "Bold", "Story-driven", "Data-led", "Friendly"],
        selection_mode="single", default="Thoughtful", key="tone",
    ) or "Thoughtful"

    # ── 2 · models ──────────────────────────────────────────────
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown(step_html(2, "Choose your models", "Pick any provider for each role. Several have free tiers."), unsafe_allow_html=True)
    mc1, mc2 = st.columns(2, gap="medium")
    with mc1:
        model_picker("w", "Writer", "Drafts and revises the post", "pen", "")
    with mc2:
        model_picker("r", "Reviewer", "Judges each draft strictly", "eye", "rv")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.segmented_control(
        "Web research with Tavily", ["On", "Off"], default="On", key="research",
        help="The writer can search the web for current facts. Turn off to skip the Tavily key.",
    )

    with st.expander("Settings"):
        a, c = st.columns([1.4, 1])
        a.text_input("Display name", key="name")
        c.segmented_control("Max drafts", [1, 2, 3, 4, 5], default=3, key="max_attempts")

    # ── 3 · keys ────────────────────────────────────────────────
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown(step_html(3, "Connect your keys", "Only the providers you picked need a key."), unsafe_allow_html=True)

    needed = needed_ids()
    creds = ss.setdefault("creds", {})
    missing = [k for k in needed if k not in creds]
    status_slot = st.empty()

    if not missing:
        chips = "".join(f'<span class="vchip">{svg("check")}{esc(key_label(k).replace(" API key", ""))}</span>' for k in needed)
        st.markdown(
            h(f"""<div class="keys-done"><div class="ic">{svg("check")}</div>
            <div><b>API keys verified</b>
            <span class="t">Your keys are not stored on any server. They stay in this browser session only
            and disappear when you close the tab.</span><div class="vchips">{chips}</div></div></div>"""),
            unsafe_allow_html=True,
        )
        st.button("Change keys", on_click=_clear_keys, key="change_keys")
    else:
        links = " · ".join(f'<a href="{key_url(k)}" target="_blank" rel="noopener">{esc(key_label(k))}</a>' for k in missing)
        with st.form("keys_form", border=False):
            st.markdown(
                '<div class="form-title">Add your API keys</div>'
                '<div class="keynote">Paste the keys, then press <b>Verify keys</b>. They are checked once and never '
                f"written to disk. Need one? {links}</div>",
                unsafe_allow_html=True,
            )
            entered = {
                k: st.text_input(key_label(k), key=f"in_{k}", type="password", autocomplete="off", placeholder=key_ph(k))
                for k in missing
            }
            submitted = st.form_submit_button("Verify keys", use_container_width=True)

        if submitted:
            clean = {k: v.strip() for k, v in entered.items()}
            status_slot.markdown(
                '<div class="keys-wait"><span class="dots"><i></i><i></i><i></i></span>Checking your keys…</div>',
                unsafe_allow_html=True,
            )
            results = validate_keys(clean)
            if all(ok for ok, _ in results.values()):
                creds.update(clean)
                ss["creds"] = creds
                st.rerun()
            else:
                chips = "".join(
                    f'<span class="{"ok" if ok else "no"}">{"✓" if ok else "✕"} '
                    f'{esc(key_label(k))}{"" if ok else " — " + esc(msg)}</span>'
                    for k, (ok, msg) in results.items()
                )
                status_slot.markdown(h(f'<div class="keyres">{chips}</div>'), unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    keys_ready = not missing
    go = st.button("Generate post", type="primary", disabled=not keys_ready, use_container_width=True)

if not keys_ready:
    st.markdown(
        h('<div class="alert"><b>Verify your API keys to continue.</b> '
          "Paste them in step 3 and press <b>Verify keys</b>.</div>"),
        unsafe_allow_html=True,
    )

# placeholders — filled live while the graph streams
ph = {"pipe": st.empty()}
left, right = st.columns([1.25, 1], gap="large")
with left:
    ph["post"] = st.empty()
    actions = st.container()
with right:
    ph["notes"] = st.empty()

if go:
    if not ss.topic.strip():
        st.toast("Add a topic first.", icon="✏️")
        render(ph, new_run(max_attempts=max_attempts(), research=research_on()))
    elif not resolve("w")[1] or not resolve("r")[1]:
        st.toast("Enter a custom model ID, or pick one from the list.", icon="🧩")
        render(ph, new_run(max_attempts=max_attempts(), research=research_on()))
    else:
        run = new_run(ss.topic.strip(), tone, max_attempts(), research_on())
        run.update(stage="writing", fresh=True)
        ss.run = run
        render(ph, run)

        init = {
            "topic": run["topic"], "tone": tone, "messages": [], "draft": "",
            "review_feedback": "", "is_approved": False, "attempt": 0,
            "max_attempts": run["max_attempts"],
        }
        try:
            graph = get_graph()
            for event in graph.stream(init, config={"recursion_limit": 40}, stream_mode="updates"):
                for node, upd in event.items():
                    upd = upd or {}
                    if node == "writer":
                        if "attempt" in upd:  # a fresh attempt just started
                            run["attempt"] = upd["attempt"]
                            run["queries_now"], run["searched_now"] = [], False
                        msg = upd["messages"][-1]
                        calls = getattr(msg, "tool_calls", None)
                        if calls:
                            run["stage"] = "searching"
                            run["searched_now"] = True
                            run["queries_now"] += [
                                str(c.get("args", {}).get("query", "")) for c in calls
                            ]
                        else:
                            run["draft"] = _text(msg.content)
                            run["stage"] = "reviewing"
                    elif node == "tools":
                        run["stage"] = "writing_post"
                    elif node == "extract_draft":
                        run["draft"] = upd.get("draft", run["draft"])
                        run["stage"] = "reviewing"
                    elif node == "reviewer":
                        approved = bool(upd.get("is_approved"))
                        run["history"].append({
                            "attempt": run["attempt"], "draft": run["draft"],
                            "queries": list(run["queries_now"]), "approved": approved,
                            "feedback": upd.get("review_feedback", ""),
                        })
                        if approved or run["attempt"] >= run["max_attempts"]:
                            run["stage"], run["approved"] = "done", approved
                        else:
                            run["attempt"] += 1
                            run["stage"] = "writing"
                            run["queries_now"], run["searched_now"] = [], False
                    render(ph, run, animate=run["stage"] == "done")
        except Exception as exc:  # noqa: BLE001
            err_text = str(exc)
            # never echo API keys back to the screen
            for v in (ss.get("creds") or {}).values():
                if v:
                    err_text = err_text.replace(v, "••••")
            run["error"] = err_text
            run["stage"] = "done" if run["draft"] else "idle"
            render(ph, run)
            st.markdown(
                h(f'<div class="alert"><b>Something went wrong.</b> {esc(run["error"][:400])} '
                  "If this mentions tools or function calling, pick a different writer model or turn web research off.</div>"),
                unsafe_allow_html=True,
            )
        run["fresh"] = False
        ss.run = run
else:
    run = ss.get("run") or new_run(max_attempts=max_attempts(), research=research_on())
    render(ph, run)

# actions once a draft exists
final = ss.get("run")
if final and final["stage"] == "done" and final["draft"]:
    with actions:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            copy_button(final["draft"], dark)
        try:
            pdf_bytes = build_pdf(final, ss.get("name", ""))
            c2.download_button(
                "Download PDF", pdf_bytes, file_name="linkedin_post.pdf",
                mime="application/pdf", use_container_width=True, key="dl",
            )
        except ImportError:
            c2.button("Download PDF", disabled=True, use_container_width=True, key="dl_off")
            st.caption("PDF export needs reportlab — run `pip install reportlab`.")
        c3.button("Start over", on_click=_reset, use_container_width=True, key="reset")

st.markdown('<div class="foot">Draftwise runs on your own keys. Nothing is stored beyond this session.</div>', unsafe_allow_html=True)