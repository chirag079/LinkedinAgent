"""
Draftwise — LinkedIn post writer + reviewer loop (LangGraph + Streamlit)

Run:  streamlit run app.py
Keys: entered in the UI (OpenAI, Tavily, Groq), verified on submit, kept only in the session.
Needs: streamlit>=1.40, requests, reportlab  (pip install reportlab)
"""

import html
import io
import json
import re
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

# id -> (label, placeholder)
KEY_FIELDS = {
    "openai": ("OpenAI API key", "sk-…"),
    "tavily": ("Tavily API key", "tvly-…"),
    "groq": ("Groq API key", "gsk_…"),
}

# ════════════════════════════════════════════════════════════════════════
#  BACKEND — LangGraph pipeline (built per session, keys never cached)
# ════════════════════════════════════════════════════════════════════════

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
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


def build_graph(openai_key: str, tavily_key: str, groq_key: str):
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_tavily import TavilySearch
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    tools = [TavilySearch(max_results=5, tavily_api_key=tavily_key)]

    writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.75, api_key=openai_key)
    writer_llm_with_tool = writer_llm.bind_tools(tools)
    reviewer_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=groq_key,
        temperature=0.25,
    )

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
            response = writer_llm_with_tool.invoke(
                [("system", WRITER_SYSTEM_PROMPT)] + msgs[last_human:]
            )
            return {"messages": [response]}

        attempt = state.get("attempt", 0) + 1
        topic = state["topic"]
        tone = state.get("tone") or "Thoughtful"

        if attempt == 1:
            user_message = (
                f"Write a LinkedIn post on this topic: {topic}. "
                f"Tone: {tone}. "
                f"If you need current info, search the web first."
            )
        else:
            user_message = (
                f"Your previous draft on '{topic}' was rejected. "
                f"Here is the reviewer's feedback:\n\n{state['review_feedback']}\n\n"
                f"Tone: {tone}. Write a new, improved draft that fixes every issue "
                f"mentioned. Do not repeat the same mistake."
            )

        response = writer_llm_with_tool.invoke(
            [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]
        )
        return {"messages": [("human", user_message), response], "attempt": attempt}

    def extract_draft_node(state: State) -> dict:
        return {"draft": _text(state["messages"][-1].content)}

    def reviewer_node(state: State) -> dict:
        prompt = f"Review this LinkedIn post draft:\n{state['draft']}\nGive your review."
        response = reviewer_llm.invoke(
            [("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)]
        )
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
        return "tools" if getattr(last, "tool_calls", None) else "extract_draft"

    def should_stop_looping(state: State):
        if state["is_approved"] or state["attempt"] >= state.get("max_attempts", 3):
            return END
        return "writer"

    g = StateGraph(State)
    g.add_node("writer", writer_node)
    g.add_node("tools", ToolNode(tools))
    g.add_node("extract_draft", extract_draft_node)
    g.add_node("reviewer", reviewer_node)

    g.add_edge(START, "writer")
    g.add_conditional_edges(
        "writer", should_use_tool, {"tools": "tools", "extract_draft": "extract_draft"}
    )
    # after searching, go back to the writer so it can actually write the post
    g.add_edge("tools", "writer")
    g.add_edge("extract_draft", "reviewer")
    g.add_conditional_edges(
        "reviewer", should_stop_looping, {END: END, "writer": "writer"}
    )
    return g.compile()


def get_graph():
    """Build the graph for THIS session only; rebuild if keys change."""
    ss = st.session_state
    creds = ss["creds"]
    sig = tuple(creds[k] for k in KEY_FIELDS)
    if ss.get("_graph_sig") != sig:
        ss["_graph"] = build_graph(*sig)
        ss["_graph_sig"] = sig
    return ss["_graph"]


def _clear_keys():
    for k in ("creds", "_graph", "_graph_sig"):
        st.session_state.pop(k, None)


# ── key validation ──────────────────────────────────────────────────────

def _status(method: str, url: str, key: str, **kw):
    """HTTP status code, or None if the service couldn't be reached."""
    try:
        r = requests.request(
            method, url, headers={"Authorization": f"Bearer {key}"}, timeout=12, **kw
        )
        return r.status_code
    except requests.RequestException:
        return None


def _verdict(code):
    if code is None:
        return False, "Couldn’t reach the service"
    if code == 200:
        return True, ""
    if code in (401, 403):
        return False, "Key was rejected"
    return False, f"Unexpected response ({code})"


def validate_keys(keys: dict) -> dict:
    """Returns {id: (ok, message)}. Uses cheap, read-only endpoints."""
    out = {}
    for k, v in keys.items():
        if not v:
            out[k] = (False, "Missing")
    if "openai" not in out:
        out["openai"] = _verdict(_status("GET", "https://api.openai.com/v1/models", keys["openai"]))
    if "groq" not in out:
        out["groq"] = _verdict(
            _status("GET", "https://api.groq.com/openai/v1/models", keys["groq"])
        )
    if "tavily" not in out:
        code = _status("GET", "https://api.tavily.com/usage", keys["tavily"])
        if code not in (200, 401, 403):  # endpoint unavailable → fall back to a 1-result search
            code = _status(
                "POST", "https://api.tavily.com/search", keys["tavily"],
                json={"query": "ping", "max_results": 1},
            )
        out["tavily"] = _verdict(code)
    return out


# ════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
#  Color  paper #F3F5F9 · surface #FFF · ink #101828 · cobalt #3B5BFF
#         mint #12B76A (approved) · rose #F04438 (sent back) · amber #F79009
#  Type   Bricolage Grotesque (display) + Figtree (interface)
# ════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600;700&display=swap');

:root{
  --paper:#F3F5F9; --surface:#fff; --ink:#101828; --ink-2:#475467; --ink-3:#98A2B3;
  --line:#E4E8F0; --cobalt:#3B5BFF; --cobalt-soft:#E8ECFF;
  --mint:#12B76A; --mint-soft:#E6F7EE; --rose:#F04438; --rose-soft:#FEECEB;
  --amber:#F79009; --amber-soft:#FEF3E2;
  --display:'Bricolage Grotesque',system-ui,sans-serif; --ui:'Figtree',system-ui,sans-serif;
  --ease:cubic-bezier(.22,.8,.3,1);
}

/* ── page chrome ───────────────────────────────── */
.stApp{
  background-color:var(--paper);
  background-image:radial-gradient(#D5DAE5 1px,transparent 1px);
  background-size:26px 26px; font-family:var(--ui); color:var(--ink);
}
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stSidebar"]{display:none!important}
.block-container{max-width:1120px!important;padding:2.2rem 1.5rem 5rem!important}

/* UI font everywhere — :where() keeps specificity low so our own classes can override it */
.stApp :where(p,label,div,span,button,input,textarea,li){font-family:var(--ui)}
/* FIX: Streamlit's icons (eye toggle, expander arrows…) are font ligatures. The old blanket
   `.stApp span{font-family}` replaced the icon font, so the ligature name ("visibility") showed as text. */
.stApp [data-testid="stIconMaterial"],.stApp .material-symbols-rounded{font-family:"Material Symbols Rounded"!important}
.stApp .brand,.stApp .notes-h,.stApp .avatar{font-family:var(--display)}

/* ── top bar ───────────────────────────────────── */
.topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:21px;letter-spacing:-.02em}
.brand .mark{width:34px;height:34px;border-radius:10px;background:var(--ink);color:#fff;display:grid;place-items:center}
.brand .mark svg{width:18px;height:18px;animation:spin 14s linear infinite}
.models{display:flex;gap:8px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:7px;padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:#fff;font-size:12.5px;color:var(--ink-2);font-weight:500}
.chip i{width:7px;height:7px;border-radius:50%;background:var(--cobalt)}
.chip i.g{background:var(--mint)}

/* ── hero (the one entrance) ───────────────────── */
.hero{padding:54px 0 30px;max-width:760px}
.hero h1{font-family:var(--display);font-weight:700;font-size:clamp(38px,6vw,64px);line-height:1.02;letter-spacing:-.035em;margin:0 0 16px;color:var(--ink)}
.hero p{font-size:18px;line-height:1.55;color:var(--ink-2);margin:0;max-width:600px}
.hero h1,.hero p{animation:rise .8s var(--ease) both}
.hero p{animation-delay:.12s}

/* ── composer ──────────────────────────────────── */
.st-key-composer{background:#fff;border:1px solid var(--line);border-radius:22px;padding:22px 24px 14px;
  box-shadow:0 1px 2px rgba(16,24,40,.04),0 24px 48px -24px rgba(16,24,40,.16)}
.st-key-composer label p{font-weight:600;font-size:14px;color:var(--ink)}
div[data-baseweb="textarea"]{border-radius:14px!important;border:1px solid var(--line)!important;background:#FAFBFD!important;transition:border-color .2s,box-shadow .2s,background .2s}
div[data-baseweb="textarea"]:focus-within{border-color:var(--cobalt)!important;background:#fff!important;box-shadow:0 0 0 4px var(--cobalt-soft)}
div[data-baseweb="textarea"] textarea{background:transparent!important;font-size:16.5px!important;line-height:1.5!important;color:var(--ink)!important}
div[data-baseweb="input"]{border-radius:12px!important;border:1px solid var(--line)!important}
div[data-baseweb="input"]:focus-within{border-color:var(--cobalt)!important;box-shadow:0 0 0 4px var(--cobalt-soft)}
div[data-baseweb="input"] input{font-size:15px!important;letter-spacing:.02em}
.stExpander{border:1px solid var(--line)!important;border-radius:14px!important;background:#FAFBFD;margin-bottom:10px}
.stExpander summary p{font-weight:600;font-size:14px}
[data-testid="stForm"]{border:1px solid var(--line)!important;border-radius:16px!important;background:#FAFBFD;padding:18px 20px 14px!important;margin-bottom:10px}
.form-title{font-weight:700;font-size:15px;margin:0 0 2px;color:var(--ink)}

/* buttons */
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{font-family:var(--ui);font-weight:600;border-radius:999px;transition:transform .18s var(--ease),box-shadow .2s,background .2s,border-color .2s,color .2s}
button[data-testid="stBaseButton-secondary"],button[data-testid="stBaseButton-secondaryFormSubmit"]{background:#fff;border:1px solid var(--line);color:var(--ink-2);font-size:13.5px;padding:.35rem .9rem;min-height:0}
button[data-testid="stBaseButton-secondary"]:hover,button[data-testid="stBaseButton-secondaryFormSubmit"]:hover{border-color:var(--cobalt);color:var(--cobalt);background:var(--cobalt-soft);transform:translateY(-1px)}
button[data-testid="stBaseButton-secondary"]:active{transform:scale(.97)}
button[data-testid="stBaseButton-primary"]{background:var(--ink);border:none;color:#fff;font-size:16px;padding:.8rem 1.6rem;border-radius:14px;box-shadow:0 8px 20px -8px rgba(16,24,40,.5)}
button[data-testid="stBaseButton-primary"]:hover{background:var(--cobalt);transform:translateY(-2px);box-shadow:0 14px 28px -10px rgba(59,91,255,.6)}
button[data-testid="stBaseButton-primary"]:active{transform:scale(.98)}
button[data-testid="stBaseButton-primary"]:disabled{background:#D0D5DD;box-shadow:none;transform:none}

/* ── keys verified ─────────────────────────────── */
.keys-done{display:flex;align-items:center;gap:14px;padding:16px 18px;border-radius:16px;background:var(--mint-soft);
  border:1px solid #B7E8CF;color:#067647;overflow:hidden;animation:settle .8s var(--ease) both;margin-bottom:10px}
.keys-done .ic{width:38px;height:38px;border-radius:50%;background:var(--mint);color:#fff;display:grid;place-items:center;flex:none;
  animation:pop .6s .15s var(--ease) both}
.keys-done .ic svg{width:19px;height:19px}
.keys-done b{display:block;font-size:14.5px;margin-bottom:2px}
.keys-done span{display:block;font-size:13px;line-height:1.5;color:#12684A}
.keys-done .ic path{stroke-dasharray:24;stroke-dashoffset:24;animation:draw .5s .45s var(--ease) forwards}
.keys-wait{display:flex;align-items:center;gap:12px;padding:14px 18px;border-radius:14px;background:var(--cobalt-soft);color:var(--cobalt);font-size:14px;font-weight:600;margin-bottom:10px;animation:rise .3s var(--ease) both}
.keyres{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 10px;animation:rise .4s var(--ease) both}
.keyres span{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;font-size:12.5px;font-weight:600}
.keyres .ok{background:var(--mint-soft);color:#067647}.keyres .no{background:var(--rose-soft);color:#B42318}

/* ── agent pipeline ────────────────────────────── */
.pipe{display:flex;align-items:center;background:#fff;border:1px solid var(--line);border-radius:20px;padding:18px 22px;
  box-shadow:0 1px 2px rgba(16,24,40,.04);margin:30px 0 22px}
.node{display:flex;align-items:center;gap:12px;min-width:0}
.dot{width:42px;height:42px;border-radius:13px;display:grid;place-items:center;border:1.5px solid var(--line);color:var(--ink-3);background:#fff;transition:all .35s var(--ease);flex:none}
.dot svg{width:20px;height:20px}
.txt b{display:block;font-size:14px;font-weight:600;color:var(--ink)}
.txt span{display:block;font-size:12.5px;color:var(--ink-3);white-space:nowrap}
.node.active .dot{border-color:var(--cobalt);color:var(--cobalt);background:var(--cobalt-soft);animation:pulse 1.7s ease-out infinite}
.node.active .txt span{color:var(--cobalt);font-weight:600}
.node.done .dot{background:var(--ink);border-color:var(--ink);color:#fff}
.node.done .txt span{color:var(--ink-2)}
.node.good .dot{background:var(--mint);border-color:var(--mint);color:#fff;animation:pop .5s var(--ease)}
.node.good .txt span{color:var(--mint);font-weight:600}
.node.bad .dot{background:var(--amber);border-color:var(--amber);color:#fff}
.node.bad .txt span{color:var(--amber);font-weight:600}
.node.skip .dot{border-style:dashed;opacity:.7}
.link{flex:1;height:2px;margin:0 14px;min-width:16px;background:var(--line);border-radius:2px;position:relative;overflow:hidden}
.link.fill{background:var(--ink)}
.link.flow::after{content:"";position:absolute;top:0;bottom:0;width:45%;background:linear-gradient(90deg,transparent,var(--cobalt),transparent);animation:slide 1s linear infinite}
.tries{margin-left:18px;padding-left:20px;border-left:1px solid var(--line);flex:none}
.tries small{display:block;font-size:12.5px;color:var(--ink-2);font-weight:600;margin-bottom:6px;white-space:nowrap}
.tries div{display:flex;gap:5px}
.tries em{width:18px;height:6px;border-radius:4px;background:var(--line);transition:background .4s,transform .4s var(--ease)}
.tries em.on{background:var(--ink)}
.tries em.cur{background:var(--cobalt);animation:breathe 1.4s ease-in-out infinite}

/* ── post preview ──────────────────────────────── */
.post{background:#fff;border:1px solid var(--line);border-radius:20px;overflow:hidden;position:relative;
  box-shadow:0 1px 2px rgba(16,24,40,.04),0 24px 48px -24px rgba(16,24,40,.18)}
.post-head{display:flex;gap:12px;align-items:center;padding:20px 22px 4px}
.avatar{width:50px;height:50px;border-radius:50%;background:var(--ink);color:#fff;display:grid;place-items:center;font-weight:700;font-size:18px;flex:none}
.who b{display:block;font-size:15px;font-weight:600}
.who span{display:block;font-size:12.5px;color:var(--ink-2);line-height:1.35}
.badge-draft{margin-left:auto;font-size:12px;font-weight:600;padding:4px 10px;border-radius:999px;background:var(--cobalt-soft);color:var(--cobalt);white-space:nowrap}
.post-body{position:relative;padding:12px 22px 20px;font-size:15.5px;line-height:1.62;color:#1D2433;min-height:210px;overflow:hidden}
.post-body p{margin:0 0 .95em}
.post-body p:first-child{font-weight:600;font-size:16.5px}
.reveal p{opacity:0;animation:rise .55s var(--ease) forwards}
.scan::after{content:"";position:absolute;left:0;right:0;top:0;height:90px;background:linear-gradient(180deg,transparent,rgba(59,91,255,.13),transparent);border-bottom:1.5px solid rgba(59,91,255,.35);animation:scan 2.4s ease-in-out infinite}
.skel .ln{height:12px;border-radius:7px;margin:13px 0;background:linear-gradient(90deg,#EEF1F6 25%,#F8F9FC 50%,#EEF1F6 75%);background-size:200% 100%;animation:shimmer 1.4s linear infinite}
.skel.still .ln{animation:none}
.skel .cap{font-size:13.5px;color:var(--ink-2);margin:0 0 14px;font-weight:500}
.post-meta{display:flex;align-items:center;gap:14px;padding:12px 22px;border-top:1px solid var(--line);font-size:12.5px;color:var(--ink-2)}
.meter{flex:1;height:5px;background:var(--line);border-radius:4px;overflow:hidden;max-width:180px}
.meter i{display:block;height:100%;background:var(--mint);border-radius:4px;transition:width .8s var(--ease)}
.meter i.warn{background:var(--amber)}
.banner{display:flex;align-items:center;gap:12px;padding:14px 22px;font-size:13.5px;animation:rise .5s var(--ease) both}
.banner b{display:block;font-size:14px}
.banner span{color:var(--ink-2)}
.banner.ok{background:var(--mint-soft);color:#067647}
.banner.bad{background:var(--amber-soft);color:#B54708}
.banner.live{background:var(--cobalt-soft);color:var(--cobalt)}
.banner .ic{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;color:#fff;flex:none}
.banner.ok .ic{background:var(--mint)} .banner.bad .ic{background:var(--amber)}
.banner svg{width:16px;height:16px}
.tick path{stroke-dasharray:24;stroke-dashoffset:24;animation:draw .5s .15s var(--ease) forwards}
.dots i{display:inline-block;width:5px;height:5px;border-radius:50%;background:currentColor;margin-left:3px;animation:blink 1.2s infinite}
.dots i:nth-child(2){animation-delay:.2s}.dots i:nth-child(3){animation-delay:.4s}

/* ── review notes ──────────────────────────────── */
.notes-h{font-weight:700;font-size:20px;letter-spacing:-.02em;margin:2px 0 14px}
.note{background:#fff;border:1px solid var(--line);border-left:4px solid var(--line);border-radius:16px;padding:16px 18px;margin-bottom:12px;animation:rise .5s var(--ease) both}
.note.ok{border-left-color:var(--mint)} .note.bad{border-left-color:var(--rose)} .note.live{border-left-color:var(--cobalt)}
.note-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.note-h b{font-size:14.5px}
.pill{font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px}
.pill.ok{background:var(--mint-soft);color:#067647}.pill.bad{background:var(--rose-soft);color:#B42318}.pill.live{background:var(--cobalt-soft);color:var(--cobalt)}
.note .fb{font-size:14px;line-height:1.55;color:var(--ink-2);margin:0 0 10px}
.note .meta{font-size:12.5px;color:var(--ink-3);margin-bottom:6px}
.note details summary{cursor:pointer;font-size:13px;font-weight:600;color:var(--cobalt);list-style:none;display:inline-flex;align-items:center;gap:6px;padding:2px 0;transition:gap .2s}
.note details summary::-webkit-details-marker{display:none}
.note details summary::before{content:"+";display:inline-grid;place-items:center;width:16px;height:16px;border-radius:50%;background:var(--cobalt-soft);font-size:13px;line-height:1;transition:transform .25s var(--ease)}
.note details[open] summary::before{transform:rotate(45deg)}
.note details .draftbox{margin-top:10px;padding:12px 14px;background:#FAFBFD;border:1px solid var(--line);border-radius:12px;font-size:13.5px;line-height:1.55;color:#344054;animation:rise .35s var(--ease) both}
.note details .draftbox p{margin:0 0 .7em}
.empty-note{border:1.5px dashed #CDD3DF;border-radius:16px;padding:22px;color:var(--ink-2);font-size:14px;line-height:1.55;background:rgba(255,255,255,.55)}

.alert{background:var(--rose-soft);border:1px solid #FECDCA;color:#B42318;border-radius:14px;padding:14px 18px;font-size:14px;margin-top:14px;animation:rise .4s var(--ease) both}
.alert code{background:#fff;padding:1px 6px;border-radius:6px}
.keynote{font-size:13px;color:var(--ink-2);line-height:1.5;margin:2px 0 12px}

/* ── keyframes ─────────────────────────────────── */
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes settle{from{max-height:380px;opacity:0;transform:scale(.97)}to{max-height:220px;opacity:1;transform:none}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(59,91,255,.4)}70%{box-shadow:0 0 0 12px rgba(59,91,255,0)}100%{box-shadow:0 0 0 0 rgba(59,91,255,0)}}
@keyframes pop{0%{transform:scale(.7)}60%{transform:scale(1.12)}100%{transform:scale(1)}}
@keyframes slide{from{left:-45%}to{left:100%}}
@keyframes shimmer{from{background-position:200% 0}to{background-position:-200% 0}}
@keyframes scan{0%{top:-90px}100%{top:100%}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}
@keyframes breathe{0%,100%{transform:scaleX(1)}50%{transform:scaleX(1.25)}}
@keyframes spin{to{transform:rotate(360deg)}}

@media (max-width:760px){
  .txt span{display:none}.pipe{padding:14px}.tries{display:none}.hero{padding-top:32px}
}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01ms!important;animation-iteration-count:1!important;transition:none!important}}
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


def paras(text: str, animate: bool = False) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    out = []
    for i, b in enumerate(blocks):
        delay = f' style="animation-delay:{i * 0.14:.2f}s"' if animate else ""
        out.append(f"<p{delay}>{esc(b).replace(chr(10), '<br>')}</p>")
    return "".join(out)


def new_run(topic="", tone="Thoughtful", max_attempts=3) -> dict:
    return {
        "topic": topic, "tone": tone, "max_attempts": max_attempts,
        "stage": "idle",          # idle | writing | searching | writing_post | reviewing | done
        "attempt": 1, "draft": "", "queries_now": [], "searched_now": False,
        "history": [], "approved": False, "fresh": False, "error": "",
    }


def pipeline_html(run: dict) -> str:
    s, att, mx = run["stage"], run["attempt"], run["max_attempts"]
    idle = s == "idle"

    w = "idle" if idle else ("active" if s in ("writing", "writing_post") else "done")
    if s == "searching":
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
        "r": {"idle": "When needed", "active": "Searching the web", "skip": "Not needed",
              "done": f"{nq} search{'es' if nq != 1 else ''}"}[r],
        "v": {"idle": "Strict critic", "active": f"Reading draft {att}", "done": "Verdict in"}[v],
        "f": {"idle": "Publish-ready", "good": "Approved", "bad": "Out of drafts"}[f],
    }
    icon = {"w": "pen", "r": "search", "v": "eye", "f": "flag"}
    icon_done = lambda st_, k: svg("check") if st_ in ("done", "good") else svg(icon[k])

    def node(k, label, st_):
        return (
            f'<div class="node {st_}"><div class="dot">{icon_done(st_, k)}</div>'
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

    return h(
        f"""<div class="pipe">
        {node("w", "Writer", w)}{link(w, r)}
        {node("r", "Research", r)}{link(r if r != "skip" else w, v)}
        {node("v", "Reviewer", v)}{link(v, f)}
        {node("f", "Ready", f)}
        <div class="tries"><small>{tries}</small><div>{dots}</div></div>
        </div>"""
    )


def short(t: str, n: int = 70) -> str:
    t = " ".join(t.split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def post_html(run: dict, animate: bool) -> str:
    s, att = run["stage"], run["attempt"]
    name = st.session_state.get("name") or "Your Name"
    # one shared source: whatever is typed in the topic box also shows in the preview byline
    topic = (run.get("topic") if s != "idle" else "") or st.session_state.get("topic", "")
    headline = short(topic) if topic.strip() else "Your topic appears here"
    initials = "".join(p[0] for p in name.split()[:2]).upper() or "Y"

    badge = f'<span class="badge-draft">Draft {att}</span>' if s != "idle" else ""
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
    return h(f'<div class="post">{head}{body}{meta}{banner}</div>')


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


def copy_button(text: str):
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@600&display=swap');
          body{{margin:0}}
          button{{width:100%;height:42px;border-radius:999px;border:1px solid #E4E8F0;background:#fff;color:#475467;
                 font:600 13.5px Figtree,system-ui,sans-serif;cursor:pointer;transition:all .2s cubic-bezier(.22,.8,.3,1)}}
          button:hover{{border-color:#3B5BFF;color:#3B5BFF;background:#E8ECFF;transform:translateY(-1px)}}
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
#  PDF EXPORT (reportlab)
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

    # the post, in a card
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

    # summary strip
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

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    h(
        f"""<div class="topbar">
        <div class="brand"><div class="mark">{svg("loop")}</div>Draftwise</div>
        <div class="models">
          <span class="chip"><i></i>Writer · GPT-4o mini</span>
          <span class="chip"><i class="g"></i>Reviewer · gpt-oss-120b</span>
        </div></div>
        <div class="hero"><h1>Posts that survive a second reader.</h1>
        <p>A writer agent drafts. A strict reviewer pushes back. The loop repeats until your LinkedIn post is ready to publish.</p></div>"""
    ),
    unsafe_allow_html=True,
)

# defaults
st.session_state.setdefault("topic", "")
st.session_state.setdefault("name", "Your Name")
st.session_state.setdefault("max_attempts", 3)

EXAMPLES = [
    "AI pilots that stall after the demo",
    "Lessons from my first year managing engineers",
    "Async work done right",
    "The case for boring technology",
]


def _use_example(t):
    st.session_state.topic = t


def _reset():
    st.session_state.pop("run", None)
    st.session_state.topic = ""


KEYS_DONE_HTML = h(
    f"""<div class="keys-done"><div class="ic">{svg("check")}</div>
    <div><b>API keys verified</b>
    <span>Your keys are not stored on any server — they stay in this browser session only
    and disappear when you close the tab.</span></div></div>"""
)


def _show_keys_done(slot):
    with slot.container():
        st.markdown(KEYS_DONE_HTML, unsafe_allow_html=True)
        st.button("Change keys", on_click=_clear_keys, key="change_keys")


with st.container(key="composer"):
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

    with st.expander("Settings"):
        a, c = st.columns([1.4, 1])
        a.text_input("Display name", key="name")
        c.slider("Max drafts", 1, 5, key="max_attempts")

    # ── API keys: verify on submit, then collapse into a short note ──
    keys_slot = st.empty()
    status_slot = st.empty()

    if st.session_state.get("creds"):
        _show_keys_done(keys_slot)
    else:
        with keys_slot.container():
            with st.form("keys_form", border=False):
                st.markdown(
                    '<div class="form-title">Connect your API keys</div>'
                    '<div class="keynote">Paste your keys, then press <b>Verify keys</b>. '
                    "They’re checked once and never written to disk.</div>",
                    unsafe_allow_html=True,
                )
                entered = {}
                for kid, (label, ph_) in KEY_FIELDS.items():
                    entered[kid] = st.text_input(
                        label, key=f"in_{kid}", type="password",
                        autocomplete="off", placeholder=ph_,
                    )
                submitted = st.form_submit_button("Verify keys", use_container_width=True)

        if submitted:
            clean = {k: v.strip() for k, v in entered.items()}
            status_slot.markdown(
                '<div class="keys-wait"><span class="dots"><i></i><i></i><i></i></span>'
                "Checking your keys…</div>",
                unsafe_allow_html=True,
            )
            results = validate_keys(clean)
            if all(ok for ok, _ in results.values()):
                st.session_state["creds"] = clean
                status_slot.empty()
                keys_slot.empty()
                _show_keys_done(keys_slot)
            else:
                chips = "".join(
                    f'<span class="{"ok" if ok else "no"}">{"✓" if ok else "✕"} '
                    f'{esc(KEY_FIELDS[k][0])}{"" if ok else " — " + esc(msg)}</span>'
                    for k, (ok, msg) in results.items()
                )
                status_slot.markdown(h(f'<div class="keyres">{chips}</div>'), unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    keys_ready = bool(st.session_state.get("creds"))
    go = st.button("Generate post", type="primary", disabled=not keys_ready, use_container_width=True)

if not keys_ready:
    st.markdown(
        h('<div class="alert"><b>Verify your API keys to continue.</b> '
          "Paste them in the form above and press <b>Verify keys</b>.</div>"),
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
    if not st.session_state.topic.strip():
        st.toast("Add a topic first.", icon="✏️")
        render(ph, new_run(max_attempts=st.session_state.max_attempts))
    else:
        run = new_run(st.session_state.topic.strip(), tone, st.session_state.max_attempts)
        run.update(stage="writing", fresh=True)
        st.session_state.run = run
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
            for v in (st.session_state.get("creds") or {}).values():
                if v:
                    err_text = err_text.replace(v, "••••")
            run["error"] = err_text
            run["stage"] = "done" if run["draft"] else "idle"
            render(ph, run)
            st.markdown(
                h(f'<div class="alert"><b>Something went wrong.</b> {esc(run["error"][:400])}</div>'),
                unsafe_allow_html=True,
            )
        run["fresh"] = False
        st.session_state.run = run
else:
    run = st.session_state.get("run") or new_run(max_attempts=st.session_state.max_attempts)
    render(ph, run)

# actions once a draft exists
final = st.session_state.get("run")
if final and final["stage"] == "done" and final["draft"]:
    with actions:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            copy_button(final["draft"])
        try:
            pdf_bytes = build_pdf(final, st.session_state.get("name", ""))
            c2.download_button(
                "Download PDF", pdf_bytes, file_name="linkedin_post.pdf",
                mime="application/pdf", use_container_width=True, key="dl",
            )
        except ImportError:
            c2.button("Download PDF", disabled=True, use_container_width=True, key="dl_off")
            st.caption("PDF export needs reportlab — run `pip install reportlab`.")
        c3.button("Start over", on_click=_reset, use_container_width=True, key="reset")