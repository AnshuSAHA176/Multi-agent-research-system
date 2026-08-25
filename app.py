import streamlit as st

from src.pipelines.pipelines import research_pipeline


st.set_page_config(
    page_title="Research Workspace",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        @import url(
            'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap'
        );

        * {
            font-family: "DM Sans", sans-serif;
        }

        .stApp {
            background: #0c0d0f;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }

        section[data-testid="stSidebar"] {
            background: #101114;
            border-right: 1px solid #24262b;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 2rem;
        }

        .brand {
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: -0.2px;
        }

        .brand-subtitle {
            color: #777b84;
            font-size: 0.75rem;
            margin-top: 0.2rem;
        }

        .sidebar-section {
            color: #6f737c;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            margin-top: 2rem;
            margin-bottom: 0.8rem;
        }

        .pipeline-step {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #a5a8ae;
            font-size: 0.82rem;
            margin: 12px 0;
        }

        .step-number {
            width: 22px;
            height: 22px;
            border: 1px solid #30333a;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.68rem;
            color: #888c95;
        }

        .page-label {
            color: #777b84;
            font-size: 0.75rem;
            font-family: "IBM Plex Mono", monospace;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .page-title {
            font-size: 2.35rem;
            line-height: 1.1;
            font-weight: 600;
            letter-spacing: -1.5px;
            color: #f2f2f3;
            margin-bottom: 0.6rem;
        }

        .page-description {
            color: #858992;
            max-width: 680px;
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 2.2rem;
        }

        .section-title {
            color: #e4e5e7;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.65rem;
        }

        .status-box {
            background: #111317;
            border: 1px solid #282b31;
            border-radius: 8px;
            padding: 14px 16px;
            color: #a4a8b0;
            font-size: 0.82rem;
            font-family: "IBM Plex Mono", monospace;
        }

        .agent-panel {
            background: #111317;
            border: 1px solid #282b31;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 24px;
        }

        .agent-row {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 14px 16px;
            border-bottom: 1px solid #202228;
        }

        .agent-row:last-child {
            border-bottom: none;
        }

        .agent-number {
            width: 28px;
            height: 28px;
            border: 1px solid #30333a;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #777b84;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.68rem;
            flex-shrink: 0;
        }

        .agent-info {
            flex: 1;
        }

        .agent-name {
            color: #d7d9dd;
            font-size: 0.84rem;
            font-weight: 500;
        }

        .agent-state {
            margin-top: 3px;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.68rem;
        }

        .agent-state.waiting {
            color: #666a73;
        }

        .agent-state.working {
            color: #d6d8dc;
        }

        .agent-state.completed {
            color: #9da2aa;
        }

        .agent-state.failed {
            color: #c8cbd1;
        }

        .report-container {
            background: #101114;
            border: 1px solid #282b31;
            border-radius: 8px;
            padding: 32px 38px;
            line-height: 1.75;
        }

        .report-container h1,
        .report-container h2,
        .report-container h3 {
            color: #eeeeef;
        }

        .report-container p,
        .report-container li {
            color: #b4b7bd;
        }

        .source-label {
            color: #696d76;
            font-size: 0.68rem;
            font-family: "IBM Plex Mono", monospace;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        div[data-testid="stTextArea"] textarea {
            background: #111317;
            border: 1px solid #292c32;
            border-radius: 8px;
            color: #e7e7e9;
            font-size: 0.92rem;
            line-height: 1.5;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #555961;
            box-shadow: none;
        }

        div.stButton > button {
            background: #e8e8e8;
            color: #111214;
            border: none;
            border-radius: 7px;
            height: 2.7rem;
            font-weight: 600;
            font-size: 0.84rem;
        }

        div.stButton > button:hover {
            background: #ffffff;
            color: #111214;
        }

        button[data-baseweb="tab"] {
            font-size: 0.8rem;
        }

        [data-testid="stMetric"] {
            background: #111317;
            border: 1px solid #282b31;
            border-radius: 8px;
            padding: 14px;
        }

        .footer {
            color: #565a62;
            font-size: 0.7rem;
            font-family: "IBM Plex Mono", monospace;
            margin-top: 3rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(
        """
        <div class="brand">Research Workspace</div>
        <div class="brand-subtitle">
            Multi-agent research system
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Pipeline</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("01", "Source discovery"),
        ("02", "Content extraction"),
        ("03", "Report synthesis"),
        ("04", "Quality review"),
    ]

    for number, label in steps:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <div class="step-number">{number}</div>
                <div>{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-section">System</div>',
        unsafe_allow_html=True,
    )

    st.caption("Search")
    st.caption("Web extraction")
    st.caption("LLM synthesis")
    st.caption("Critical review")

    st.markdown("---")

    st.markdown(
        """
        <div class="brand-subtitle">
            Research Workspace v0.1
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="page-label">Research</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-title">Investigate a question.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-description">
        Submit a research question and the system will discover sources,
        extract evidence, synthesize findings, and evaluate the resulting
        report.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="section-title">Research question</div>',
    unsafe_allow_html=True,
)

topic = st.text_area(
    "",
    placeholder=(
        "e.g. What are the major technical challenges "
        "of deploying autonomous AI agents in production?"
    ),
    height=110,
    label_visibility="collapsed",
)


run_col, info_col = st.columns([1, 3])

with run_col:
    run_research = st.button(
        "Run research",
        type="primary",
        use_container_width=True,
    )

with info_col:
    st.markdown(
        """
        <div class="status-box">
            Sources → Evidence → Synthesis → Review
        </div>
        """,
        unsafe_allow_html=True,
    )


if run_research:

    if not topic.strip():
        st.warning("Enter a research question.")
        st.stop()

    st.divider()

    st.markdown(
        '<div class="section-title">Agent activity</div>',
        unsafe_allow_html=True,
    )

    agent_status = {
        "Search Agent": "waiting",
        "Scrape Agent": "waiting",
        "Writer Agent": "waiting",
        "Critic Agent": "waiting",
    }

    agent_number = {
        "Search Agent": "01",
        "Scrape Agent": "02",
        "Writer Agent": "03",
        "Critic Agent": "04",
    }

    activity = st.empty()

    def render_agents():
        rows = []

        for agent, current_status in agent_status.items():

            if current_status == "working":
                icon = "●"
                label = "Working"
                css_class = "working"

            elif current_status == "completed":
                icon = "✓"
                label = "Completed"
                css_class = "completed"

            elif current_status == "failed":
                icon = "×"
                label = "Failed"
                css_class = "failed"

            else:
                icon = "○"
                label = "Waiting"
                css_class = "waiting"

            rows.append(
                f"""
                <div class="agent-row">
                    <div class="agent-number">
                        {agent_number[agent]}
                    </div>

                    <div class="agent-info">
                        <div class="agent-name">
                            {agent}
                        </div>

                        <div class="agent-state {css_class}">
                            {icon} {label}
                        </div>
                    </div>
                </div>
                """
            )

        activity.markdown(
            f"""
            <div class="agent-panel">
                {"".join(rows)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def on_agent_step(agent: str, status: str):
        agent_status[agent] = status
        render_agents()

    render_agents()

    try:

        result = research_pipeline(
            topic,
            on_step=on_agent_step,
        )

        st.session_state["research_result"] = result

    except Exception as exc:

        for agent, current_status in agent_status.items():
            if current_status == "working":
                agent_status[agent] = "failed"

        render_agents()

        st.error(f"Research failed: {exc}")
        st.stop()


if "research_result" in st.session_state:

    result = st.session_state["research_result"]

    st.divider()

    st.markdown(
        '<div class="page-label">Output</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Research report</div>',
        unsafe_allow_html=True,
    )

    report = result.get("report")

    if report:

        st.markdown(
            '<div class="report-container">',
            unsafe_allow_html=True,
        )

        st.markdown(report)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    tabs = st.tabs(
        [
            "Sources",
            "Evidence",
            "Review",
        ]
    )

    with tabs[0]:

        st.markdown(
            '<div class="source-label">Search output</div>',
            unsafe_allow_html=True,
        )

        search_result = result.get("search_result")

        if search_result:
            st.markdown(search_result)
        else:
            st.info("No search results available.")

    with tabs[1]:

        st.markdown(
            '<div class="source-label">Extracted material</div>',
            unsafe_allow_html=True,
        )

        scrape_result = result.get("scrape_result")

        if scrape_result:
            st.markdown(scrape_result)
        else:
            st.info("No extracted evidence available.")

    with tabs[2]:

        st.markdown(
            '<div class="source-label">Critical assessment</div>',
            unsafe_allow_html=True,
        )

        feedback = result.get("feedback")

        if feedback:
            st.markdown(feedback)
        else:
            st.info("No review available.")


st.markdown(
    """
    <div class="footer">
        Research Workspace · Search · Extraction · Synthesis · Review
    </div>
    """,
    unsafe_allow_html=True,
)