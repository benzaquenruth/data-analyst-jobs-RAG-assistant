# WHAT THIS FILE DOES
# This is the Streamlit app the USER interacts with: a simple web page
# where someone types a question (e.g. "jobs for a junior data analyst
# in Tel Aviv") and gets back an answer from our RAG assistant.
#
# STEP 8 of the monitoring plan: on top of "ask -> answer" (step 1) and
# saving metrics (step 5), this version also:
#   - asks the LLM judge (judge.py) to grade the answer's relevance
#   - saves that judge verdict as feedback (db_feedback.py)
#   - shows two buttons (+1 / -1) so a real user can rate the answer,
#     which also gets saved as feedback
#
# HOW TO RUN THIS FILE:
#   uv run streamlit run app.py

import streamlit as st

from assistant import create_assistant
from db_save import save_conversation
from db_feedback import save_feedback
from judge import evaluate_relevance
from guardrails import check_input



# testing the conection between the app and the BigQuery client for Steamlit live app
#from bigquery_client import get_bigquery_client

#client = get_bigquery_client()

#query = """
#SELECT COUNT(*) AS total
#FROM `massive-bliss-481811-d8.rag_monitoring.conversations`
#"""

#result = client.query(query).result()
#row = next(result)
#st.write(f"✅ BigQuery connected! Conversations in BigQuery: {row.total}")



# create_assistant() builds our RAG object (search index + OpenAI client).
# Streamlit re-runs this whole script top-to-bottom every time the user
# clicks something on the page, so this line runs again each time too —
# that's fine for now (matches how we learned it in the course); we can
# optimize it later with @st.cache_resource if it turns out to be slow.
assistant = create_assistant()

st.title("Data Analyst Job Seeker Assistant")

# unsafe_allow_html=True below only enables the one <a> tag for "data
# pipeline" further down — everything else here is still plain markdown
# (**bold**, emoji, etc.), that part is untouched.
st.markdown("""
Curious what's happening in the Israeli data analyst job market? 👀  
This assistant searches **real job postings**, collected daily through an automated <a href="https://github.com/benzaquenruth/data_analyst_job_seeker_automation" style="color: inherit; text-decoration: underline; font-style: italic;">data pipeline</a>.

📊 The data is automatically updated every day at 10:00 AM!

Ask about roles, skills, locations, seniority, or what jobs fit your background 🚀

**It's a job-matching assistant, not a statistics tool!** it won't answer dataset-wide questions like *"How many jobs are open in Tel Aviv?"*
""", unsafe_allow_html=True)

st.markdown("### Try one of these questions:")

example_questions = [
    "Analyst jobs for someone with data engineering knowledge. Show me what skills, tools, and experience are employers looking for" ,
    "For financial analyst-related roles, what skills, tools, and experience are employers looking for?",
    "Show me gaming company job postings and what they’re looking for",
    "For fraud and risk analytics roles, what skills, tools, and experience are employers looking for?",
    "For marketing analytics roles, what are employers looking for? Show me also job postings and their requirements",
    "What jobs would fit someone with experience in BigQuery, ETL pipelines, and data integration?",
    "Best matches for a data / business analyst background",
    "Best maches for someone with SQL and python skills"
]

selected_question = None

# make each example question a button; if the user clicks one, save it in
for question in example_questions:
    if st.button(question, use_container_width=True):
        st.session_state["question"] = question
        selected_question = question

# 💡 tip that links to the "Recent conversations" section at the bottom of
# the Monitoring dashboard page. It's a real HTML link (<a href=...>) —
# that's what gives it the hand cursor on hover — the inline style just
# strips the usual blue color/underline so it still reads as plain text.
# "Monitoring_Dashboard" is the URL Streamlit gives pages/1_Monitoring_Dashboard.py,
# and "#recent-conversations" is the anchor Streamlit auto-generates for the
# st.subheader("Recent conversations") already at the bottom of that page.
st.markdown(
    '💡 <a href="Monitoring_Dashboard#recent-conversations" '
    'style="color: inherit; text-decoration: none;">'
    '<i>Curious what others asked? Check the latest Q&As on the Monitoring dashboard!</i></a>',
    unsafe_allow_html=True,
)
st.write("")

# A text_input's built-in label can't be made bigger (only bold via
# markdown), so instead we print the label ourselves as a heading — headings
# are bold by default and bigger than normal text — then hide the input's
# own label (label_visibility="collapsed") so it doesn't show up twice.
st.markdown("#### Ask about data analyst jobs in Israel:")

# A single text box where the user types their question.
user_input = st.text_input(
    "Ask about data analyst jobs in Israel:",
    key="question",
    label_visibility="collapsed"
)

# This is the previus version of the text input, which we replaced with the one above.
# typed_question = st.text_input("Ask about data analyst jobs in Israel:")
# user_input = selected_question or typed_question

# The app only does anything once the user clicks "Ask" or one of the example questions. 
if selected_question or st.button("Ask"):
    user_input = selected_question or user_input
    
    try:
        with st.spinner("Checking your questions..."):
            guardrail = check_input(user_input, assistant.llm_client)
    except Exception: 
        st.error("I couldn't check this question right now. Please try again.")
    else: 
        if guardrail.fail : 
            st.warning(
                "That question can't be processed. Please ask an appropriate question about data analyst jobs in Israel."
            )
        else:  
            with st.spinner("Searching job listings and thinking..."):
                # assistant.rag() is the full pipeline from rag_helper.py:
                # hybrid search (keyword + vector) -> build prompt -> call the LLM.
                answer = assistant.rag(user_input)

            # Because assistant is a RAGWithMetrics (see metrics.py), the call
            # above also filled in assistant.last_call with everything about
            # this one exchange: response time, token counts, and cost.
            record = assistant.last_call

            # Save this question + answer + metrics as one row in monitoring.db.
            conversation_id = save_conversation(record, user_input)

            # Ask the LLM judge (judge.py) to grade this answer's relevance to
            # the question, then save that verdict as a "judge" feedback row
            # linked to the conversation we just saved.
            relevance, explanation = evaluate_relevance(user_input, answer)
            save_feedback(
                    conversation_id,
                    "judge",
                    relevance=relevance,
                    explanation=explanation
                )

            # Stash everything needed to redraw this answer on screen into
            # st.session_state, instead of just local variables. Streamlit
            # re-runs this whole file top-to-bottom on every click (e.g. the
            # 👍/👎 buttons below) — a plain local variable would be lost on
            # that rerun, but st.session_state survives across reruns within
            # the same browser tab. This is what keeps the answer visible after
            # giving feedback, instead of it vanishing.
            st.session_state.conversation_id = conversation_id
            st.session_state.last_answer = answer
            st.session_state.last_record = record
            st.session_state.last_relevance = relevance
            st.session_state.last_explanation = explanation


# This block redraws the most recent answer + metrics + judge verdict on
# EVERY rerun (not only right after clicking "Ask"), by reading them back
# from st.session_state instead of local variables. That's what makes the
# answer stay on screen when you click 👍/👎 below — it only changes when
# a new question is asked (session_state gets overwritten above) or the
# page is actually refreshed in the browser (a fresh page load starts a
# new Streamlit session, so session_state is empty again).
if "last_answer" in st.session_state:
    record = st.session_state.last_record
       
    st.success("Done!")
    st.write(st.session_state.last_answer)
    # st.write(f"Response time: {record.response_time:.2f}s")
    # st.write(f"Prompt tokens: {record.prompt_tokens}")
    # st.write(f"Completion tokens: {record.completion_tokens}")
    # st.write(f"Cost: ${record.cost:.4f}")
    # st.write(f"Judge relevance: {st.session_state.last_relevance}")
    # st.write(f"Judge explanation: {st.session_state.last_explanation}")
 
 
# These two buttons live outside both blocks above, so they stay on the
# page across reruns. They only work once a question has been asked at
# least once (i.e. once st.session_state.conversation_id exists) —
# that's what "conversation_id" in st.session_state checks.
st.divider()
# Columns for the two buttons + an empty spacer column, so they sit
# close together on the left instead of spread across the full width.
# (wide enough that "Not helpful" doesn't wrap onto a second line)
col1, col2, _ = st.columns([2, 2, 3])
 
with col1:
    if st.button("👍 Helpful"):
        if "conversation_id" in st.session_state:
            save_feedback(st.session_state.conversation_id, "user", score=1)
            st.write("Thanks for the feedback!")
        else:
            st.write("Ask a question first.")
 
with col2:
    if st.button("👎 Not helpful"):
        if "conversation_id" in st.session_state:
            save_feedback(st.session_state.conversation_id, "user", score=-1)
            st.write("Thanks for the feedback!")
        else:
            st.write("Ask a question first.")
            