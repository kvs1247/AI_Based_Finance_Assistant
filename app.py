import os
import streamlit as st
from document_parser import extract_text_from_pdf
from embed_store import (
    embed_and_store,
    load_documents_and_index,
    get_company_from_filename
)
from langchain.docstore.document import Document
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Finance AI Assistant")
st.title("📊 Finance Portfolio AI Assistant")

# --- Conversation Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Multi-document Upload ---
st.sidebar.header("Upload your financial documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
)

# --- Display all uploaded files ---
st.sidebar.markdown("### Uploaded Files")
user_docs_folder = "data/user_docs"
if os.path.exists(user_docs_folder):
    uploaded_filenames = os.listdir(user_docs_folder)
    if uploaded_filenames:
        for filename in sorted(uploaded_filenames):
            st.sidebar.write(f"- {filename}")
    else:
        st.sidebar.info("No files uploaded yet.")
else:
    st.sidebar.info("No files uploaded yet.")

all_documents = []

if uploaded_files:
    os.makedirs("data/user_docs", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = f"data/user_docs/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.success(f"File {uploaded_file.name} saved!")

        # Step 1: Extract text
        raw_text = extract_text_from_pdf(file_path)
        # Step 2: Chunk text and create Document objects with company metadata
        text_chunks = raw_text.split("\n\n")
        company = get_company_from_filename(uploaded_file.name)
        documents = [
            Document(page_content=chunk, metadata={"company": company})
            for chunk in text_chunks if chunk.strip()
        ]
        all_documents.extend(documents)
        # Step 3: Embed and store
        embed_and_store(text_chunks, documents)

    st.success("✅ All files processed and embedded.")

# --- Compare Two Documents in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("Compare Two Documents")
compare_files = st.sidebar.file_uploader(
    "Select two PDFs to compare", type=["pdf"], accept_multiple_files=True, key="compare"
)
if compare_files and len(compare_files) == 2:
    texts = [extract_text_from_pdf(f) for f in compare_files]
    prompt = f"""Compare the following two financial documents. Highlight key differences and similarities in performance, revenue, and any notable trends.

Document 1:
{texts[0][:4000]}

Document 2:
{texts[1][:4000]}
"""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    st.subheader("Comparison Analysis:")
    st.write(response.choices[0].message.content.strip())

# --- Load index and documents for Q&A ---
try:
    index, documents = load_documents_and_index()
except FileNotFoundError:
    st.warning("Please upload and embed documents first.")
    index, documents = None, None

# --- Enhanced Summarization ---
st.header("Document Summarization")
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = f"data/user_docs/{uploaded_file.name}"
        raw_text = extract_text_from_pdf(file_path)
        
        if st.button(f"Analyze {uploaded_file.name}"):
            enhanced_prompt = f"""You are a senior financial analyst. Analyze the following financial report and provide the following sections:

1. **Consolidated Financial Snapshot (Latest FY):**
   - Revenue
   - EBITDA
   - PAT
   - Net Worth
   - Net Block (Assets)
   - Total Debt
   - Debt + Long-term Liabilities
   - Current Ratio
   - Quick Ratio
   (If any data is missing, write: "Data not available")

2. **Operational Capacity & Infrastructure:**
   - Number of offices
   - Number of employees
   - Manufacturing facilities and other infrastructure details

3. **Key Observations:**
   - Year-on-year revenue trend, and comment on increases/decreases
   - Gross margin changes and their impact on outlook
   - PAT changes and their impact
   - Any information about receivables, payables, working capital, etc.

4. **Product Portfolio & Market Details:**
   - Key products/services
   - Major customers
   - Collaborations/partnerships
   - Market share and expansion plans

If any section's data is not present, state "Data not available". After all sections, ask: "Let me know if you want a one-pager or a pitch deck summary."

Here is the document content:
\"\"\"
{raw_text[:6000]}
\"\"\"
"""
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a chartered financial analyst."},
                    {"role": "user", "content": enhanced_prompt}
                ]
            )
            st.session_state.current_analysis = response.choices[0].message.content.strip()
            st.session_state.current_file = uploaded_file.name

# Display analysis and format options
if 'current_analysis' in st.session_state:
    st.subheader(f"Analysis of {st.session_state.current_file}")
    st.write(st.session_state.current_analysis)

    format_choice = st.radio(
        "Select output format:",
        ("One-Pager", "Pitch Deck"),
        index=None
    )
    if format_choice:
        format_prompt = f"Create {'a one-page executive summary' if format_choice == 'One-Pager' else 'a 10-slide pitch deck'} based on this analysis: {st.session_state.current_analysis}"
            
        
        formatted_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial communications expert."},
                {"role": "user", "content": format_prompt}
            ]
        )
        st.subheader(f"{format_choice} Format")
        st.write(formatted_response.choices[0].message.content.strip())

# --- Company-specific Query ---
if documents:
    company_names = sorted(list(set(doc.metadata.get("company", "Unknown") for doc in documents)))
    selected_company = st.selectbox("Filter by company for Q&A", ["All"] + company_names)
    if selected_company != "All":
        filtered_docs = [doc for doc in documents if doc.metadata.get("company") == selected_company]
    else:
        filtered_docs = documents
else:
    filtered_docs = []

# --- Chat Interface with Memory (Chat below response and Send button) ---
st.header("Ask a Question")

# Display chat history
with st.container():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AI:** {msg['content']}")

# Input box and Send button
query = st.text_input("Type your question:", key="user_input")
send_clicked = st.button("Send")

if send_clicked and query and index and filtered_docs:
    from rag_pipeline import answer_query
    st.session_state.chat_history.append({"role": "user", "content": query})
    response = answer_query(query, index, filtered_docs, openai_client)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()
