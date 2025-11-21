"""
Haas Mill Operator's Manual Assistant
A RAG-powered Q&A system for the Haas Mill Next Generation Control Manual
"""
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

st.set_page_config(
    page_title="Haas Mill Assistant",
    page_icon="🔧",
    layout="wide"
)

def check_password():
    """Returns `True` if the user has entered the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "Keith2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔧 Keith Manufacturing - Machine Assistant")
        st.markdown("### Secure Access")
        st.text_input(
            "Enter Password:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.info("👋 Enter the password to access the machine manuals.")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔧 Keith Manufacturing - Machine Assistant")
        st.markdown("### Secure Access")
        st.text_input(
            "Enter Password:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("❌ Incorrect password. Please try again.")
        return False
    else:
        return True

if not check_password():
    st.stop()

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
INDEX_NAME = 'haas-mill-manual'

@st.cache_resource
def init_clients():
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    return openai_client, index

def get_query_embedding(query, client):
    """Generate embedding for user query"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

def search_manual(query, index, client, top_k=5):
    """Search the manual for relevant sections"""
    query_embedding = get_query_embedding(query, client)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results

def generate_response(query, context_chunks, client):
    """Generate response using GPT-4"""
    context = "\n\n---\n\n".join([
        f"[Page {match['metadata']['page']}]\n{match['metadata']['text']}"
        for match in context_chunks
    ])
    
    prompt = f"""You are a helpful assistant for Haas Mill operators. Answer the user's question based on the following sections from the Haas Mill Operator's Manual.

If the answer is in the manual sections provided, give a clear, detailed answer and cite the relevant page numbers.
If the information isn't in the provided sections, say so and suggest what section of the manual might contain that information.

Always prioritize safety and accurate information. If something involves machine operation or safety, emphasize following proper procedures.

Manual Sections:
{context}

User Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are an expert Haas Mill operator assistant. Provide clear, accurate, and safety-conscious answers based on the official operator's manual."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

st.title("🔧 Haas Mill Operator's Manual Assistant")
st.markdown("### Next Generation Control - 15\" LCD (96-8210)")
st.caption("🔒 Secure access for Keith Manufacturing employees")

st.markdown("""
Ask questions about operating your Haas Mill! This assistant searches the 550-page operator's manual 
to provide accurate answers with page references.

**Example questions:**
- How do I set up a work offset?
- What is the proper procedure for tool changes?
- How do I use the probe system?
- What do the alarm codes mean?
- How do I calibrate the machine?
""")

st.divider()

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.markdown(f"**Page {source['page']}** (Relevance: {source['score']:.2%})")
                    st.text(source['text'][:300] + "...")
                    st.divider()

if prompt := st.chat_input("Ask about the Haas Mill operation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            try:
                openai_client, index = init_clients()
                search_results = search_manual(prompt, index, openai_client)
                
                if not search_results.matches:
                    response = "I couldn't find relevant information in the manual for that question. Please try rephrasing or ask about a different topic."
                    sources = []
                else:
                    response = generate_response(prompt, search_results.matches, openai_client)
                    sources = [
                        {
                            "page": match.metadata['page'],
                            "score": match.score,
                            "text": match.metadata['text']
                        }
                        for match in search_results.matches
                    ]
                
                st.markdown(response)
                
                if sources:
                    with st.expander("📚 View Sources"):
                        for source in sources:
                            st.markdown(f"**Page {source['page']}** (Relevance: {source['score']:.2%})")
                            st.text(source['text'][:300] + "...")
                            st.divider()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please make sure your API keys are correctly configured in Streamlit secrets.")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This assistant uses RAG (Retrieval Augmented Generation) to answer questions about the 
    **Haas Mill Operator's Manual** (Revision U, December 2024).
    
    The system:
    - Searches through 550 pages of documentation
    - Provides accurate answers with page references
    - Cites sources from the official manual
    
    **Manual Details:**
    - Model: Next Generation Control
    - Screen: 15" LCD
    - Part #: 96-8210
    - Revision: U
    - Date: December 2024
    """)
    
    st.divider()
    
    st.header("Safety Notice")
    st.warning("""
    ⚠️ Always verify critical operations in the physical manual. 
    This assistant is a helpful tool but should not replace proper training and certification.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    if st.button("🔓 Logout"):
        st.session_state["password_correct"] = False
        st.rerun()
    
    st.divider()
    st.caption("🔒 Secure System for Keith Manufacturing")
    st.caption("Built with Streamlit • OpenAI • Pinecone")
