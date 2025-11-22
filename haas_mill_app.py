"""
Haas Mill Operator's Manual Assistant - Enterprise Version
A secure RAG-powered Q&A system for the Haas Mill Next Generation Control Manual

Version 2.0 - Enhanced with security features for IT approval
"""
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import time
from datetime import datetime
import csv
import os

st.set_page_config(
    page_title="Haas Mill Assistant",
    page_icon="🔧",
    layout="wide"
)

# ============================================================================
# USER ACCOUNTS DATABASE
# ============================================================================
# In production, move this to a secure database or secrets management
USERS = {
    "john.smith": {"password": "Keith2025", "name": "John Smith", "role": "Operator"},
    "sarah.jones": {"password": "Keith2025", "name": "Sarah Jones", "role": "Operator"},
    "mike.wilson": {"password": "Keith2025", "name": "Mike Wilson", "role": "Supervisor"},
    "admin": {"password": "Keith2025Admin", "name": "Administrator", "role": "Admin"}
}

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================
def log_activity(username, action, details=""):
    """Log user activity for audit trail"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "action": action,
            "details": details
        }
        
        # In production, write to database or proper logging service
        # For now, store in session state
        if 'activity_log' not in st.session_state:
            st.session_state.activity_log = []
        st.session_state.activity_log.append(log_entry)
        
    except Exception as e:
        # Silent fail - don't break app if logging fails
        pass

def log_query(username, question, response, sources, feedback=None):
    """Log questions and responses for quality monitoring"""
    try:
        query_log = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "question": question,
            "response_length": len(response),
            "num_sources": len(sources),
            "sources": [s['page'] for s in sources],
            "feedback": feedback
        }
        
        if 'query_log' not in st.session_state:
            st.session_state.query_log = []
        st.session_state.query_log.append(query_log)
        
    except Exception as e:
        pass

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================
def check_session_timeout():
    """Check if session has timed out (30 minutes)"""
    if 'last_activity' in st.session_state:
        if time.time() - st.session_state.last_activity > 1800:  # 30 minutes
            st.session_state.authenticated = False
            st.session_state.username = None
            return True
    return False

def update_activity():
    """Update last activity timestamp"""
    st.session_state.last_activity = time.time()

# ============================================================================
# RATE LIMITING
# ============================================================================
def check_rate_limit(username):
    """Check if user has exceeded daily query limit"""
    if 'query_counts' not in st.session_state:
        st.session_state.query_counts = {}
    
    today = datetime.now().date().isoformat()
    user_key = f"{username}_{today}"
    
    if user_key not in st.session_state.query_counts:
        st.session_state.query_counts[user_key] = 0
    
    return st.session_state.query_counts[user_key] < 100

def increment_query_count(username):
    """Increment user's query count for today"""
    today = datetime.now().date().isoformat()
    user_key = f"{username}_{today}"
    
    if 'query_counts' not in st.session_state:
        st.session_state.query_counts = {}
    
    if user_key not in st.session_state.query_counts:
        st.session_state.query_counts[user_key] = 0
    
    st.session_state.query_counts[user_key] += 1

# ============================================================================
# AUTHENTICATION
# ============================================================================
def authenticate_user():
    """Handle user authentication with individual accounts"""
    
    def login_submitted():
        """Process login attempt"""
        username = st.session_state.login_username
        password = st.session_state.login_password
        
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_name = USERS[username]["name"]
            st.session_state.user_role = USERS[username]["role"]
            st.session_state.last_activity = time.time()
            st.session_state.login_attempts = 0
            log_activity(username, "login", "Successful login")
        else:
            st.session_state.authenticated = False
            st.session_state.login_attempts = st.session_state.get('login_attempts', 0) + 1
            log_activity(username, "login_failed", "Failed login attempt")

    # Check for timeout
    if check_session_timeout():
        st.warning("⏰ Your session has expired for security. Please log in again.")
        log_activity(st.session_state.get('username', 'unknown'), "timeout", "Session timeout")
    
    # Login form
    if not st.session_state.get("authenticated", False):
        st.title("🔧 Keith Manufacturing - Machine Assistant")
        st.markdown("### Secure Login")
        
        # Safety disclaimer before login
        st.info("""
        👋 **Welcome to the Haas Mill Manual Assistant**
        
        This system provides quick access to the 550-page operator's manual using AI.
        """)
        
        with st.form("login_form"):
            st.text_input("Username:", key="login_username")
            st.text_input("Password:", type="password", key="login_password")
            submitted = st.form_submit_button("Login", on_click=login_submitted)
        
        # Show error if login failed
        if st.session_state.get('login_attempts', 0) > 0:
            st.error("❌ Invalid username or password. Please try again.")
            
        if st.session_state.get('login_attempts', 0) >= 3:
            st.warning("⚠️ Multiple failed login attempts. Contact IT if you need help: brian@keithmfg.com")
        
        st.caption("For access, contact your supervisor or IT department")
        return False
    
    return True

# ============================================================================
# SAFETY DISCLAIMER
# ============================================================================
def show_safety_disclaimer():
    """Show safety disclaimer that user must acknowledge"""
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state.disclaimer_accepted = False
    
    if not st.session_state.disclaimer_accepted:
        st.title("⚠️ Important Safety Notice")
        
        st.error("""
        **CRITICAL: READ BEFORE USING THIS SYSTEM**
        
        This AI assistant is a REFERENCE TOOL ONLY.
        """)
        
        st.warning("""
        **Safety Requirements:**
        
        ✓ Always verify critical procedures in the physical operator's manual
        
        ✓ Never rely solely on AI for safety-critical operations
        
        ✓ Consult your supervisor for any unclear procedures
        
        ✓ In case of emergency, follow official safety protocols
        
        ✓ This tool does NOT replace proper training or certification
        
        **Liability Notice:**
        
        This system provides information from the operator's manual but may contain 
        errors or outdated information. Keith Manufacturing and the system developers 
        are not liable for accidents, injuries, or damage resulting from use of this system.
        
        **By clicking "I Understand and Agree" below, you acknowledge:**
        - You have read and understand these warnings
        - You will verify all critical operations in the official manual
        - You will not use this as the sole source for safety decisions
        - You accept responsibility for proper machine operation
        """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ I Understand and Agree", use_container_width=True):
                st.session_state.disclaimer_accepted = True
                log_activity(st.session_state.username, "disclaimer_accepted", "User acknowledged safety disclaimer")
                st.rerun()
            
            if st.button("❌ I Do Not Agree", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.disclaimer_accepted = False
                st.info("You must accept the safety disclaimer to use this system.")
                st.stop()
        
        st.stop()

# ============================================================================
# API CONFIGURATION
# ============================================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
INDEX_NAME = 'haas-mill-manual'

@st.cache_resource
def init_clients():
    """Initialize API clients"""
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        return openai_client, index
    except Exception as e:
        st.error(f"Failed to initialize API clients: {str(e)}")
        st.info("Please contact IT support: brian@keithmfg.com")
        st.stop()

# ============================================================================
# RAG FUNCTIONS
# ============================================================================
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
    
    prompt = f"""You are an expert assistant for Haas Mill operators at Keith Manufacturing. Answer questions based on the official operator's manual.

CRITICAL RULES:
1. Only answer based on the manual sections provided below
2. If information isn't in the provided sections, say "I don't see that specific information in the manual sections I found. Please check the physical manual or consult your supervisor."
3. ALWAYS cite page numbers for your answers
4. For safety-critical operations, emphasize verifying with the physical manual and supervisor
5. If a question is unclear, ask for clarification

RESPONSE FORMAT:
- Start with the direct answer
- Provide step-by-step instructions if applicable  
- Cite page numbers in parentheses: (page 45)
- End with relevant safety warnings if applicable
- Use clear, simple language

Manual Sections:
{context}

User Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are an expert Haas Mill operator assistant. Provide clear, accurate, and safety-conscious answers based on the official operator's manual. Always prioritize safety and encourage verification of critical procedures."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Authentication check
if not authenticate_user():
    st.stop()

# Safety disclaimer
show_safety_disclaimer()

# Update activity timestamp
update_activity()

# Main UI
st.title("🔧 Haas Mill Operator's Manual Assistant")
st.markdown("### Next Generation Control - 15\" LCD (96-8210)")
st.caption(f"🔒 Logged in as: {st.session_state.user_name} ({st.session_state.user_role})")

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

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.markdown(f"**Page {source['page']}** (Relevance: {source['score']:.2%})")
                    st.text(source['text'][:300] + "...")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about the Haas Mill operation..."):
    
    # Check rate limit
    if not check_rate_limit(st.session_state.username):
        st.error("⚠️ You've reached the daily limit of 100 questions. If you need more access, contact your supervisor.")
        st.stop()
    
    # Increment query count
    increment_query_count(st.session_state.username)
    
    # Update activity
    update_activity()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            try:
                openai_client, index = init_clients()
                search_results = search_manual(prompt, index, openai_client)
                
                if not search_results.matches:
                    response = "I couldn't find relevant information in the manual for that question. Please try rephrasing, check the physical manual, or ask your supervisor."
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
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
                # Log the query
                log_query(st.session_state.username, prompt, response, sources)
                
                # Feedback buttons
                st.divider()
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                        log_query(st.session_state.username, prompt, response, sources, feedback="positive")
                        st.success("Thanks for the feedback!")
                with col2:
                    if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                        log_query(st.session_state.username, prompt, response, sources, feedback="negative")
                        st.success("Thanks for the feedback! We'll work to improve.")
                
            except Exception as e:
                st.error("⚠️ An error occurred while processing your question.")
                st.error(f"Error details: {str(e)}")
                st.info("""
                **What to do:**
                1. Try rephrasing your question
                2. If the problem persists, contact IT support
                3. Email: brian@keithmfg.com
                4. In the meantime, refer to the physical operator's manual
                """)
                log_activity(st.session_state.username, "error", f"Error processing query: {str(e)}")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📊 System Status")
    
    # Usage stats
    today = datetime.now().date().isoformat()
    user_key = f"{st.session_state.username}_{today}"
    queries_today = st.session_state.get('query_counts', {}).get(user_key, 0)
    
    st.metric("Your Questions Today", f"{queries_today}/100")
    
    # Estimated costs (approximate)
    estimated_cost = queries_today * 0.024
    st.metric("Estimated Cost Today", f"${estimated_cost:.2f}")
    
    st.caption(f"Last activity: {datetime.fromtimestamp(st.session_state.last_activity).strftime('%H:%M:%S')}")
    
    st.divider()
    
    st.header("About")
    st.markdown("""
    This assistant uses RAG (Retrieval Augmented Generation) to answer questions about the 
    **Haas Mill Operator's Manual** (Revision U, December 2024).
    
    **The system:**
    - Searches through 550 pages of documentation
    - Provides accurate answers with page references
    - Cites sources from the official manual
    
    **Manual Details:**
    - Model: Next Generation Control
    - Screen: 15" LCD
    - Part #: 96-8210
    - Revision: U
    - Date: December 2024
    
    **Security Features:**
    - Individual user accounts
    - Activity logging
    - 30-minute session timeout
    - Daily query limits (100/day)
    - Usage monitoring
    """)
    
    st.divider()
    
    st.header("⚠️ Safety Reminder")
    st.error("""
    **Always verify critical operations in the physical manual.**
    
    This is a reference tool only. For safety-critical operations, 
    consult the official manual and your supervisor.
    """)
    
    st.divider()
    
    # Action buttons
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔓 Logout"):
        log_activity(st.session_state.username, "logout", "User logged out")
        st.session_state.authenticated = False
        st.session_state.disclaimer_accepted = False
        st.session_state.username = None
        st.rerun()
    
    st.divider()
    
    st.caption("🔒 Secure System for Keith Manufacturing")
    st.caption("Built with Streamlit • OpenAI • Pinecone")
    st.caption("Version 2.0 - Enterprise Edition")
    
    # Admin view (only for admin users)
    if st.session_state.get('user_role') == 'Admin':
        st.divider()
        st.header("👨‍💼 Admin Panel")
        if st.button("View Activity Logs"):
            st.write(st.session_state.get('activity_log', []))
        if st.button("View Query Logs"):
            st.write(st.session_state.get('query_log', []))
