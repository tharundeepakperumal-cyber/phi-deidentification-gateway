
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from collections import Counter

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_URL = os.getenv(
    "PHI_API_URL",
    "http://127.0.0.1:8001"
).rstrip("/")

API_KEY = os.getenv("PHI_API_KEY")

st.set_page_config(
    page_title="PHI Guardian | De-identification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #10243e;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .main-title {
        color: #163454;
        font-size: 34px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .status-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .result-card {
        background: white;
        border: 1px solid #dbe4ee;
        border-radius: 12px;
        padding: 18px;
        min-height: 180px;
    }

    .success-text {
        color: #15803d;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"] {
        background-color: #1769aa;
        border-radius: 9px;
        border: none;
        color: white;
        font-weight: 650;
        min-height: 45px;
    }

    h2, h3 {
        color: #163454;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "processed_count" not in st.session_state:
    st.session_state.processed_count = 0

# --------------------------------------------------
# API HEALTH CHECK
# --------------------------------------------------

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 🛡️ PHI Guardian")
    st.caption("Healthcare Data Protection")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "PHI De-identification",
            "Security & API",
        ],
    )

    st.divider()

    st.markdown("### API STATUS")

    api_online = check_api()

    if api_online:
        st.success("Backend Online")
    else:
        st.error("Backend Offline")

    st.caption("FastAPI • Localhost:8001")

    st.divider()

    st.caption("PHI De-identification Gateway")
    st.caption("Development Prototype")


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Dashboard Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Monitor your healthcare data protection gateway.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "API STATUS",
            "ONLINE" if api_online else "OFFLINE",
        )

    with col2:
        st.metric(
            "SESSION PROCESSES",
            st.session_state.processed_count,
        )

    with col3:
        st.metric(
            "SUPPORTED TEXT LENGTH",
            "20,000 chars",
        )

    st.divider()

    st.subheader("Welcome to PHI Guardian")

    st.write(
        "This dashboard connects to your existing "
        "FastAPI-based PHI De-identification Gateway."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔐 Data Protection")
        st.write(
            "Detect and replace sensitive information "
            "in clinical notes."
        )

    with col2:
        st.markdown("### ⚙️ API Integration")
        st.write(
            "Process text through your authenticated "
            "FastAPI backend."
        )

    st.info(
        "Select PHI De-identification from the sidebar "
        "to process clinical text."
    )

# --------------------------------------------------
# DE-IDENTIFICATION PAGE
# --------------------------------------------------

elif page == "PHI De-identification":

    st.markdown(
        '<div class="main-title">PHI De-identification</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter clinical text to detect and mask sensitive information.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not API_KEY:
        st.error(
            "API key not configured. Check your local .env file."
        )
        st.stop()

    if not api_online:
        st.error(
            "FastAPI backend is offline. Start the API server "
            "on port 8001 before processing."
        )
        st.stop()

    with st.form("phi_form"):

        clinical_text = st.text_area(
            "ORIGINAL CLINICAL TEXT",
            placeholder=(
                "Example: Patient Michael Anderson visited "
                "the hospital. Contact him at "
                "michael@example.com. Phone: 9876543210."
            ),
            height=190,
            max_chars=20000,
        )

        submitted = st.form_submit_button(
            "🛡️ De-identify Clinical Text",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        if not clinical_text.strip():
            st.warning("Please enter clinical text.")
        else:

            try:
                with st.spinner(
                    "Detecting and de-identifying PHI..."
                ):

                    response = requests.post(
                        f"{API_URL}/deidentify",
                        headers={
                            "Content-Type": "application/json",
                            "X-API-Key": API_KEY,
                        },
                        json={
                            "text": clinical_text,
                        },
                        timeout=180,
                    )

                if response.status_code == 200:

                    data = response.json()

                    st.session_state.result = {
                        "original": clinical_text,
                        "deidentified": data["deidentified_text"],
                        "entities": data["detected_entities"],
                    }

                    st.session_state.processed_count += 1

                    st.success(
                        "De-identification completed successfully!"
                    )

                elif response.status_code == 401:
                    st.error(
                        "Authentication failed. Check the API key."
                    )

                else:
                    st.error(
                        f"API returned HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )

            except requests.Timeout:
                st.error("API request timed out. Try again.")

            except requests.RequestException:
                st.error(
                    "Could not connect to the FastAPI backend."
                )

            except (ValueError, KeyError):
                st.error(
                    "The API returned an unexpected response."
                )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    result = st.session_state.result

    if result:

        st.divider()

        st.subheader("Detection Results")

        entities = result["entities"]

        preferred = [
            "PERSON",
            "EMAIL",
            "PHONE",
            "MRN",
        ]

        entity_types = list(
            dict.fromkeys(preferred + list(entities.keys()))
        )

        total_entities = sum(entities.values())

        cols = st.columns(len(entity_types) + 1)

        with cols[0]:
            st.metric("TOTAL DETECTED", total_entities)

        for index, entity in enumerate(entity_types, start=1):
            with cols[index]:
                st.metric(
                    entity,
                    entities.get(entity, 0),
                )

        st.divider()

        st.subheader("Text Comparison")

        left, right = st.columns(2)

        with left:
            st.markdown("### Original Text")
            st.text_area(
                "Original",
                result["original"],
                height=230,
                disabled=True,
                label_visibility="collapsed",
                key="original_output",
            )

        with right:
            st.markdown("### Protected Text")
            st.text_area(
                "De-identified",
                result["deidentified"],
                height=230,
                disabled=True,
                label_visibility="collapsed",
                key="protected_output",
            )

            st.download_button(
                "⬇️ Download Protected Text",
                data=result["deidentified"],
                file_name="deidentified_output.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if st.button("Clear displayed results"):
            st.session_state.result = None
            st.rerun()

        st.caption(
            "Only the de-identified output is offered for download. "
            "Clear results when finished."
        )

# --------------------------------------------------
# SECURITY PAGE
# --------------------------------------------------

elif page == "Security & API":

    st.markdown(
        '<div class="main-title">Security & API Monitoring</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Backend connectivity and security configuration.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Backend",
            "Connected" if api_online else "Disconnected",
        )

    with col2:
        st.metric(
            "Authentication",
            "Configured" if API_KEY else "Missing",
        )

    st.divider()

    st.subheader("API Configuration")

    st.write(f"API URL: `{API_URL}`")
    st.write("Health endpoint: `/health`")
    st.write("De-identification endpoint: `/deidentify`")
    st.write("Authentication header: `X-API-Key`")

    st.divider()

    st.subheader("Security Guidelines")

    st.success("API key is read from the local environment.")

    st.markdown("""
    - Never commit `.env` files.
    - Never expose API keys in browser-side code.
    - Do not use real patient information for demonstrations.
    - Protect original clinical notes and re-identification mappings.
    - Use HTTPS and appropriate access controls before deployment.
    - Avoid retaining clinical text unnecessarily.
    """)

    st.warning(
        "This is a development prototype. "
        "The dashboard is not independently validated "
        "or certified for clinical production use."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PHI Guardian | PHI De-identification Gateway | "
    "Local Development Prototype"
)

