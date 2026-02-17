
# Streamlit Application Specification: AI-Augmented 10-K Analysis for Investment Professionals

## 1. Application Overview

### Purpose
This Streamlit application serves as a development blueprint for "Idea Generation with GPT: Unlocking 10-K Insights for Investment Professionals." Its primary purpose is to demonstrate how Large Language Models (LLMs) can act as an "AI junior analyst" to efficiently extract, audit, and compare material risks and opportunities from SEC 10-K filings. The application aims to automate preliminary research, enhance consistency across analyses, and free up senior analysts for higher-value qualitative judgment and client engagement, all while adhering to CFA ethical standards, particularly regarding diligence and the human-in-the-loop framework.

### High-Level Story Flow

The application guides the user, Sarah Chen (CFA Charterholder and Senior Equity Research Analyst), through a structured workflow:

1.  **Welcome & Setup:** Sarah is introduced to the application and sets up her OpenAI API key, acknowledging data privacy considerations.
2.  **Data Acquisition:** She selects a universe of companies (ticker, year) for analysis. The application then loads the "Item 1A: Risk Factors" sections of their 10-K filings, automatically handling text chunking for LLM context window limits.
3.  **Prompt Engineering:** Sarah reviews and can configure pre-designed prompt templates for both risk and opportunity extraction, understanding how different LLM `temperature` settings impact output for factual extraction versus nuanced signal identification.
4.  **LLM Extraction:** With a single click, the LLM processes the selected filings, extracting structured JSON outputs for risks (category, severity, novelty, quote, implication) and opportunities (category, quote, risk-to-opportunity, timeframe).
5.  **Hallucination Audit:** A crucial step for compliance, the application performs an automated audit to verify LLM-extracted quotes against the original text, flagging potential hallucinations and calculating verification rates.
6.  **Comparative Analysis:** The extracted data is aggregated into comparison tables and visualized using stacked bar charts (risk/opportunity category distribution) and heatmaps (risk severity) to identify industry trends and company-specific outliers.
7.  **Cost Management:** The application provides a clear breakdown of LLM API token usage and estimated costs, justifying the ROI of AI augmentation to budget-conscious stakeholders.
8.  **Analyst Review & Compliance:** Finally, a first-draft analyst review checklist is generated for a selected company, emphasizing the professional obligation of human review and validation, in line with CFA Standards V(A) and I(C), ensuring a human-in-the-loop framework.

This workflow simulates a practical, AI-augmented research process, reinforcing the professional obligation of human review and validation.

## 2. Code Requirements

### Import Statement

```python
from source import *
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # Necessary for API key management and file system operations within Streamlit
import json # For displaying JSON outputs
import tiktoken # For token counting if not already wrapped by source functions
from openai import OpenAI # For initializing client
```

### `st.session_state` Design

All `st.session_state` keys will be initialized at the start of the `app.py` script. A helper function `_initialize_session_state()` will be used to ensure keys exist before access.

*   `st.session_state.current_page`: `str`, default `'Home'`. Controls which main content section is displayed.
*   `st.session_state.openai_api_key`: `str` or `None`, stores the user-provided OpenAI API key. Default `None`.
*   `st.session_state.client`: `OpenAI` client object. Initialized after API key is set. Default `None`.
*   `st.session_state.companies_for_selection`: `list[dict]`, the predefined list of companies from `source.py`. Initialized from `source.companies_data`.
*   `st.session_state.selected_company_tickers`: `list[str]`, tickers of companies selected by the user for analysis. Default `['AAPL', 'JPM']`.
*   `st.session_state.selected_company_years`: `dict[str, int]`, mapping ticker to selected year. Default `{t['ticker']: t['year'] for t in source.companies_data}`.
*   `st.session_state.filings_raw`: `dict[str, dict]`, stores raw text, chunks, and token counts for each selected company. Default `{}`. Populated by "Load Filings" button.
*   `st.session_state.risk_extraction_temperature`: `float`, LLM temperature for risk extraction. Default `0.1`.
*   `st.session_state.opp_extraction_temperature`: `float`, LLM temperature for opportunity extraction. Default `0.7`.
*   `st.session_state.all_risks_data`: `dict[str, list[dict]]`, stores extracted risks for each company. Default `{}`. Populated after extraction.
*   `st.session_state.all_opportunities_data`: `dict[str, list[dict]]`, stores extracted opportunities for each company. Default `{}`. Populated after extraction.
*   `st.session_state.total_input_tokens`: `int`, cumulative input tokens for all LLM calls. Default `0`.
*   `st.session_state.total_output_tokens`: `int`, cumulative output tokens for all LLM calls. Default `0`.
*   `st.session_state.all_audit_results`: `dict[str, pd.DataFrame]`, stores audit results for each company. Default `{}`. Populated after audit.
*   `st.session_state.comparison_risks_df`: `pd.DataFrame`, aggregated risks for cross-company comparison. Default `pd.DataFrame()`.
*   `st.session_state.comparison_opportunities_df`: `pd.DataFrame`, aggregated opportunities for cross-company comparison. Default `pd.DataFrame()`.
*   `st.session_state.total_cost`: `float`, estimated total LLM API cost. Default `0.0`.
*   `st.session_state.selected_company_for_checklist`: `str`, ticker for which to generate the analyst checklist. Default `None` (or first available ticker after data load).

### UI Interactions, Function Invocation, and State Management

#### Initialization (`_initialize_session_state()` function)

```python
def _initialize_session_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if "client" not in st.session_state:
        st.session_state.client = None
        if st.session_state.openai_api_key:
            try:
                st.session_state.client = OpenAI(api_key=st.session_state.openai_api_key)
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {e}")
                st.session_state.client = None
    if "companies_for_selection" not in st.session_state:
        st.session_state.companies_for_selection = source.companies_data # Using source.companies_data directly
    if "selected_company_tickers" not in st.session_state:
        st.session_state.selected_company_tickers = [c['ticker'] for c in st.session_state.companies_for_selection]
    if "selected_company_years" not in st.session_state:
        st.session_state.selected_company_years = {c['ticker']: c['year'] for c in st.session_state.companies_for_selection}
    if "filings_raw" not in st.session_state:
        st.session_state.filings_raw = {}
    if "risk_extraction_temperature" not in st.session_state:
        st.session_state.risk_extraction_temperature = 0.1
    if "opp_extraction_temperature" not in st.session_state:
        st.session_state.opp_extraction_temperature = 0.7
    if "all_risks_data" not in st.session_state:
        st.session_state.all_risks_data = {}
    if "all_opportunities_data" not in st.session_state:
        st.session_state.all_opportunities_data = {}
    if "total_input_tokens" not in st.session_state:
        st.session_state.total_input_tokens = 0
    if "total_output_tokens" not in st.session_state:
        st.session_state.total_output_tokens = 0
    if "all_audit_results" not in st.session_state:
        st.session_state.all_audit_results = {}
    if "comparison_risks_df" not in st.session_state:
        st.session_state.comparison_risks_df = pd.DataFrame()
    if "comparison_opportunities_df" not in st.session_state:
        st.session_state.comparison_opportunities_df = pd.DataFrame()
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "selected_company_for_checklist" not in st.session_state:
        st.session_state.selected_company_for_checklist = None
```

#### Global Streamlit Structure

```python
# Call _initialize_session_state() at the very beginning of app.py
_initialize_session_state()

# Sidebar Navigation
st.sidebar.title("AI Analyst Workflow")
page_selection = st.sidebar.selectbox(
    "Navigate",
    ["Home", "1. Setup & Data Loading", "2. LLM Prompts & Extraction",
     "3. Hallucination Audit", "4. Comparative Analysis",
     "5. Cost & ROI", "6. Analyst Review & Compliance"],
    index=["Home", "1. Setup & Data Loading", "2. LLM Prompts & Extraction",
           "3. Hallucination Audit", "4. Comparative Analysis",
           "5. Cost & ROI", "6. Analyst Review & Compliance"].index(st.session_state.current_page)
)
if page_selection != st.session_state.current_page:
    st.session_state.current_page = page_selection
    st.experimental_rerun()

# Main Content Area
if st.session_state.current_page == "Home":
    # Home Page Content
    # ...
elif st.session_state.current_page == "1. Setup & Data Loading":
    # Data Loading Page Content
    # ...
# ... and so on for other pages
```

#### Page: Home

##### Markdown Content

```python
st.title("💡 Idea Generation with GPT: Unlocking 10-K Insights")
st.markdown(f"## Welcome, Sarah Chen, CFA Charterholder")
st.markdown(f"As a Senior Equity Research Analyst at 'Global Markets Insight' (GMI), your daily workflow involves sifting through numerous SEC 10-K filings to identify material risks and opportunities.")
st.markdown(f"Manually extracting these insights from dense legal prose is incredibly time-consuming and can lead to overlooked nuances.")
st.markdown(f"This application empowers you to leverage Large Language Models (LLMs) as an 'AI junior analyst' to automate preliminary research, enhance consistency, and free up your time for higher-value qualitative judgment and client engagement.")
st.markdown(f"---")
st.markdown(f"### Persona: Sarah Chen, CFA Charterholder")
st.markdown(f"Sarah leads a team of equity analysts covering the technology and healthcare sectors. GMI aims to integrate AI to improve efficiency and consistency in financial analysis.")
st.markdown(f"### Scenario:")
st.markdown(f"Your goal is to efficiently extract, audit, and compare critical information from 10-K filings using AI, maintaining the highest standards of diligence and ethics.")
st.markdown(f"---")
st.markdown(f"### CFA Curriculum Connection")
st.markdown(f"This workflow directly supports **Financial Statement Analysis** (Levels I-II) by automating the first pass of risk factor analysis, freeing you to focus on judgment-intensive interpretation.")
st.markdown(f"It aligns with **Equity Investments** (Level II) by structuring risk and opportunity identification, mirroring an analyst's mental model.")
st.markdown(f"Crucially, it addresses **Ethics (CFA Standard V(A) – Diligence)** by emphasizing human review and validation to satisfy the 'reasonable basis' requirement for AI-assisted analysis.")
st.markdown(f"It also touches on **Ethics (CFA Standard I(C) – Misrepresentation)** by advocating for clear disclosure of AI-assisted work.")
st.markdown(f"---")
st.warning("⚠️ **Practitioner Warning: Data Privacy is Non-Negotiable**")
st.markdown(f"While SEC 10-K filings are public, many firms analyze internal, confidential documents. **Never send confidential text to external LLM APIs.** Always check your firm's data classification policy.")
st.markdown(f"This case study uses only public filings. For confidential data, local or on-premises LLM deployments (e.g., Ollama, vLLM) are required.")
```

#### Page: 1. Setup & Data Loading

##### Widgets & UI

*   `st.text_input`: For OpenAI API Key (`st.session_state.openai_api_key`). On change, update `st.session_state.openai_api_key` and re-initialize `st.session_state.client`.
*   `st.multiselect`: To select company tickers (`st.session_state.selected_company_tickers`).
*   For each selected company, `st.number_input` for the fiscal year (`st.session_state.selected_company_years[ticker]`).
*   `st.button`: "Load Filings".

##### Markdown Content

```python
st.title("1. Setting Up the Environment and Loading Data")
st.markdown(f"As a CFA Charterholder, Sarah values efficiency and accuracy. The first step in this AI-augmented workflow is to set up her Python environment and prepare the raw 10-K text for analysis.")
st.markdown(f"For this lab, we'll work with pre-downloaded text files to ensure consistent results and focus on the LLM interaction.")
st.markdown(f"### API Key Configuration")
st.info(f"Please ensure your OpenAI API key is configured. You can set it as an environment variable (recommended) or paste it below.")
# API key input widget
api_key_input = st.text_input("OpenAI API Key:", type="password", value=st.session_state.openai_api_key, key="api_key_input")
if api_key_input and api_key_input != st.session_state.openai_api_key:
    st.session_state.openai_api_key = api_key_input
    os.environ["OPENAI_API_KEY"] = api_key_input
    try:
        st.session_state.client = OpenAI(api_key=api_key_input)
        st.success("OpenAI client initialized!")
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.session_state.client = None
elif not api_key_input:
    st.session_state.client = None # Clear client if key is removed

st.markdown(f"### Story: Preparing the Digital Stack of 10-K Filings")
st.markdown(f"Sarah needs to analyze a peer group of companies. Before the LLM can process these documents, they need to be loaded and, if excessively long, chunked to fit within the LLM's context window.")
st.markdown(f"This is analogous to a junior analyst physically organizing filing documents and highlighting relevant sections.")
st.markdown(f"The token limit is a critical consideration for LLMs. If a document exceeds the model's context window (e.g., 128k tokens for `gpt-4o`), it must be split into smaller, overlapping chunks.")
st.markdown(f"Overlap helps maintain continuity, much like reading a chapter summary with a few sentences from the previous chapter to avoid losing context.")
st.markdown(r"$$ N_{{\text{{tokens}}}} $$")
st.markdown(r"where $N_{{\text{{tokens}}}}$ is the number of tokens, distinct from the number of words or characters, directly impacting API costs and context limits.")
st.markdown(f"The token calculation is based on the LLM's internal tokenizer.")

st.markdown(f"### Select Companies and Years")
# Company selection widgets
selected_tickers = st.multiselect(
    "Select Companies for Analysis:",
    options=[c['ticker'] for c in st.session_state.companies_for_selection],
    default=st.session_state.selected_company_tickers
)
if selected_tickers:
    st.session_state.selected_company_tickers = selected_tickers
    st.markdown(f"Configure Fiscal Year for Selected Companies:")
    temp_years = st.session_state.selected_company_years.copy()
    for ticker in st.session_state.selected_company_tickers:
        default_year = next((c['year'] for c in st.session_state.companies_for_selection if c['ticker'] == ticker), 2024)
        year = st.number_input(f"Fiscal Year for {ticker}:", min_value=2000, max_value=datetime.now().year, value=temp_years.get(ticker, default_year), key=f"year_{ticker}")
        temp_years[ticker] = year
    st.session_state.selected_company_years = temp_years

if st.button("Load Filings"):
    if not st.session_state.client:
        st.error("Please provide a valid OpenAI API Key first.")
    elif not st.session_state.selected_company_tickers:
        st.warning("Please select at least one company.")
    else:
        st.session_state.filings_raw = {}
        for ticker in st.session_state.selected_company_tickers:
            year = st.session_state.selected_company_years.get(ticker, 2024) # Default year if not found
            with st.spinner(f"Loading and processing {ticker}'s 10-K for FY{year}..."):
                text = source.get_10k_risk_factors(ticker, year)
                if text:
                    n_tokens = source.count_tokens(text)
                    st.write(f"{ticker}: {len(text):,} characters, {n_tokens:,} tokens")
                    chunks = [text]
                    if n_tokens > 8000: # Example threshold for chunking
                        chunks = source.chunk_text(text, max_tokens=6000, overlap=500)
                        st.write(f" -> Split into {len(chunks)} chunks due to context window limits.")
                    st.session_state.filings_raw[ticker] = {'text': text, 'chunks': chunks, 'token_count': n_tokens}
                else:
                    st.error(f"Could not load filing for {ticker} ({year}). Please ensure the dummy file exists or update the filepath.")
        if st.session_state.filings_raw:
            st.success("Filings loaded and processed!")
            st.markdown(f"### Explanation of Output")
            st.markdown(f"Sarah sees the character and token counts for each company's 'Risk Factors' section. This gives her a practical understanding of the data volume. For very long documents, the output confirms that the text was automatically chunked, a crucial step to manage LLM context windows and avoid errors or truncated responses. This ensures the 'junior analyst' can process even the most verbose filings.")
        else:
            st.warning("No filings were successfully loaded.")

if st.session_state.filings_raw:
    st.markdown("### Loaded Filings Summary:")
    for ticker, data in st.session_state.filings_raw.items():
        st.markdown(f"- **{ticker} (FY{st.session_state.selected_company_years.get(ticker, 'N/A')}):** {len(data['text']):,} characters, {data['token_count']:,} tokens, {len(data['chunks'])} chunks.")
```

#### Page: 2. LLM Prompts & Extraction

##### Widgets & UI

*   `st.slider`: For `risk_extraction_temperature` (`st.session_state.risk_extraction_temperature`).
*   `st.slider`: For `opp_extraction_temperature` (`st.session_state.opp_extraction_temperature`).
*   `st.expander` with `st.code` to display prompt templates.
*   `st.button`: "Extract Risks & Opportunities".

##### Markdown Content

```python
st.title("2. Crafting the Expert Prompt & Unleashing the Junior Analyst")
st.markdown(f"Sarah knows that the quality of the LLM's output heavily depends on the instructions it receives. This step is about designing a sophisticated prompt that assigns the LLM the role of an expert equity analyst, enforces anti-hallucination rules, and specifies structured JSON output for easy programmatic parsing.")
st.markdown(f"### Story: Architecting the \"Junior Analyst's\" Mindset")
st.markdown(f"Sarah, leveraging her deep experience, designs a \"system prompt\" to define the LLM's persona and critical guidelines, and a \"task prompt\" to specify the exact information to extract. This is similar to training a new junior analyst on how to approach 10-K analysis: what to look for, how to verify facts, and how to format their findings.")
st.markdown(f"The `temperature` parameter is key here; a low value (e.g., $T=0.1$) is chosen for factual extraction to minimize creativity and maximize deterministic, accurate output.")
st.markdown(r"The autoregressive nature of LLMs means they predict the next token based on previous tokens and a probability distribution. The temperature $T$ parameter controls the \"peakiness\" of this distribution:")
st.markdown(r"$$ P(\text{{token}}_t | \text{{token}}_1, ..., \text{{token}}_{{t-1}}) = \text{{softmax}}\left(\frac{{h_t}}{{T}}\right) $$")
st.markdown(r"where $h_t$ is the hidden state at position $t$ (from the transformer's attention layers). A lower $T$ makes the distribution sharper, favoring higher-probability tokens and thus more deterministic output, which is essential for factual extraction tasks like risk analysis.")

st.markdown(f"### Prompt Templates")
with st.expander("View Risk Extraction Prompts"):
    st.markdown("#### System Prompt for Risk Extraction (`SYSTEM_PROMPT_RISK`)")
    st.code(source.SYSTEM_PROMPT_RISK)
    st.markdown("#### Task Prompt for Risk Extraction (`TASK_PROMPT_RISK_TEMPLATE`)")
    st.code(source.TASK_PROMPT_RISK_TEMPLATE)

with st.expander("View Opportunity Identification Prompts"):
    st.markdown("#### System Prompt for Opportunity Identification (`SYSTEM_PROMPT_OPPS`)")
    st.code(source.SYSTEM_PROMPT_OPPS)
    st.markdown("#### Task Prompt for Opportunity Identification (`TASK_PROMPT_OPPS_TEMPLATE`)")
    st.code(source.TASK_PROMPT_OPPS_TEMPLATE)

st.markdown(f"### Configure LLM Temperature")
st.session_state.risk_extraction_temperature = st.slider(
    "Temperature for Risk Extraction (0.0 - 1.0, lower for factual):",
    min_value=0.0, max_value=1.0, value=st.session_state.risk_extraction_temperature, step=0.1
)
st.session_state.opp_extraction_temperature = st.slider(
    "Temperature for Opportunity Extraction (0.0 - 1.0, higher for creative):",
    min_value=0.0, max_value=1.0, value=st.session_state.opp_extraction_temperature, step=0.1
)

st.markdown(f"### Story: Automated Risk and Opportunity Identification")
st.markdown(f"Sarah executes the extraction functions, which send the carefully crafted prompts and 10-K text to the `gpt-4o` model. She sets a low `temperature` for risks to ensure factual extraction and a slightly higher one for opportunities to identify nuanced signals.")
st.markdown(f"The LLM acts as a diligent junior analyst, rapidly sifting through the text and summarizing key risks and opportunities into a structured format.")

if st.button("Extract Risks & Opportunities"):
    if not st.session_state.client:
        st.error("OpenAI client not initialized. Please configure your API key on the 'Setup & Data Loading' page.")
    elif not st.session_state.filings_raw:
        st.warning("No filings loaded. Please load filings on the 'Setup & Data Loading' page first.")
    else:
        st.session_state.all_risks_data = {}
        st.session_state.all_opportunities_data = {}
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0

        for ticker, data in st.session_state.filings_raw.items():
            text_to_process = data['chunks'][0] if len(data['chunks']) > 0 else data['text']
            company_year = st.session_state.selected_company_years.get(ticker, 2024)

            st.markdown(f"#### Processing {ticker} (FY{company_year})...")
            with st.spinner(f"Extracting risks for {ticker}..."):
                # Pass explicit temperature to extract_risks
                risks, input_t_risk, output_t_risk = source.extract_risks(
                    ticker, company_year, text_to_process, temperature=st.session_state.risk_extraction_temperature
                )
                st.session_state.all_risks_data[ticker] = risks
                st.session_state.total_input_tokens += input_t_risk
                st.session_state.total_output_tokens += output_t_risk
                st.success(f"{ticker}: Extracted {len(risks)} risk factors.")
                if risks:
                    st.json(risks[0]) # Show first risk as example

            with st.spinner(f"Extracting opportunities for {ticker}..."):
                # Pass explicit temperature to extract_opportunities
                opportunities, input_t_opp, output_t_opp = source.extract_opportunities(
                    ticker, company_year, text_to_process, temperature=st.session_state.opp_extraction_temperature
                )
                st.session_state.all_opportunities_data[ticker] = opportunities
                st.session_state.total_input_tokens += input_t_opp
                st.session_state.total_output_tokens += output_t_opp
                st.success(f"{ticker}: Extracted {len(opportunities)} opportunities.")
                if opportunities:
                    st.json(opportunities[0]) # Show first opportunity as example

        st.success("All extractions complete!")
        st.markdown(f"### Explanation of Output")
        st.markdown(f"Sarah now has a structured list of risks and opportunities for each company. The immediate, structured JSON output contrasts sharply with the hours it would take to manually read and summarize.")
        st.markdown(f"The `OpenAI` client handles the API communication, and `json.loads()` effectively converts the LLM's text response into Python lists of dictionaries, ready for further analysis.")
        st.markdown(f"For opportunity extraction, a slightly higher `temperature` was used compared to risk extraction. This allows the LLM a bit more creativity to identify nuanced signals from management's forward-looking statements, without veering into outright fabrication. This dual-lens approach provides a more holistic preliminary view.")

if st.session_state.all_risks_data or st.session_state.all_opportunities_data:
    st.markdown("### Extracted Data Summary:")
    for ticker in st.session_state.selected_company_tickers:
        num_risks = len(st.session_state.all_risks_data.get(ticker, []))
        num_opps = len(st.session_state.all_opportunities_data.get(ticker, []))
        st.markdown(f"- **{ticker}:** {num_risks} Risks, {num_opps} Opportunities")
```

#### Page: 3. Hallucination Audit

##### Widgets & UI

*   `st.button`: "Run Hallucination Audit".
*   `st.dataframe`: To display audit results for each company.
*   `st.plotly_chart` or `st.pyplot`: To display the Hallucination Audit Dashboard (V3).

##### Markdown Content

```python
st.title("3. Verifying the \"Junior Analyst\": Hallucination Audit")
st.markdown(f"As a CFA Charterholder, Sarah understands the ethical imperative of diligence (CFA Standard V(A) - Diligence). The LLM's output is a draft and must be audited for hallucinations (fabrications). This step cross-checks extracted quotes against the original text to ensure factual accuracy.")
st.markdown(f"### Story: Quality Control - Catching AI Fabrications")
st.markdown(f"Sarah's firm adheres to strict compliance frameworks. She cannot blindly trust the AI. This audit function acts as a critical second pair of eyes, automatically checking if the LLM's \"supporting quotes\" actually appear in the original 10-K text. This quantifies the trustworthiness of the AI's output, allowing her to prioritize human review for flagged items.")
st.markdown(r"The **Hallucination Rate ($H$)** quantifies the proportion of extracted items that cannot be verified in the source text.")
st.markdown(r"$$ H = \frac{{N_{{\text{{flagged}}}}}}{{N_{{\text{{total extracted}}}}}} $$")
st.markdown(r"where $N_{{\text{{flagged}}}}$ are items with a low word-match percentage (e.g., < 75%) or no exact substring match, and $N_{{\text{{total extracted}}}}$ is the total number of items extracted by the LLM.")
st.markdown(r"A target for $H$ in structured extraction is typically $< 0.05$ (less than 5%).")

st.markdown(r"**Precision ($P$)** measures how many of the LLM's extracted items are actually correct and relevant.")
st.markdown(r"$$ P = \frac{{N_{{\text{{verified and relevant}}}}}}{{N_{{\text{{total extracted}}}}}} $$")
st.markdown(r"**Recall ($R$)** measures how many of the truly relevant items in the document were extracted by the LLM.")
st.markdown(r"$$ R = \frac{{N_{{\text{{verified and relevant}}}}}}{{N_{{\text{{true risks in filing}}}}}} $$")
st.markdown(r"Note: Recall ($R$) often requires human judgment to determine $N_{{\text{{true risks in filing}}}}$, which is the total number of material risks a human analyst would identify.")

if st.button("Run Hallucination Audit"):
    if not st.session_state.all_risks_data:
        st.warning("No risks extracted. Please run LLM extraction on the previous page.")
    elif not st.session_state.filings_raw:
        st.warning("No raw filings loaded. Please load filings on the 'Setup & Data Loading' page.")
    else:
        st.session_state.all_audit_results = {}
        audit_summary_data = []
        for ticker in st.session_state.selected_company_tickers:
            risks = st.session_state.all_risks_data.get(ticker, [])
            source_text = st.session_state.filings_raw.get(ticker, {}).get('text', '')
            if risks and source_text:
                with st.spinner(f"Auditing risks for {ticker}..."):
                    audit_df = source.audit_hallucinations(risks, source_text)
                    st.session_state.all_audit_results[ticker] = audit_df
                    n_verified = audit_df['VERIFIED'].sum()
                    n_total = len(audit_df)
                    hallucination_rate = 1 - (n_verified / n_total) if n_total > 0 else 0
                    audit_summary_data.append({'Company': ticker, 'Verified Risks': n_verified, 'Total Risks': n_total, 'Hallucination Rate': f"{hallucination_rate:.2%}"})
            else:
                st.info(f"Skipping audit for {ticker}: No risks extracted or no source text found.")
        
        if audit_summary_data:
            audit_summary_df = pd.DataFrame(audit_summary_data)
            st.markdown("### Hallucination Audit Summary (V3 Dashboard)")
            st.dataframe(audit_summary_df)

            # V3 Hallucination Audit Dashboard Visualization (example: bar chart of verification rates)
            if not audit_summary_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                audit_summary_df['Verification_Rate_Pct'] = audit_summary_df['Verified Risks'] / audit_summary_df['Total Risks'] * 100
                sns.barplot(x='Company', y='Verification_Rate_Pct', data=audit_summary_df, palette='viridis', ax=ax)
                ax.set_title('Risk Verification Rate by Company', fontsize=16)
                ax.set_ylabel('Verification Rate (%)', fontsize=12)
                ax.set_xlabel('Company', fontsize=12)
                ax.set_ylim(0, 100)
                st.pyplot(fig) # Displaying the plot
                plt.close(fig) # Close the figure to free up memory

            st.markdown(f"### Explanation of Output")
            st.markdown(f"Sarah receives a detailed audit report for each company. The dashboard clearly visualizes the verification rate, showing how frequently the LLM's claims are directly supported by the source text.")
            st.markdown(f"Any \"POTENTIAL HALLUCINATION\" flags items for Sarah's immediate human review. This rigorous verification process directly addresses CFA Standard V(A) by ensuring a \"reasonable basis\" for the AI-assisted analysis, building trust in the tool and enhancing the quality of her team's preliminary research.")
        else:
            st.info("No audit data to display.")

if st.session_state.all_audit_results:
    st.markdown("### Detailed Audit Results")
    for ticker, audit_df in st.session_state.all_audit_results.items():
        st.markdown(f"#### {ticker} Audit Report")
        st.dataframe(audit_df)
```

#### Page: 4. Comparative Analysis

##### Widgets & UI

*   Displays `st.dataframe` for `st.session_state.comparison_risks_df` and `st.session_state.comparison_opportunities_df`.
*   Displays `st.pyplot` for Risk Category Distribution Bar Chart (V1).
*   Displays `st.pyplot` for Severity Heatmap (V2).
*   Displays `st.pyplot` for Opportunity Category Distribution Bar Chart.

##### Markdown Content

```python
st.title("4. Peer Group Analysis: Comparing Risks and Opportunities")
st.markdown(f"After extracting data from individual filings, Sarah's next crucial step is to aggregate and compare these insights across the peer group. This allows her to identify common industry risks, unique company-specific threats, and differentiated growth strategies.")
st.markdown(f"### Story: Uncovering Industry Trends and Outliers")
st.markdown(f"Sarah transforms the individual JSON outputs into comprehensive pandas DataFrames, which are the cornerstones for quantitative comparison. She then visualizes these comparisons, quickly revealing patterns that would be tedious to spot manually.")
st.markdown(f"This cross-company analysis is vital for portfolio managers who need to understand relative positioning and industry-wide exposures.")

if not st.session_state.all_risks_data and not st.session_state.all_opportunities_data:
    st.warning("No risks or opportunities extracted. Please run LLM extraction on the 'LLM Prompts & Extraction' page first.")
else:
    # Aggregate all risks into a single DataFrame
    all_risks_flat = []
    for ticker, risks in st.session_state.all_risks_data.items():
        for risk in risks:
            risk_copy = risk.copy()
            risk_copy['Company'] = ticker
            all_risks_flat.append(risk_copy)
    st.session_state.comparison_risks_df = pd.DataFrame(all_risks_flat)

    # Aggregate all opportunities into a single DataFrame
    all_opportunities_flat = []
    for ticker, opportunities in st.session_state.all_opportunities_data.items():
        for opp in opportunities:
            opp_copy = opp.copy()
            opp_copy['Company'] = ticker
            all_opportunities_flat.append(opp_copy)
    st.session_state.comparison_opportunities_df = pd.DataFrame(all_opportunities_flat)

    st.markdown("### Cross-Company Risk Comparison Table")
    if not st.session_state.comparison_risks_df.empty:
        st.dataframe(st.session_state.comparison_risks_df[['Company', 'risk_name', 'category', 'severity', 'novel', 'investment_implication']])
    else:
        st.info("No risk data to display for comparison.")

    st.markdown("### Risk Category Distribution by Company (V1)")
    if not st.session_state.comparison_risks_df.empty:
        risk_category_pivot = st.session_state.comparison_risks_df.groupby(['Company', 'category']).size().unstack(fill_value=0)
        fig_v1, ax_v1 = plt.subplots(figsize=(14, 7))
        risk_category_pivot.plot(kind='bar', stacked=True, colormap='Set2', edgecolor='black', ax=ax_v1)
        ax_v1.set_title('Risk Category Distribution by Company', fontsize=16)
        ax_v1.set_ylabel('Number of Risk Factors', fontsize=12)
        ax_v1.set_xlabel('Company', fontsize=12)
        ax_v1.set_xticklabels(ax_v1.get_xticklabels(), rotation=45, ha='right')
        ax_v1.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig_v1, use_container_width=True)
        plt.close(fig_v1)
    else:
        st.info("Not enough data to generate Risk Category Distribution chart.")

    st.markdown("### Average Risk Severity by Company and Category (V2)")
    if not st.session_state.comparison_risks_df.empty:
        severity_map = {'High': 3, 'Medium': 2, 'Low': 1}
        temp_df = st.session_state.comparison_risks_df.copy()
        temp_df['severity_num'] = temp_df['severity'].map(severity_map)
        severity_pivot = temp_df.pivot_table(
            values='severity_num', index='Company', columns='category', aggfunc='mean'
        ).fillna(0)
        fig_v2, ax_v2 = plt.subplots(figsize=(12, 6))
        sns.heatmap(severity_pivot, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5, linecolor='black', vmin=1, vmax=3, ax=ax_v2)
        ax_v2.set_title('Average Risk Severity by Company and Category', fontsize=16)
        ax_v2.set_ylabel('Company', fontsize=12)
        ax_v2.set_xlabel('Risk Category', fontsize=12)
        st.pyplot(fig_v2, use_container_width=True)
        plt.close(fig_v2)
    else:
        st.info("Not enough data to generate Severity Heatmap.")

    st.markdown("### Cross-Company Opportunity Comparison Table")
    if not st.session_state.comparison_opportunities_df.empty:
        st.dataframe(st.session_state.comparison_opportunities_df[['Company', 'opportunity_name', 'category', 'risk_to_opportunity', 'timeframe']])
    else:
        st.info("No opportunity data to display for comparison.")

    st.markdown("### Opportunity Category Distribution by Company")
    if not st.session_state.comparison_opportunities_df.empty:
        opportunity_category_pivot = st.session_state.comparison_opportunities_df.groupby(['Company', 'category']).size().unstack(fill_value=0)
        fig_opps, ax_opps = plt.subplots(figsize=(14, 7))
        opportunity_category_pivot.plot(kind='bar', stacked=True, colormap='Paired', edgecolor='black', ax=ax_opps)
        ax_opps.set_title('Opportunity Category Distribution by Company', fontsize=16)
        ax_opps.set_ylabel('Number of Opportunities', fontsize=12)
        ax_opps.set_xlabel('Company', fontsize=12)
        ax_opps.set_xticklabels(ax_opps.get_xticklabels(), rotation=45, ha='right')
        ax_opps.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig_opps, use_container_width=True)
        plt.close(fig_opps)
    else:
        st.info("Not enough data to generate Opportunity Category Distribution chart.")

    st.markdown(f"### Explanation of Output")
    st.markdown(f"Sarah quickly gains an aggregate view of the peer group. The stacked bar chart (`Risk Category Distribution`) immediately shows the distribution of risk categories across companies, allowing her to identify industry-specific concerns (e.g., \"Regulatory\" risks for JPM). The heatmap (`Average Risk Severity`) highlights categories with higher average severity for specific companies, indicating concentrated threats.")
    st.markdown(f"Similarly, the opportunity visualizations (`Opportunity Category Distribution`) illuminate distinct growth strategies. This efficient aggregation and visualization replace hours of manual data compilation and interpretation, enabling faster, more informed comparative analysis.")
```

#### Page: 5. Cost & ROI

##### Widgets & UI

*   Displays total input/output tokens.
*   Displays estimated total cost.
*   Displays `st.pyplot` for Token Cost Breakdown (V4).

##### Markdown Content

```python
st.title("5. Cost Management and ROI Justification")
st.markdown(f"Sarah, as a senior analyst, also has budgetary responsibilities. Demonstrating the cost-effectiveness and return on investment (ROI) of AI tools is crucial for gaining internal buy-in from the investment committee. This step quantifies the API costs.")
st.markdown(f"### Story: Budgeting for AI - Demonstrating ROI")
st.markdown(f"Sarah needs to present a clear cost analysis to GMI's investment committee. The cost of LLM API calls is primarily based on the number of input and output tokens. She calculates the total cost for processing all filings, demonstrating that the expenditure is minimal compared to the hundreds of analyst hours saved.")
st.markdown(r"The API cost ($C$) for a single LLM call is calculated as:")
st.markdown(r"$$ C = (N_{{\text{{input}}}} \times P_{{\text{{in}}}}) + (N_{{\text{{output}}}} \times P_{{\text{{out}}}}) $$")
st.markdown(r"where:")
st.markdown(r"*   $N_{{\text{{input}}}}$ = Number of tokens in the input prompt and filing text.")
st.markdown(r"*   $P_{{\text{{in}}}}$ = Price per input token (e.g., for GPT-4o, $5.00 per 1M tokens).")
st.markdown(r"*   $N_{{\text{{output}}}}$ = Number of tokens in the LLM's generated response.")
st.markdown(r"*   $P_{{\text{{out}}}}$ = Price per output token (e.g., for GPT-4o, $15.00 per 1M tokens).")

if not st.session_state.total_input_tokens and not st.session_state.total_output_tokens:
    st.warning("No token usage data available. Please run LLM extraction on the 'LLM Prompts & Extraction' page first.")
else:
    st.markdown(f"### LLM API Cost Estimation")
    st.markdown(f"Total Input Tokens: {st.session_state.total_input_tokens:,}")
    st.markdown(f"Total Output Tokens: {st.session_state.total_output_tokens:,}")

    # Calculate total cost using the source function
    st.session_state.total_cost = source.estimate_llm_cost(st.session_state.total_input_tokens, st.session_state.total_output_tokens)
    st.markdown(f"**Estimated Total Cost for this Analysis: ${st.session_state.total_cost:.4f}**")

    # V4: Pie chart for token cost breakdown
    cost_input = (st.session_state.total_input_tokens / 1_000_000) * source.GPT4O_PRICE_INPUT_PER_MILLION
    cost_output = (st.session_state.total_output_tokens / 1_000_000) * source.GPT4O_PRICE_OUTPUT_PER_MILLION
    
    cost_breakdown_data = {
        'Category': ['Input Tokens Cost', 'Output Tokens Cost'],
        'Cost': [cost_input, cost_output]
    }
    cost_breakdown_df = pd.DataFrame(cost_breakdown_data)

    fig_v4, ax_v4 = plt.subplots(figsize=(8, 8))
    ax_v4.pie(cost_breakdown_df['Cost'], labels=cost_breakdown_df['Category'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'])
    ax_v4.set_title('LLM API Cost Breakdown (Input vs. Output Tokens)', fontsize=16)
    ax_v4.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig_v4)
    plt.close(fig_v4)

    st.markdown(f"### ROI Justification for Investment Committee")
    st.markdown(f"Manually analyzing {len(st.session_state.selected_company_tickers)} x 10-K filings can easily take an analyst 10-20 hours (2-4 hours per filing).")
    st.markdown(f"At an average analyst hourly rate (e.g., $100/hr), this could cost $1,000 - $2,000 in labor.")
    st.markdown(f"The LLM-assisted analysis cost for this batch is ~${st.session_state.total_cost:.2f}.")
    st.markdown(f"This represents a significant time-saving and cost reduction, allowing analysts to focus on deeper qualitative judgment, client communication, and higher-value tasks, rather than repetitive data extraction.")
    st.markdown(f"The value proposition is not 'replace the analyst' but 'free the analyst to focus on judgment and client communication.'")

st.markdown(f"### Explanation of Output")
st.markdown(f"Sarah now has concrete figures on the API costs, broken down by input and output tokens. The pie chart visually reinforces the cost structure. She can confidently present to the investment committee that the financial outlay for this AI tool is negligible compared to the hundreds of hours of analyst time saved annually.")
st.markdown(f"This argument directly addresses budget-conscious stakeholders and frames the AI as an augmentation, not a replacement, for human expertise.")
```

#### Page: 6. Analyst Review & Compliance

##### Widgets & UI

*   `st.selectbox`: To select a company for generating the checklist (`st.session_state.selected_company_for_checklist`).
*   `st.button`: "Generate Analyst Review Checklist".
*   `st.text_area`: To display the generated checklist.

##### Markdown Content

```python
st.title("6. Human-in-the-Loop: The Analyst Review Checklist and Ethical AI")
st.markdown(f"The final, and most critical, step for Sarah is the human review. As per CFA Standards, AI-generated analysis is a *draft* and requires human validation to ensure accuracy, relevance, and compliance. This section generates a structured checklist to guide that essential human oversight.")
st.markdown(f"### Story: The Analyst's Final Judgment - Upholding Ethical Standards")
st.markdown(f"Sarah understands that while AI accelerates preliminary work, the ultimate responsibility for investment recommendations rests with the human analyst. The AI-generated risk and opportunity profiles, along with the hallucination audit, form the basis of a first draft. Sarah uses a structured review checklist to methodically verify the LLM's output, apply her judgment on financial materiality (which LLMs currently lack), and ensure compliance with CFA Standards V(A) (Diligence and Reasonable Basis) and I(C) (Misrepresentation).")
st.markdown(f"This ensures the \"AI as junior analyst\" model integrates ethically and effectively into GMI's workflow.")

if not st.session_state.selected_company_tickers:
    st.warning("No companies selected for analysis. Please start from 'Setup & Data Loading'.")
elif not st.session_state.all_risks_data and not st.session_state.all_opportunities_data:
    st.warning("No risks or opportunities extracted. Please run LLM extraction on the 'LLM Prompts & Extraction' page first.")
else:
    # Update default selection for checklist if not set or if current selection is no longer valid
    if st.session_state.selected_company_for_checklist not in st.session_state.selected_company_tickers:
        st.session_state.selected_company_for_checklist = st.session_state.selected_company_tickers[0] if st.session_state.selected_company_tickers else None

    if st.session_state.selected_company_for_checklist:
        selected_company_for_checklist = st.selectbox(
            "Select Company for Analyst Review Checklist:",
            options=st.session_state.selected_company_tickers,
            index=st.session_state.selected_company_tickers.index(st.session_state.selected_company_for_checklist)
        )
        st.session_state.selected_company_for_checklist = selected_company_for_checklist

        if st.button("Generate Analyst Review Checklist"):
            ticker_for_checklist = st.session_state.selected_company_for_checklist
            risks_for_checklist = st.session_state.all_risks_data.get(ticker_for_checklist, [])
            opps_for_checklist = st.session_state.all_opportunities_data.get(ticker_for_checklist, [])
            audit_df_for_checklist = st.session_state.all_audit_results.get(ticker_for_checklist, pd.DataFrame())

            if not risks_for_checklist and not opps_for_checklist:
                st.warning(f"No risks or opportunities found for {ticker_for_checklist} to generate checklist.")
            else:
                with st.spinner(f"Generating checklist for {ticker_for_checklist}..."):
                    checklist_content = source.generate_review_checklist(
                        ticker_for_checklist, risks_for_checklist, opps_for_checklist, audit_df_for_checklist
                    )
                    st.text_area("Analyst Review Checklist", checklist_content, height=700)
                    st.download_button(
                        label="Download Checklist",
                        data=checklist_content.encode('utf-8'),
                        file_name=f"{ticker_for_checklist}_analyst_review_checklist.txt",
                        mime="text/plain"
                    )
                    st.markdown(f"### Explanation of Output")
                    st.markdown(f"Sarah receives a comprehensive `Analyst Review Checklist` for {ticker_for_checklist}. This document highlights the audit summary, explicitly lists items for human judgment (e.g., verifying high-severity risks, checking flagged hallucinations, assessing missed risks, validating investment implications), and includes placeholders for her signature and date.")
                    st.markdown(f"This structured workflow operationalizes the firm's compliance framework, ensuring that AI-assisted research meets regulatory standards and upholds the CFA Code of Ethics. The provided Prompt Template Library empowers her team to adapt these methods for their own coverage universe, ensuring consistent and scalable AI-augmented research across GMI.")
```
