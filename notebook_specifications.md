
# Idea Generation with GPT: Unlocking 10-K Insights for Investment Professionals

**Persona:** Sarah Chen, CFA Charterholder and Senior Equity Research Analyst at 'Global Markets Insight' (GMI), a leading investment management firm.

**Scenario:** Sarah leads a team of equity analysts covering the technology and healthcare sectors. Her daily workflow involves sifting through numerous SEC 10-K filings to identify material risks and opportunities for investment decision-making. Manually extracting these insights from dense legal prose is incredibly time-consuming, often leading to overlooked nuances or inconsistencies across analyses, especially when covering a wide universe of companies. GMI is eager to integrate AI as a "junior analyst" to automate preliminary research, enhance consistency, and free up senior analysts like Sarah for higher-value qualitative judgment and client engagement.

This notebook will walk Sarah through a practical workflow to leverage Large Language Models (LLMs) to efficiently extract, audit, and compare critical information from 10-K filings.

---

## 1. Setting Up the Environment and Loading Data

As a CFA Charterholder, Sarah values efficiency and accuracy. The first step in this AI-augmented workflow is to set up her Python environment and prepare the raw 10-K text for analysis. While `sec-edgar-downloader` can retrieve filings programmatically, for this lab, we'll work with pre-downloaded text files to ensure consistent results and focus on the LLM interaction.

**Learning Outcomes:**
*   Install necessary Python libraries.
*   Understand basic text loading and preprocessing for LLM context management.

### Installation of Required Libraries

```python
!pip install openai pandas matplotlib seaborn tiktoken # sec-edgar-downloader
```

### Import Required Dependencies

```python
import os
import json
import re
import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from datetime import datetime

# Configure OpenAI API key (ensure it's set as an environment variable)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Uncomment and set your API key if not using environment variables
client = OpenAI() # Initializes the OpenAI client using the API key from environment variables
```

### Story: Preparing the Digital Stack of 10-K Filings

Sarah needs to analyze a peer group of companies. For this exercise, her team focuses on a mix of tech, financial, and energy giants. Before the LLM can process these documents, they need to be loaded and, if excessively long, chunked to fit within the LLM's context window. This is analogous to a junior analyst physically organizing filing documents and highlighting relevant sections.

The token limit is a critical consideration for LLMs. If a document exceeds the model's context window (e.g., 128k tokens for `gpt-4o`), it must be split into smaller, overlapping chunks. Overlap helps maintain continuity, much like reading a chapter summary with a few sentences from the previous chapter to avoid losing context.

The token calculation is based on the LLM's internal tokenizer. The number of tokens ($N_{\text{tokens}}$) is distinct from the number of words or characters and directly impacts API costs and context limits.

```python
def count_tokens(text: str, model: str = 'gpt-4o') -> int:
    """
    Counts tokens in a text using the specified model's tokenizer.
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def chunk_text(text: str, max_tokens: int = 6000, overlap: int = 500) -> list[str]:
    """
    Splits text into chunks that fit within the model's context window,
    with overlap for continuity.
    """
    enc = tiktoken.encoding_for_model('gpt-4o')
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start = end - overlap # Overlap for context continuity
    return chunks

def get_10k_risk_factors(ticker: str, year: int, filepath_template: str = 'filings/{ticker}_{year}_10K_risk_factors.txt') -> str | None:
    """
    Retrieves the Risk Factors section from a pre-downloaded 10-K filing.
    For this lab, we assume files are pre-downloaded locally.
    """
    filepath = filepath_template.format(ticker=ticker, year=year)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print(f"File not found for {ticker} ({year}): {filepath}. Please ensure it's in the 'filings/' directory.")
        return None

# Define the companies and their details
companies_data = [
    {'ticker': 'AAPL', 'cik': '0000320193', 'year': 2024},
    {'ticker': 'JPM', 'cik': '0000019617', 'year': 2024},
    {'ticker': 'TSLA', 'cik': '0001318605', 'year': 2024},
    {'ticker': 'PFE', 'cik': '0000078003', 'year': 2024},
    {'ticker': 'XOM', 'cik': '0000034088', 'year': 2024},
]

filings_raw = {}
print("Loading and preparing 10-K filings:")
for company in companies_data:
    ticker = company['ticker']
    year = company['year']
    text = get_10k_risk_factors(ticker, year)
    if text:
        n_tokens = count_tokens(text)
        print(f"{ticker}: {len(text):,} characters, {n_tokens:,} tokens")

        if n_tokens > 8000: # Example threshold for chunking
            chunks = chunk_text(text, max_tokens=6000, overlap=500)
            print(f" -> Split into {len(chunks)} chunks due to context window limits.")
        else:
            chunks = [text]
        filings_raw[ticker] = {'text': text, 'chunks': chunks, 'token_count': n_tokens}
```

### Explanation of Output

Sarah sees the character and token counts for each company's 'Risk Factors' section. This gives her a practical understanding of the data volume. For very long documents, the output confirms that the text was automatically chunked, a crucial step to manage LLM context windows and avoid errors or truncated responses. This ensures the "junior analyst" can process even the most verbose filings.

---

## 2. Crafting the Expert Prompt: Guiding the AI "Junior Analyst"

Sarah knows that the quality of the LLM's output heavily depends on the instructions it receives. This step is about designing a sophisticated prompt that assigns the LLM the role of an expert equity analyst, enforces anti-hallucination rules, and specifies structured JSON output for easy programmatic parsing. This is the art of "prompt engineering."

**Learning Outcomes:**
*   Design effective LLM prompts with role assignment and anti-hallucination rules.
*   Enforce structured JSON output for downstream processing.

### Story: Architecting the "Junior Analyst's" Mindset for Risk Extraction

Sarah, leveraging her deep experience, designs a "system prompt" to define the LLM's persona and critical guidelines, and a "task prompt" to specify the exact information to extract. This is similar to training a new junior analyst on how to approach 10-K analysis: what to look for, how to verify facts, and how to format their findings. The `temperature` parameter is key here; a low value (e.g., $T=0.1$) is chosen for factual extraction to minimize creativity and maximize deterministic, accurate output.

The autoregressive nature of LLMs means they predict the next token based on previous tokens and a probability distribution. The temperature $T$ parameter controls the "peakiness" of this distribution:
$$ P(\text{token}_t | \text{token}_1, ..., \text{token}_{t-1}) = \text{softmax}\left(\frac{h_t}{T}\right) $$
Where $h_t$ is the hidden state at position $t$ (from the transformer's attention layers). A lower $T$ makes the distribution sharper, favoring higher-probability tokens and thus more deterministic output, which is essential for factual extraction tasks like risk analysis.

```python
# System prompt for risk factor extraction
SYSTEM_PROMPT_RISK = """You are an expert equity research analyst with 15 years of experience analyzing SEC filings for a top-tier investment bank. You specialize in identifying material risk factors that could affect investment decisions. Your goal is to provide concise, actionable insights for senior portfolio managers.

CRITICAL RULES FOR EXTRACTION:
1. ONLY cite risks EXPLICITLY mentioned in the provided text.
2. Do NOT invent, infer, or hallucinate any risk not directly supported by the text.
3. If unsure whether a risk is present or material, DO NOT include it.
4. QUOTE the specific sentence or phrase (max 50 words) from the filing that supports each risk.
5. Categorize each risk using the provided taxonomy ONLY: [Operational, Financial, Regulatory, Competitive, Macroeconomic, Technology, ESG, Legal].
6. Assign severity (High, Medium, Low) based on the language intensity in the quote.
7. Identify novelty: is this a NEW risk or recurring boilerplate for this company, based on typical industry disclosures?
8. Provide a concise investment implication (one sentence) for each risk.
"""

# Task prompt for risk factor extraction (formatted for a single chunk)
TASK_PROMPT_RISK_TEMPLATE = """
Analyze the following 'Item 1A: Risk Factors' section from {company}'s 10-K filing (FY{year}).

Extract the TOP 10 most material risk factors. For each risk, provide the following details as a JSON array:

[
  {{
    "risk_name": "A concise 5-10 word title for the risk",
    "category": "One of (Operational, Financial, Regulatory, Competitive, Macroeconomic, Technology, ESG, Legal)",
    "severity": "High, Medium, Low",
    "novel": "Yes/No (Is this a NEW risk vs. recurring boilerplate?)",
    "supporting_quote": "The exact sentence or phrase from the filing (max 50 words) supporting this risk.",
    "investment_implication": "One sentence on what this risk means for investors"
  }},
  // ... up to 10 risk factors
]

Return the result as a JSON array. Ensure strict adherence to the critical rules.

FILING TEXT:
---
{filing_text}
---
"""
```

### Explanation of Output

Sarah has now defined the precise instructions for her AI assistant. The `SYSTEM_PROMPT_RISK` establishes the "junior analyst's" identity and ethical boundaries (e.g., "Do NOT invent"). The `TASK_PROMPT_RISK_TEMPLATE` dictates the specific information fields and forces JSON output, making the extracted data highly structured and ready for quantitative analysis. The critical rules are embedded to minimize hallucination, a common LLM pitfall.

---

## 3. Unleashing the Junior Analyst: Extracting Investment Risks

With the prompt engineered, Sarah can now deploy her AI "junior analyst" to process the 10-K filings. This involves making API calls to the LLM and parsing its structured JSON output.

**Learning Outcomes:**
*   Interact with an LLM via API programmatically.
*   Parse structured JSON output from LLM responses.

### Story: Automated Risk Identification

Sarah executes the `extract_risks` function, which sends the carefully crafted prompt and 10-K text to the `gpt-4o` model. She sets `temperature=0.1` to ensure the model focuses on factual extraction rather than creative interpretation, aligning with the firm's need for high-fidelity data. The LLM acts as a diligent junior analyst, rapidly sifting through the text and summarizing key risks into a structured format.

```python
def extract_risks(company: str, year: int, filing_text: str, model: str = 'gpt-4o') -> list[dict]:
    """
    Extracts structured risk factors from 10-K text using an LLM.
    Returns: list of risk dictionaries.
    """
    prompt = TASK_PROMPT_RISK_TEMPLATE.format(company=company, year=year, filing_text=filing_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RISK},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Low temperature for factual extraction
            max_tokens=4000, # Max tokens for output to prevent truncation
            response_format={"type": "json_object"} # Force JSON output
        )
        output = response.choices[0].message.content
        risks = json.loads(output)

        # Handle cases where LLM might wrap the JSON array in an object
        if isinstance(risks, dict) and 'risks' in risks:
            risks = risks['risks']
        elif isinstance(risks, dict) and 'Risk Factors' in risks: # common LLM variation
            risks = risks['Risk Factors']
        elif not isinstance(risks, list):
            print(f"Warning: LLM returned non-list JSON for {company}. Attempting direct parsing.")
            return [] # Or attempt more robust parsing if needed

        # Add token usage for cost estimation
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return risks, input_tokens, output_tokens

    except json.JSONDecodeError as e:
        print(f"JSON parse failed for {company}: {e}\nOutput: {output[:500]}...")
        return [], 0, 0
    except Exception as e:
        print(f"LLM API call failed for {company}: {e}")
        return [], 0, 0

all_risks_data = {}
total_input_tokens = 0
total_output_tokens = 0

print("\nExtracting risk factors using GPT-4o:")
for ticker, data in filings_raw.items():
    # Process only the first chunk or the full text if not chunked
    # This simplifies the initial extraction for demonstration.
    # For full analysis, one would iterate through all chunks and aggregate.
    text_to_process = data['chunks'][0] if len(data['chunks']) > 1 else data['text']
    
    risks, input_t, output_t = extract_risks(ticker, companies_data[0]['year'], text_to_process) # Assuming same year for all
    all_risks_data[ticker] = risks
    total_input_tokens += input_t
    total_output_tokens += output_t
    print(f"{ticker}: Extracted {len(risks)} risk factors.")

# Display an example of the structured output for one company (e.g., Apple)
if 'AAPL' in all_risks_data and all_risks_data['AAPL']:
    print("\n--- Example of Extracted Risk Factors (AAPL) ---")
    print(json.dumps(all_risks_data['AAPL'][0], indent=2)) # Print first risk for AAPL
```

### Explanation of Output

Sarah now has a structured list of risks for each company. The output for AAPL shows a single risk factor, complete with its name, category, severity, novelty, supporting quote, and investment implication. This immediate, structured output contrasts sharply with the hours it would take to manually read and summarize. The `.create()` method from the `OpenAI` client handles the API communication, and `json.loads()` effectively converts the LLM's text response into a Python list of dictionaries, ready for further analysis.

---

## 4. Verifying the "Junior Analyst": Hallucination Audit

As a CFA Charterholder, Sarah understands the ethical imperative of diligence (CFA Standard V(A) - Diligence). The LLM's output is a draft and must be audited for hallucinations (fabrications). This step cross-checks extracted quotes against the original text to ensure factual accuracy.

**Learning Outcomes:**
*   Implement a hallucination audit mechanism involving fuzzy and exact text matching.
*   Understand and calculate hallucination metrics like Hallucination Rate, Precision, and Recall.

### Story: Quality Control - Catching AI Fabrications

Sarah's firm adheres to strict compliance frameworks. She cannot blindly trust the AI. This audit function acts as a critical second pair of eyes, automatically checking if the LLM's "supporting quotes" actually appear in the original 10-K text. This quantifies the trustworthiness of the AI's output, allowing her to prioritize human review for flagged items.

The **Hallucination Rate ($H$)** quantifies the proportion of extracted items that cannot be verified in the source text.
$$ H = \frac{N_{\text{flagged}}}{N_{\text{total extracted}}} $$
Where $N_{\text{flagged}}$ are items with a low word-match percentage (e.g., < 75%) or no exact substring match, and $N_{\text{total extracted}}$ is the total number of items extracted by the LLM.
A target for $H$ in structured extraction is typically $< 0.05$ (less than 5%).

**Precision ($P$)** measures how many of the LLM's extracted items are actually correct and relevant.
$$ P = \frac{N_{\text{verified and relevant}}}{N_{\text{total extracted}}} $$
**Recall ($R$)** measures how many of the truly relevant items in the document were extracted by the LLM.
$$ R = \frac{N_{\text{verified and relevant}}}{N_{\text{true risks in filing}}} $$
Note: Recall ($R$) often requires human judgment to determine $N_{\text{true risks in filing}}$, which is the total number of material risks a human analyst would identify.

```python
def audit_hallucinations(risks: list[dict], source_text: str) -> pd.DataFrame:
    """
    Verifies each extracted risk's supporting quote appears in the source text.
    Flags potential hallucinations.
    """
    audit_results = []
    source_lower = source_text.lower()
    source_words = set(source_lower.split())

    for i, risk in enumerate(risks):
        quote = risk.get('supporting_quote', '').lower()
        risk_name = risk.get('risk_name', '')
        
        quote_words = set(w for w in quote.split() if w.strip()) # Filter out empty strings from split
        
        match_pct = 0.0
        if len(quote_words) > 0:
            found_words = sum(1 for w in quote_words if w in source_words)
            match_pct = found_words / len(quote_words)
        
        # Exact substring check (more stringent)
        exact_match = (quote in source_lower) if quote else False

        # Define VERIFIED based on thresholds (e.g., 75% word match or exact substring match)
        verified = (match_pct > 0.75) or exact_match
        flag = 'OK' if verified else 'POTENTIAL HALLUCINATION'
        
        audit_results.append({
            'risk_idx': i,
            'risk_name': risk_name,
            'quote_word_match': round(match_pct, 2),
            'exact_substring_match': exact_match,
            'VERIFIED': verified,
            'FLAG': flag
        })
    
    audit_df = pd.DataFrame(audit_results)
    return audit_df

all_audit_results = {}
print("\n--- Running Hallucination Audit ---")
for ticker, risks in all_risks_data.items():
    if risks and ticker in filings_raw:
        audit_df = audit_hallucinations(risks, filings_raw[ticker]['text'])
        all_audit_results[ticker] = audit_df
        n_verified = audit_df['VERIFIED'].sum()
        n_flagged = len(audit_df) - n_verified
        print(f"\n{ticker} Hallucination Audit:")
        print(f"Audit: {n_verified}/{len(audit_df)} risks verified, {n_flagged} flagged for review.")
        print(audit_df[['risk_name', 'quote_word_match', 'exact_substring_match', 'FLAG']])
    else:
        print(f"No risks extracted or filing text missing for {ticker}.")

# Prepare data for Hallucination Audit Dashboard (V3)
audit_summary = []
for ticker, audit_df in all_audit_results.items():
    if not audit_df.empty:
        total_items = len(audit_df)
        verified_items = audit_df['VERIFIED'].sum()
        flagged_items = total_items - verified_items
        verification_rate = (verified_items / total_items) * 100 if total_items > 0 else 0
        audit_summary.append({
            'Company': ticker,
            'Total Items': total_items,
            'Verified Items': verified_items,
            'Flagged Items': flagged_items,
            'Verification Rate (%)': verification_rate
        })
audit_summary_df = pd.DataFrame(audit_summary)

plt.figure(figsize=(12, 6))
sns.barplot(x='Company', y='Verification Rate (%)', data=audit_summary_df, palette='viridis')
plt.title('Hallucination Audit Dashboard: Verification Rate by Company', fontsize=16)
plt.ylabel('Verification Rate (%)', fontsize=12)
plt.xlabel('Company', fontsize=12)
plt.ylim(0, 100)
for index, row in audit_summary_df.iterrows():
    plt.text(index, row['Verification Rate (%)'] + 1, f"{row['Verification Rate (%)']:.1f}%", color='black', ha="center")
plt.tight_layout()
plt.savefig('hallucination_audit_dashboard.png', dpi=150)
plt.show()

# Example of an annotated output (V5) - one company's risk extraction with source citations
print("\n--- Example Annotated Output (AAPL Risk 1 with Audit Status) ---")
if 'AAPL' in all_risks_data and all_risks_data['AAPL'] and 'AAPL' in all_audit_results:
    sample_risk = all_risks_data['AAPL'][0]
    sample_audit = all_audit_results['AAPL'][0]
    
    print(f"Risk Name: {sample_risk.get('risk_name')}")
    print(f"Category: {sample_risk.get('category')}")
    print(f"Severity: {sample_risk.get('severity')}")
    print(f"Novelty: {sample_risk.get('novel')}")
    print(f"Quote (from LLM): \"{sample_risk.get('supporting_quote')}\"")
    print(f"Investment Implication: {sample_risk.get('investment_implication')}")
    print(f"--- Audit Status ---")
    print(f"Word Match (%): {sample_audit['quote_word_match'] * 100:.1f}%")
    print(f"Exact Substring Match: {sample_audit['exact_substring_match']}")
    print(f"Verification Flag: {sample_audit['FLAG']}")
```

### Explanation of Output

Sarah receives a detailed audit report for each company. The dashboard clearly visualizes the verification rate, showing how frequently the LLM's claims are directly supported by the source text. Any "POTENTIAL HALLUCINATION" flags items for Sarah's immediate human review. This rigorous verification process directly addresses CFA Standard V(A) by ensuring a "reasonable basis" for the AI-assisted analysis, building trust in the tool and enhancing the quality of her team's preliminary research.

---

## 5. Identifying Future Potential: Extracting Investment Opportunities

Beyond risks, Sarah's role also involves identifying potential investment opportunities. The "junior analyst" can be repurposed to scan for forward-looking statements in management commentary.

**Learning Outcomes:**
*   Develop a separate LLM call to identify forward-looking investment opportunities.
*   Apply prompt engineering for distinct extraction tasks.

### Story: Spotting Growth Catalysts

Sarah modifies the prompt to direct the LLM to focus on positive, forward-looking statements within the 10-K, such as new product launches, market expansions, or efficiency initiatives. This is akin to instructing a junior analyst to highlight passages indicating growth potential, rather than just threats.

```python
# System prompt for opportunity identification (can be similar or specialized)
SYSTEM_PROMPT_OPPS = """You are an expert equity research analyst with 15 years of experience specializing in identifying growth catalysts and strategic opportunities from SEC filings. Your objective is to uncover forward-looking investment opportunities.

CRITICAL RULES FOR EXTRACTION:
1. ONLY identify opportunities EXPLICITLY mentioned in the provided text.
2. Do NOT invent, infer, or hallucinate any opportunity not directly supported by the text.
3. If unsure whether an opportunity is genuine or merely boilerplate, DO NOT include it.
4. QUOTE the specific sentence or phrase (max 50 words) from the filing that supports each opportunity.
5. Categorize each opportunity using the provided taxonomy ONLY: [Growth, Efficiency, Strategic, Innovation, Market Expansion].
6. Outline a key risk-to-opportunity (what could prevent it from materializing?)
7. Estimate a timeframe: [Near-term (<1yr), Medium (1-3yr), Long-term (>3yr)].
"""

# Task prompt for opportunity identification
TASK_PROMPT_OPPS_TEMPLATE = """
Analyze the following text from {company}'s 10-K filing (FY{year}).

Identify UP TO 5 forward-looking investment opportunities mentioned by management. Focus on:
- New product launches or market expansions
- Cost reduction or efficiency initiatives
- Strategic acquisitions or partnerships
- Competitive advantages being developed

For each opportunity, provide the following details as a JSON array:

[
  {{
    "opportunity_name": "A concise title for the opportunity",
    "category": "One of (Growth, Efficiency, Strategic, Innovation, Market Expansion)",
    "evidence_quote": "Supporting sentence or phrase from the filing (max 50 words).",
    "risk_to_opportunity": "What could prevent this opportunity from materializing?",
    "timeframe": "Near-term (<1yr), Medium (1-3yr), Long-term (>3yr)"
  }},
  // ... up to 5 opportunities
]

Return as a JSON array. Only include opportunities explicitly described in the text. Do NOT fabricate opportunities.

FILING TEXT:
---
{filing_text}
---
"""

def extract_opportunities(company: str, year: int, filing_text: str, model: str = 'gpt-4o') -> list[dict]:
    """
    Identifies structured investment opportunities from 10-K text using an LLM.
    Returns: list of opportunity dictionaries.
    """
    prompt = TASK_PROMPT_OPPS_TEMPLATE.format(company=company, year=year, filing_text=filing_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_OPPS},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Higher temperature for creative brainstorming/identifying nuanced signals
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        output = response.choices[0].message.content
        opportunities = json.loads(output)
        
        # Handle cases where LLM might wrap the JSON array in an object
        if isinstance(opportunities, dict) and 'opportunities' in opportunities:
            opportunities = opportunities['opportunities']
        elif not isinstance(opportunities, list):
            print(f"Warning: LLM returned non-list JSON for {company}. Attempting direct parsing.")
            return [], 0, 0

        # Add token usage for cost estimation
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return opportunities, input_tokens, output_tokens

    except json.JSONDecodeError as e:
        print(f"JSON parse failed for {company}: {e}\nOutput: {output[:500]}...")
        return [], 0, 0
    except Exception as e:
        print(f"LLM API call failed for {company}: {e}")
        return [], 0, 0

all_opportunities_data = {}
print("\nExtracting investment opportunities using GPT-4o:")
for ticker, data in filings_raw.items():
    # Using the same text segment for simplicity
    text_to_process = data['chunks'][0] if len(data['chunks']) > 1 else data['text']

    opportunities, input_t, output_t = extract_opportunities(ticker, companies_data[0]['year'], text_to_process)
    all_opportunities_data[ticker] = opportunities
    total_input_tokens += input_t
    total_output_tokens += output_t
    print(f"{ticker}: Extracted {len(opportunities)} opportunities.")

# Display an example of the structured output for one company (e.g., Tesla)
if 'TSLA' in all_opportunities_data and all_opportunities_data['TSLA']:
    print("\n--- Example of Extracted Investment Opportunities (TSLA) ---")
    print(json.dumps(all_opportunities_data['TSLA'][0], indent=2))
```

### Explanation of Output

Sarah now has a structured list of potential opportunities for each company, similar to the risk factors. Notice that for opportunity extraction, a slightly higher `temperature` ($T=0.7$) was used compared to risk extraction. This allows the LLM a bit more creativity to identify nuanced signals from management's forward-looking statements, which might not be as explicitly stated as risks, without veering into outright fabrication. This dual-lens approach (risks and opportunities) provides a more holistic preliminary view of each company.

---

## 6. Peer Group Analysis: Comparing Risks and Opportunities

After extracting data from individual filings, Sarah's next crucial step is to aggregate and compare these insights across the peer group. This allows her to identify common industry risks, unique company-specific threats, and differentiated growth strategies.

**Learning Outcomes:**
*   Aggregate structured LLM output into pandas DataFrames.
*   Generate comparative visualizations (stacked bar charts, heatmaps).

### Story: Uncovering Industry Trends and Outliers

Sarah transforms the individual JSON outputs into a comprehensive pandas DataFrame, which is the cornerstone for quantitative comparison. She then visualizes these comparisons, quickly revealing patterns that would be tedious to spot manually. This cross-company analysis is vital for portfolio managers who need to understand relative positioning and industry-wide exposures.

```python
# Aggregate all risks into a single DataFrame
all_risks_flat = []
for ticker, risks in all_risks_data.items():
    for risk in risks:
        risk_copy = risk.copy()
        risk_copy['Company'] = ticker
        all_risks_flat.append(risk_copy)
comparison_risks_df = pd.DataFrame(all_risks_flat)

# Aggregate all opportunities into a single DataFrame
all_opportunities_flat = []
for ticker, opportunities in all_opportunities_data.items():
    for opp in opportunities:
        opp_copy = opp.copy()
        opp_copy['Company'] = ticker
        all_opportunities_flat.append(opp_copy)
comparison_opportunities_df = pd.DataFrame(all_opportunities_flat)

print("\n--- Cross-Company Risk Comparison Table ---")
# Display a sample of the comparison DataFrame
print(comparison_risks_df[['Company', 'risk_name', 'category', 'severity']].head())

# --- Visualizations (V1 & V2) ---

# V1: Risk Category Distribution Bar Chart
if not comparison_risks_df.empty:
    risk_category_pivot = comparison_risks_df.groupby(['Company', 'category']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(14, 7))
    risk_category_pivot.plot(kind='bar', stacked=True, colormap='Set2', edgecolor='black', ax=plt.gca())
    plt.title('Risk Category Distribution by Company', fontsize=16)
    plt.ylabel('Number of Risk Factors', fontsize=12)
    plt.xlabel('Company', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('risk_profile_comparison_stacked_bar.png', dpi=150)
    plt.show()

# V2: Severity Heatmap
if not comparison_risks_df.empty:
    severity_map = {'High': 3, 'Medium': 2, 'Low': 1}
    comparison_risks_df['severity_num'] = comparison_risks_df['severity'].map(severity_map)

    severity_pivot = comparison_risks_df.pivot_table(
        values='severity_num', index='Company', columns='category', aggfunc='mean'
    ).fillna(0) # Fill NaN for categories not present with 0 or a neutral value

    plt.figure(figsize=(12, 6))
    sns.heatmap(severity_pivot, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5, linecolor='black', vmin=1, vmax=3)
    plt.title('Average Risk Severity by Company and Category', fontsize=16)
    plt.ylabel('Company', fontsize=12)
    plt.xlabel('Risk Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('risk_severity_heatmap.png', dpi=150)
    plt.show()

print("\n--- Cross-Company Opportunity Comparison Table ---")
print(comparison_opportunities_df[['Company', 'opportunity_name', 'category', 'timeframe']].head())

# Visualize opportunity categories (similar to risks, but for opportunities)
if not comparison_opportunities_df.empty:
    opportunity_category_pivot = comparison_opportunities_df.groupby(['Company', 'category']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(14, 7))
    opportunity_category_pivot.plot(kind='bar', stacked=True, colormap='Paired', edgecolor='black', ax=plt.gca())
    plt.title('Opportunity Category Distribution by Company', fontsize=16)
    plt.ylabel('Number of Opportunities', fontsize=12)
    plt.xlabel('Company', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('opportunity_profile_comparison_stacked_bar.png', dpi=150)
    plt.show()
```

### Explanation of Output

Sarah quickly gains an aggregate view of the peer group. The stacked bar chart (`risk_profile_comparison_stacked_bar.png`) immediately shows the distribution of risk categories across companies, allowing her to identify industry-specific concerns (e.g., "Regulatory" risks for JPM). The heatmap (`risk_severity_heatmap.png`) highlights categories with higher average severity for specific companies, indicating concentrated threats. Similarly, the opportunity visualizations (`opportunity_profile_comparison_stacked_bar.png`) illuminate distinct growth strategies. This efficient aggregation and visualization replace hours of manual data compilation and interpretation, enabling faster, more informed comparative analysis.

---

## 7. Cost Management and ROI Justification

Sarah, as a senior analyst, also has budgetary responsibilities. Demonstrating the cost-effectiveness and return on investment (ROI) of AI tools is crucial for gaining internal buy-in from the investment committee. This step quantifies the API costs.

**Learning Outcomes:**
*   Implement token-based cost estimation for LLM API usage.
*   Justify ROI based on cost savings.

### Story: Budgeting for AI - Demonstrating ROI

Sarah needs to present a clear cost analysis to GMI's investment committee. The cost of LLM API calls is primarily based on the number of input and output tokens. She calculates the total cost for processing all filings, demonstrating that the expenditure is minimal compared to the hundreds of analyst hours saved.

The API cost ($C$) for a single LLM call is calculated as:
$$ C = (N_{\text{input}} \times P_{\text{in}}) + (N_{\text{output}} \times P_{\text{out}}) $$
Where:
*   $N_{\text{input}}$ = Number of tokens in the input prompt and filing text.
*   $P_{\text{in}}$ = Price per input token (e.g., for GPT-4o, $5.00 per 1M tokens).
*   $N_{\text{output}}$ = Number of tokens in the LLM's generated response.
*   $P_{\text{out}}$ = Price per output token (e.g., for GPT-4o, $15.00 per 1M tokens).

```python
# Define GPT-4o pricing (as of a hypothetical 2024/2025 rate)
GPT4O_PRICE_INPUT_PER_MILLION = 5.00 # $5.00 per 1M input tokens
GPT4O_PRICE_OUTPUT_PER_MILLION = 15.00 # $15.00 per 1M output tokens

def estimate_llm_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Estimates the cost of an LLM API call based on token usage and predefined prices.
    """
    cost_input = (input_tokens / 1_000_000) * GPT4O_PRICE_INPUT_PER_MILLION
    cost_output = (output_tokens / 1_000_000) * GPT4O_PRICE_OUTPUT_PER_MILLION
    return cost_input + cost_output

# Calculate total cost
total_cost = estimate_llm_cost(total_input_tokens, total_output_tokens)
print(f"\n--- LLM API Cost Estimation ---")
print(f"Total Input Tokens: {total_input_tokens:,}")
print(f"Total Output Tokens: {total_output_tokens:,}")
print(f"Estimated Total Cost for this Analysis: ${total_cost:.4f}")

# Pie chart for token cost breakdown (V4)
cost_breakdown_data = {
    'Category': ['Input Tokens Cost', 'Output Tokens Cost'],
    'Cost': [
        (total_input_tokens / 1_000_000) * GPT4O_PRICE_INPUT_PER_MILLION,
        (total_output_tokens / 1_000_000) * GPT4O_PRICE_OUTPUT_PER_MILLION
    ]
}
cost_breakdown_df = pd.DataFrame(cost_breakdown_data)

plt.figure(figsize=(8, 8))
plt.pie(cost_breakdown_df['Cost'], labels=cost_breakdown_df['Category'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'])
plt.title('LLM API Cost Breakdown (Input vs. Output Tokens)', fontsize=16)
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('llm_cost_breakdown_pie_chart.png', dpi=150)
plt.show()

# ROI Justification (conceptual explanation)
print("\n--- ROI Justification for Investment Committee ---")
print("Manually analyzing 5 x 10-K filings can easily take an analyst 10-20 hours (2-4 hours per filing).")
print(f"At an average analyst hourly rate (e.g., $100/hr), this could cost $1,000 - $2,000 in labor.")
print(f"The LLM-assisted analysis cost for this batch is ~${total_cost:.2f}.")
print(f"This represents a significant time-saving and cost reduction, allowing analysts to focus on deeper qualitative judgment, client communication, and higher-value tasks, rather than repetitive data extraction.")
print("The value proposition is not 'replace the analyst' but 'free the analyst to focus on judgment and client communication.'")
```

### Explanation of Output

Sarah now has concrete figures on the API costs, broken down by input and output tokens. The pie chart (`llm_cost_breakdown_pie_chart.png`) visually reinforces the cost structure. She can confidently present to the investment committee that the financial outlay for this AI tool is negligible compared to the hundreds of hours of analyst time saved annually. This argument directly addresses budget-conscious stakeholders and frames the AI as an augmentation, not a replacement, for human expertise.

---

## 8. Human-in-the-Loop: The Analyst Review Checklist and Ethical AI

The final, and most critical, step for Sarah is the human review. As per CFA Standards, AI-generated analysis is a *draft* and requires human validation to ensure accuracy, relevance, and compliance. This section generates a structured checklist to guide that essential human oversight.

**Learning Outcomes:**
*   Generate a formatted analyst review checklist for human validation.
*   Understand and operationalize human-in-the-loop requirements for ethical AI adoption, specifically CFA Standards V(A) and I(C).

### Story: The Analyst's Final Judgment - Upholding Ethical Standards

Sarah understands that while AI accelerates preliminary work, the ultimate responsibility for investment recommendations rests with the human analyst. The AI-generated risk and opportunity profiles, along with the hallucination audit, form the basis of a first draft. Sarah uses a structured review checklist to methodically verify the LLM's output, apply her judgment on financial materiality (which LLMs currently lack), and ensure compliance with CFA Standards V(A) (Diligence and Reasonable Basis) and I(C) (Misrepresentation). This ensures the "AI as junior analyst" model integrates ethically and effectively into GMI's workflow.

```python
def generate_review_checklist(ticker: str, risks: list[dict], opportunities: list[dict], audit_df: pd.DataFrame) -> str:
    """
    Generates a structured review document for a human analyst.
    """
    report = []
    report.append(f"{'= '*30}")
    report.append("AI-ASSISTED RESEARCH NOTE -- FOR ANALYST REVIEW")
    report.append(f"Company: {ticker} | Source: 10-K FY{companies_data[0]['year']}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model: GPT-4o | Temperature: 0.1 (Risks), 0.7 (Opportunities)")
    report.append(f"{'= '*30}")
    report.append("\n--- HALLUCINATION AUDIT SUMMARY ---")
    if not audit_df.empty:
        n_verified = audit_df['VERIFIED'].sum()
        n_total = len(audit_df)
        report.append(f" {n_verified}/{n_total} risk items verified against source.")
        flagged_risks = audit_df[audit_df['FLAG'] == 'POTENTIAL HALLUCINATION']
        if not flagged_risks.empty:
            report.append(" Potentially Hallucinated Risks to Review:")
            for _, row in flagged_risks.iterrows():
                report.append(f"    - [ ] Risk '{row['risk_name']}' (Word Match: {row['quote_word_match']:.1f}, Exact: {row['exact_substring_match']})")
        else:
            report.append(" No potential hallucinations flagged for risks.")
    else:
        report.append(" No audit data available or no risks extracted.")


    report.append("\n--- ANALYST REVIEW REQUIRED ---")
    report.append(" [ ] Verify all HIGH severity risks against filing for accuracy and financial materiality.")
    report.append(" [ ] Check flagged items for hallucination (as per audit summary above).")
    report.append(" [ ] Assess whether any material risks were MISSED by the LLM.")
    report.append(" [ ] Validate investment implications for all extracted risks.")
    report.append(" [ ] Review extracted opportunities for genuine signals vs. boilerplate.")
    report.append(" [ ] Evaluate 'risk-to-opportunity' and 'timeframe' for opportunities.")
    report.append(" [ ] Approve for use in preliminary research output, ensuring compliance with CFA Standards V(A) and I(C).")
    report.append("\n--- EXTRACTED RISKS (for reference) ---")
    if risks:
        for i, risk in enumerate(risks):
            flag = audit_df.iloc[i]['FLAG'] if i < len(audit_df) else 'N/A'
            report.append(f"\nRisk {i+1}: {risk.get('risk_name', 'N/A')}")
            report.append(f"  Category: {risk.get('category', 'N/A')}")
            report.append(f"  Severity: {risk.get('severity', 'N/A')}")
            report.append(f"  Novelty: {risk.get('novel', 'N/A')}")
            report.append(f"  Audit: {flag}")
            report.append(f"  Quote: \"{risk.get('supporting_quote', 'N/A')}\"")
            report.append(f"  Implication: {risk.get('investment_implication', 'N/A')}")
    else:
        report.append(" No risks extracted.")

    report.append("\n--- EXTRACTED OPPORTUNITIES (for reference) ---")
    if opportunities:
        for i, opp in enumerate(opportunities):
            report.append(f"\nOpportunity {i+1}: {opp.get('opportunity_name', 'N/A')}")
            report.append(f"  Category: {opp.get('category', 'N/A')}")
            report.append(f"  Quote: \"{opp.get('evidence_quote', 'N/A')}\"")
            report.append(f"  Risk-to-Opportunity: {opp.get('risk_to_opportunity', 'N/A')}")
            report.append(f"  Timeframe: {opp.get('timeframe', 'N/A')}")
    else:
        report.append(" No opportunities extracted.")

    report.append(f"\n{'-'*60}")
    report.append("Analyst Signature: _________________________ Date: _________")
    report.append(f"{'-'*60}")

    return '\n'.join(report)

# Generate and print the checklist for one company (e.g., Apple)
target_ticker = 'AAPL'
if target_ticker in all_risks_data and target_ticker in all_audit_results and target_ticker in all_opportunities_data:
    checklist_report = generate_review_checklist(
        target_ticker,
        all_risks_data[target_ticker],
        all_opportunities_data[target_ticker],
        all_audit_results[target_ticker]
    )
    print(f"\n--- Analyst Review Checklist for {target_ticker} ---")
    print(checklist_report)
else:
    print(f"\nCould not generate checklist for {target_ticker}. Data might be missing.")

# Prompt Template Library (O5)
print("\n--- Prompt Template Library ---")
print("These templates can be reused and adapted for future analyses:")
print("\n**Risk Factor Extraction System Prompt:**")
print("```markdown")
print(SYSTEM_PROMPT_RISK)
print("```")
print("\n**Risk Factor Extraction Task Prompt Template:**")
print("```markdown")
print(TASK_PROMPT_RISK_TEMPLATE)
print("```")
print("\n**Opportunity Identification System Prompt:**")
print("```markdown")
print(SYSTEM_PROMPT_OPPS)
print(f"```")
print("\n**Opportunity Identification Task Prompt Template:**")
print("```markdown")
print(TASK_PROMPT_OPPS_TEMPLATE)
print("```")
```

### Explanation of Output

Sarah receives a comprehensive `Analyst Review Checklist` for AAPL. This document highlights the audit summary, explicitly lists items for human judgment (e.g., verifying high-severity risks, checking flagged hallucinations, assessing missed risks, validating investment implications), and includes placeholders for her signature and date. This structured workflow operationalizes the firm's compliance framework, ensuring that AI-assisted research meets regulatory standards and upholds the CFA Code of Ethics. The provided Prompt Template Library at the end empowers her team to adapt these methods for their own coverage universe, ensuring consistent and scalable AI-augmented research across GMI.

