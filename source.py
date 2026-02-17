"""
Source module for QuLab: Lab 11 - Idea Generation with GPT (Generative AI)
Contains helper functions, prompt templates, and data for 10-K analysis.
"""

import os
import json
import re
import pandas as pd
import tiktoken
from openai import OpenAI
from pydantic import BaseModel

# ==============================================================================
# DATA: Company Information
# ==============================================================================

companies_data = [
    {'ticker': 'AAPL', 'cik': '0000320193', 'year': 2024},
    {'ticker': 'JPM', 'cik': '0000019617', 'year': 2024},
    {'ticker': 'TSLA', 'cik': '0001318605', 'year': 2024},
    {'ticker': 'PFE', 'cik': '0000078003', 'year': 2024},
    {'ticker': 'XOM', 'cik': '0000034088', 'year': 2024},
]

# ==============================================================================
# PRICING CONSTANTS
# ==============================================================================

# GPT-4o pricing (as of early 2025)
GPT4O_PRICE_INPUT_PER_MILLION = 2.50  # $2.50 per 1M input tokens
GPT4O_PRICE_OUTPUT_PER_MILLION = 10.00  # $10.00 per 1M output tokens

# ==============================================================================
# PYDANTIC MODELS FOR STRUCTURED RESPONSES
# ==============================================================================


class RiskFactor(BaseModel):
    """Pydantic model for a single risk factor."""
    risk_name: str
    category: str
    severity: str
    novel: str
    supporting_quote: str
    investment_implication: str


class RiskFactorsList(BaseModel):
    """Pydantic model for a list of risk factors."""
    risks: list[RiskFactor]


class Opportunity(BaseModel):
    """Pydantic model for a single opportunity."""
    opportunity_name: str
    category: str
    evidence_quote: str
    risk_to_opportunity: str
    timeframe: str


class OpportunityList(BaseModel):
    """Pydantic model for a list of opportunities."""
    opportunities: list[Opportunity]

# ==============================================================================
# PROMPT TEMPLATES FOR RISK EXTRACTION
# ==============================================================================


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

TASK_PROMPT_RISK_TEMPLATE = """
Analyze the following 'Item 1A: Risk Factors' section from {company}'s 10-K filing (FY{year}).

Extract the TOP 10 most material risk factors. For each risk, provide the following details as a JSON array:

{{
  "risks": [
    {{
      "risk_name": "A concise 5-10 word title for the risk",
      "category": "One of (Operational, Financial, Regulatory, Competitive, Macroeconomic, Technology, ESG, Legal)",
      "severity": "High, Medium, Low",
      "novel": "Yes/No (Is this a NEW risk vs. recurring boilerplate?)",
      "supporting_quote": "The exact sentence or phrase from the filing (max 50 words) supporting this risk.",
      "investment_implication": "One sentence on what this risk means for investors"
    }}
  ]
}}

Return the result as a JSON object with a "risks" key containing the array. Ensure strict adherence to the critical rules.

FILING TEXT:
---
{filing_text}
---
"""

# ==============================================================================
# PROMPT TEMPLATES FOR OPPORTUNITY IDENTIFICATION
# ==============================================================================

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

TASK_PROMPT_OPPS_TEMPLATE = """
Analyze the following text from {company}'s 10-K filing (FY{year}).

Identify UP TO 5 forward-looking investment opportunities mentioned by management. Focus on:
- New product launches or market expansions
- Cost reduction or efficiency initiatives
- Strategic acquisitions or partnerships
- Competitive advantages being developed

For each opportunity, provide the following details as a JSON object:

{{
  "opportunities": [
    {{
      "opportunity_name": "A concise title for the opportunity",
      "category": "One of (Growth, Efficiency, Strategic, Innovation, Market Expansion)",
      "evidence_quote": "Supporting sentence or phrase from the filing (max 50 words).",
      "risk_to_opportunity": "What could prevent this opportunity from materializing?",
      "timeframe": "Near-term (<1yr), Medium (1-3yr), Long-term (>3yr)"
    }}
  ]
}}

Return as a JSON object with an "opportunities" key containing the array. Only include opportunities explicitly described in the text. Do NOT fabricate opportunities.

FILING TEXT:
---
{filing_text}
---
"""

# ==============================================================================
# TOKEN COUNTING AND TEXT CHUNKING
# ==============================================================================


def count_tokens(text: str, model: str = 'gpt-4o') -> int:
    """
    Counts tokens in a text using the specified model's tokenizer.

    Args:
        text: The text to count tokens for
        model: The model name (default: 'gpt-4o')

    Returns:
        int: Number of tokens in the text
    """
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback: rough estimate
        return len(text) // 4


def chunk_text(text: str, max_tokens: int = 6000, overlap: int = 500) -> list:
    """
    Splits text into chunks that fit within the model's context window,
    with overlap for continuity.

    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk (default: 6000)
        overlap: Number of tokens to overlap between chunks (default: 500)

    Returns:
        list: List of text chunks
    """
    try:
        enc = tiktoken.encoding_for_model('gpt-4o')
        tokens = enc.encode(text)

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(enc.decode(chunk_tokens))
            start = end - overlap  # Overlap for context continuity
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        # Fallback: return original text as single chunk
        return [text]


# ==============================================================================
# DATA LOADING
# ==============================================================================

def get_10k_risk_factors(ticker: str, year: int, filepath_template: str = 'filings/{ticker}_{year}_10K_risk_factors.txt') -> str:
    """
    Retrieves the Risk Factors section from a pre-downloaded 10-K filing.
    For this lab, we assume files are pre-downloaded locally.

    Args:
        ticker: Company ticker symbol
        year: Fiscal year
        filepath_template: Path template for the filing files

    Returns:
        str: The risk factors text, or mock data if file not found
    """
    filepath = filepath_template.format(ticker=ticker, year=year)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print(
            f"File not found for {ticker} ({year}): {filepath}. Using mock data.")
        # Return mock data for demonstration purposes
        return generate_mock_10k_data(ticker, year)


def generate_mock_10k_data(ticker: str, year: int) -> str:
    """
    Generates mock 10-K risk factors data for demonstration purposes.

    Args:
        ticker: Company ticker symbol
        year: Fiscal year

    Returns:
        str: Mock risk factors text
    """
    mock_data = {
        'AAPL': f"""ITEM 1A. RISK FACTORS

The Company's business, reputation, results of operations, financial condition and stock price can be affected by a number of factors, whether currently known or unknown, including those described below. When any one or more of these risks materialize, the Company's business, reputation, results of operations, financial condition and stock price can be materially and adversely affected.

Macroeconomic and Industry Risks

Global economic conditions could materially adversely affect the Company. The Company's operations and performance depend significantly on global and regional economic conditions and their impact on consumer and business spending. Economic weakness, uncertainty about future economic conditions, and tighter credit can make it more difficult for customers to purchase products, delay or reduce purchases, and impair the Company's ability to collect receivables.

The Company participates in highly competitive markets and faces significant competition. The markets for the Company's products and services are highly competitive, and are characterized by aggressive price competition and resulting downward pressure on gross margins, frequent introduction of new products, short product life cycles, evolving industry standards, continual improvement in product price/performance characteristics, rapid adoption of technological advancements by competitors, and price sensitivity on the part of consumers and businesses.

Technology, Information and Information Security Risks

The Company's products and services may be affected by hardware and software defects. Despite testing, the Company's hardware and software products can contain defects affecting their performance, which could cause interruptions in the availability and functionality of the Company's products and services, or damage to data, devices or property. Such defects could result in extensive repair costs, product recalls, liability claims, negative publicity and reputational damage.

The Company is subject to risks of information security breaches. While the Company takes extensive measures to protect against data security breaches and protect information, such measures cannot provide absolute security. Computer hackers may attempt to penetrate the Company's network security or may attempt to obtain personal information from the Company's customers. The Company's information systems and those of its service providers are vulnerable to breaches, computer viruses, security vulnerabilities, and attacks by hackers.

Operational Risks

The Company relies on third parties for certain components and services. The Company relies on sole-source suppliers and manufacturers for some components, manufacturing equipment and services. Any change in these supply relationships could negatively impact the Company's business. Manufacturing delays could result in significant revenue shortfalls and inventory shortages, potentially causing the Company to lose market share.

The Company is exposed to disruptions in its supply chain. The Company's business can be impacted by natural disasters, epidemic disease, labor disputes, political unrest, terrorism, or conflicts, all of which could disrupt the Company's supply chain or the sale of its products.""",

        'JPM': f"""ITEM 1A. RISK FACTORS

An investment in JPMorgan Chase & Co. involves a number of risks. The risks and uncertainties described below are not the only risks that may have a material adverse effect on JPMorgan Chase.

Strategic and Competitive Risks

The Firm operates in a highly competitive environment. The Firm competes with commercial banks, investment banks, securities firms, insurance companies, financial technology companies, merchant processors, and other companies offering financial services in the U.S. and internationally. Competition is intense in all of the Firm's businesses. Competition may increase further as regulatory restrictions are modified and new technologies enable greater competitive entry into the financial services industry.

Regulatory and Legal Risks

JPMorgan Chase is subject to extensive regulation, which significantly affects its businesses. The Firm is subject to extensive and comprehensive regulation under U.S. federal and state laws, as well as the laws of the jurisdictions outside the U.S. in which it operates. Banking and financial services laws, regulations and policies currently governing the Firm and its subsidiaries have recently undergone significant changes, and the future direction and impact of these laws, regulations and policies is difficult to predict.

Changes in regulation or law may materially and adversely impact the Firm's business. Regulation and the interpretation of regulation continues to evolve. Changes in regulation or legislation can affect the value of assets held under custody, the underlying value of certain assets, the revenue the Firm generates, and the business and operations of the Firm generally. In particular, the Firm is subject to capital and liquidity requirements that are more stringent than generally applicable regulatory requirements.

Financial and Credit Risks

JPMorgan Chase's business may be adversely affected by credit risk. Credit risk is the risk that customers or counterparties will be unable or unwilling to meet their contractual obligations. Credit risk exists for all transactions that give rise to actual, contingent or potential claims against a counterparty, borrower or obligor. Credit risk is one of the Firm's most significant risks.

Market risk could adversely affect the Firm's revenues. The Firm's market-making, investing and lending activities make it susceptible to changes in the level and volatility of interest rates, foreign exchange rates, market volatility, equity prices, credit spreads, and commodity prices. Market risk is inherent in the financial instruments associated with many of the Firm's operations and activities.""",

        'TSLA': f"""ITEM 1A. RISK FACTORS

Our business and financial performance are subject to numerous risks and uncertainties, including those highlighted below.

Operational Risks

We may experience delays in launching our products and features, which could harm our business and adversely affect our brand, business, prospects, financial condition and operating results. We often announce new product and feature releases or updates before they are commercially available. We may discover problems after introducing new products and features. Any delay in the release of new products or features could result in adverse publicity, loss of sales, and damage to our business and reputation.

Manufacturing capacity constraints and production difficulties may limit our ability to deliver our products. Production difficulties at our manufacturing facilities can reduce production volumes, result in the loss of revenue, and damage our reputation. We have experienced difficulties in ramping production to meet demand. Interruptions or inefficiencies in our supply chain or manufacturing could impact our production capabilities.

Financial Risks

We are subject to substantial regulation and unfavorable changes to, or failure by us to comply with, government laws and regulations could substantially harm our business and operating results. We are subject to federal, state, local, and foreign regulations specific to electric vehicles, vehicle safety standards, vehicle emissions, fuel economy standards, environmental regulations, consumer protection regulations, and competition laws. Changes in existing regulations or the introduction of new regulations could increase our operating costs.

Technology and Market Risks

The automotive market is highly competitive, and we may not be successful in competing in this industry. The worldwide automotive industry is highly competitive. We face competition from established and new automobile manufacturers, many of which have significantly greater financial and other resources than we do. Some of our competitors have established or are establishing battery, component, or vehicle production facilities and are investing significantly in the development of electric vehicles.""",

        'PFE': f"""ITEM 1A. RISK FACTORS

Our business faces significant risks and uncertainties that could have a material adverse effect on our business, financial condition, cash flows, and results of operations.

Commercial and Reputational Risks

We face intense competition from branded, generic, and biosimilar products. Our products face substantial competition from proprietary and generic products developed by our competitors. Generic or biosimilar competition can occur before the expiration of a product's patent protection due to patent challenges. Loss of exclusivity of one or more of our major products could significantly negatively impact our revenues and profitability.

Regulatory Risks

The development and commercialization of new products are subject to extensive regulation. Regulatory agencies regulate the research, development, manufacturing, safety, efficacy, record-keeping, labeling, storage, approval, advertising, promotion, sale, and distribution of biopharmaceutical products. We incur substantial costs and time delays in obtaining regulatory approvals, which can affect our business results.

We are subject to uncertainties relating to healthcare legislative and regulatory reform. Changes to healthcare laws and regulations may lower reimbursement for our products, requiring us to reduce our prices or experience reduced demand for our products. Government and private insurers increasingly are seeking price discounts and limiting access to drugs through formulary controls and reimbursement restrictions.

Operational and Manufacturing Risks

Manufacturing difficulties, delays, or disruptions could result in product shortages. Manufacturing biopharmaceuticals is complex. Technical issues or regulatory matters can lead to production delays, product recalls, or withdrawals. We rely on third-party manufacturers for certain products and raw materials. Disruptions at our manufacturing facilities or those of third parties could interrupt our supply and adversely affect our business.""",

        'XOM': f"""ITEM 1A. RISK FACTORS

Risk factors that could materially and adversely affect our business, financial condition, cash flows, and operating results include the following.

Commodity Price and Market Risks

Commodity prices, particularly crude oil and natural gas prices, are volatile and can significantly affect our revenues, cash flows, and profitability. Crude oil and natural gas prices are subject to volatile and sometimes wide fluctuations in response to changes in supply and demand, market uncertainty, and geopolitical events. Sustained periods of low commodity prices can reduce our revenues, cash flow, and profitability.

Operational and Supply Chain Risks

Our operations are subject to disruptions that could affect our production and sales. Our operations involve hazards inherent to hydrocarbon exploration, development, production, refining, and transportation. Operational disruptions from accidents, natural disasters, adverse weather, unscheduled downtime, labor disputes, or equipment failures could impact production volumes and financial performance.

Regulatory and Climate Risks

We are subject to extensive environmental, health, and safety laws and regulations. We are subject to numerous federal, state, local, and international environmental, health, and safety laws and regulations. Compliance with these requirements increases our costs. Changes in environmental regulations, including those related to climate change and greenhouse gas emissions, could require us to incur additional capital expenditures and increase operating costs.

Increasing attention to climate change and transition to a lower-carbon economy may adversely impact our business. Governmental and societal responses to concerns about climate change could result in regulations that have a significant impact on our operations and demand for our products. Our business may face physical risks from climate change, including impacts from extreme weather events."""
    }

    return mock_data.get(ticker, f"Mock risk factors data for {ticker} ({year}). This is placeholder text for demonstration purposes. In production, actual 10-K filing data would be loaded from files.")


# ==============================================================================
# RISK EXTRACTION
# ==============================================================================

def extract_risks(company: str, year: int, filing_text: str, temperature: float = 0.1, model: str = 'gpt-4o', openai_client=None):
    """
    Extracts structured risk factors from 10-K text using an LLM with Pydantic models.

    Args:
        company: Company name/ticker
        year: Fiscal year
        filing_text: The 10-K text to analyze
        temperature: LLM temperature (default: 0.1 for factual extraction)
        model: Model name (default: 'gpt-4o')
        openai_client: OpenAI client instance

    Returns:
        tuple: (list of risk dicts, input_tokens, output_tokens)
    """
    print(
        f"\n[DEBUG extract_risks] Starting extraction for {company} (FY{year})")
    print(
        f"[DEBUG extract_risks] Client provided: {openai_client is not None}")
    print(
        f"[DEBUG extract_risks] Filing text length: {len(filing_text)} chars")
    print(f"[DEBUG extract_risks] Temperature: {temperature}")

    if openai_client is None:
        print(f"[ERROR] OpenAI client not initialized for {company}")
        return [], 0, 0

    prompt = TASK_PROMPT_RISK_TEMPLATE.format(
        company=company, year=year, filing_text=filing_text)

    try:
        print(f"[DEBUG extract_risks] Calling OpenAI API for {company}...")
        response = openai_client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_RISK},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            text_format=RiskFactorsList
        )

        # Get the parsed Pydantic object
        parsed_output = response.output_parsed
        print(
            f"[DEBUG extract_risks] API call successful. Risks found: {len(parsed_output.risks)}")

        # Convert Pydantic models to dictionaries
        risks = [risk.model_dump() for risk in parsed_output.risks]

        # Add token usage for cost estimation
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        print(
            f"[DEBUG extract_risks] Token usage - Input: {input_tokens}, Output: {output_tokens}")
        return risks, input_tokens, output_tokens

    except Exception as e:
        print(f"[ERROR] LLM API call failed for {company}: {e}")
        import traceback
        traceback.print_exc()
        return [], 0, 0


# ==============================================================================
# OPPORTUNITY EXTRACTION
# ==============================================================================

def extract_opportunities(company: str, year: int, filing_text: str, temperature: float = 0.7, model: str = 'gpt-4o', openai_client=None):
    """
    Identifies structured investment opportunities from 10-K text using an LLM with Pydantic models.

    Args:
        company: Company name/ticker
        year: Fiscal year
        filing_text: The 10-K text to analyze
        temperature: LLM temperature (default: 0.7 for more creativity)
        model: Model name (default: 'gpt-4o')
        openai_client: OpenAI client instance

    Returns:
        tuple: (list of opportunity dicts, input_tokens, output_tokens)
    """
    print(
        f"\n[DEBUG extract_opportunities] Starting extraction for {company} (FY{year})")
    print(
        f"[DEBUG extract_opportunities] Client provided: {openai_client is not None}")
    print(
        f"[DEBUG extract_opportunities] Filing text length: {len(filing_text)} chars")
    print(f"[DEBUG extract_opportunities] Temperature: {temperature}")

    if openai_client is None:
        print(f"[ERROR] OpenAI client not initialized for {company}")
        return [], 0, 0

    prompt = TASK_PROMPT_OPPS_TEMPLATE.format(
        company=company, year=year, filing_text=filing_text)

    try:
        print(
            f"[DEBUG extract_opportunities] Calling OpenAI API for {company}...")
        response = openai_client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_OPPS},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            text_format=OpportunityList
        )

        # Get the parsed Pydantic object
        parsed_output = response.output_parsed
        print(
            f"[DEBUG extract_opportunities] API call successful. Opportunities found: {len(parsed_output.opportunities)}")

        # Convert Pydantic models to dictionaries
        opportunities = [opp.model_dump()
                         for opp in parsed_output.opportunities]

        # Add token usage for cost estimation
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        print(
            f"[DEBUG extract_opportunities] Token usage - Input: {input_tokens}, Output: {output_tokens}")
        return opportunities, input_tokens, output_tokens

    except Exception as e:
        print(f"[ERROR] LLM API call failed for {company}: {e}")
        import traceback
        traceback.print_exc()
        return [], 0, 0


# ==============================================================================
# HALLUCINATION AUDIT
# ==============================================================================

def audit_hallucinations(risks: list, source_text: str) -> pd.DataFrame:
    """
    Verifies each extracted risk's supporting quote appears in the source text.
    Flags potential hallucinations.

    Args:
        risks: List of risk dictionaries
        source_text: Original 10-K text

    Returns:
        pd.DataFrame: Audit results with verification flags
    """
    audit_results = []
    source_lower = source_text.lower()
    source_words = set(source_lower.split())

    for i, risk in enumerate(risks):
        quote = risk.get('supporting_quote', '').lower()
        risk_name = risk.get('risk_name', '')

        quote_words = set(w for w in quote.split() if w.strip())

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


# ==============================================================================
# COST ESTIMATION
# ==============================================================================

def estimate_llm_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Estimates the LLM API cost based on token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        float: Estimated cost in USD
    """
    cost_input = (input_tokens / 1_000_000) * GPT4O_PRICE_INPUT_PER_MILLION
    cost_output = (output_tokens / 1_000_000) * GPT4O_PRICE_OUTPUT_PER_MILLION
    return cost_input + cost_output


# ==============================================================================
# ANALYST REVIEW CHECKLIST GENERATION
# ==============================================================================

def generate_review_checklist(ticker: str, risks: list, opportunities: list, audit_df: pd.DataFrame) -> str:
    """
    Generates a comprehensive analyst review checklist for human oversight.

    Args:
        ticker: Company ticker symbol
        risks: List of extracted risk dictionaries
        opportunities: List of extracted opportunity dictionaries
        audit_df: Audit results DataFrame

    Returns:
        str: Formatted checklist text
    """
    checklist = []
    checklist.append("=" * 80)
    checklist.append(f"ANALYST REVIEW CHECKLIST: {ticker}")
    checklist.append("=" * 80)
    checklist.append("")
    checklist.append(
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    checklist.append("")
    checklist.append("PURPOSE:")
    checklist.append(
        "This checklist ensures that the AI-assisted risk and opportunity analysis")
    checklist.append(
        "meets the firm's standards for diligence and complies with CFA Institute")
    checklist.append(
        "Standards of Professional Conduct (especially Standard V(A) - Diligence")
    checklist.append("and Reasonable Basis).")
    checklist.append("")
    checklist.append("=" * 80)
    checklist.append("SECTION 1: HALLUCINATION AUDIT SUMMARY")
    checklist.append("=" * 80)
    checklist.append("")

    if not audit_df.empty:
        total_items = len(audit_df)
        verified_items = audit_df['VERIFIED'].sum()
        flagged_items = total_items - verified_items
        verification_rate = (verified_items / total_items) * \
            100 if total_items > 0 else 0

        checklist.append(f"Total Risks Extracted: {total_items}")
        checklist.append(f"Verified Risks: {verified_items}")
        checklist.append(f"Flagged for Review: {flagged_items}")
        checklist.append(f"Verification Rate: {verification_rate:.1f}%")
        checklist.append("")

        if flagged_items > 0:
            checklist.append("RISKS FLAGGED FOR MANUAL VERIFICATION:")
            for idx, row in audit_df[~audit_df['VERIFIED']].iterrows():
                checklist.append(
                    f"  - Risk #{row['risk_idx'] + 1}: {row['risk_name'][:60]}")
                checklist.append(
                    f"    Word Match: {row['quote_word_match']*100:.0f}%, Exact Match: {row['exact_substring_match']}")
            checklist.append("")
    else:
        checklist.append("No audit results available.")
        checklist.append("")

    checklist.append("=" * 80)
    checklist.append("SECTION 2: HIGH-SEVERITY RISK REVIEW")
    checklist.append("=" * 80)
    checklist.append("")
    checklist.append(
        "ACTION REQUIRED: Manually validate each high-severity risk below.")
    checklist.append("")

    high_severity_risks = [r for r in risks if r.get(
        'severity', '').lower() == 'high']
    if high_severity_risks:
        for i, risk in enumerate(high_severity_risks, 1):
            checklist.append(f"{i}. {risk.get('risk_name', 'N/A')}")
            checklist.append(f"   Category: {risk.get('category', 'N/A')}")
            checklist.append(f"   Severity: {risk.get('severity', 'N/A')}")
            checklist.append(f"   Novel: {risk.get('novel', 'N/A')}")
            checklist.append(
                f"   Supporting Quote: \"{risk.get('supporting_quote', 'N/A')[:100]}...\"")
            checklist.append(
                f"   Investment Implication: {risk.get('investment_implication', 'N/A')}")
            checklist.append("")
            checklist.append("   [ ] Verified quote in original filing")
            checklist.append(
                "   [ ] Assessed financial materiality (quantitative impact if available)")
            checklist.append(
                "   [ ] Considered risk in broader portfolio context")
            checklist.append("")
    else:
        checklist.append("No high-severity risks identified by AI.")
        checklist.append("")

    checklist.append("=" * 80)
    checklist.append("SECTION 3: OPPORTUNITIES REVIEW")
    checklist.append("=" * 80)
    checklist.append("")
    checklist.append("ACTION REQUIRED: Validate identified opportunities.")
    checklist.append("")

    if opportunities:
        for i, opp in enumerate(opportunities, 1):
            checklist.append(f"{i}. {opp.get('opportunity_name', 'N/A')}")
            checklist.append(f"   Category: {opp.get('category', 'N/A')}")
            checklist.append(f"   Timeframe: {opp.get('timeframe', 'N/A')}")
            checklist.append(
                f"   Evidence Quote: \"{opp.get('evidence_quote', 'N/A')[:100]}...\"")
            checklist.append(
                f"   Risk to Opportunity: {opp.get('risk_to_opportunity', 'N/A')}")
            checklist.append("")
            checklist.append("   [ ] Verified quote in original filing")
            checklist.append(
                "   [ ] Assessed feasibility and competitive dynamics")
            checklist.append("")
    else:
        checklist.append("No opportunities identified by AI.")
        checklist.append("")

    checklist.append("=" * 80)
    checklist.append("SECTION 4: MISSED RISKS & OPPORTUNITIES")
    checklist.append("=" * 80)
    checklist.append("")
    checklist.append(
        "ACTION REQUIRED: After reviewing the original 10-K filing, identify any")
    checklist.append(
        "material risks or opportunities that the AI may have missed.")
    checklist.append("")
    checklist.append("Missed Risks:")
    checklist.append("_" * 70)
    checklist.append("")
    checklist.append("_" * 70)
    checklist.append("")
    checklist.append("Missed Opportunities:")
    checklist.append("_" * 70)
    checklist.append("")
    checklist.append("_" * 70)
    checklist.append("")

    checklist.append("=" * 80)
    checklist.append("SECTION 5: COMPLIANCE SIGN-OFF")
    checklist.append("=" * 80)
    checklist.append("")
    checklist.append("By signing below, I certify that:")
    checklist.append("1. I have personally reviewed the AI-generated analysis")
    checklist.append(
        "2. I have verified the supporting quotes against the original 10-K filing")
    checklist.append(
        "3. I have assessed the materiality of identified risks and opportunities")
    checklist.append(
        "4. I have a reasonable and adequate basis for any recommendations")
    checklist.append(
        "5. This analysis meets CFA Standards V(A) (Diligence) and I(C) (Misrepresentation)")
    checklist.append("")
    checklist.append("Analyst Name: _______________________________")
    checklist.append("")
    checklist.append("Analyst Signature: _______________________________")
    checklist.append("")
    checklist.append("Date: _______________________________")
    checklist.append("")
    checklist.append("=" * 80)
    checklist.append("END OF CHECKLIST")
    checklist.append("=" * 80)

    return "\n".join(checklist)
