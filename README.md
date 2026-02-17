Here is a comprehensive `README.md` file for your Streamlit application lab project.

---

# QuLab: Lab 11 - Idea Generation with GPT (Generative AI)

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Lab 11 - Idea Generation with GPT (Generative AI): Unlocking 10-K Insights**

This Streamlit application, developed as part of the QuantUniversity Lab series (QuLab Lab 11), empowers financial analysts to leverage Large Language Models (LLMs) for efficient and compliant extraction of material risks and opportunities from SEC 10-K filings.

The application simulates the workflow of a Senior Equity Research Analyst, Sarah Chen, CFA Charterholder, at 'Global Markets Insight' (GMI). It automates the preliminary research phase, enhances consistency, and frees up human analysts for higher-value qualitative judgment, critical review, and client engagement. It emphasizes a "human-in-the-loop" approach, focusing on ethical AI integration, hallucination auditing, and compliance with CFA Institute Standards of Professional Conduct.

## Features

This application guides users through a structured AI-augmented analytical workflow, featuring:

1.  **Home Page**: Introduction to the project, scenario, persona (Sarah Chen, CFA), connections to the CFA curriculum, and a critical warning on data privacy.
2.  **Setup & Data Loading**:
    *   Configuration of OpenAI API Key.
    *   Selection of companies and fiscal years for analysis.
    *   Automated loading and processing of 10-K risk factors (or relevant sections) for selected companies.
    *   Token counting and intelligent text chunking to manage LLM context window limits.
3.  **LLM Prompts & Extraction**:
    *   Pre-defined expert system and task prompts to guide the LLM (GPT-4o) as an "AI junior analyst."
    *   Configurable `temperature` parameter for risk (low for factual) and opportunity (slightly higher for nuanced insight) extraction.
    *   Automated extraction of structured risk factors and opportunities (name, category, severity/risk-to-opportunity, implications, supporting quotes) into JSON format.
    *   Tracking of total input and output tokens.
4.  **Hallucination Audit**:
    *   Automated verification of LLM-generated supporting quotes against the original 10-K text to detect "hallucinations" (fabrications).
    *   Calculation and visualization of verification rates and potential hallucination flags.
    *   Quantitative metrics for assessing AI trustworthiness (Hallucination Rate, Precision, Recall definitions).
5.  **Comparative Analysis**:
    *   Transformation of extracted JSON data into pandas DataFrames for structured comparison.
    *   Cross-company comparison tables for risks and opportunities.
    *   Visualizations (stacked bar charts, heatmaps) to identify industry trends, common risks, unique opportunities, and severity distributions across peer groups.
6.  **Cost & ROI**:
    *   Real-time estimation of LLM API costs based on token usage.
    *   Breakdown of costs by input and output tokens.
    *   Justification for Return on Investment (ROI) by contrasting AI costs with saved human analyst hours.
7.  **Analyst Review & Compliance**:
    *   Generation of a comprehensive "Analyst Review Checklist" for human oversight.
    *   Emphasis on ethical AI, human judgment, and adherence to CFA Standards (Diligence, Reasonable Basis, Misrepresentation).
    *   Facilitates structured human validation, material risk assessment, and compliance reporting.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   An OpenAI API Key. You can obtain one from the [OpenAI Developer Platform](https://platform.openai.com/).

### Installation

1.  **Clone the Repository (or create project structure):**
    ```bash
    git clone https://github.com/your-repo/qu-lab11-ai-idea-generation.git
    cd qu-lab11-ai-idea-generation
    ```
    *(Note: If this is a direct code snippet, assume `app.py` and `source.py` are in the same directory.)*

2.  **Create a Virtual Environment:**
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit>=1.30.0
    pandas>=2.0.0
    matplotlib>=3.0.0
    seaborn>=0.12.0
    openai>=1.0.0
    tiktoken>=0.5.0 # For token counting
    fuzzywuzzy>=0.18.0 # Often used for string matching in audit functions
    python-Levenshtein>=0.21.0 # Dependency for fuzzywuzzy for speed
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables

Set your OpenAI API key as an environment variable (recommended for security):

*   **On macOS/Linux:**
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
*   **On Windows (Command Prompt):**
    ```bash
    set OPENAI_API_KEY="your_openai_api_key_here"
    ```
*   **On Windows (PowerShell):**
    ```bash
    $env:OPENAI_API_KEY="your_openai_api_key_here"
    ```
    *(Alternatively, you can paste the API key directly into the Streamlit app on the "1. Setup & Data Loading" page, but using an environment variable is more secure for persistent use.)*

## Usage

1.  **Run the Streamlit Application:**
    Navigate to the project directory in your terminal (with the virtual environment activated) and run:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

2.  **Navigate the Workflow:**
    The application presents a sidebar with navigation steps:
    *   **Home**: Get an overview of the project, persona, and ethical considerations.
    *   **1. Setup & Data Loading**: Input your OpenAI API key, select companies (e.g., AAPL, MSFT, JPM) and their fiscal years. Click "Load Filings" to process the mock 10-K data.
    *   **2. LLM Prompts & Extraction**: Review the prompt templates and configure LLM temperature. Click "Extract Risks & Opportunities" to send the data to GPT-4o for analysis.
    *   **3. Hallucination Audit**: Click "Run Hallucination Audit" to verify the LLM's extracted quotes against the original text, identifying potential fabrications.
    *   **4. Comparative Analysis**: Explore the aggregated data in tables and visualizations to compare risks and opportunities across selected companies.
    *   **5. Cost & ROI**: Review the estimated API costs and the ROI justification for using AI in this workflow.
    *   **6. Analyst Review & Compliance**: Select a company and generate a comprehensive checklist to guide human review and ensure compliance.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── source.py               # Contains helper functions, prompt templates, and data
└── data/                   # Directory for dummy 10-K filing text files
    ├── AAPL_2023_risk_factors.txt
    ├── MSFT_2023_risk_factors.txt
    ├── JPM_2023_risk_factors.txt
    └── ...                 # Other company filing data
├── requirements.txt        # Python dependencies
└── README.md               # This README file
```

### `app.py`
This file contains the Streamlit UI, handles session state, orchestrates page navigation, and calls functions from `source.py` to perform the core logic.

### `source.py`
This crucial file encapsulates the backend logic and data, including:
*   `companies_data`: Initial list of companies for selection.
*   Prompt Templates: `SYSTEM_PROMPT_RISK`, `TASK_PROMPT_RISK_TEMPLATE`, `SYSTEM_PROMPT_OPPS`, `TASK_PROMPT_OPPS_TEMPLATE`.
*   LLM Pricing Constants: `GPT4O_PRICE_INPUT_PER_MILLION`, `GPT4O_PRICE_OUTPUT_PER_MILLION`.
*   Core Functions:
    *   `get_10k_risk_factors(ticker, year)`: Loads dummy 10-K text for a given company and year.
    *   `count_tokens(text)`: Estimates token count using `tiktoken`.
    *   `chunk_text(text, max_tokens, overlap)`: Chunks long texts for LLM context windows.
    *   `extract_risks(...)`, `extract_opportunities(...)`: Functions to call OpenAI API for extraction, using defined prompts and temperature.
    *   `audit_hallucinations(extracted_items, source_text)`: Verifies extracted quotes against source text.
    *   `estimate_llm_cost(input_tokens, output_tokens)`: Calculates API cost.
    *   `generate_review_checklist(...)`: Creates the human review checklist.

### `data/`
This directory is expected to contain pre-downloaded or dummy 10-K text files. For example, `AAPL_2023_risk_factors.txt` would hold the extracted risk factors section from Apple's 2023 10-K filing.

## Technology Stack

*   **Python**: Programming language
*   **Streamlit**: For building the interactive web application UI
*   **OpenAI API (GPT-4o)**: Large Language Model for text extraction and generation
*   **Pandas**: For data manipulation and analysis
*   **Matplotlib / Seaborn**: For data visualization
*   **`tiktoken`**: OpenAI's tokenizer for token counting
*   **`fuzzywuzzy` & `python-Levenshtein`**: Likely used for string matching and similarity in hallucination audit.

## Contributing

Contributions to this lab project are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and ensure the code adheres to best practices.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*(Note: You will need to create a `LICENSE` file in the root of your project.)*

## Contact

For questions or feedback regarding this QuLab project, please contact:

*   **QuantUniversity**
*   **Website**: [www.quantuniversity.com](https://www.quantuniversity.com/)
*   **Email**: info@quantuniversity.com

---