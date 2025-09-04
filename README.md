# Agentic Bias & Issue Classifier

A real-time, agent-powered tool for detecting bias and data issues in any CSV dataset. Upload your CSV, select columns, and let AI agents analyze each row for quality, completeness, and bias—no matter the domain or language.

## Features
- **Dynamic CSV Support:** Works with any CSV, not just translation datasets.
- **Agentic Analysis:** Multiple AI agents review each row for issues and bias.
- **Configurable Types:** Issue and bias types are managed via `types.json` for easy updates.
- **Streamlit UI:** Upload CSVs, select columns/ranges, and view results interactively.
- **Robust Parsing:** Handles agent responses with explanations, markdown, or formatting.
- **Flexible & Extensible:** Add new issue/bias types by editing `types.json`—no code changes needed.

## How It Works
1. **Upload CSV:** Use the Streamlit UI to upload your dataset.
2. **Select Columns/Range:** Choose which columns and rows to analyze.
3. **Agentic Processing:** AI agents analyze each row, selecting from predefined issue and bias types.
4. **Results:** Clean rows are saved to `GOOD_CSV`, flagged rows to `ISSUES_CSV`, and logs are written for traceability.

## Setup
1. **Clone the repo:**
   ```sh
   git clone https://github.com/yourusername/agentic-bias-classifier.git
   cd agentic-bias-classifier
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Configure API keys:**
   - Add your Groq/OpenAI API keys to a `.env` file:
     ```env
     GROQ_API_KEY_1=your_key_1
     GROQ_API_KEY_2=your_key_2
     GROQ_API_KEY_3=your_key_3
     ```
4. **Edit `types.json` (optional):**
   - Add or remove issue/bias types as needed.

## Usage
1. **Start the UI:**
   ```sh
   streamlit run User_Interface.py
   ```
2. **Upload your CSV and run analysis.**
3. **Check output files:**
   - `GOOD_CSV`: Rows with no issues or bias.
   - `ISSUES_CSV`: Rows flagged for issues or bias.
   - `LOG_FILE`: Processing logs and errors.

## File Structure
- `User_Interface.py` — Streamlit UI for upload and analysis
- `Agents.py` — Agent pool, prompt logic, and CSV processing
- `types.json` — Configurable issue and bias types
- `requirements.txt` — Python dependencies
- `.env` — API keys (not committed)
- `README.md` — Project documentation

## Customization
- **Add new types:** Edit `types.json`.
- **Change agent logic:** Modify prompts in `Agents.py`.
- **UI tweaks:** Update `User_Interface.py` for new features.

## License
MIT

## Contributing
Pull requests and suggestions welcome! Open an issue or submit a PR.

## Contact
For questions or support, open a GitHub issue or contact the maintainer.
