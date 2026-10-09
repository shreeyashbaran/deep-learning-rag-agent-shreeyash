# Part 1 RAG Workshop — completion and submission

This implementation follows the supplied RAG_INSTRUCTIONS text guide. The separately named PART01_RAG_Instructions.md and instructor test_rag.md were not included in the attachments. The included examples/lstm_intermediate.md is an alternative technical note, permitted by the supplied guide.

## Apply the completion package on Windows

Save RAG_Part1_Completion.zip in Downloads. If your browser adds a suffix such as (1), adjust the filename below. Close any running Streamlit server and save your current edits first. Run:

```powershell
cd C:\Users\shrey\deep-learning-rag-agent
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\RAG_Part1_Completion.zip" -DestinationPath "$env:USERPROFILE\Downloads\RAG_Part1_Completion" -Force
uv run python "$env:USERPROFILE\Downloads\RAG_Part1_Completion\apply_completion.py" --repo .
uv add langchain-text-splitters
uv run pytest tests -q
```

Run these one at a time; stop on an error. The installer verifies package hashes, backs up replaced source files beside your project, and copies only the listed completion files. It preserves .env, .git, pyproject.toml, uv.lock, and your existing database. It replaces the listed source files, including the two factories and the Chroma initialization you already wrote. Other local edits in those files remain available in the backup.

You already installed torchvision and configured .env. Keep these values locally:

```dotenv
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=your_own_key
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Never paste your actual key into source code or screenshots. The uv configuration and LangChain embedding deprecation warnings are allowed by this starter workflow; warnings alone do not mean execution failed.

## Optional actual-model check

```powershell
uv run python scripts/check_part1.py --groq
```

This creates a temporary database, uses your actual local embedding model, verifies ingestion and duplicate handling, retrieves the forget-gate material, and makes one request using your actual Groq key. It does not populate the Streamlit application's database. Inspect the displayed answer for accuracy against the note. On Windows, Chroma may keep a temporary database open until Python exits.

## Run the application and collect the required evidence

```powershell
uv run streamlit run src/rag_agent/ui/app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

1. Upload examples/lstm_intermediate.md in the sidebar.
2. Click Ingest Documents. Confirm chunks were added and Total Chunks is greater than zero.
3. Click Ingest Documents again. Confirm duplicates were skipped and the total chunk count stays unchanged.
4. Leave Topic and Difficulty set to their All options.
5. Ask: What does the LSTM forget gate do?
6. Confirm the answer describes retaining or removing information from the previous cell state. The note explicitly states that near-zero gate values remove information and near-one values retain it.
7. Confirm Retrieved sources shows lstm_intermediate.md. Source labels identify the context supplied to the model; they are not automated claim-level verification.
8. Capture a real screenshot showing the question, generated answer, and source information. Use Windows + Shift + S. Save it as screenshots/rag_demo.png inside your repository. If needed, reduce browser zoom or scroll the chat pane so the question and answer are both visible.

Create the screenshot folder from a second PowerShell terminal if necessary:

```powershell
cd C:\Users\shrey\deep-learning-rag-agent
mkdir screenshots
Test-Path screenshots/rag_demo.png
```

The final check must print True after you save the screenshot. This package does not contain a fabricated screenshot. Stop Streamlit with Ctrl+C when finished.

If an answer says no relevant context, inspect the note, filters, and chunk count first. If a Groq request fails, read the local terminal traceback. Do not share your API key. If you edit a note and re-upload it under the same name, use Remove first to avoid retaining its old chunks.

## Publish to your own GitHub repository

On GitHub, create an empty public repository named deep-learning-rag-agent-shreeyash (unless the instructor requires private). Do not initialize it with a README or license.

Back in PowerShell:

```powershell
cd C:\Users\shrey\deep-learning-rag-agent
Test-Path screenshots/rag_demo.png
git check-ignore .env
git ls-files -- .env
git remote -v
```

The screenshot check must be True; the ignore check should print .env; the tracked-files check should print nothing. Inspect the listed remotes. If origin still points to gitmystuff, use:

```powershell
git remote rename origin upstream
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/deep-learning-rag-agent-shreeyash.git
```

Replace YOUR_GITHUB_USERNAME before running the command. If origin already points to your own repository, skip the rename/add commands.

Stage and inspect the submission:

```powershell
git add src tests examples scripts docs README.md pyproject.toml uv.lock screenshots/rag_demo.png
git diff --cached --stat
git status
```

Make sure .env and local database files are not staged, and no source file contains your real key. Then:

```powershell
git commit -m "Complete Part 1 Markdown RAG workshop"
git branch -M main
git push -u origin main
```

Open your own GitHub repository and verify the code, note, and screenshot are present and .env is absent. Submit that repository URL in Canvas. No Git commit, push, or Canvas submission was performed by the assistant.

## What was completed

- Local embedding factory and Groq client factory, with missing-key validation.
- Persistent Chroma initialization, batched ingestion, duplicate detection, cosine retrieval, thresholds, and topic/difficulty filters.
- Corpus statistics, document listing, chunk viewing in original order, and document deletion.
- Markdown heading-aware chunking and filename-based metadata inference.
- Cached Streamlit models, multiple Markdown uploads, chat history, grounded-context prompts, and displayed sources.
- A context-budget limit and a no-retrieval response that makes no LLM request.
- A sample study note, actual-model smoke-check command, and executable tests.

PDF support, alternate model providers, and the unused chunk_files helper remain starter stubs as permitted by the supplied instructions. The LangGraph graph and nodes are reserved for later parts and are not called in Part 1. A separate qa.py helper keeps direct Part 1 retrieval and generation independently testable.

## Verification and remaining local checks

26 checks passed in the assistant's isolated Python 3.12 environment using ChromaDB 1.5.9, LangChain Core 1.6.9, langchain-groq 1.1.3, langchain-text-splitters 1.1.3, and Streamlit 1.65.0. Tests use real persistent Chroma and Streamlit AppTest, with deterministic embedding and LLM test doubles. They verify database behavior and prompt/source wiring; they do not measure real-model semantic quality or Groq availability.

Your earlier Python 3.11 checks already confirmed that the actual local model produces 384-dimensional vectors and your configured Groq model responds. The integrated real-model smoke check, actual UI screenshot, GitHub push, and Canvas submission still need to be performed on your laptop.

The checked tests cover deterministic IDs, duplicate uploads, filtering, ordering, persistence, deletion, error reporting, Markdown metadata, source-bearing prompts, context limits, no-context behavior, and UI chat history. No unit tests are skipped.
