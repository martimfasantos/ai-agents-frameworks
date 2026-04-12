# Comparison and Study of AI Agent Frameworks

A comprehensive study comparing **14 AI agent frameworks** using a standardized agent interface. Each framework is evaluated on implementation complexity, response time, token usage, tool calling, RAG retrieval accuracy, and multi-agent coordination.

**Frameworks included:**

| Framework | Basic Agent | RAG+API Agent | Multi-Agent |
|---|---|---|---|
| [Agno](https://docs.agno.com/introduction) | `agno_agent.py` | `agno_rag_api_agent.py` | - |
| [AG2](https://docs.ag2.ai/) | `ag2_agent.py` | - | - |
| [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-agent-sdk) | `claude_sdk_agent.py` | - | - |
| [CrewAI](https://docs.crewai.com/) | `crewai_agent.py` | `crewai_rag_api_agent.py` | Yes |
| [Google ADK](https://google.github.io/adk-docs/) | `google_adk_agent.py` | - | - |
| [LangChain](https://python.langchain.com/docs/) | `langchain_agent.py` | `langchain_rag_api_agent.py` | - |
| [LangGraph](https://www.langchain.com/langgraph) | `langgraph_agent.py` | `langgraph_rag_api_agent.py` | Yes |
| [LlamaIndex](https://docs.llamaindex.ai/en/stable/) | `llama_index_agent.py` | `llama_index_rag_api_agent.py` | - |
| [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) | `microsoft_agent.py` | - | - |
| [OpenAI (raw API)](https://platform.openai.com/docs/guides/function-calling) | `openai_agent.py` | - | - |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | `openai_agents_sdk_agent.py` | - | Yes |
| [PydanticAI](https://ai.pydantic.dev/) | `pydantic_ai_agent.py` | - | - |
| [Smolagents](https://huggingface.co/docs/smolagents/) | `smolagents_agent.py` | - | - |
| [Strands Agents SDK](https://strandsagents.com/) | `strands_agent.py` | - | - |

> For a detailed analysis, see [REPORT.md](REPORT.md).

---

## Setup

### Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Navigate to the study directory
cd study-agents-differences

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Required keys depend on your provider:
- **Azure OpenAI**: `azure_endpoint`, `azure_deployment_name`, `azure_api_version`, `azure_api_key`
- **OpenAI**: `openai_api_key`, `openai_model_name`
- **Anthropic** (Claude SDK only): `anthropic_api_key`
- **Google** (Google ADK only): `google_api_key`
- **Tavily** (web search): `tavily_api_key`

---

## Running Agents

Each agent can be run independently as a CLI chatbot:

```bash
# Run with default settings (Azure provider, memory enabled)
python agno_agent.py

# Run with OpenAI provider
python crewai_agent.py --provider openai

# Run in metrics mode (single question, outputs timing + tokens)
python pydantic_ai_agent.py --mode metrics --provider openai

# Run benchmark loop (N iterations of the same question)
python langgraph_agent.py --mode metrics-loop --iter 50 --provider azure --no-memory --create
```

### CLI Flags

| Flag | Description | Default |
|---|---|---|
| `--provider [azure\|openai\|other]` | LLM provider | `azure` |
| `--mode [metrics\|metrics-loop]` | Execution mode | None (chatbot) |
| `--iter N` | Iterations for metrics-loop | 1 |
| `--no-memory` | Disable conversation memory | False |
| `--create` | Recreate agent each iteration | False |
| `--verbose` | Show agent logs and responses | Auto |
| `--file PATH` | Save output to file | None |

---

## Benchmark Suite

### Standard Benchmarks

The `benchmark_runner.py` script runs standardized tests across all agents:

```bash
# Run all benchmarks on all agents
python benchmark_runner.py --provider openai --benchmark all

# Run only basic benchmarks with 20 iterations
python benchmark_runner.py --provider openai --benchmark basic --iterations 20

# Run RAG benchmarks on specific agents
python benchmark_runner.py --provider azure --benchmark rag --agents agno langgraph llama_index

# Save results to custom directory
python benchmark_runner.py --provider openai --benchmark target --output my_results/
```

**Benchmark suites:**

| Suite | Questions | Tests |
|---|---|---|
| `basic` | 5 | Web search (explicit/implicit), date query, greeting, factual |
| `rag` | 3 | Ball possession, match score, Arsenal score |
| `api` | 2 | Combined Metro+F1 query, single F1 query |
| `target` | 3 | US president (factual), train distance (reasoning), AI regulation (web search) |
| `all` | 13 | All of the above |

**Agents registered:**
- Basic (14): agno, ag2, claude_sdk, crewai, google_adk, langchain, langgraph, llama_index, microsoft, openai, openai_agents_sdk, pydantic_ai, smolagents, strands
- RAG+API (5): agno, crewai, langchain, langgraph, llama_index

### Multi-Agent Benchmarks

```bash
# Run multi-agent benchmarks
python benchmark_multi_agent.py --provider openai --iterations 5
```

Tests CrewAI (researcher + analyst crew), LangGraph (tool-augmented agent), and OpenAI Agents SDK (agent handoffs) on coordination tasks.

---

## Results

> The results below were collected with Azure OpenAI (GPT-4o-mini) on the original four frameworks. Run `benchmark_runner.py` to generate updated results across all 14 frameworks.

### Response Time with Memory

**Prompt:** _search the web for who won the Champions League final in 2024?_

| Iterations | Agno | LangGraph | LlamaIndex |
|---|---|---|---|
| 20x | 5.41 +/- 1.19s | 6.04 +/- 2.61s | 5.36 +/- 2.02s |
| 50x | 4.24 +/- 0.78s | 8.48 +/- 2.56s | 3.00 +/- 3.24s |
| 100x | 4.39 +/- 0.73s | 9.45 +/- 4.73s | 2.64 +/- 2.29s |

**LangGraph** shows degrading performance over iterations due to memory accumulation. **LlamaIndex** remains stable.

### Response Time without Memory

**Prompt:** _search the web for who won the Champions League final in 2024?_

| Iterations | Agno | LangGraph | LlamaIndex | OpenAI (raw) |
|---|---|---|---|---|
| 50x | 4.58 +/- 1.03s | 4.22 +/- 1.11s | 4.12 +/- 1.01s | 3.83 +/- 0.99s |
| 100x | 4.28 +/- 0.76s | 3.31 +/- 0.59s | 3.63 +/- 0.66s | 3.61 +/- 0.83s |

### Token Usage (100x, without Memory)

| Metric | Agno | LangGraph | LlamaIndex | OpenAI (raw) |
|---|---|---|---|---|
| Prompt Tokens | 1999.2 | 1946.1 | 2121.7 | 1888.5 |
| Completion Tokens | 65.3 | 53.5 | 76.9 | 58.3 |
| Total Tokens | 2064.5 | 1999.7 | 2198.6 | 1946.7 |

### RAG (100x)

**Prompt:** _Ball possession in Benfica's game?_

| Metric | Agno | LangGraph | LlamaIndex |
|---|---|---|---|
| Response Time | 3.30 +/- 0.75s | 2.68 +/- 1.35s | 2.86 +/- 1.05s |
| Total Tokens | 4439.3 | 4877.2 | 3279.9 |
| Misses | 2/100 | 4/100 | 2/100 |

### API (100x)

**Prompt:** _Tell me the waiting time at the CG station and the status of the red line, and also give me information about Formula 1 driver number 44!_

| Metric | Agno | LangGraph | LlamaIndex |
|---|---|---|---|
| Response Time | 5.49 +/- 1.40s | 4.24 +/- 1.35s | 6.41 +/- 2.47s |
| Total Tokens | 1849.2 | 1412.2 | 3913.4 |
| Misses | 0/100 | 0/100 | 0/100 |

---

## Implementation Complexity

Lines of code per basic agent (sorted by simplicity):

```
openai_agents_sdk_agent.py  137  ████████
claude_sdk_agent.py         140  ████████
crewai_agent.py             144  █████████
ag2_agent.py                150  █████████
microsoft_agent.py          150  █████████
strands_agent.py            156  █████████
langchain_agent.py          161  ██████████
smolagents_agent.py         166  ██████████
google_adk_agent.py         170  ██████████
pydantic_ai_agent.py        172  ██████████
agno_agent.py               203  ████████████
llama_index_agent.py        217  █████████████
langgraph_agent.py          243  ██████████████
openai_agent.py             281  ████████████████
```

---

## UI

Run the Streamlit chat interface to interact with all agents:

```bash
streamlit run agent-ui.py
```

The UI auto-discovers all `*_agent.py` files and presents them in a sidebar for selection.

---

## Project Structure

```
study-agents-differences/
├── settings.py              # Environment configuration
├── prompts.py               # Shared system prompts
├── utils.py                 # CLI parsing & execution utilities
├── shared_functions/        # Shared API tools (F1, Metro)
├── knowledge_base/          # RAG data (Champions League matches)
│
├── agno_agent.py            # Agno basic agent
├── agno_rag_api_agent.py    # Agno RAG+API agent
├── ag2_agent.py             # AG2 basic agent
├── claude_sdk_agent.py      # Claude Agent SDK basic agent
├── crewai_agent.py          # CrewAI basic agent
├── crewai_rag_api_agent.py  # CrewAI RAG+API agent
├── google_adk_agent.py      # Google ADK basic agent
├── langchain_agent.py       # LangChain basic agent
├── langchain_rag_api_agent.py # LangChain RAG+API agent
├── langgraph_agent.py       # LangGraph basic agent
├── langgraph_rag_api_agent.py # LangGraph RAG+API agent
├── llama_index_agent.py     # LlamaIndex ReAct agent
├── llama_index_fc_agent.py  # LlamaIndex Function Calling agent
├── llama_index_rag_api_agent.py # LlamaIndex RAG+API agent
├── microsoft_agent.py       # Microsoft Agent Framework basic agent
├── openai_agent.py          # OpenAI raw API agent
├── openai_agents_sdk_agent.py # OpenAI Agents SDK basic agent
├── pydantic_ai_agent.py     # PydanticAI basic agent
├── smolagents_agent.py      # Smolagents basic agent
├── strands_agent.py         # Strands Agents SDK basic agent
│
├── benchmark_runner.py      # Standard benchmark suite
├── benchmark_multi_agent.py # Multi-agent benchmark
├── agent-ui.py              # Streamlit chat UI
│
├── REPORT.md                # Detailed comparison report
├── pyproject.toml           # Project dependencies
├── tests/                   # Historical test outputs
└── res/                     # Screenshots
```
