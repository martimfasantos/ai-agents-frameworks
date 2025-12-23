# AI Agent Frameworks - Development Guide

This repository contains hands-on examples comparing 10+ AI agent frameworks. Each framework lives in its own top-level directory with standalone examples demonstrating key capabilities.

## Repository Architecture

### Framework Organization Pattern
Each framework directory (`pydantic-ai/`, `crewai/`, `autogen/`, etc.) follows this structure:
- **Numbered examples**: `0_hello_world.py`, `1_tools.py`, `2_streaming.py` - Progressive feature demonstrations
- **`settings.py`**: Pydantic-based settings loader reading from `.env` (see `.env.example`)
- **`requirements.txt`**: Python dependencies (some frameworks use PDM with `pyproject.toml`)
- **`utils.py`**: Shared helper functions where applicable
- **Framework-specific subdirectories**: Some have `*-project/` or `*-examples/` subdirectories

### Dependency Management
- **Most frameworks**: Use `requirements.txt` → Install with `pip install -r requirements.txt` in activated venv
- **PDM projects**: Subdirectories with `pdm.lock` (e.g., `crewai-project/`, `langgraph-project/`)
- **Always check each framework's `README.md`** for specific setup instructions

## Development Workflow

### Running Examples
1. **Navigate to framework directory**: `cd pydantic-ai/` (or relevant framework)
2. **Create/activate venv**: 
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   ```
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Configure environment**: Copy `.env.example` to `.env` and add API keys (OpenAI, Azure, etc.)
5. **Run examples**: `python3 0_hello_world.py` (or use the numbered file pattern)

### Environment Configuration
All frameworks use **Pydantic `BaseSettings`** to load environment variables from `.env`:
```python
# Standard pattern in settings.py
class Settings(BaseSettings):
    OPENAI_API_KEY: pydantic.SecretStr
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    
    class Config:
        env_file = ".env"
```

**Critical**: `.env` files are at the **repository root** for most frameworks, not in each framework directory.

## Code Style & Conventions

### File Naming
- **Numbered examples**: `0_hello_world.py`, `1_tools.py`, `2_streaming.py` - numbers indicate progression
- **Multi-word names**: Use underscores (e.g., `5_human_in_the_loop.py`)

### Example Structure
All examples follow this pattern:
```python
from dotenv import load_dotenv
from settings import settings
load_dotenv()

"""
-------------------------------------------------------
In this example, we explore a simple Hello World agent
-------------------------------------------------------
"""

# 1. Define agent/tools with numbered comments
# 2. Create instance with model from settings.OPENAI_MODEL_NAME
# 3. Run/execute
# 4. Print results
```

### Docstrings & Comments
- **Module docstring**: Use horizontal rule style (`-------` or `-------------`) for multi-line descriptions
- **Section comments**: Number steps with `# --- 1. Create agent ---` format
- **Section separators**: Print statements with `print("=== Section Title ===")`  or `print("\n" + "="*60 + "\n")`
- **Keep it simple**: Examples should be concise and easy to follow

### Tool Definitions
- **Type hints**: Always include for function parameters and return types
- **Docstrings**: Required for tools - the LLM reads these for schema generation
  ```python
  def get_weather(location: str) -> WeatherData:
      """Get current weather for a location.
      
      Args:
          location: The city name to get weather for
      """
  ```

### Utility Functions
When frameworks have repeated patterns, extract to `utils.py`:
- Keep function signatures consistent across examples
- Use descriptive names that indicate purpose
- Examples: `print_all_messages()`, `show_metrics()`, `print_new_section()`

## Framework-Specific Patterns

### Pydantic-AI
- **Tool decorators**: `@agent.tool` (context-aware) vs `@agent.tool_plain` (standalone)
- **RunContext**: `ctx: RunContext[T]` provides typed dependencies to tools
- **Metrics**: All runs return `RunUsage` - access with `result.usage()`
- **Async patterns**: Use `async def main()` with `agent.run()` (not `run_sync()`)

### CrewAI
- **Structure**: Agent → Task → Crew (three-tier)
- **Verbose mode**: `verbose=True` on agents (no manual printing needed)
- **Execution**: `crew.kickoff()` for workflows, `agent.execute_task(task)` for single tasks
- **Environment**: Set `os.environ["OPENAI_API_KEY"]` from settings

### Autogen
- **v0.4 architecture**: Event-driven, asynchronous (Core API + AgentChat API)
- **Legacy notice**: v0.4 replaces v0.2 - use new patterns

### Study-Agents-Differences
Cross-framework comparison tools with unified interfaces:
- **CLI flags**: `--provider`, `--mode`, `--iter`, `--no-memory`, `--create`, `--verbose`, `--file`
- **Streamlit UI**: `streamlit run agent-ui.py`
- **Usage**: `python [script].py --mode metrics-loop --iter 30 --create --no-memory --verbose --file results.txt`

## Environment & Integration

### Environment Variables
- **OpenAI**: `OPENAI_API_KEY`, `OPENAI_MODEL_NAME` (typically `gpt-4o-mini` or `gpt-4o`)
- **Azure**: `azure_endpoint`, `azure_deployment_name`, `azure_api_version`, `azure_api_key`
- **Optional tools**: `tavily_api_key` for web search

### Settings Pattern
All frameworks use Pydantic `BaseSettings`:
```python
class Settings(BaseSettings):
    OPENAI_API_KEY: pydantic.SecretStr
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Memory & Message History
- Access via framework-specific methods (e.g., `result.all_messages()` in Pydantic-AI)
- Be aware: memory storage can impact performance at scale
- Some frameworks support tool call history to avoid redundant API calls

## Common Pitfalls

1. **Missing `.env`**: Copy `.env.example` to `.env` and add API keys (examples fail silently without this)
2. **Virtual environment**: Must activate venv before running examples or installing dependencies
3. **Dependency manager**: Check README - some use pip, others use PDM
4. **Model names**: Azure vs OpenAI use different naming formats

## Adding New Examples

Follow these conventions for consistency:

1. **Naming**: Use numbered format `N_feature_name.py` (e.g., `14_new_feature.py`)
2. **Docstring**: Add horizontal-rule style module docstring at top
3. **Comments**: Number steps with `# --- 1. Step description ---`
4. **Settings**: Import from `settings.py` - never hardcode API keys
5. **Simplicity**: Keep examples concise and focused on one concept
6. **Separators**: Use `print("=== Section Title ===")` between major sections
7. **README**: Update if adding new dependencies
