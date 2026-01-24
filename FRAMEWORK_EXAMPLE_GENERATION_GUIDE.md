# Universal Agentic Framework Example Generation Guide

This guide provides a complete methodology for creating straightforward, feature-focused example suites for any agentic AI framework. It was developed through the process of restructuring and enhancing the LlamaIndex examples and can be applied to any framework.

---

## Overview

You are creating a suite of examples that helps developers understand a framework's **core features** through **simple, clear demonstrations**. Each example should focus on ONE feature and showcase it in the most straightforward way possible.

### Key Principles

1. **One Feature Per Example** - Each file demonstrates exactly one core capability
2. **Straightforward Over Clever** - Direct demonstration beats advanced patterns
3. **Feature-Focused** - The feature is the hero, not the code complexity
4. **Beginner-Friendly** - Minimal prerequisites, maximum clarity
5. **Well-Documented** - Official source links for every feature

---

## Phase 1: Framework Analysis

### Step 1.1: Documentation Discovery

Start by gathering comprehensive information about the framework:

**Primary Sources (in priority order):**
1. Official documentation (e.g., `https://docs.[framework].ai/`)
2. Official GitHub repository (`https://github.com/[org]/[framework]`)
3. Official blog/announcements
4. Example repositories and starter templates

**What to Look For:**
- Main landing page - what does the framework claim to do best?
- Getting started guide - what are the first features introduced?
- API reference - what are the core classes and methods?
- Examples directory - what features are showcased?
- Blog posts - what features are highlighted in announcements?

**Document Structure to Create:**

```markdown
## Framework: [Name]

**Official Documentation:** [URL]
**GitHub Repository:** [URL]
**Primary Strength:** [What makes this framework unique]

**Key Documentation Sections:**
- Getting Started: [URL]
- Core Concepts: [URL]
- API Reference: [URL]
- Examples: [URL]
- Blog/Tutorials: [URL]
```

### Step 1.2: Feature Identification

Create a comprehensive feature matrix. Focus on **main features** - the capabilities that define the framework, not every possible option.

**Feature Matrix Template:**

| Feature | Description | Official Docs | Code Examples | Why It Matters |
|---------|-------------|---------------|---------------|----------------|
| [Feature 1] | [1-2 sentence description] | [Specific URL] | [GitHub URL or docs example] | [Framework strength this demonstrates] |

**Example from LlamaIndex:**

| Feature | Description | Official Docs | Code Examples | Why It Matters |
|---------|-------------|---------------|---------------|----------------|
| QueryEngineTool | Wraps query engines as agent tools for document search | https://docs.llamaindex.ai/... | https://github.com/run-llama/llama_index/examples/ | Core to agentic RAG pattern |
| Custom Workflows | Event-driven workflows with @step decorator | https://developers.llamaindex.ai/.../workflows/ | [examples URL] | Unique workflow orchestration |

**Categories to Consider:**
- Document/Data Loading
- Indexing and Retrieval
- Agent Creation and Tools
- Memory Management
- Streaming and Events
- Workflow Orchestration
- Structured Outputs
- Multi-Agent Systems

### Step 1.3: Feature Prioritization

Not all features are equal. Prioritize based on:

**Priority 1 (Must Have):**
- Framework's primary strength/differentiator
- Features used in getting started guides
- Core concepts that enable everything else
- Features that appear in official examples

**Priority 2 (Should Have):**
- Common use cases
- Features that build on Priority 1
- Agent capabilities (if agentic framework)
- Integration patterns

**Priority 3 (Nice to Have):**
- Advanced patterns
- Optimization features
- Edge case handling
- Enterprise features

**For Example Suite:** Focus on Priority 1 and Priority 2 features only.

---

## Phase 2: Example Architecture Design

### Step 2.1: Map Features to Examples

Create one example per main feature. Number them sequentially starting from 0.

**Naming Convention:** `NN_feature_name.py`
- Use two digits (00, 01, 02...) for proper sorting
- Use descriptive feature names
- Keep names concise but clear

**Example Architecture Template:**

```
00_hello_world.py          - Basic framework introduction (essential first example)
01_[core_feature_1].py     - First main feature
02_[core_feature_2].py     - Second main feature
03_[core_feature_3].py     - Third main feature
...
```

**LlamaIndex Example:**

```
00_hello_world.py          - Basic RAG with document loading
01_tools.py                - Function tools and agents
02_structured_outputs.py   - Pydantic models for type-safe outputs
03_memory.py               - Conversation memory
04_streaming.py            - Event-based streaming
05_memory_advanced.py      - Memory with initial context
06_agentic_rag.py          - Multiple query engines
07_async_patterns.py       - Async execution
08_agent_delegation.py     - Agents as tools
09_router_engine.py        - Query routing
10_workflow_custom.py      - Custom workflows
```

### Step 2.2: Define Example Progression

Examples should flow naturally:

1. **Foundation** (Examples 00-02): Core concepts needed for everything else
2. **Capabilities** (Examples 03-07): Main features that define the framework
3. **Advanced** (Examples 08-10): Patterns that combine features

**Each Example Must:**
- Be independently runnable
- Require only .env configuration
- Have clear, focused output
- Demonstrate one feature clearly
- Include official documentation links

---

## Phase 3: Code Structure Standards

### Universal File Structure

Every example file MUST follow this exact structure:

```python
# --- SECTION 1: IMPORTS ---
import asyncio  # or other execution imports
from framework_core import MainClass, HelperClass
from framework_specific import FeatureClass
from settings import settings  # Local settings file

# --- SECTION 2: MODULE DOCSTRING ---
"""
-------------------------------------------------------
In this example, we explore [Framework Name] with the following features:
- [Main feature being demonstrated]
- [Specific aspect of the feature]
- [Why this matters]

[Brief description of what makes this feature valuable in this framework]

For more details, visit:
[Official documentation URL]
-------------------------------------------------------
"""

# --- SECTION 3: HELPER FUNCTIONS (if needed) ---
def helper_function(param: type) -> return_type:
    """Brief description of what this does."""
    # Simple, focused implementation
    return result

# --- SECTION 4: MAIN FUNCTION ---
async def main():  # or def main() for sync
    # --- 1. Configure/Initialize ---
    # Clear step comment
    initialization_code()
    
    print("Configuration complete")
    print("-" * 50)
    
    # --- 2. Demonstrate Feature ---
    # Clear step comment explaining what happens
    feature_demonstration()
    
    print("Feature output:")
    print("-" * 50)
    
    # --- 3. Show Results ---
    print(f"Result: {result}")
    print("-" * 50)
    
    """
    Expected output:
    - What should happen
    - What this demonstrates
    - Key takeaway
    """

# --- SECTION 5: ENTRY POINT ---
if __name__ == "__main__":
    asyncio.run(main())  # or main() for sync
```

### Module Docstring Requirements

The module docstring is critical for understanding. It must include:

```python
"""
-------------------------------------------------------
In this example, we explore [Framework Name] with the following features:
- [Main feature name and brief description]
- [Specific capability demonstrated]
- [Framework-specific advantage]

[2-3 sentences about why this feature matters for this framework.
What makes this feature special? What problems does it solve?]

For more details, visit:
[Official documentation URL - MUST be specific to this feature]
-------------------------------------------------------
"""
```

**Real Example from LlamaIndex:**

```python
"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Multiple query engines (vector search vs summary)
- QueryEngineTool for wrapping engines as agent tools
- Agentic RAG pattern with tool selection
- similarity_top_k for controlling retrieval

This demonstrates LlamaIndex's strength: giving agents the ability to intelligently
select different retrieval strategies based on the question type.

For more details, visit:
https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6
-------------------------------------------------------
"""
```

### Step Comment Format

Use numbered step comments with descriptive text:

```python
# --- 1. Initialize the feature ---
# Brief explanation of what happens in this step

# --- 2. Demonstrate main capability ---
# Why we're doing this and what to expect

# --- 3. Show results ---
# What the output means
```

### Expected Output Section

Every example should end with an expected output comment:

```python
"""
Expected output:
- [What should be displayed]
- [What this demonstrates about the feature]
- [Key concept reinforced]
"""
```

---

## Phase 4: Writing Straightforward Examples

### The Straightforward Checklist

Before writing code, ask yourself:

- [ ] Can someone NEW to this framework understand this example?
- [ ] Is the feature obvious without reading comments?
- [ ] Am I using the SIMPLEST way to demonstrate this feature?
- [ ] Is every line of code necessary?
- [ ] Does the output clearly show the feature working?
- [ ] Would removing ANY code make it clearer?
- [ ] Is there only ONE feature being demonstrated?

### Simplicity Guidelines

**DO:**
- Use the most direct API calls
- Include only necessary configuration
- Show immediate, obvious results
- Use clear variable names
- Print intermediate steps
- Demonstrate the feature in action
- Use official/recommended patterns

**DON'T:**
- Add error handling unless feature-critical
- Show optimization techniques
- Combine multiple features
- Use advanced patterns
- Include production-ready code
- Add complexity "for completeness"
- Show edge cases

### Feature-Focused Code Example

**BAD (Too Complex):**
```python
# Don't do this - too many features mixed together
async def main():
    # Setup with many options
    llm = OpenAI(
        model=settings.MODEL,
        temperature=0.7,
        max_tokens=500,
        timeout=30,
        retry_policy=custom_retry,
    )
    
    # Multiple indices
    vector_index = VectorStoreIndex.from_documents(docs)
    summary_index = SummaryIndex.from_documents(docs)
    graph_index = KnowledgeGraphIndex.from_documents(docs)
    
    # Complex routing logic
    router = CustomRouter(
        indices=[vector_index, summary_index, graph_index],
        selector=ComplexSelector(),
        fallback_strategy="best_match"
    )
    
    # Advanced query with many parameters
    result = await router.query(
        query_text,
        top_k=5,
        rerank=True,
        filter_metadata=complex_filter,
        post_process=custom_processor
    )
```

**GOOD (Straightforward):**
```python
# Do this - focused on ONE feature: QueryEngineTool
async def main():
    # --- 1. Configure LLM ---
    llm = OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )
    Settings.llm = llm
    
    # --- 2. Create index and query engine ---
    documents = SimpleDirectoryReader(input_files=["res/doc.pdf"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    # --- 3. Wrap as tool (THIS IS THE FEATURE) ---
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="search_doc",
            description="Search the document for information"
        )
    )
    
    # --- 4. Use the tool with an agent ---
    agent = FunctionAgent(llm=llm, tools=[query_tool])
    response = await agent.achat("What is this document about?")
    
    print(f"Response: {response}")
```

### Balancing Simplicity with Framework Features

**The Golden Rule:** Keep it simple BUT showcase what makes the framework special.

**Example: LlamaIndex's Multiple Query Engines**

We could have made agentic RAG even simpler with just ONE query engine, but that would miss LlamaIndex's strength: intelligent tool selection between different retrieval strategies.

**The Balance:**
- Simple: Only 2 query engines (vector + summary), not 5
- Feature-Rich: Shows the intelligent selection capability
- Clear: Each engine has obvious use case described

```python
# Simple but showcases framework strength
vector_tool = QueryEngineTool(
    query_engine=vector_engine,
    metadata=ToolMetadata(
        name="vector_search",
        description=(
            "Useful for searching specific information and facts. "
            "Use this when you need precise details."
        )
    )
)

summary_tool = QueryEngineTool(
    query_engine=summary_engine,
    metadata=ToolMetadata(
        name="document_summary",
        description=(
            "Useful for getting high-level overviews. "
            "Use this when you need general understanding."
        )
    )
)
```

### Framework-Specific Parameters

Include framework-specific parameters that showcase unique capabilities:

**LlamaIndex Example:**
```python
# Show similarity_top_k - this is a LlamaIndex-specific control
vector_engine = vector_index.as_query_engine(
    similarity_top_k=3,      # LlamaIndex feature
    response_mode="compact"  # LlamaIndex response strategy
)

summary_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"  # LlamaIndex hierarchical summarization
)
```

**What This Shows:**
- `similarity_top_k` demonstrates retrieval control
- `response_mode` shows different strategies
- Both are LlamaIndex-specific capabilities

---

## Phase 5: Documentation and Links

### Official Documentation Requirements

Every example MUST link to official sources:

**In Module Docstring:**
```python
"""
...feature description...

For more details, visit:
https://docs.[framework].ai/en/stable/path/to/feature/
"""
```

**Link Quality Checklist:**
- [ ] URL is from official documentation or repository
- [ ] URL is specific to the feature (not homepage)
- [ ] URL contains relevant information about the feature
- [ ] URL is accessible and works
- [ ] URL uses https://
- [ ] URL is not shortened or redirected

### Where to Find Documentation Links

**Priority 1: Official Documentation**
- Feature-specific pages in docs
- API reference for the class/method
- Concept explanations

**Priority 2: Official Blog/Guides**
- Feature announcement posts
- Tutorial articles by framework authors
- Official blog posts

**Priority 3: Official Examples**
- GitHub examples directory
- Example notebooks
- Official starter templates

### Documentation Link Examples

**GOOD Examples:**
```python
# Specific feature page
https://docs.llamaindex.ai/en/stable/module_guides/querying/router/

# Specific blog post about feature
https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6

# Specific API reference
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
```

**BAD Examples:**
```python
# Too general - just homepage
https://docs.llamaindex.ai/

# External tutorial (not official)
https://medium.com/some-tutorial

# Shortened URL
https://bit.ly/some-link
```

---

## Phase 6: README.md Creation

### README Structure

```markdown
# [Framework Name]

- Repo: [GitHub URL]
- Documentation: [Docs URL]

## What is [Framework Name]?

[2-3 paragraphs describing the framework, its primary strength, and key features]

This folder contains **simple and straightforward examples** that demonstrate [Framework]'s core features. Each example is focused, minimal, and easy to understand.

Key strengths include:
- [Strength 1]: [Brief explanation]
- [Strength 2]: [Brief explanation]
- [Strength 3]: [Brief explanation]

## [Framework] Examples

### How to setup

#### Virtual environment

Create a simple virtual environment with:

\`\`\`bash
python3 -m venv .venv
\`\`\`

Then activate it with:
\`\`\`bash
# On Linux/macOS
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
\`\`\`

And install the requirements with:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

#### .env

See .env.example and create a .env (on the root of the repository).
You need to get an OpenAI endpoint and key and fill them in.

### Example Progression

**Core Fundamentals** (Required):
- `00_example_name.py` - [One sentence description]
- `01_example_name.py` - [One sentence description]
- `02_example_name.py` - [One sentence description]

**Main Capabilities** (Recommended):
- `03_example_name.py` - [One sentence description]
- `04_example_name.py` - [One sentence description]
- `05_example_name.py` - [One sentence description]

**Advanced Patterns** (Optional):
- `06_example_name.py` - [One sentence description]
- `07_example_name.py` - [One sentence description]

### Key [Framework] Differentiators

| Aspect | [Framework] Specialty |
|--------|----------------------|
| **[Category 1]** | [What makes it special] |
| **[Category 2]** | [What makes it special] |
| **[Category 3]** | [What makes it special] |

### Documentation References

- Main Docs: [URL]
- [Feature Category]: [URL]
- [Feature Category]: [URL]
```

### Example Descriptions

Each example should have ONE sentence that describes:
1. What feature it demonstrates
2. What specific capability it shows

**Examples from LlamaIndex:**
- `00_hello_world.py` - Basic RAG with document loading and querying
- `06_agentic_rag.py` - Multiple query engines (vector vs summary) with intelligent tool selection
- `10_workflow_custom.py` - Custom Workflow with @step decorator, Events, and Context state

---

## Phase 7: settings.py Configuration

### Standard Settings File

Create a consistent settings.py file:

```python
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # OpenAI Configuration (most common)
    OPENAI_API_KEY: SecretStr
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_EMBEDDINGS_MODEL: str = "text-embedding-3-small"
    
    # Framework-specific settings
    # Add as needed for the specific framework


settings = Settings()
```

### .env.example File

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small

# Framework-specific variables
# Add as needed
```

---

## Phase 8: Quality Validation

### Pre-Delivery Checklist

For EVERY example file:

**Code Quality:**
- [ ] Feature is immediately obvious
- [ ] Code is minimal and focused
- [ ] No unnecessary complexity
- [ ] Variable names are clear
- [ ] Follows standard structure
- [ ] Has proper module docstring
- [ ] Uses step comments

**Documentation:**
- [ ] Official documentation link present
- [ ] Link is specific to feature
- [ ] Link is accessible and works
- [ ] Feature is well described
- [ ] Expected output documented

**Functionality:**
- [ ] Imports resolve correctly
- [ ] Runs with .env configuration
- [ ] Produces clear output
- [ ] Feature works as documented
- [ ] Independent and runnable

**Straightforwardness:**
- [ ] Beginner can understand it
- [ ] Purpose is crystal clear
- [ ] Only ONE feature demonstrated
- [ ] Output shows feature clearly
- [ ] Easy to modify and experiment

### Common Issues to Fix

**Issue: Feature Not Obvious**
- **Fix:** Rename variables, add comments, simplify code
- **Test:** Can someone unfamiliar identify the feature in 10 seconds?

**Issue: Too Complex**
- **Fix:** Remove optional parameters, use defaults, cut extra features
- **Test:** Count lines - can you remove 20% without losing clarity?

**Issue: Multiple Features**
- **Fix:** Split into separate examples
- **Test:** Can you describe it in one sentence with "and" appearing?

**Issue: Poor Documentation**
- **Fix:** Find specific feature documentation page
- **Test:** Does clicking the link show information about THIS feature?

---

## Phase 9: Implementation Workflow

### Complete Process Timeline

**Day 1: Analysis**
1. Read framework documentation (2-3 hours)
2. Create feature matrix (1-2 hours)
3. Design example architecture (1 hour)
4. Validate with framework strengths (30 mins)

**Day 2: Core Examples**
1. Create settings.py and .env.example (30 mins)
2. Write 00_hello_world.py (1 hour)
3. Write examples 01-03 (3-4 hours)
4. Test all examples (1 hour)

**Day 3: Main Examples**
1. Write examples 04-07 (4-5 hours)
2. Test all examples (1 hour)
3. Refine based on testing (1-2 hours)

**Day 4: Advanced & Documentation**
1. Write examples 08-10 (3-4 hours)
2. Create README.md (1-2 hours)
3. Final testing (1 hour)
4. Quality validation (1 hour)

**Day 5: Polish**
1. Review all docstrings (1 hour)
2. Verify all links (30 mins)
3. Test example progression (1 hour)
4. Final review and delivery (1 hour)

### Iteration Process

After creating initial examples:

1. **Test Run**: Execute every example
2. **Simplicity Check**: Can they be simpler?
3. **Feature Check**: Is the feature obvious?
4. **Documentation Check**: Are links helpful?
5. **Progression Check**: Do they flow naturally?
6. **Refine**: Improve based on findings
7. **Repeat**: Until quality checklist passes

---

## Phase 10: Framework-Specific Adaptations

### Adapting to Different Framework Types

**Document-Centric Frameworks (e.g., LlamaIndex):**
- Focus on indexing, retrieval, query engines
- Emphasize RAG patterns
- Show different index types
- Demonstrate query routing

**Agent-Centric Frameworks (e.g., AutoGen, CrewAI):**
- Focus on agent creation and configuration
- Emphasize tool calling
- Show multi-agent patterns
- Demonstrate conversation flows

**Workflow-Centric Frameworks:**
- Focus on step definition
- Emphasize state management
- Show event handling
- Demonstrate orchestration

**Function-Calling Frameworks (e.g., Pydantic AI):**
- Focus on tool definition
- Emphasize type safety
- Show validation patterns
- Demonstrate structured outputs

### Common Patterns to Include

**Core Patterns (Almost Every Framework):**
1. Hello World / Basic usage
2. Tool/Function definition and calling
3. Structured outputs (Pydantic models)
4. Memory/Context management
5. Streaming (if supported)

**Agent Patterns:**
1. Single agent with tools
2. Agent delegation
3. Multi-agent systems
4. Human-in-the-loop

**Advanced Patterns:**
1. Custom workflows/orchestration
2. Callbacks and events
3. Error handling and retries
4. Performance optimization

---

## Real-World Example: LlamaIndex Journey

### What We Did

**Initial State:**
- 2 files: `1_router_engine.py`, `2_tool_calling.py`
- Inconsistent numbering
- Missing core features

**Process:**
1. Analyzed LlamaIndex documentation
2. Identified core features (RAG, workflows, memory, etc.)
3. Created 11 examples (00-10)
4. Initially made them too simple (lost framework features)
5. Enhanced to showcase LlamaIndex specifics while keeping clear

**Final State:**
- 11 examples, numbered 00-10
- Each focuses on ONE feature
- Balance of simplicity and framework strengths
- Complete documentation

### Key Decisions

**Decision 1: Multiple Query Engines in Agentic RAG**
- **Why:** Shows LlamaIndex's strength in intelligent retrieval
- **Trade-off:** Slightly more complex than one engine
- **Result:** Clear demonstration of framework differentiator

**Decision 2: Context State in Workflows**
- **Why:** Context is unique to LlamaIndex workflows
- **Trade-off:** Requires explaining Context API
- **Result:** Shows powerful state management capability

**Decision 3: Initial Messages in Memory**
- **Why:** Demonstrates persistent context feature
- **Trade-off:** More setup than basic memory
- **Result:** Shows practical user profile pattern

### Lessons Learned

1. **First Pass Too Simple**: Lost framework identity
2. **Balance is Key**: Simple code + framework features
3. **Real Use Cases Matter**: Show practical patterns
4. **Documentation is Critical**: Links guide deeper learning
5. **Iteration Improves Quality**: Multiple refinements needed

---

## Appendix A: Feature Discovery Techniques

### Documentation Mining

**Look for these sections:**
- "Core Concepts"
- "Key Features"
- "Getting Started"
- "Why [Framework]?"
- "Examples" or "Tutorials"
- "API Reference"

**Identify patterns:**
- What classes appear most?
- What methods are highlighted?
- What examples are provided?
- What blog posts exist?

### GitHub Repository Analysis

**Examine:**
- `/examples` directory structure
- README.md feature lists
- Issue labels and common questions
- Pull request titles (new features)
- Documentation source files

### Community Resources

**Check:**
- Official Discord/Slack announcements
- Twitter/X from framework authors
- Conference talks and demos
- Tutorial videos (official)
- Starter templates

---

## Appendix B: Common Mistakes to Avoid

### Mistake 1: Over-Simplification

**Problem:** Removing so much that framework's unique value is lost

**Example:**
```python
# Too simple - doesn't show LlamaIndex's strength
query_engine = index.as_query_engine()
response = query_engine.query("question")
```

**Fix:**
```python
# Shows framework features while staying simple
vector_engine = index.as_query_engine(
    similarity_top_k=3,      # Retrieval control
    response_mode="compact"  # Response strategy
)
```

### Mistake 2: Feature Mixing

**Problem:** Combining multiple features in one example

**Example:**
```python
# Bad: mixing memory, streaming, and tools
agent = Agent(
    tools=[tool1, tool2, tool3],
    memory=complex_memory,
    streaming_handler=custom_handler,
    callbacks=[callback1, callback2]
)
```

**Fix:** Split into separate examples, each focusing on one feature

### Mistake 3: Poor Documentation Links

**Problem:** Generic or broken links

**Bad:**
```python
# For more details, visit: https://docs.framework.ai
```

**Good:**
```python
# For more details, visit: 
# https://docs.framework.ai/en/stable/module_guides/feature_name/
```

### Mistake 4: Missing Context

**Problem:** Not explaining WHY a feature matters

**Bad:**
```python
"""
This example shows QueryEngineTool.
"""
```

**Good:**
```python
"""
This example shows QueryEngineTool for wrapping query engines as agent tools.
This demonstrates LlamaIndex's strength: giving agents the ability to 
intelligently select different retrieval strategies based on question type.
"""
```

---

## Appendix C: Templates and Checklists

### Quick Start Template

```python
import asyncio
from [framework] import MainClass
from settings import settings

"""
-------------------------------------------------------
In this example, we explore [Framework] with the following features:
- [Main feature]
- [Specific capability]

[Why this matters - 2-3 sentences]

For more details, visit:
[Official docs URL]
-------------------------------------------------------
"""

async def main():
    # --- 1. Initialize ---
    instance = MainClass(
        api_key=settings.API_KEY.get_secret_value()
    )
    
    print("Initialized [Framework]")
    print("-" * 50)
    
    # --- 2. Demonstrate Feature ---
    result = await instance.feature_method()
    
    print(f"Result: {result}")
    print("-" * 50)
    
    """
    Expected output:
    - [What should happen]
    - [What this shows]
    """

if __name__ == "__main__":
    asyncio.run(main())
```

### Example Review Checklist

Print this and check off for each example:

```
Example: _______________

STRUCTURE:
☐ Has proper imports
☐ Has module docstring with feature description
☐ Has official documentation link
☐ Uses step comments (# --- N. Description ---)
☐ Has expected output section
☐ Has entry point

CLARITY:
☐ Feature is immediately obvious
☐ Code is minimal
☐ Purpose is clear
☐ Variable names are descriptive
☐ Output is meaningful

QUALITY:
☐ Runs without errors
☐ Demonstrates ONE feature
☐ Shows framework strength
☐ Beginner can understand
☐ Links work and are relevant

FRAMEWORK SPECIFIC:
☐ Uses framework's recommended patterns
☐ Shows framework-specific capabilities
☐ Highlights what makes framework unique
☐ Includes framework-specific parameters
```

---

## Conclusion

This guide provides everything needed to create high-quality, straightforward example suites for any agentic framework. The key is balancing simplicity with feature showcase - make it easy to understand while highlighting what makes the framework special.

**Remember:**
- One feature per example
- Straightforward over clever
- Feature-focused demonstration
- Official documentation links
- Clear, beginner-friendly code

**Success Metrics:**
- Someone new to the framework can understand each example
- Each example clearly demonstrates a valuable feature
- Examples progress naturally from simple to advanced
- All documentation links work and are helpful
- The suite showcases the framework's unique strengths

Use this guide as a reference throughout the process, validate against the checklists, and iterate until quality standards are met.
