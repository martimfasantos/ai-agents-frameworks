"""
Multi-Agent Benchmark Runner

Benchmarks multi-agent systems across frameworks that support them.
Measures: response time, number of agent calls/handoffs, token usage,
and quality of coordination.

Usage:
    python benchmark_multi_agent.py --provider openai --iterations 5
"""

import time
import json
import os
import argparse
import numpy as np
from datetime import datetime


# ---- Multi-Agent Test Questions ----

MULTI_AGENT_QUESTIONS = [
    {
        "id": "multi_research_analyze",
        "prompt": (
            "Research the latest developments in quantum computing and then "
            "analyze how they might impact the cybersecurity industry. "
            "Provide a structured summary with key findings and implications."
        ),
        "category": "research_and_analysis",
        "requires_coordination": True,
    },
    {
        "id": "multi_plan_execute",
        "prompt": (
            "Search the web for the top 3 most popular programming languages in 2025, "
            "then for each one, find the most commonly used web framework. "
            "Present the results in a structured comparison."
        ),
        "category": "multi_step_research",
        "requires_coordination": True,
    },
    {
        "id": "multi_simple_delegation",
        "prompt": (
            "What is today's date and what day of the week is it? "
            "Also search the web for any major events happening today."
        ),
        "category": "simple_multi_tool",
        "requires_coordination": False,
    },
]


def benchmark_crewai_multi_agent(provider: str, question: dict, iterations: int):
    """Benchmark CrewAI multi-agent system."""
    try:
        from crewai import Agent, Task, Crew, Process
        from crewai.tools import tool as crewai_tool
        from tavily import TavilyClient
        from settings import settings
        from datetime import date
        import json as json_mod

        @crewai_tool("web_search_tool")
        def web_search(query: str) -> str:
            """Searches the web for information."""
            client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
            return json_mod.dumps(client.search(query).get("results", []))

        @crewai_tool("date_tool")
        def get_date() -> str:
            """Gets the current date."""
            return date.today().strftime("%B %d, %Y")

        llm = (
            f"openai/{settings.openai_model_name}"
            if provider == "openai"
            else f"azure/{settings.azure_deployment_name}"
        )

        researcher = Agent(
            role="Researcher",
            goal="Research and gather information from the web",
            backstory="Expert at finding relevant information online",
            tools=[web_search, get_date],
            llm=llm,
            verbose=False,
        )
        analyst = Agent(
            role="Analyst",
            goal="Analyze information and provide structured insights",
            backstory="Expert at analyzing data and creating summaries",
            llm=llm,
            verbose=False,
        )

        results = _run_multi_agent_iterations(
            lambda q: _run_crewai_crew(researcher, analyst, q, llm),
            question,
            iterations,
        )
        return results

    except Exception as e:
        return {"error": str(e)}


def _run_crewai_crew(researcher, analyst, question, llm):
    from crewai import Task, Crew, Process

    research_task = Task(
        description=f"Research: {question['prompt']}",
        expected_output="Detailed research findings",
        agent=researcher,
    )
    analysis_task = Task(
        description="Analyze the research findings and create a structured summary",
        expected_output="Structured analysis with key findings",
        agent=analyst,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=False,
    )

    start = time.perf_counter()
    result = crew.kickoff()
    end = time.perf_counter()

    usage = result.token_usage if hasattr(result, "token_usage") else None
    return {
        "response": str(result)[:500],
        "exec_time": end - start,
        "agent_calls": 2,
        "tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }


def benchmark_langgraph_multi_agent(provider: str, question: dict, iterations: int):
    """Benchmark LangGraph multi-agent system."""
    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
        from langchain_core.tools import Tool
        from tavily import TavilyClient
        from settings import settings
        from datetime import date
        import json as json_mod

        if provider == "azure":
            model = AzureChatOpenAI(
                base_url=f"{settings.azure_endpoint}/deployments/{settings.azure_deployment_name}",
                api_version=settings.azure_api_version,
                api_key=settings.azure_api_key.get_secret_value(),
            )
        else:
            model = ChatOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.openai_model_name,
            )

        tavily = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
        tools = [
            Tool(
                name="web_search",
                func=lambda q: json_mod.dumps(tavily.search(q).get("results", [])),
                description="Search the web",
            ),
            Tool(
                name="date_tool",
                func=lambda _={}: date.today().strftime("%B %d, %Y"),
                description="Get today's date",
            ),
        ]

        graph = create_react_agent(model=model, tools=tools)

        def run_fn(q):
            start = time.perf_counter()
            events = list(
                graph.stream(
                    {"messages": [("user", q["prompt"])]},
                    stream_mode="values",
                )
            )
            end = time.perf_counter()

            response = ""
            total_tokens = 0
            agent_calls = 0
            if events:
                last = events[-1]
                if "messages" in last:
                    response = last["messages"][-1].content
                    for msg in last["messages"]:
                        if hasattr(msg, "response_metadata") and msg.response_metadata:
                            usage = msg.response_metadata.get("token_usage", {})
                            total_tokens += usage.get("total_tokens", 0)
                            agent_calls += 1

            return {
                "response": response[:500],
                "exec_time": end - start,
                "agent_calls": agent_calls,
                "tokens": total_tokens,
            }

        return _run_multi_agent_iterations(run_fn, question, iterations)

    except Exception as e:
        return {"error": str(e)}


def benchmark_openai_agents_sdk_multi_agent(
    provider: str, question: dict, iterations: int
):
    """Benchmark OpenAI Agents SDK multi-agent system with handoffs."""
    try:
        from agents import Agent, Runner, function_tool
        from tavily import TavilyClient
        from settings import settings
        from datetime import date
        import json as json_mod

        @function_tool
        def web_search(query: str) -> str:
            """Search the web."""
            client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
            return json_mod.dumps(client.search(query).get("results", []))

        @function_tool
        def get_date() -> str:
            """Get today's date."""
            return date.today().strftime("%B %d, %Y")

        researcher = Agent(
            name="researcher",
            instructions="You research information from the web. Be thorough and factual.",
            model=settings.openai_model_name,
            tools=[web_search, get_date],
        )
        analyst = Agent(
            name="analyst",
            instructions="You analyze information and create structured summaries.",
            model=settings.openai_model_name,
            handoffs=[researcher],
        )

        def run_fn(q):
            start = time.perf_counter()
            result = Runner.run_sync(analyst, q["prompt"])
            end = time.perf_counter()

            prompt_tokens = 0
            completion_tokens = 0
            agent_calls = 0
            for resp in getattr(result, "raw_responses", []):
                if hasattr(resp, "usage") and resp.usage:
                    prompt_tokens += getattr(resp.usage, "prompt_tokens", 0)
                    completion_tokens += getattr(resp.usage, "completion_tokens", 0)
                    agent_calls += 1

            return {
                "response": result.final_output[:500] if result.final_output else "",
                "exec_time": end - start,
                "agent_calls": agent_calls,
                "tokens": prompt_tokens + completion_tokens,
            }

        return _run_multi_agent_iterations(run_fn, question, iterations)

    except Exception as e:
        return {"error": str(e)}


def _run_multi_agent_iterations(run_fn, question, iterations):
    """Generic helper to run multi-agent benchmarks for N iterations."""
    exec_times = []
    token_counts = []
    agent_call_counts = []
    sample_response = ""
    errors = 0

    for i in range(iterations):
        try:
            result = run_fn(question)
            exec_times.append(result["exec_time"])
            token_counts.append(result["tokens"])
            agent_call_counts.append(result["agent_calls"])
            if i == 0:
                sample_response = result["response"]
        except Exception as e:
            errors += 1
            if i == 0:
                sample_response = f"ERROR: {e}"

    return {
        "question_id": question["id"],
        "prompt": question["prompt"],
        "iterations": iterations,
        "errors": errors,
        "response_time_mean": float(np.mean(exec_times)) if exec_times else 0.0,
        "response_time_std": float(np.std(exec_times)) if exec_times else 0.0,
        "avg_tokens": float(np.mean(token_counts)) if token_counts else 0.0,
        "avg_agent_calls": float(np.mean(agent_call_counts))
        if agent_call_counts
        else 0.0,
        "sample_response": sample_response,
    }


MULTI_AGENT_BENCHMARKS = {
    "CrewAI": benchmark_crewai_multi_agent,
    "LangGraph": benchmark_langgraph_multi_agent,
    "OpenAI Agents SDK": benchmark_openai_agents_sdk_multi_agent,
}


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Benchmark Runner")
    parser.add_argument(
        "--provider", type=str, choices=["azure", "openai"], default="openai"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Iterations per question"
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory"
    )
    parser.add_argument("--frameworks", nargs="*", help="Specific frameworks to test")
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"  Multi-Agent Benchmark")
    print(f"  Provider: {args.provider}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    all_results = {}

    for framework_name, bench_fn in MULTI_AGENT_BENCHMARKS.items():
        if args.frameworks and framework_name not in args.frameworks:
            continue

        print(f"\n{'=' * 60}")
        print(f"  Multi-Agent: {framework_name}")
        print(f"{'=' * 60}")

        framework_results = []
        for question in MULTI_AGENT_QUESTIONS:
            print(f"  [{question['id']}] Running {args.iterations}x...")
            result = bench_fn(args.provider, question, args.iterations)

            if "error" in result:
                print(f"    -> ERROR: {result['error'][:80]}")
                framework_results.append(result)
            else:
                print(
                    f"    -> Time: {result['response_time_mean']:.2f}s "
                    f"| Agents: {result['avg_agent_calls']:.1f} "
                    f"| Tokens: {result['avg_tokens']:.0f}"
                )
                framework_results.append(result)

        all_results[framework_name] = framework_results

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(args.output, f"multi_agent_benchmark_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")

    # Print summary table
    try:
        from tabulate import tabulate

        print(f"\n{'=' * 80}")
        print("  MULTI-AGENT SUMMARY")
        print(f"{'=' * 80}\n")

        for qid in [q["id"] for q in MULTI_AGENT_QUESTIONS]:
            headers = [
                "Framework",
                "Time (mean +/- std)",
                "Avg Agent Calls",
                "Avg Tokens",
                "Errors",
            ]
            rows = []
            for fw, fw_results in all_results.items():
                for r in fw_results:
                    if isinstance(r, dict) and r.get("question_id") == qid:
                        rows.append(
                            [
                                fw,
                                f"{r['response_time_mean']:.2f} +/- {r['response_time_std']:.2f}s",
                                f"{r['avg_agent_calls']:.1f}",
                                f"{r['avg_tokens']:.0f}",
                                r["errors"],
                            ]
                        )
            if rows:
                q_prompt = next(
                    q["prompt"] for q in MULTI_AGENT_QUESTIONS if q["id"] == qid
                )
                print(f"  Q: {q_prompt[:80]}...")
                print(tabulate(rows, headers=headers, tablefmt="grid"))
                print()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
