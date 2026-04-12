"""
Benchmark Runner for AI Agent Framework Comparison

This script runs standardized benchmarks across all agent implementations,
collecting metrics on response time, token usage, and response quality.

Usage:
    python benchmark_runner.py --provider openai --iterations 10 --output results/
    python benchmark_runner.py --provider azure --iterations 50 --agents agno langgraph
    python benchmark_runner.py --provider openai --iterations 30 --benchmark all
"""

import importlib
import time
import json
import os
import argparse
import numpy as np
from datetime import datetime
from typing import Optional


# ---- Agent Registry ----
# Maps agent module names to their class names and categories

BASIC_AGENTS = {
    "agno_agent": ("Agent", "Agno"),
    "langgraph_agent": ("Agent", "LangGraph"),
    "llama_index_agent": ("Agent", "LlamaIndex"),
    "openai_agent": ("Agent", "OpenAI (Raw)"),
    "pydantic_ai_agent": ("Agent", "PydanticAI"),
    "crewai_agent": ("Agent", "CrewAI"),
    "google_adk_agent": ("Agent", "Google ADK"),
    "smolagents_agent": ("Agent", "Smolagents"),
    "strands_agent": ("Agent", "Strands"),
    "claude_sdk_agent": ("Agent", "Claude SDK"),
    "ag2_agent": ("Agent", "AG2"),
    "microsoft_agent": ("Agent", "Microsoft AF"),
    "langchain_agent": ("Agent", "LangChain"),
    "openai_agents_sdk_agent": ("Agent", "OpenAI Agents SDK"),
}

RAG_API_AGENTS = {
    "agno_rag_api_agent": ("AgnoRAGandAPIAgent", "Agno"),
    "langgraph_rag_api_agent": ("LangGraphRAGandAPIAgent", "LangGraph"),
    "llama_index_rag_api_agent": ("LlamaIndexRAGandAPIAgent", "LlamaIndex"),
    "crewai_rag_api_agent": ("CrewAIRAGandAPIAgent", "CrewAI"),
    "langchain_rag_api_agent": ("LangChainRAGandAPIAgent", "LangChain"),
}


# ---- Benchmark Questions ----

BASIC_QUESTIONS = [
    {
        "id": "web_search_explicit",
        "prompt": "search the web for who won the Champions League final in 2024?",
        "category": "web_search",
        "expects_tool": True,
    },
    {
        "id": "web_search_implicit",
        "prompt": "who won the Champions League final in 2024?",
        "category": "web_search",
        "expects_tool": True,
    },
    {
        "id": "date_query",
        "prompt": "what is today's date?",
        "category": "date",
        "expects_tool": True,
    },
    {
        "id": "greeting",
        "prompt": "hello, how are you?",
        "category": "no_tool",
        "expects_tool": False,
    },
    {
        "id": "factual_knowledge",
        "prompt": "what is the capital of Portugal?",
        "category": "no_tool",
        "expects_tool": False,
    },
]

RAG_QUESTIONS = [
    {
        "id": "rag_possession",
        "prompt": "Ball possession in Benfica's game?",
        "category": "rag",
        "expected_answer_contains": ["52%", "48%"],
    },
    {
        "id": "rag_score",
        "prompt": "Benfica's UCL match score?",
        "category": "rag",
        "expected_answer_contains": ["0-1", "0 - 1", "Benfica 0", "Barcelona"],
    },
    {
        "id": "rag_arsenal",
        "prompt": "What was the score of Arsenal's match in the Champions League round of 16?",
        "category": "rag",
        "expected_answer_contains": ["1-7", "1 - 7", "7"],
    },
]

API_QUESTIONS = [
    {
        "id": "api_combined",
        "prompt": "Tell me the waiting time at the CG station and the status of the red line, and also give me information about Formula 1 driver number 44!",
        "category": "api",
        "expects_multiple_tools": True,
    },
    {
        "id": "api_f1",
        "prompt": "Give me information about Formula 1 driver number 1",
        "category": "api",
        "expects_tool": True,
    },
]

TARGET_QUESTIONS = [
    {
        "id": "target_factual",
        "prompt": "Who is the current President of the United States?",
        "category": "factual",
    },
    {
        "id": "target_reasoning",
        "prompt": "If a train travels at 60 km/h for 2.5 hours, how far does it go?",
        "category": "reasoning",
    },
    {
        "id": "target_web_search",
        "prompt": "Search the web for the latest news about artificial intelligence regulation in Europe.",
        "category": "web_search",
    },
]


def load_agent(
    module_name: str,
    class_name: str,
    provider: str,
    memory: bool = False,
    tokens: bool = True,
):
    """Dynamically load and instantiate an agent."""
    try:
        module = importlib.import_module(module_name)
        AgentClass = getattr(module, class_name)
        agent = AgentClass(
            provider=provider,
            memory=memory,
            verbose=False,
            tokens=tokens,
        )
        return agent
    except Exception as e:
        print(f"  [ERROR] Failed to load {module_name}: {e}")
        return None


def run_single_benchmark(agent, question: dict, iterations: int = 1) -> dict:
    """Run a single benchmark question against an agent for N iterations."""
    response_times = []
    token_counts = {
        "prompt_llm_token_count": [],
        "completion_llm_token_count": [],
        "total_llm_token_count": [],
    }
    responses = []
    errors = 0
    misses = 0

    for i in range(iterations):
        try:
            result = agent.chat(question["prompt"])

            if isinstance(result, tuple) and len(result) == 3:
                response, exec_time, tokens = result
            elif isinstance(result, tuple) and len(result) == 2:
                response, exec_time = result
                tokens = {}
            else:
                response = str(result)
                exec_time = 0.0
                tokens = {}

            response_times.append(exec_time)

            if tokens:
                for key in token_counts:
                    token_counts[key].append(tokens.get(key, 0))

            # Check for misses (RAG questions)
            if "expected_answer_contains" in question:
                hit = any(
                    expected.lower() in str(response).lower()
                    for expected in question["expected_answer_contains"]
                )
                if not hit:
                    misses += 1

            if i == 0:  # Store first response as sample
                responses.append(str(response)[:500])

        except Exception as e:
            errors += 1
            if i == 0:
                responses.append(f"ERROR: {str(e)[:200]}")

    result = {
        "question_id": question["id"],
        "prompt": question["prompt"],
        "category": question["category"],
        "iterations": iterations,
        "errors": errors,
        "response_time_mean": float(np.mean(response_times)) if response_times else 0.0,
        "response_time_std": float(np.std(response_times)) if response_times else 0.0,
        "response_time_min": float(np.min(response_times)) if response_times else 0.0,
        "response_time_max": float(np.max(response_times)) if response_times else 0.0,
        "sample_response": responses[0] if responses else "",
    }

    if any(token_counts["total_llm_token_count"]):
        result["avg_prompt_tokens"] = float(
            np.mean(token_counts["prompt_llm_token_count"])
        )
        result["avg_completion_tokens"] = float(
            np.mean(token_counts["completion_llm_token_count"])
        )
        result["avg_total_tokens"] = float(
            np.mean(token_counts["total_llm_token_count"])
        )

    if "expected_answer_contains" in question:
        result["misses"] = misses
        result["miss_rate"] = f"{misses}/{iterations}"

    return result


def run_benchmark_suite(
    agents_registry: dict,
    questions: list,
    provider: str,
    iterations: int,
    memory: bool = False,
    create_new: bool = True,
    filter_agents: Optional[list] = None,
) -> dict:
    """Run a full benchmark suite across all agents and questions."""
    results = {}

    for module_name, (class_name, display_name) in agents_registry.items():
        if filter_agents and module_name not in filter_agents:
            continue

        print(f"\n{'=' * 60}")
        print(f"  Benchmarking: {display_name} ({module_name})")
        print(f"{'=' * 60}")

        agent = load_agent(module_name, class_name, provider, memory=memory)
        if agent is None:
            results[display_name] = {"error": "Failed to load agent"}
            continue

        agent_results = []
        for question in questions:
            print(
                f"  [{question['id']}] Running {iterations}x: {question['prompt'][:60]}..."
            )

            if create_new:
                agent = load_agent(module_name, class_name, provider, memory=memory)
                if agent is None:
                    continue

            bench_result = run_single_benchmark(agent, question, iterations)
            agent_results.append(bench_result)

            time_str = f"{bench_result['response_time_mean']:.2f} +/- {bench_result['response_time_std']:.2f}s"
            tokens_str = f"tokens: {bench_result.get('avg_total_tokens', 'N/A')}"
            miss_str = f"misses: {bench_result.get('miss_rate', 'N/A')}"
            print(f"    -> Time: {time_str} | {tokens_str} | {miss_str}")

        results[display_name] = agent_results

    return results


def save_results(results: dict, output_dir: str, benchmark_name: str):
    """Save benchmark results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{benchmark_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")
    return filepath


def print_comparison_table(results: dict, title: str):
    """Print a formatted comparison table."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("Install tabulate for prettier tables: pip install tabulate")
        return

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")

    # Group by question
    all_questions = set()
    for agent_results in results.values():
        if isinstance(agent_results, list):
            for r in agent_results:
                all_questions.add(r["question_id"])

    for qid in sorted(all_questions):
        headers = [
            "Framework",
            "Time (mean +/- std)",
            "Tokens (avg)",
            "Misses",
            "Errors",
        ]
        rows = []
        for agent_name, agent_results in results.items():
            if isinstance(agent_results, dict) and "error" in agent_results:
                rows.append(
                    [agent_name, "FAILED", "-", "-", agent_results["error"][:30]]
                )
                continue
            if isinstance(agent_results, list):
                for r in agent_results:
                    if r["question_id"] == qid:
                        time_str = f"{r['response_time_mean']:.2f} +/- {r['response_time_std']:.2f}s"
                        tokens = (
                            f"{r.get('avg_total_tokens', '-'):.1f}"
                            if "avg_total_tokens" in r
                            else "-"
                        )
                        misses = r.get("miss_rate", "-")
                        rows.append([agent_name, time_str, tokens, misses, r["errors"]])

        if rows:
            prompt = next(
                (
                    r["prompt"]
                    for ar in results.values()
                    if isinstance(ar, list)
                    for r in ar
                    if r["question_id"] == qid
                ),
                qid,
            )
            print(f"  Question: {prompt}")
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            print()


def main():
    parser = argparse.ArgumentParser(description="AI Agent Framework Benchmark Runner")
    parser.add_argument(
        "--provider", type=str, choices=["azure", "openai"], default="openai"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations per question"
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--agents", nargs="*", help="Specific agent modules to test (default: all)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["basic", "rag", "api", "target", "all"],
        default="basic",
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--memory", action="store_true", help="Enable memory for agents"
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Reuse agent instance across iterations",
    )
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"  AI Agent Framework Benchmark")
    print(f"  Provider: {args.provider}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Memory: {'ON' if args.memory else 'OFF'}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"{'#' * 60}")

    all_results = {}

    # --- Basic Agent Benchmarks ---
    if args.benchmark in ["basic", "all"]:
        print("\n\n>>> BASIC AGENT BENCHMARKS <<<")
        basic_results = run_benchmark_suite(
            BASIC_AGENTS,
            BASIC_QUESTIONS,
            args.provider,
            args.iterations,
            memory=args.memory,
            create_new=not args.no_create,
            filter_agents=args.agents,
        )
        all_results["basic"] = basic_results
        print_comparison_table(basic_results, "Basic Agent Comparison")
        save_results(basic_results, args.output, "basic_benchmark")

    # --- Target Questions (Response Quality) ---
    if args.benchmark in ["target", "all"]:
        print("\n\n>>> TARGET QUESTIONS BENCHMARK <<<")
        target_results = run_benchmark_suite(
            BASIC_AGENTS,
            TARGET_QUESTIONS,
            args.provider,
            args.iterations,
            memory=args.memory,
            create_new=not args.no_create,
            filter_agents=args.agents,
        )
        all_results["target"] = target_results
        print_comparison_table(target_results, "Target Questions Comparison")
        save_results(target_results, args.output, "target_benchmark")

    # --- RAG + API Benchmarks ---
    if args.benchmark in ["rag", "all"]:
        print("\n\n>>> RAG & API BENCHMARKS <<<")

        # RAG questions
        rag_results = run_benchmark_suite(
            RAG_API_AGENTS,
            RAG_QUESTIONS,
            args.provider,
            args.iterations,
            memory=args.memory,
            create_new=not args.no_create,
            filter_agents=args.agents,
        )
        all_results["rag"] = rag_results
        print_comparison_table(rag_results, "RAG Comparison")
        save_results(rag_results, args.output, "rag_benchmark")

    # --- API Benchmarks ---
    if args.benchmark in ["api", "all"]:
        print("\n\n>>> API BENCHMARKS <<<")
        api_results = run_benchmark_suite(
            RAG_API_AGENTS,
            API_QUESTIONS,
            args.provider,
            args.iterations,
            memory=args.memory,
            create_new=not args.no_create,
            filter_agents=args.agents,
        )
        all_results["api"] = api_results
        print_comparison_table(api_results, "API Comparison")
        save_results(api_results, args.output, "api_benchmark")

    # Save combined results
    if args.benchmark == "all":
        save_results(all_results, args.output, "full_benchmark")

    print("\n\nBenchmark complete!")


if __name__ == "__main__":
    main()
