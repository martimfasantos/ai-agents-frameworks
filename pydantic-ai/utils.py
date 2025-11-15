from pydantic_ai import ModelMessage, RunUsage


def print_all_messages(messages: list[ModelMessage]) -> None:
    """Structured print of all messages."""
    for i, message in enumerate(messages):
        print(f"   Message {i+1}: {type(message).__name__}")
        if hasattr(message, 'parts'):
            for j, part in enumerate(message.parts):
                part_type = type(part).__name__
                if 'ToolCall' in part_type:
                    print(f"      Part {j+1}: {part_type} - {getattr(part, 'tool_name', 'Unknown tool')}")
                elif 'ToolReturn' in part_type:
                    content = getattr(part, 'content', '')
                    print(f"      Part {j+1}: {part_type} - {content[:50]}...")
                else:
                    print(f"      Part {j+1}: {part_type}")

def show_metrics(usage: RunUsage) -> None:
    print(f"\n📈 Final Metrics:")
    print(f"   - Total requests: {usage.requests}")
    print(f"   - Tool calls executed: {usage.tool_calls}")
    print(f"   - Input tokens used: {usage.input_tokens}")
    print(f"   - Output tokens generated: {usage.output_tokens}")
    print("\n" + "="*60 + "\n")