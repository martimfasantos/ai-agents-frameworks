import os
import asyncio
import random

import numpy as np

from agents import Agent, function_tool, set_tracing_disabled
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
---------------------------------------------------------------------
Voice Agent Pipeline (STT → Agent → TTS)

Features demonstrated:
- VoicePipeline: orchestrates speech-to-text → agent → text-to-speech
- SingleAgentVoiceWorkflow: wraps an Agent for voice use
- AudioInput: provides audio buffer to the pipeline
- Streaming voice events: audio chunks, lifecycle events, errors
- Tool usage within a voice agent
- Non-interactive execution (no microphone/speaker required)

The voice pipeline is a 3-step process:
1. Transcribe input audio via STT (e.g., OpenAI Whisper)
2. Run an agent workflow to produce a text response
3. Convert the response to audio via TTS (e.g., OpenAI TTS)

This example sends 3 seconds of silence to demonstrate the pipeline
mechanics without requiring audio hardware. In production, you would
feed real microphone input and play back the audio output.

Requires: pip install 'openai-agents[voice]' numpy sounddevice
---------------------------------------------------------------------
"""

# --- Tool ---


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"  [tool] get_weather called with city={city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


# --- Agent ---

agent = Agent(
    name="VoiceAssistant",
    instructions=(
        "You are a helpful voice assistant. Be concise and conversational. "
        "You can check the weather for any city using the get_weather tool."
    ),
    model=settings.OPENAI_MODEL_NAME,
    tools=[get_weather],
)


# --- Main ---


async def main():
    set_tracing_disabled(True)

    # 1. Create the voice pipeline with a single-agent workflow
    workflow = SingleAgentVoiceWorkflow(agent)
    pipeline = VoicePipeline(workflow=workflow)

    print("Voice Pipeline created")
    print(f"  Workflow: {type(workflow).__name__}")
    print(f"  Agent: {agent.name}")
    print(f"  Model: {settings.OPENAI_MODEL_NAME}")
    print()

    # 2. Create audio input (3 seconds of silence at 24kHz)
    #    In production, this would be real microphone data
    duration_seconds = 3
    sample_rate = 24000
    buffer = np.zeros(sample_rate * duration_seconds, dtype=np.int16)
    audio_input = AudioInput(buffer=buffer)

    print(f"Audio input: {duration_seconds}s of silence at {sample_rate}Hz")
    print("Running pipeline (STT → Agent → TTS)...")
    print()

    # 3. Run the pipeline
    result = await pipeline.run(audio_input)

    # 4. Collect audio output events (no playback — headless environment)
    audio_chunks = 0
    audio_bytes = 0
    lifecycle_events = []

    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            audio_chunks += 1
            audio_bytes += len(event.data)
        elif event.type == "voice_stream_event_lifecycle":
            lifecycle_events.append(event.event)
            print(f"  [lifecycle] {event.event}")
        elif event.type == "voice_stream_event_error":
            print(f"  [error] {event.error}")

    print()
    print("Pipeline complete!")
    print(f"  Audio chunks received: {audio_chunks}")
    print(f"  Total audio bytes: {audio_bytes}")
    print(f"  Lifecycle events: {lifecycle_events}")

    # Note: With silent input, STT may produce empty transcription,
    # resulting in no agent response and thus no TTS output.
    # With real audio input, you would see audio chunks streamed back.


if __name__ == "__main__":
    asyncio.run(main())
