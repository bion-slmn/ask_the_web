import streamlit as st
import time
from typing import Dict


def update_telemetry(
        start_time: float, usage_metadata: Dict[str, int]) -> None:
    """
    Updates the telemetry information in the Streamlit sidebar,
      including the latency and token usage.

    This function calculates the total latency by measuring
    the time elapsed since the start of the query.
    It also extracts and displays token usage details such as
    input tokens, output tokens, and total tokens.

    Args:
        start_time (float): The time when the query was initiated.
        usage_metadata (Dict[str, int]): A dict containing token usage with:
            - "input_tokens" (int)
            - "output_tokens" (int)
            - "total_tokens" (int)

    Returns:
        None: This function does not return any value.
        It updates the Streamlit sidebar with telemetry data.
    """
    latency = round(time.time() - start_time, 2)

    prompt_tokens = usage_metadata.get("input_tokens", 0)
    output_tokens = usage_metadata.get("output_tokens", 0)
    total_tokens = usage_metadata.get("total_tokens", 0)

    with st.sidebar:
        st.markdown("### 🔧 Telemetry")
        st.metric("⏱ Latency (s)", latency)
        st.metric("🧠 Input Tokens", prompt_tokens)
        st.metric("💬 Output Tokens", output_tokens)
        st.metric("📊 Total Tokens", total_tokens)
