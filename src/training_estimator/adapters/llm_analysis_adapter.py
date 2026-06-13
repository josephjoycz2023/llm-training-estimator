class LLMAnalysisAdapter:
    """
    Adapter for llm-analysis.

    Responsibility:
    - Convert our unified YAML config to llm-analysis config.
    - Run theoretical training latency / GPU-hour estimation.
    - Return normalized time result.
    """

    def estimate_time(self, case: dict) -> dict:
        raise NotImplementedError
