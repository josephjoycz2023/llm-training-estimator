class GPUMemCalculatorAdapter:
    """
    Adapter for gpu-mem-calculator.

    Responsibility:
    - Convert our unified YAML config to gpu-mem-calculator config.
    - Run memory calculation.
    - Return normalized memory result.
    """

    def estimate_memory(self, case: dict) -> dict:
        raise NotImplementedError
