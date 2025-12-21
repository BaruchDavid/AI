ISSUE_STYLE: dict[str, tuple[str, str]] = {
    "normal": ("🟢", "green"),
    "slow_response": ("🟡", "orange"),
    "truncated_response": ("🟠", "orange"),
    "context_loss": ("🔴", "red"),
    "hallucination_risk": ("🔴", "red"),
}
