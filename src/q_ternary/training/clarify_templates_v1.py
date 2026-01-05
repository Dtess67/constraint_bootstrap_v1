from typing import Optional, List

def get_weak_knowledge_question(top1_act: str, top2_act: str) -> str:
    """Template for when knowledge exists but is below prediction threshold."""
    return f"I'm close — is the correct act {top1_act} or {top2_act}?"

def get_conflict_question(top1_act: str, top2_act: str) -> str:
    """Template for when multiple strong handles compete with low margin."""
    return f"I'm seeing both {top1_act} and {top2_act} strongly — which one matches your intent/context?"

def get_ambiguous_oracle_question(top1_act: str, top2_act: str) -> str:
    """Template for when the oracle response itself is ambiguous (multi-label)."""
    return f"This looks ambiguous (multiple valid acts). Which should I choose: {top1_act} or {top2_act}?"

def format_act(act_sig: str) -> str:
    """Helper to format signature for human reading."""
    if not act_sig or act_sig == "0":
        return "[]"
    return f"[{act_sig.replace(',', ' ')}]"
