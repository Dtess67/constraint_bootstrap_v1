from dataclasses import dataclass
from enum import Enum

class ErrorCategory(Enum):
    NONE = "none"
    ALIAS = "alias"
    METAPHOR = "metaphor"
    NA_MISSED = "na-missed"
    POLARITY_MISSED = "polarity-missed"
    OTHER = "other"

@dataclass
class ErrorClassification:
    category: ErrorCategory
    description: str

def classify_error(pred_sig: str, act_sig: str, err: float) -> ErrorClassification:
    """
    Classify the error between predicted and actual signatures.
    In the toy world:
    - ALIAS: predicted something that is mathematically equivalent or a known alias (not really defined yet)
    - METAPHOR: predicted something that has the right 'shape' but wrong values (e.g. same length)
    - NA_MISSED: error is in the 'Na' range but prediction was empty or very different
    - POLARITY_MISSED: prediction was empty when act was not, or vice versa.
    """
    if err == 0.0:
        return ErrorClassification(ErrorCategory.NONE, "No error")
    
    if pred_sig == "0" and act_sig != "0":
        return ErrorClassification(ErrorCategory.POLARITY_MISSED, "False negative (silent when should have responded)")
    
    if pred_sig != "0" and act_sig == "0":
        return ErrorClassification(ErrorCategory.POLARITY_MISSED, "False positive (responded when should have been silent)")

    pred_len = len(pred_sig.split(",")) if pred_sig != "0" else 0
    act_len = len(act_sig.split(",")) if act_sig != "0" else 0
    
    if pred_len == act_len:
        return ErrorClassification(ErrorCategory.METAPHOR, "Same length, different values")
    
    if 0.0 < err < 1.5:
        return ErrorClassification(ErrorCategory.NA_MISSED, "Small error in Na range")
        
    return ErrorClassification(ErrorCategory.OTHER, "General error")
