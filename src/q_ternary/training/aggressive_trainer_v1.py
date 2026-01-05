import json
import os
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1, _sig, Seq
from constraint_bootstrap.alien_partners_v1 import make_partner
from q_ternary.lane_v1 import Lane, Decision
from q_ternary.training.clarify_templates_v1 import (
    get_ambiguous_oracle_question, 
    get_weak_knowledge_question, 
    get_conflict_question, 
    format_act
)
from .error_taxonomy_v1 import classify_error, ErrorCategory

@dataclass
class TrainingSample:
    sent: Seq
    sent_sig: str
    synthetic_drill: bool = False

@dataclass
class TrainingResult:
    sample: TrainingSample
    decision: Decision
    actual: Seq
    error: float
    uncertainty: float
    category: str
    corrected: bool = False
    oracle_ambiguous: bool = False
    is_trainable_oracle: bool = False
    update_type: Optional[str] = None

class AggressiveTrainerV1:
    def __init__(self, agent: BootstrapAgentV1, partner_name: str = "mixed", seed: int = 42):
        self.agent = agent
        self.partner = make_partner(partner_name)
        self.rng = random.Random(seed)
        self.drill_queue: List[TrainingSample] = []
        self.history: List[TrainingResult] = []

    def generate_batch(self, size: int) -> List[TrainingSample]:
        batch = []
        # First pull from drill queue
        while self.drill_queue and len(batch) < size:
            batch.append(self.drill_queue.pop(0))
        
        # Then generate new ones
        while len(batch) < size:
            # Fixed pulses for testing or random for diversity
            if self.rng.random() < 0.2:
                sent = (1, 1) # Good for [1,1] tests
            else:
                n = self.rng.randint(1, 10)
                sent = tuple(self.rng.randint(1, 12) for _ in range(n))
            batch.append(TrainingSample(sent=sent, sent_sig=_sig(sent)))
        
        return batch

    def compute_uncertainty(self, sent: Seq) -> float:
        sent_s = _sig(sent)
        # Use strongest first for consistency with predict
        candidates = [h for h in self.agent.handles if h.sent_sig == sent_s and h.strength >= self.agent.min_strength_to_predict]
        if not candidates:
            return 1.0 # Max uncertainty if no matches
        
        # Respect topk
        if self.agent.compete_topk > 0:
            candidates = candidates[:self.agent.compete_topk]
            
        s1 = candidates[0].strength
        s2 = candidates[1].strength if len(candidates) > 1 else 0.0
        
        # We ensure uncertainty is a positive value reflecting the margin
        # Smaller margin = Higher uncertainty
        margin = s1 - s2
        return margin

    def run_inference(self, sample: TrainingSample) -> TrainingResult:
        decision = self.agent.predict(sample.sent)
        actual = self.partner.respond(sample.sent)
        
        from constraint_bootstrap.metrics_v1 import response_error
        # response_error still uses raw pred for now, but we'll adapt
        pred_sig = decision.act if decision.lane == Lane.SPEAK else (999,) if decision.lane == Lane.QUESTION else ()
        err = response_error(pred_sig, actual)
        uncertainty = self.compute_uncertainty(sample.sent)
        
        classification = classify_error(_sig(pred_sig), _sig(actual), err)
        
        # Check if oracle has multi-label (ambiguous)
        oracle_ambiguous = len(actual) > 1
        
        # Define is_trainable_oracle = oracle_act is single-label AND representable.
        # If oracle has comma (e.g. "5,7") => treat as ambiguous => QUESTION/NA; no accuracy; NO core update.
        oracle_sig = _sig(actual)
        is_trainable_oracle = (len(actual) == 1) and ("," not in oracle_sig)

        return TrainingResult(
            sample=sample,
            decision=decision,
            actual=actual,
            error=err,
            uncertainty=uncertainty,
            category=classification.category.value,
            oracle_ambiguous=oracle_ambiguous,
            is_trainable_oracle=is_trainable_oracle
        )

    def apply_boundary_drills(self, result: TrainingResult, n: int):
        if result.error == 0.0:
            return
        
        sent = list(result.sample.sent)
        if not sent:
            return

        for _ in range(n):
            # Mutate one pulse
            idx = self.rng.randrange(len(sent))
            new_sent = list(sent)
            delta = self.rng.choice([-1, 1])
            new_sent[idx] = max(1, new_sent[idx] + delta)
            
            # Or add/remove a pulse
            if self.rng.random() < 0.3:
                if self.rng.random() < 0.5:
                    new_sent.append(self.rng.randint(1, 12))
                elif len(new_sent) > 1:
                    new_sent.pop(self.rng.randrange(len(new_sent)))
            
            drill_sent = tuple(new_sent)
            self.drill_queue.append(TrainingSample(sent=drill_sent, sent_sig=_sig(drill_sent), synthetic_drill=True))

    def train_round(self, batch_size: int, drill_n: int, uncertainty_threshold: float, fixed_batch: List[TrainingSample] = None, question_credit: float = 0.25, question_preferred: bool = True) -> Dict[str, Any]:
        batch = fixed_batch if fixed_batch else self.generate_batch(batch_size)
        results = []
        corrections = 0
        
        for sample in batch:
            res = self.run_inference(sample)
            
            # Uncertainty flagging (low margin)
            is_uncertain = res.uncertainty < uncertainty_threshold
            
            # Decide if we route to QUESTION lane if oracle is ambiguous or we are uncertain
            if (res.oracle_ambiguous or is_uncertain) and res.decision.lane != Lane.SPEAK:
                if question_preferred:
                    # If we were NA/SILENT but uncertain, force QUESTION template if possible
                    if res.decision.lane in [Lane.NA, Lane.SILENT]:
                        # Generate a question template based on what we know
                        top_h = self.agent.handles[:2] if self.agent.handles else []
                        top1_act = format_act(top_h[0].resp_sig) if top_h else "[]"
                        top2_act = format_act(top_h[1].resp_sig) if len(top_h) > 1 else "0"
                        
                        if res.oracle_ambiguous:
                            q_text = get_ambiguous_oracle_question(top1_act, top2_act)
                        else:
                            q_text = get_weak_knowledge_question(top1_act, top2_act)
                        
                        # Update the decision in the result
                        res.decision = Decision(lane=Lane.QUESTION, question=q_text, meta={"forced": True})

            # Core weight update condition must be ONLY:
            # - lane == SPEAK
            # - is_trainable_oracle == true
            # - pred_act != oracle_act (error > 0 or uncertainty below threshold for refinement)
            # => update_type="correction_truth"
            
            should_correct_truth = False
            if res.decision.lane == Lane.SPEAK and res.is_trainable_oracle:
                if res.error > 0.0 or is_uncertain:
                    should_correct_truth = True

            if should_correct_truth:
                self.agent.observe(sample.sent, res.actual, learn=True, update_truth=True)
                res.corrected = True
                corrections += 1
                res.update_type = "correction_truth"
                
                # Apply boundary drills if it was a real SPEAK error
                if res.error > 0.0:
                    self.apply_boundary_drills(res, drill_n)
            else:
                # Absolutely NO core weight updates on lanes: QUESTION / NA / SILENT.
                # If silence_penalty/missed_opportunity shaping exists, it may ONLY adjust gate/calibration
                # and must be logged as update_type="eligibility_nudge" (separate from correction).
                
                if res.decision.lane in [Lane.QUESTION, Lane.NA, Lane.SILENT]:
                    if self.agent.silence_penalty > 0.0 and res.is_trainable_oracle:
                        # This triggers the silence_penalty logic in agent.observe
                        # which now only boosts eligibility (because update_truth=False).
                        self.agent.observe(sample.sent, res.actual, learn=True, update_truth=False)
                        res.update_type = "eligibility_nudge"
                    else:
                        # No core update (except for promotion logic)
                        self.agent.observe(sample.sent, res.actual, learn=False)
            
            results.append(res)
            self.history.append(res)

        # Metrics computation (v1.1)
        n = len(results)
        speak_count = sum(1 for r in results if r.decision.lane == Lane.SPEAK)
        speak_correct = sum(1 for r in results if r.decision.lane == Lane.SPEAK and r.error == 0.0)
        question_count = sum(1 for r in results if r.decision.lane == Lane.QUESTION)
        na_count = sum(1 for r in results if r.decision.lane == Lane.NA)
        silent_count = sum(1 for r in results if r.decision.lane == Lane.SILENT)
        
        precision = speak_correct / speak_count if speak_count > 0 else 0.0
        speak_rate = speak_count / n
        question_rate = question_count / n
        utility = (speak_rate * precision) + (question_rate * question_credit)
        
        # Accuracy: Only on trainable oracle samples where lane == SPEAK
        trainable_count = sum(1 for r in results if r.is_trainable_oracle)
        accuracy = sum(1 for r in results if r.is_trainable_oracle and r.decision.lane == Lane.SPEAK and r.error == 0.0) / trainable_count if trainable_count > 0 else 0.0

        # Avg eligibility and truth for top handles
        # Use _handles directly to avoid property overhead in metrics
        top_h = sorted(self.agent._handles, key=lambda h: h.strength, reverse=True)[:10]
        avg_eligibility = sum(h.eligibility for h in top_h) / len(top_h) if top_h else 0.0
        avg_truth = sum(h.truth for h in top_h) / len(top_h) if top_h else 0.0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "speak_rate": speak_rate,
            "question_rate": question_rate,
            "na_rate": na_count / n,
            "utility": utility,
            "corrections": corrections,
            "drill_queue_size": len(self.drill_queue),
            "top_errors": self._get_top_errors(results),
            "avg_eligibility": avg_eligibility,
            "avg_truth": avg_truth
        }
        return metrics

    def _get_top_errors(self, results: List[TrainingResult]) -> List[Tuple[str, int]]:
        error_cats = {}
        for r in results:
            if r.error > 0.0 and r.decision.lane == Lane.SPEAK:
                error_cats[r.category] = error_cats.get(r.category, 0) + 1
        return sorted(error_cats.items(), key=lambda x: x[1], reverse=True)[:5]

    def save_run(self, metadata: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"data/training_runs/run_{timestamp}.json"
        
        report = {
            "metadata": metadata,
            "summary": {
                "total_samples": len(self.history),
                "final_accuracy": sum(1 for r in self.history[-100:] if r.error == 0.0) / min(100, len(self.history)) if self.history else 0
            },
            "last_results": [
                {
                    "sent": _sig(r.sample.sent),
                    "lane": r.decision.lane.value,
                    "oracle_act": _sig(r.actual),
                    "pred_act": _sig(r.decision.act) if r.decision.act is not None else "0",
                    "err": r.error,
                    "uncertainty_margin": r.uncertainty,
                    "question": r.decision.question,
                    "cat": r.category,
                    "drill": r.sample.synthetic_drill,
                    "corrected": r.corrected,
                    "update_type": r.update_type,
                    "meta": r.decision.meta,
                    "top_candidates": [
                        {
                            "hid": h.hid,
                            "eligibility": h.eligibility,
                            "truth": h.truth,
                            "combined_strength": h.strength
                        } for h in self.agent.handles[:3] if h.sent_sig == _sig(r.sample.sent)
                    ]
                } for r in self.history[-100:]
            ]
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        return filename
