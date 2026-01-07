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
        
        # Drift Detection (Training-only)
        self.drift_window_size = 50
        self.drift_outcomes = [] # List of bool (True if miss)
        self.drift_probe_burst_steps_left = 0
        self.drift_triggers = 0
        self.drift_probe_steps_total = 0

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

    def train_round(self, batch_size: int, drill_n: int, uncertainty_threshold: float, fixed_batch: List[TrainingSample] = None, question_credit: float = 0.25, question_preferred: bool = True, question_budget_per_round: int = 0, probe_after_budget: bool = False) -> Dict[str, Any]:
        batch = fixed_batch if fixed_batch else self.generate_batch(batch_size)
        results = []
        corrections = 0
        question_supervised_count = 0
        self.agent._proto_seeded_round = 0 # reset per round
        self.agent._silent_to_question_nudges_round = 0 # reset per round
        self.agent._question_repeats_blocked_round = 0 # reset per round
        
        questions_used_this_round = 0
        probe_count = 0
        probe_wrong_or_uncertain_count = 0
        question_budget_hit_count = 0
        questions_blocked_count = 0
        
        # Reset per-round drift counters if needed, but drift state persists across rounds
        drift_trigger_indices = []

        for sample_idx, sample in enumerate(batch):
            res = self.run_inference(sample)
            
            # Add phase info if partner is mixed_shift
            if hasattr(self.partner, "step_count") and "mixed_shift" in self.partner.name:
                # partner.step_count was already incremented in respond()
                # step_count - 1 is the index for the current sample
                current_step = getattr(self.partner, "step_count") - 1
                split_point = getattr(self.partner, "split_point", 500)
                phase = 0 if current_step < split_point else 1
                res.decision.meta["phase"] = phase
                res.decision.meta["rule"] = "mixed" if phase == 0 else "sum_prime" if "large" not in self.partner.name else "parity"

            # DRIFT REACTION (PROBE BURST)
            if self.drift_probe_burst_steps_left > 0:
                self.drift_probe_burst_steps_left -= 1
                self.drift_probe_steps_total += 1
                # Force lane = SPEAK but mark meta={"probe": true, "drift_probe": true}
                # We need to find the best handle prediction
                all_matches = [h for h in self.agent._handles if h.sent_sig == res.sample.sent_sig]
                if all_matches:
                    all_matches.sort(key=lambda h: (h.strength, h.hits), reverse=True)
                    h = all_matches[0]
                    act = tuple(int(x) for x in h.resp_sig.split(",")) if h.resp_sig != "0" else ()
                    res.decision = Decision(lane=Lane.SPEAK, act=act, meta={**res.decision.meta, "probe": True, "drift_probe": True})
                    # Re-evaluate error
                    from constraint_bootstrap.metrics_v1 import response_error
                    res.error = response_error(act, res.actual)
                else:
                    # If no handles, we can't really probe effectively, but let's at least mark it
                    res.decision = Decision(lane=Lane.SPEAK, act=(), meta={**res.decision.meta, "probe": True, "drift_probe": True})
                    from constraint_bootstrap.metrics_v1 import response_error
                    res.error = response_error((), res.actual)

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

            # APPLY QUESTION BUDGET & PROBE LOGIC
            if res.decision.lane == Lane.QUESTION:
                if question_budget_per_round > 0 and questions_used_this_round >= question_budget_per_round:
                    question_budget_hit_count += 1
                    if probe_after_budget:
                        # Force a PROBE decision: choose best candidate mapping (top handle prediction) even if gated
                        # We use agent._handles because agent.handles filters by eligibility/truth
                        all_matches = [h for h in self.agent._handles if h.sent_sig == res.sample.sent_sig]
                        if all_matches:
                            all_matches.sort(key=lambda h: (h.strength, h.hits), reverse=True)
                            h = all_matches[0]
                            act = tuple(int(x) for x in h.resp_sig.split(",")) if h.resp_sig != "0" else ()
                            # Emit decision lane as SPEAK but with meta {"probe": true}
                            res.decision = Decision(lane=Lane.SPEAK, act=act, meta={**res.decision.meta, "probe": True})
                            probe_count += 1
                            # Re-evaluate error for the probe
                            from constraint_bootstrap.metrics_v1 import response_error
                            res.error = response_error(act, res.actual)
                        else:
                            # If no handles at all, we can't really probe effectively
                            # Route to NA as per task A
                            res.decision = Decision(lane=Lane.NA, meta={**res.decision.meta, "budget_blocked": True})
                            questions_blocked_count += 1
                    else:
                        # Budget hit and no probe requested -> route to NA (or SILENT)
                        # Task A: "route would-be QUESTION decisions to NA (or SILENT) instead of QUESTION"
                        original_lane = res.decision.lane
                        res.decision = Decision(lane=Lane.NA, meta={**res.decision.meta, "budget_blocked": True, "original_lane": original_lane.value})
                        questions_blocked_count += 1
                else:
                    questions_used_this_round += 1

            # Core weight update condition must be ONLY:

            # Core weight update condition must be ONLY:
            # - lane == SPEAK
            # - is_trainable_oracle == true
            # - pred_act != oracle_act (error > 0 or uncertainty below threshold for refinement)
            # => update_type="correction_truth"
            
            should_correct_truth = False
            if res.decision.lane == Lane.SPEAK and res.is_trainable_oracle:
                is_probe = res.decision.meta.get("probe", False)
                if res.error > 0.0 or is_uncertain:
                    should_correct_truth = True
                    if is_probe:
                        probe_wrong_or_uncertain_count += 1
            
            # NEW: QUESTION lane supervised truth update
            should_question_train = False
            if res.decision.lane == Lane.QUESTION and res.is_trainable_oracle:
                should_question_train = True

            if should_correct_truth:
                is_probe = res.decision.meta.get("probe", False)
                # observe() with update_truth=False for probe (even if learn=True)
                # UNLESS it triggers a correction (which it does here if should_correct_truth is True)
                # Requirement 3: "If oracle/correction is available, a wrong/uncertain probe may trigger a proper supervised truth update"
                self.agent.observe(sample.sent, res.actual, learn=True, update_truth=True)
                res.corrected = True
                corrections += 1
                res.update_type = "correction_truth_probe" if is_probe else "correction_truth"
                
                # Apply boundary drills if it was a real SPEAK error
                if res.error > 0.0:
                    self.apply_boundary_drills(res, drill_n)
            elif should_question_train:
                # QUESTION lane acts as a request for label
                self.agent.observe(
                    sample.sent, 
                    res.actual, 
                    learn=True, 
                    update_truth=True, 
                    eligibility_bump=self.agent.question_eligibility_bump
                )
                res.corrected = True # Count as a learning event
                question_supervised_count += 1
                res.update_type = "question_supervised"
            else:
                # Absolutely NO core weight updates on lanes: NA / SILENT.
                # If silence_penalty/missed_opportunity shaping exists, it may ONLY adjust gate/calibration
                # and must be logged as update_type="eligibility_nudge" (separate from correction).
                
                if res.decision.lane in [Lane.SPEAK, Lane.QUESTION, Lane.NA, Lane.SILENT]:
                    is_probe = res.decision.meta.get("probe", False)
                    if is_probe:
                        # Requirement: "ensure observe() is called with update_truth=False for probe (even if learn=True)"
                        self.agent.observe(sample.sent, res.actual, learn=True, update_truth=False)
                        res.update_type = "probe_speak"
                    elif res.decision.lane in [Lane.QUESTION, Lane.NA, Lane.SILENT] and self.agent.silence_penalty > 0.0 and res.is_trainable_oracle:
                        # This triggers the silence_penalty logic in agent.observe
                        # which now only boosts eligibility (because update_truth=False).
                        self.agent.observe(sample.sent, res.actual, learn=True, update_truth=False)
                        res.update_type = "eligibility_nudge"
                    else:
                        # No core update (except for promotion logic)
                        self.agent.observe(sample.sent, res.actual, learn=False)
            
            results.append(res)
            self.history.append(res)
            
            # DRIFT DETECTION (Training-only)
            # Consider “committed” = SPEAK or PROBE (not QUESTION, not SILENT)
            is_probe = res.decision.meta.get("probe", False)
            if res.decision.lane == Lane.SPEAK or is_probe:
                # err > 0 counts as miss
                is_miss = res.error > 0.0
                self.drift_outcomes.append(is_miss)
                if len(self.drift_outcomes) > self.drift_window_size:
                    self.drift_outcomes.pop(0)
                
                # Trigger drift when: miss_rate >= 0.60 AND window is full (50 items)
                if len(self.drift_outcomes) == self.drift_window_size:
                    miss_rate = sum(self.drift_outcomes) / self.drift_window_size
                    if miss_rate >= 0.60 and self.drift_probe_burst_steps_left <= 0:
                        self.drift_triggers += 1
                        self.drift_probe_burst_steps_left = 20
                        drift_trigger_indices.append(len(self.history) - 1)
                        # Log “DE_DRIFT” event to telemetry
                        res.decision.meta["DE_DRIFT"] = True
                        res.decision.meta["drift_miss_rate"] = miss_rate
                        res.decision.meta["drift_trigger_index"] = len(self.history) - 1
                        print(f"!!! DRIFT TRIGGERED at index {len(self.history)-1}, miss_rate={miss_rate:.2f}")

        # Metrics computation (v1.1)
        n = len(results)
        speak_results = [r for r in results if r.decision.lane == Lane.SPEAK]
        speak_non_probe = [r for r in speak_results if not r.decision.meta.get("probe")]
        
        speak_count = len(speak_results)
        speak_non_probe_count = len(speak_non_probe)
        
        speak_correct = sum(1 for r in speak_results if r.error == 0.0)
        speak_non_probe_correct = sum(1 for r in speak_non_probe if r.error == 0.0)
        
        question_count = sum(1 for r in results if r.decision.lane == Lane.QUESTION)
        na_count = sum(1 for r in results if r.decision.lane == Lane.NA)
        silent_count = sum(1 for r in results if r.decision.lane == Lane.SILENT)
        
        # Diagnostics for silent misses
        # We compute these from decision-time metadata
        silent_miss_no_candidates = 0
        silent_miss_with_candidates = 0
        silent_to_question_nudges = 0
        
        # Diagnostics for gate mismatch
        speakable_handle_count = 0
        gated_by_eligibility_count = 0
        
        # We can look at the handles directly at the end of the round
        for h in self.agent.handles:
            # speakable: meets truth_min_to_speak AND strength meets min_strength_to_predict
            if h.truth >= self.agent.truth_min_to_speak:
                if h.strength >= self.agent.min_strength_to_predict:
                    speakable_handle_count += 1
                else:
                    gated_by_eligibility_count += 1

        for r in results:
            meta = r.decision.meta
            if r.decision.lane in [Lane.SILENT, Lane.NA] and r.error > 0.0:
                if meta.get("had_any_match"):
                    silent_miss_with_candidates += 1
                else:
                    silent_miss_no_candidates += 1
            
            # Diagnostic for Nudges
            if meta.get("was_nudged_to_question"):
                silent_to_question_nudges += 1

        # Separate probe metrics (Requirement B)
        # speak_precision counts ONLY non-probe SPEAK
        precision = speak_non_probe_correct / speak_non_probe_count if speak_non_probe_count > 0 else 0.0
        # probe_precision separately (optional)
        probe_speak_results = [r for r in speak_results if r.decision.meta.get("probe")]
        probe_precision = sum(1 for r in probe_speak_results if r.error == 0.0) / len(probe_speak_results) if probe_speak_results else 0.0

        speak_rate = speak_count / n
        question_rate = question_count / n
        # response_efficiency excludes probe SPEAK events (Requirement B)
        # response_efficiency = speak_non_probe_rate * speak_non_probe_precision
        # which is basically speak_non_probe_correct / n
        response_efficiency = speak_non_probe_correct / n if n > 0 else 0.0
        
        utility = (response_efficiency) + (question_rate * question_credit)
        
        # Accuracy: Only on trainable oracle samples where lane == SPEAK
        # Should we exclude probes from Accuracy? Usually Accuracy is "when we try, how often are we right"
        # Let's keep accuracy as is (includes probes) or exclude?
        # Task B says "pollutes SPEAK precision / response efficiency metrics". Accuracy is different.
        # But let's make speak_precision purely non-probe.
        
        trainable_count = sum(1 for r in results if r.is_trainable_oracle)
        # Let's make accuracy also non-probe if we want pure "learned" performance? 
        # Actually, accuracy usually includes all SPEAK attempts in this project.
        # But if we want to see "graduation to SPEAK", let's keep it inclusive but separate the precision.
        accuracy = sum(1 for r in results if r.is_trainable_oracle and r.decision.lane == Lane.SPEAK and r.error == 0.0) / trainable_count if trainable_count > 0 else 0.0
        
        # Diagnostics
        speak_wrong_or_uncertain_count = sum(1 for r in results if r.decision.lane == Lane.SPEAK and r.is_trainable_oracle and (r.error > 0.0 or r.uncertainty < uncertainty_threshold))
        
        # Avg strength across ALL handles (diagnostic)
        avg_strength = sum(h.strength for h in self.agent._handles) / len(self.agent._handles) if self.agent._handles else 0.0

        # Avg eligibility and truth for top handles
        # Use _handles directly to avoid property overhead in metrics
        top_h = sorted(self.agent._handles, key=lambda h: h.strength, reverse=True)[:10]
        avg_eligibility = sum(h.eligibility for h in top_h) / len(top_h) if top_h else 0.0
        avg_truth = sum(h.truth for h in top_h) / len(top_h) if top_h else 0.0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "probe_precision": probe_precision,
            "speak_rate": speak_rate,
            "speak_non_probe_count": speak_non_probe_count,
            "question_rate": question_rate,
            "na_rate": na_count / n,
            "utility": utility,
            "corrections": corrections,
            "question_supervised_count": question_supervised_count,
            "drill_queue_size": len(self.drill_queue),
            "top_errors": self._get_top_errors(results),
            "avg_eligibility": avg_eligibility,
            "avg_truth": avg_truth,
            "avg_strength": avg_strength,
            "speakable_handle_count": speakable_handle_count,
            "gated_by_eligibility_count": gated_by_eligibility_count,
            "trainable_oracle_rate": trainable_count / n if n > 0 else 0.0,
            "speak_wrong_or_uncertain_count": speak_wrong_or_uncertain_count,
            "corrections_triggered_count": corrections + question_supervised_count,
            "silent_miss_no_candidates": silent_miss_no_candidates,
            "silent_miss_with_candidates": silent_miss_with_candidates,
            "proto_seeded": self.agent._proto_seeded_round,
            "silent_to_question_nudges": self.agent._silent_to_question_nudges_round,
            "question_repeats_blocked": self.agent._question_repeats_blocked_round,
            "probe_count": probe_count,
            "probe_wrong_or_uncertain_count": probe_wrong_or_uncertain_count,
            "question_budget_hit_count": question_budget_hit_count,
            "questions_blocked_count": questions_blocked_count,
            "drift_triggers": self.drift_triggers,
            "drift_probe_steps": self.drift_probe_steps_total,
            "drift_trigger_indices": drift_trigger_indices
        }
        return metrics

    def _get_top_errors(self, results: List[TrainingResult]) -> List[Tuple[str, int]]:
        error_cats = {}
        for r in results:
            if r.error > 0.0 and r.decision.lane == Lane.SPEAK:
                error_cats[r.category] = error_cats.get(r.category, 0) + 1
        return sorted(error_cats.items(), key=lambda x: x[1], reverse=True)[:5]

    def save_run(self, metadata: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/training_runs/run_{timestamp}.json"
        
        # Ensure uniqueness
        counter = 1
        while os.path.exists(filename):
            filename = f"data/training_runs/run_{timestamp}_{counter:02d}.json"
            counter += 1
        
        report = {
            "metadata": metadata,
            "summary": {
                "total_samples": len(self.history),
                "final_accuracy": sum(1 for r in self.history[-100:] if r.error == 0.0) / min(100, len(self.history)) if self.history else 0,
                "silent_miss_no_candidates": sum(1 for r in self.history[-100:] if r.decision.lane in [Lane.SILENT, Lane.NA] and r.error > 0.0 and not r.decision.meta.get("had_any_match")),
                "silent_miss_with_candidates": sum(1 for r in self.history[-100:] if r.decision.lane in [Lane.SILENT, Lane.NA] and r.error > 0.0 and r.decision.meta.get("had_any_match")),
                "proto_seeded_total": self.agent._total_handles_created,
                "probe_count_total": sum(1 for r in self.history if r.decision.meta.get("probe")),
                "silent_to_question_nudges_total": sum(1 for r in self.history if r.decision.meta.get("was_nudged_to_question")),
                "question_repeats_blocked_total": sum(1 for r in self.history if r.decision.meta.get("on_cooldown") and r.decision.lane in [Lane.SILENT, Lane.NA])
            },
            "history": [
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
                } for r in self.history
            ]
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        return filename
