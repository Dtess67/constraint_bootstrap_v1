from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import random

from .metrics_v1 import StepMetrics, response_error
from q_ternary.lane_v1 import Decision, Lane
from q_ternary.training.clarify_templates_v1 import get_weak_knowledge_question, get_conflict_question, format_act

Pulse = int
Seq = Tuple[Pulse, ...]

@dataclass
class Handle:
    """
    A handle is a revocable mapping:
    sent_pattern_signature -> expected response signature
    No words, just stable references.

    Split into two components:
    - eligibility: discoverability / relevance (grows on any observation)
    - truth: evidence-backed correctness (grows only on SPEAK corrections)
    """
    hid: str
    sent_sig: str
    resp_sig: str
    eligibility: float = 0.25
    truth: float = 0.1
    hits: int = 0
    misses: int = 0

    def __init__(self, hid: str, sent_sig: str, resp_sig: str, eligibility: float = 0.25, truth: float = 0.1, hits: int = 0, misses: int = 0, strength: float = None):
        self.hid = hid
        self.sent_sig = sent_sig
        self.resp_sig = resp_sig
        self.hits = hits
        self.misses = misses
        if strength is not None:
            # For backward compatibility with tests that pass 'strength'
            self.eligibility = strength
            self.truth = strength # No longer optimistic 1.0
        else:
            self.eligibility = eligibility
            self.truth = truth

    @property
    def strength(self) -> float:
        """
        Combined strength for selection. 
        Truth dominates, but eligibility is required for confidence.
        Definition: strength = truth * sigmoid_like(eligibility)
        For simplicity and stability: strength = min(eligibility, truth)
        """
        return min(self.eligibility, self.truth)

    def update(self, matched: bool, update_truth: bool = False) -> None:
        """
        Update handle components.
        eligibility always updates if matched.
        truth only updates if update_truth is True.
        """
        if matched:
            self.hits += 1
            # Eligibility grows on any match
            self.eligibility = min(1.0, self.eligibility + 0.08)
            if update_truth:
                old_t = self.truth
                self.truth = min(1.0, self.truth + 0.08)
                # print(f"DEBUG: Handle {self.hid} matched={matched} update_truth={update_truth} truth {old_t} -> {self.truth} sent_sig={self.sent_sig} resp_sig={self.resp_sig}")
        else:
            self.misses += 1
            # Both decay on mismatch
            self.eligibility = max(0.0, self.eligibility - 0.12)
            # Only update truth if requested
            if update_truth:
                self.truth = max(0.0, self.truth - 0.12)


def _sig(seq: Seq) -> str:
    """Cheap signature string used only internally (not meaning)."""
    return ",".join(map(str, seq)) if seq else "0"

@dataclass
class BootstrapAgentV1:
    """
    Constraint-first learner (minimal):
    - Explore sequences
    - Predict using best matching handle (if any)
    - Update/promote handles when stable correlations appear
    """
    seed: int | None = None
    promote_threshold: int = 4 # how many consistent matches before promotion
    min_strength_to_predict: float = 0.35
    surprise_threshold: float = 1.5
    focus_repeats: int = 4
    focus_mutate_prob: float = 0.25
    decay_rate: float = 0.0
    prune_below: float = 0.0
    compete_topk: int = 0
    inhibit_mult: float = 0.0
    silence_penalty: float = 0.0
    conflict_margin: float = 0.1
    eligibility_min_to_consider: float = 0.25
    truth_min_to_speak: float = 0.35
    seed_proto_handles: bool = False
    seed_eligibility: float = 0.25
    question_cooldown_n: int = 0 # 0 means no cooldown, N means wait N steps before asking again
    question_eligibility_bump: float = 0.0 # bump eligibility on QUESTION-supervised events

    _rng: random.Random = field(init=False)
    _handles: List[Handle] = field(default_factory=list, init=False)
    _seen_counts: Dict[Tuple[str, str], int] = field(default_factory=dict, init=False)
    _focus_seq: Seq | None = field(default=None, init=False)
    _focus_left: int = field(default=0, init=False)
    _last_question_step: Dict[str, int] = field(default_factory=dict, init=False)
    _current_step: int = field(default=0, init=False)

    # Telemetry
    total_multi_candidate_steps: int = field(default=0, init=False)
    total_inhibitions: int = field(default=0, init=False)
    sum_candidate_count: int = field(default=0, init=False)
    total_predict_calls: int = field(default=0, init=False)
    _total_handles_created: int = field(default=0, init=False)
    _proto_seeded_round: int = field(default=0, init=False)
    _silent_to_question_nudges_round: int = field(default=0, init=False)
    _question_repeats_blocked_round: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    @property
    def handles(self) -> List[Handle]:
        # return strongest first for display
        return sorted(self._handles, key=lambda h: (h.strength, h.hits), reverse=True)

    def choose_action(self, step: int) -> Seq:
        """
        Exploration policy:
        - early: vary length heavily (to catch prime rules)
        - later: try structured patterns (palindrome, ratios)
        """
        if self._focus_left > 0 and self._focus_seq is not None:
            self._focus_left -= 1
            # Usually repeat exactly; sometimes mutate one pulse by +/-1 (clamped to >=1)
            if self._rng.random() < self.focus_mutate_prob and len(self._focus_seq) > 0:
                idx = self._rng.randrange(len(self._focus_seq))
                delta = self._rng.choice([-1, 1])
                lst = list(self._focus_seq)
                lst[idx] = max(1, lst[idx] + delta)
                return tuple(lst)
            return self._focus_seq

        if step < 15:
            n = self._rng.choice([1, 2, 3, 4, 5, 6, 7])
            return tuple(self._rng.randint(1, 7) for _ in range(n))

        # Mix random + structured probes
        roll = self._rng.random()
        if roll < 0.35:
            # random length and pulses
            n = self._rng.choice([2, 3, 4, 5, 6, 7, 8])
            return tuple(self._rng.randint(1, 9) for _ in range(n))

        if roll < 0.65:
            # palindrome probe
            half = tuple(self._rng.randint(1, 9) for _ in range(2))
            mid = (self._rng.randint(1, 9),) if self._rng.random() < 0.5 else ()
            return half + mid + tuple(reversed(half))

        # 2:1 ratio probe
        b = self._rng.randint(1, 6)
        a = 2 * b
        tail = tuple(self._rng.randint(1, 7) for _ in range(self._rng.choice([0, 1, 2])))
        return (a, b) + tail

    def _update_telemetry(self, sent: Seq) -> None:
        """Update telemetry counters for a given input."""
        sent_s = _sig(sent)
        # Match by signature alone for telemetry of "discovery overlap"
        all_match = [h for h in self._handles if h.sent_sig == sent_s]
        self.total_predict_calls += 1
        self.sum_candidate_count += len(all_match)
        if len(all_match) >= 2:
            self.total_multi_candidate_steps += 1

    def predict(self, sent: Seq) -> Decision:
        self._current_step += 1
        sent_s = _sig(sent)
        
        # All potential matches in registry
        all_registry_matches = [h for h in self._handles if h.sent_sig == sent_s]
        
        was_proto_seeded = False
        if self.seed_proto_handles and not all_registry_matches:
            # Pre-decision proto-seeding: create a hypothesis handle immediately
            # We don't know the resp_sig yet, so we use a placeholder "0" (empty)
            # or we could use a special "?" but "0" is safer for existing logic.
            self._total_handles_created += 1
            hid = f"H{self._total_handles_created:03d}"
            # SEED: Eligibility = seed_eligibility, Truth = 0.0
            new_h = Handle(hid=hid, sent_sig=sent_s, resp_sig="0", eligibility=self.seed_eligibility, truth=0.0)
            self._handles.append(new_h)
            self._proto_seeded_round += 1
            all_registry_matches = [new_h]
            was_proto_seeded = True

        if not all_registry_matches:
            return Decision(lane=Lane.SILENT, meta={"candidate_count": 0, "eligible_count": 0, "had_any_match": False})

        # Check cooldown
        on_cooldown = False
        if self.question_cooldown_n > 0:
            last_step = self._last_question_step.get(sent_s, -1)
            if last_step >= 0 and (self._current_step - last_step) <= self.question_cooldown_n:
                on_cooldown = True

        # Identify all matching candidates that meet minimum eligibility to be "known"
        all_candidates = [h for h in all_registry_matches if h.eligibility >= self.eligibility_min_to_consider]
        
        meta = {
            "candidate_count": len(all_registry_matches),
            "eligible_count": len(all_candidates),
            "had_any_match": True,
            "was_proto_seeded_predecision": was_proto_seeded,
            "on_cooldown": on_cooldown
        }

        if not all_candidates:
            # SILENT/NA -> QUESTION nudge rule
            if on_cooldown:
                self._question_repeats_blocked_round += 1
                return Decision(lane=Lane.SILENT, meta=meta)

            self._silent_to_question_nudges_round += 1
            # Sort matches so we have the best one for the question
            matches_sorted = sorted(all_registry_matches, key=lambda h: (h.strength, h.hits), reverse=True)
            h1 = matches_sorted[0]
            h2_act = matches_sorted[1].resp_sig if len(matches_sorted) > 1 else "0"
            question = get_weak_knowledge_question(format_act(h1.resp_sig), format_act(h2_act))
            meta.update({"reason": "pre_eligible_nudge", "nudge": True, "top1": h1.hid, "was_nudged_to_question": True})
            self._last_question_step[sent_s] = self._current_step
            return Decision(lane=Lane.QUESTION, question=question, meta=meta)

        # Sort candidates for competition
        all_candidates.sort(key=lambda h: (h.strength, h.hits), reverse=True)

        # Use Strength gate for active candidates
        active_candidates = [h for h in all_candidates if h.strength >= self.min_strength_to_predict]
        
        if self.compete_topk > 0:
            active_candidates = active_candidates[:self.compete_topk]
            
        # Middle Lane for weak but existing knowledge
        if not active_candidates:
            if on_cooldown:
                self._question_repeats_blocked_round += 1
                return Decision(lane=Lane.NA, meta=meta)

            h1 = all_candidates[0]
            h2_act = all_candidates[1].resp_sig if len(all_candidates) > 1 else "0"
            question = get_weak_knowledge_question(format_act(h1.resp_sig), format_act(h2_act))
            meta.update({"reason": "weak_knowledge", "top1": h1.hid, "eligibility": h1.eligibility, "truth": h1.truth})
            self._last_question_step[sent_s] = self._current_step
            return Decision(lane=Lane.QUESTION, question=question, meta=meta)

        # Middle Lane for conflicting strong knowledge
        if len(active_candidates) >= 2:
            margin = active_candidates[0].strength - active_candidates[1].strength
            if margin < self.conflict_margin:
                if on_cooldown:
                    self._question_repeats_blocked_round += 1
                    return Decision(lane=Lane.NA, meta=meta)

                h1 = active_candidates[0]
                h2 = active_candidates[1]
                question = get_conflict_question(format_act(h1.resp_sig), format_act(h2.resp_sig))
                meta.update({"reason": "conflict", "margin": margin, "top1": h1.hid, "top2": h2.hid, "s1": h1.strength, "s2": h2.strength})
                self._last_question_step[sent_s] = self._current_step
                return Decision(lane=Lane.QUESTION, question=question, meta=meta)

        h = active_candidates[0]
        act = tuple(int(x) for x in h.resp_sig.split(",")) if h.resp_sig != "0" else ()
        meta.update({"hid": h.hid, "strength": h.strength, "eligibility": h.eligibility, "truth": h.truth})
        return Decision(lane=Lane.SPEAK, act=act, meta=meta)

    def _apply_handle_decay(self) -> None:
        """Apply exponential decay to all handles and optionally prune weak ones."""
        if self.decay_rate <= 0.0:
            return

        keep = []
        for h in self._handles:
            # Decay both eligibility and truth
            # eligibility: discoverability / relevance (decay slowly)
            # truth: evidence-backed correctness (decay slower or not at all)
            # Instruction: "Decay eligibility slowly; decay truth slower or not at all"
            h.eligibility *= (1.0 - self.decay_rate)
            h.truth *= (1.0 - self.decay_rate * 0.5) # slower decay for truth

            if self.prune_below > 0.0 and h.strength < self.prune_below:
                continue

            keep.append(h)

        self._handles = keep

    def observe(self, sent: Seq, received: Seq, learn: bool = True, update_truth: bool = True, eligibility_bump: float = 0.0) -> StepMetrics:
        """
        Update internals based on an exchange.
        """
        decision = self.predict(sent)
        pred = decision.act if decision.lane == Lane.SPEAK else (999,) if decision.lane == Lane.QUESTION else ()
        err = response_error(pred, received)

        # Decay handles ONCE per round to avoid metabolic inflation
        # If we are in AggressiveTrainer, it calls _apply_handle_decay() manually once per round.
        
        if not learn:
            # Still track correlation counts for promotion even if we don't update weights
            # This allows the agent to discover handles without necessarily having to SPEAK first.
            sent_s = _sig(sent)
            recv_s = _sig(received)
            key = (sent_s, recv_s)
            self._seen_counts[key] = self._seen_counts.get(key, 0) + 1
            if self._seen_counts[key] >= self.promote_threshold:
                # Check if a handle already exists for this exact mapping
                existing = [h for h in self._handles if h.sent_sig == sent_s and h.resp_sig == recv_s]
                if not existing:
                    self._total_handles_created += 1
                    hid = f"H{self._total_handles_created:03d}"
                    # New handles born with 0.25 eligibility and 0.0 truth
                    self._handles.append(Handle(hid=hid, sent_sig=sent_s, resp_sig=recv_s, truth=0.0))
                # Promotion no longer boosts truth
            # Update telemetry AFTER promotion
            self._update_telemetry(sent)
            return StepMetrics(predicted=pred, actual=received, error=err)

        self._apply_handle_decay()

        # If surprised, lock attention and repeat to verify (obsession loop)
        # Truth guardrail: Obsession loop repeats enable learning, but truth only 
        # grows if update_truth is True (which requires a trainable correction event).
        if err >= self.surprise_threshold:
            self._focus_seq = sent
            self._focus_left = self.focus_repeats

        # Update existing matching handle (if any)
        sent_s = _sig(sent)
        recv_s = _sig(received)
        updated_any = False
        
        # Identify matching candidates for competition
        candidates = [h_cand for h_cand in self._handles if h_cand.sent_sig == sent_s]
        
        # If we have exactly one candidate and it's a freshly seeded proto with placeholder "0",
        # adopt the actual response signature.
        if len(candidates) == 1 and candidates[0].resp_sig == "0" and candidates[0].hits == 0 and candidates[0].misses == 0:
            candidates[0].resp_sig = recv_s

        # Sort by strength descending
        candidates.sort(key=lambda x: (x.strength, x.hits), reverse=True)

        winner = None
        if self.compete_topk > 0:
            allowed = candidates[:self.compete_topk]
        else:
            allowed = candidates

        for h_cand in allowed:
            if h_cand.resp_sig == recv_s:
                h_cand.update(matched=True, update_truth=update_truth)
                # Extra eligibility bump if requested (e.g. for QUESTION supervised events)
                if eligibility_bump > 0.0:
                    h_cand.eligibility = min(1.0, h_cand.eligibility + eligibility_bump)
                winner = h_cand
                updated_any = True
            else:
                h_cand.update(matched=False, update_truth=update_truth)
        
        if winner is None and allowed:
            # If we had matches but none were correct, we still mark updated_any 
            # to prevent the "lightly reinforce" fallback
            updated_any = True

        # Apply inhibition if enabled
        if self.inhibit_mult > 0.0 and winner is not None:
            for h in candidates:
                if h is not winner and h.sent_sig == sent_s:
                    # Inhibition affects both? Or just eligibility?
                    # Let's say it affects both proportionally.
                    h.eligibility = max(0.0, h.eligibility - (winner.strength * self.inhibit_mult))
                    h.truth = max(0.0, h.truth - (winner.strength * self.inhibit_mult))
                    self.total_inhibitions += 1

        # Track correlation counts for promotion
        key = (sent_s, recv_s)
        self._seen_counts[key] = self._seen_counts.get(key, 0) + 1

        # Promote new handle if we see repeated stable mapping
        if self._seen_counts[key] >= self.promote_threshold:
            # Check if a handle already exists for this exact mapping
            existing = [h for h in self._handles if h.sent_sig == sent_s and h.resp_sig == recv_s]
            if not existing:
                self._total_handles_created += 1
                hid = f"H{self._total_handles_created:03d}"
                self._handles.append(Handle(hid=hid, sent_sig=sent_s, resp_sig=recv_s, truth=0.0))
            # Promotion no longer boosts truth

        # Update telemetry AFTER promotion
        self._update_telemetry(sent)

        # If nothing updated, lightly reinforce the most recent mapping handle if it exists
        if not updated_any:
            # ONLY if we didn't update anything in the standard loop (including mismatches)
            for h_m in self._handles:
                if h_m.sent_sig == sent_s and h_m.resp_sig == recv_s:
                    h_m.update(matched=True, update_truth=update_truth)
                    if eligibility_bump > 0.0:
                        h_m.eligibility = min(1.0, h_m.eligibility + eligibility_bump)
                    updated_any = True # Mark it as updated
                    break

        # If we were silent but the environment spoke, and silence_penalty is on,
        # find or create handles for the observed mapping and give them a boost.
        is_truly_silent = (decision.lane in [Lane.SILENT, Lane.NA, Lane.QUESTION])
        if self.silence_penalty > 0.0 and is_truly_silent and received:
            # We missed a chance to speak.
            # Boost any handle that WOULD have been correct
            sent_s = _sig(sent)
            recv_s = _sig(received)
            for h_s in self._handles:
                if h_s.sent_sig == sent_s and h_s.resp_sig == recv_s:
                    # silence_penalty applies ONLY to eligibility.
                    h_s.eligibility = min(1.0, h_s.eligibility + self.silence_penalty)
                    # Truth must NEVER be gifted via silence penalty.
        
        return StepMetrics(predicted=pred, actual=received, error=err)
