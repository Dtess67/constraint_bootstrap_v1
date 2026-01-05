from __future__ import annotations

import argparse

from .alien_partners_v1 import make_partner
from .bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.lane_v1 import Lane
from .channel_v1 import ChannelV1

def _fmt(seq: tuple[int, ...]) -> str:
    return "[" + " ".join(str(x) for x in seq) + "]" if seq else "[]"

def main() -> int:
    ap = argparse.ArgumentParser(description="Constraint-first bootstrap demo (v0.1)")
    ap.add_argument("--partner", default="prime", help="prime | ratio | symmetry | mixed | adversarial")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--noise-prob", type=float, default=0.0)
    ap.add_argument("--noise-jitter", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--freeze", action="store_true", help="Skip learning/handle updates (for experiments)")
    ap.add_argument("--decay-rate", type=float, default=0.0)
    ap.add_argument("--prune-below", type=float, default=0.0)
    ap.add_argument("--compete-topk", type=int, default=0)
    ap.add_argument("--inhibit-mult", type=float, default=0.0)
    ap.add_argument("--promote-threshold", type=int, default=4)

    args = ap.parse_args()

    partner = make_partner(args.partner)
    chan = ChannelV1(noise_prob=args.noise_prob, noise_jitter=args.noise_jitter, seed=args.seed)
    agent = BootstrapAgentV1(
        seed=args.seed,
        decay_rate=args.decay_rate,
        prune_below=args.prune_below,
        compete_topk=args.compete_topk,
        inhibit_mult=args.inhibit_mult,
        promote_threshold=args.promote_threshold,
    )

    print("=" * 72)
    print(f"BOOTSTRAP v0.1  partner={partner.name}  steps={args.steps}")
    print("Watch: predicted vs actual, error, and handles forming.")
    print("=" * 72)

    for t in range(1, args.steps + 1):
        sent = agent.choose_action(t)
        raw_resp = partner.respond(sent)
        ex = chan.transmit(sent, raw_resp)
        decision = agent.predict(ex.sent)
        m = agent.observe(ex.sent, ex.received, learn=not args.freeze)

        # Decay handles once per step if learning is on
        if not args.freeze:
            agent._apply_handle_decay()

        # Use lane from Decision object
        lane_str = decision.lane.value[:2] # SPEAK->SP, QUESTION->QU, etc
        if decision.lane == Lane.SPEAK:
            lane_str = "SP"
        elif decision.lane == Lane.QUESTION:
            lane_str = "QU"
        elif decision.lane == Lane.NA:
            lane_str = "NA"
        elif decision.lane == Lane.SILENT:
            lane_str = "SI"
        
        # Override with "De" if surprised (surprises usually happen on SPEAK)
        if m.error >= agent.surprise_threshold:
            lane_str = "De"

        top = agent.handles[:3]
        top_s = " | ".join(f"{h.hid}:{h.strength:.2f} hits={h.hits} ({h.sent_sig}->{h.resp_sig})" for h in top) if top else "(none)"

        print(
            f"{t:03d}  sent={_fmt(ex.sent):<22} pred={_fmt(m.predicted):<10} "
            f"act={_fmt(m.actual):<10} err={m.error:>4.1f}  lane={lane_str}  handles={top_s}"
        )

    print("-" * 72)
    print("Final handles (strongest first):")
    for h in agent.handles[:12]:
        print(f"  {h.hid}  strength={h.strength:.2f}  hits={h.hits} misses={h.misses}  {h.sent_sig} -> {h.resp_sig}")

    print("-" * 72)
    print("Telemetry:")
    print(f"  multi-candidate steps: {agent.total_multi_candidate_steps}")
    print(f"  total inhibitions:     {agent.total_inhibitions}")
    print(f"  total handles created: {agent._total_handles_created}")
    avg_cand = agent.sum_candidate_count / agent.total_predict_calls if agent.total_predict_calls > 0 else 0
    print(f"  avg candidate count:   {avg_cand:.2f}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
