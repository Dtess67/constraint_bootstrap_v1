import argparse
import sys
from constraint_bootstrap.bootstrap_agent_v1 import BootstrapAgentV1
from q_ternary.training.aggressive_trainer_v1 import AggressiveTrainerV1

def train_aggressive(args):
    agent = BootstrapAgentV1(
        seed=args.seed,
        silence_penalty=args.silence_penalty,
        min_strength_to_predict=args.min_strength,
        promote_threshold=args.promote_threshold,
        conflict_margin=args.conflict_margin,
        eligibility_min_to_consider=args.eligibility_min_to_consider,
        truth_min_to_speak=args.truth_min_to_speak,
        seed_proto_handles=args.seed_proto_handles,
        seed_eligibility=args.seed_eligibility,
        question_cooldown_n=args.question_cooldown_n,
        question_eligibility_bump=args.question_eligibility_bump
    )
    trainer = AggressiveTrainerV1(agent, partner_name=args.partner, seed=args.seed)
    
    print(f"Starting Aggressive Training Sandbox (v1.2 - Truth/Eligibility Split)...")
    print(f"Partner: {args.partner} | Batch: {args.batch} | Rounds: {args.rounds}")
    print(f"Utility Credit: {args.question_credit} | Conflict Margin: {args.conflict_margin}")
    print(f"Eligibility Min: {args.eligibility_min_to_consider} | Truth Min to Speak: {args.truth_min_to_speak}")
    print("-" * 65)
    
    for r in range(1, args.rounds + 1):
        metrics = trainer.train_round(
            batch_size=args.batch,
            drill_n=args.drill_n,
            uncertainty_threshold=args.uncertainty_threshold,
            question_credit=args.question_credit,
            question_preferred=args.question_preferred,
            question_budget_per_round=args.question_budget_per_round,
            probe_after_budget=args.probe_after_budget
        )
        
        print(f"Round {r:02d}: Acc={metrics['accuracy']:.2%} | SP={metrics['speak_rate']:.2%} | QU={metrics['question_rate']:.2%} | Prec={metrics['precision']:.2%} | Util={metrics['utility']:.2%}")
        print(f"          Avg Elig={metrics['avg_eligibility']:.3f} | Avg Truth={metrics['avg_truth']:.3f} | Avg Strength={metrics.get('avg_strength', 0):.3f}")
        print(f"          Speakable Handles={metrics.get('speakable_handle_count', 0)} | Gated by Elig={metrics.get('gated_by_eligibility_count', 0)}")
        print(f"          Corrections={metrics['corrections']} | Q-Supervised={metrics['question_supervised_count']} | Na-Rate={metrics['na_rate']:.2%}")
        print(f"          Oracle={metrics['trainable_oracle_rate']:.1%} | SpeakUnc={metrics['speak_wrong_or_uncertain_count']} | CorrTrig={metrics['corrections_triggered_count']}")
        print(f"          SilentMissNoCand={metrics['silent_miss_no_candidates']} | SilentMissWithCand={metrics['silent_miss_with_candidates']}")
        print(f"          ProtoSeeded={metrics['proto_seeded']} | Nudges={metrics['silent_to_question_nudges']} | BlockedQ={metrics['question_repeats_blocked']}")
        print(f"          ProbeCount={metrics.get('probe_count', 0)} | SpeakNonProbePrec={metrics.get('precision', 0):.2%} | SpeakProbeCount={metrics.get('probe_count', 0)}")
        print(f"          BudgetHit={metrics.get('question_budget_hit_count', 0)} | BlockedQ_Budget={metrics.get('questions_blocked_count', 0)}")
        if metrics['top_errors']:
            err_str = ", ".join([f"{cat}: {count}" for cat, count in metrics['top_errors']])
            print(f"          Top SPEAK Errors: {err_str}")
        print(f"          Drill Queue: {metrics['drill_queue_size']}")

    metadata = {
        "batch": args.batch,
        "rounds": args.rounds,
        "uncertainty_threshold": args.uncertainty_threshold,
        "conflict_margin": args.conflict_margin,
        "question_credit": args.question_credit,
        "question_preferred": args.question_preferred,
        "min_strength": args.min_strength,
        "promote_threshold": args.promote_threshold,
        "eligibility_min_to_consider": args.eligibility_min_to_consider,
        "truth_min_to_speak": args.truth_min_to_speak,
        "partner": args.partner,
        "seed": args.seed,
        "silence_penalty": args.silence_penalty,
        "seed_proto_handles": args.seed_proto_handles,
        "seed_eligibility": args.seed_eligibility,
        "question_cooldown_n": args.question_cooldown_n,
        "question_eligibility_bump": args.question_eligibility_bump,
        "question_budget_per_round": args.question_budget_per_round,
        "probe_after_budget": args.probe_after_budget
    }
    filename = trainer.save_run(metadata)
    print("-" * 60)
    print(f"Training complete. Metrics saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="QD Shell v1")
    subparsers = parser.add_subparsers(dest="command")

    # train_aggressive command
    train_parser = subparsers.add_parser("train_aggressive")
    train_parser.add_argument("--batch", type=int, default=200)
    train_parser.add_argument("--rounds", type=int, default=5)
    train_parser.add_argument("--uncertainty-threshold", type=float, default=0.1)
    train_parser.add_argument("--conflict-margin", type=float, default=0.10)
    train_parser.add_argument("--question-credit", type=float, default=0.25)
    train_parser.add_argument("--question-preferred", type=bool, default=True)
    train_parser.add_argument("--min-strength", type=float, default=0.35)
    train_parser.add_argument("--promote-threshold", type=int, default=4)
    train_parser.add_argument("--eligibility-min-to-consider", type=float, default=0.25)
    train_parser.add_argument("--truth-min-to-speak", type=float, default=0.35)
    train_parser.add_argument("--drill-n", type=int, default=3)
    train_parser.add_argument("--silence-penalty", type=float, default=0.0)
    train_parser.add_argument("--partner", default="mixed")
    train_parser.add_argument("--seed", type=int, default=123)
    train_parser.add_argument("--seed-proto-handles", action="store_true", default=True)
    train_parser.add_argument("--no-seed-proto-handles", action="store_false", dest="seed_proto_handles")
    train_parser.add_argument("--seed-eligibility", type=float, default=0.25)
    train_parser.add_argument("--question-cooldown-n", type=int, default=0)
    train_parser.add_argument("--question-eligibility-bump", type=float, default=0.0)
    train_parser.add_argument("--question-budget-per-round", type=int, default=0)
    train_parser.add_argument("--probe-after-budget", action="store_true", default=False)

    args = parser.parse_args()

    if args.command == "train_aggressive":
        train_aggressive(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
