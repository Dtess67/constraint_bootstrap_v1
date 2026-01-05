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
        truth_min_to_speak=args.truth_min_to_speak
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
            question_preferred=args.question_preferred
        )
        
        print(f"Round {r:02d}: Acc={metrics['accuracy']:.2%} | SP={metrics['speak_rate']:.2%} | QU={metrics['question_rate']:.2%} | Prec={metrics['precision']:.2%} | Util={metrics['utility']:.2%}")
        print(f"          Avg Elig={metrics['avg_eligibility']:.3f} | Avg Truth={metrics['avg_truth']:.3f}")
        print(f"          Corrections={metrics['corrections']} | Na-Rate={metrics['na_rate']:.2%}")
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
        "silence_penalty": args.silence_penalty
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

    args = parser.parse_args()

    if args.command == "train_aggressive":
        train_aggressive(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
