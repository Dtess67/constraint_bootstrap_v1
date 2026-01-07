import argparse
import json
import sys
from pathlib import Path

def print_summary(data):
    metadata = data.get("metadata", {})
    summary = data.get("summary", {})
    
    print("========================================================================")
    print("Q-TERNARY RUN SUMMARY")
    print("========================================================================")
    print(f"Partner:          {metadata.get('partner')}")
    print(f"Rounds:           {metadata.get('rounds')}")
    print(f"Batch Size:       {metadata.get('batch')}")
    print(f"Min Strength:     {metadata.get('min_strength')}")
    print("-" * 72)
    print(f"Total Samples:    {summary.get('total_samples')}")
    print(f"Final Accuracy:   {summary.get('final_accuracy', 0):.2%}")
    print(f"Proto Seeded:     {summary.get('proto_seeded_total')}")
    print(f"Probe Count:      {summary.get('probe_count_total', 0)}")
    print(f"Silent Misses:    {summary.get('silent_miss_no_candidates', 0) + summary.get('silent_miss_with_candidates', 0)}")
    
    # If there are last results, we can compute more
    results = data.get("last_results", [])
    if results:
        speak_results = [r for r in results if r.get("lane") == "SPEAK"]
        speak_non_probe = [r for r in speak_results if not r.get("meta", {}).get("probe")]
        question_count = sum(1 for r in results if r.get("lane") == "QUESTION")
        probe_count = sum(1 for r in speak_results if r.get("meta", {}).get("probe"))
        
        n = len(results)
        sp_rate = len(speak_results) / n
        qu_rate = question_count / n
        pr_rate = probe_count / n
        
        # SpeakNonProbe Precision
        snp_correct = sum(1 for r in speak_non_probe if r.get("err") == 0.0)
        snp_prec = snp_correct / len(speak_non_probe) if speak_non_probe else 0.0
        
        print(f"Last Round Lane Rates: SP={sp_rate:.1%} QU={qu_rate:.1%} Probe={pr_rate:.1%}")
        print(f"Last Round Precision:  SpeakNonProbePrec={snp_prec:.2%}")
    
    print("========================================================================")

def main():
    parser = argparse.ArgumentParser(description="Summarize q_ternary training JSON results.")
    parser.add_argument("input_file", nargs="?", help="Input JSON file path (positional).")
    parser.add_argument("--in", dest="in_flag", help="Input JSON file path (flag).")
    args = parser.parse_args()
    
    # Priority: positional argument first, then --in flag
    input_path_str = args.input_file or args.in_flag

    if not input_path_str:
        print("Error: No input file provided. Use positional argument or --in.")
        parser.print_help()
        return 1

    path = Path(input_path_str)
    if not path.exists():
        print(f"Error: File not found {path}")
        return 1
        
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print_summary(data)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
