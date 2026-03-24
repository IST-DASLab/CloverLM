
import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Upload lm_eval results JSON to wandb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("results_file", type=os.path.abspath,
                   help="Path to lm_eval results JSON file")
    p.add_argument("--wandb_project", default="cloverlm-eval")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_run_id", default=None,
                   help="Fixed run ID for resuming across invocations")
    p.add_argument("--wandb_step", type=int, default=None,
                   help="Training step (x-axis value in wandb)")
    p.add_argument("--wandb_tags", nargs="*", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="Original checkpoint path (recorded in wandb config)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.results_file) as f:
        data = json.load(f)

    results = data.get("results", {})

    # ── Extract per-task metrics ──────────────────────────────────────────
    task_metrics = {}
    for task_name, task_results in results.items():
        for metric, value in task_results.items():
            if metric == "alias" or not isinstance(value, (int, float)):
                continue
            parts = metric.rsplit(",", 1)
            if len(parts) != 2:
                continue
            metric_name = parts[0]
            task_metrics[f"{task_name}/{metric_name}"] = value

    # ── Compute averages ─────────────────────────────────────────────────
    acc_norm_keys = [k for k in task_metrics
                     if ("acc_norm" in k or "acc_mutual_info" in k)
                     and "stderr" not in k]
    acc_raw_keys = [k for k in task_metrics
                    if k.endswith("/acc") and "norm" not in k
                    and "stderr" not in k]
    f1_keys = [k for k in task_metrics
               if k.endswith("/f1") and "stderr" not in k]
    em_keys = [k for k in task_metrics
               if k.endswith("/exact_match") and "stderr" not in k]

    avg_acc_norm = None
    avg_acc = None
    avg_score = None

    if acc_norm_keys:
        avg_acc_norm = sum(task_metrics[k] for k in acc_norm_keys) / len(acc_norm_keys)
    if acc_raw_keys:
        avg_acc = sum(task_metrics[k] for k in acc_raw_keys) / len(acc_raw_keys)
    composite_keys = acc_norm_keys + f1_keys + em_keys
    if composite_keys:
        avg_score = sum(task_metrics[k] for k in composite_keys) / len(composite_keys)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  Results from: {args.results_file}")
    print(f"{'─'*60}")
    for key, value in sorted(task_metrics.items()):
        if "stderr" not in key:
            if "perplexity" in key:
                print(f"  {key:40s} = {value:.2f}")
            else:
                print(f"  {key:40s} = {value*100:.2f}%")
    if avg_acc_norm is not None:
        print(f"\n  avg_acc_norm = {avg_acc_norm*100:.2f}%")
    if avg_acc is not None:
        print(f"  avg_acc      = {avg_acc*100:.2f}%")
    if avg_score is not None:
        print(f"  avg_score    = {avg_score*100:.2f}%")

    # ── wandb ────────────────────────────────────────────────────────────
    import wandb

    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config={
            "checkpoint": args.checkpoint,
            "results_file": args.results_file,
        },
    )
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"

    init_kwargs["settings"] = wandb.Settings(init_timeout=300)
    run = wandb.init(**init_kwargs)

    wandb_log = dict(task_metrics)
    if avg_acc_norm is not None:
        wandb_log["avg_acc_norm"] = avg_acc_norm
    if avg_acc is not None:
        wandb_log["avg_acc"] = avg_acc
    if avg_score is not None:
        wandb_log["avg_score"] = avg_score

    log_kwargs = {}
    if args.wandb_step is not None:
        run.define_metric("*", step_metric="train_step")
        wandb_log["train_step"] = args.wandb_step
        log_kwargs["step"] = args.wandb_step

    run.log(wandb_log, **log_kwargs)
    run.summary.update(wandb_log)

    step_tag = f"-step{args.wandb_step}" if args.wandb_step else ""
    artifact = wandb.Artifact(
        name=f"eval-results-{run.id}{step_tag}", type="eval_results",
    )
    artifact.add_file(args.results_file)
    run.log_artifact(artifact)

    run.finish()
    print(f"\n  [wandb] Results uploaded to run {run.id}")


if __name__ == "__main__":
    main()
