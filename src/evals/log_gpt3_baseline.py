
import argparse
import wandb

GPT3_BASELINES = {
    "hellaswag/acc_norm":              0.789,
    "lambada_openai_norm/acc":         0.762,
    "lambada_openai_norm/perplexity":  3.00,
    "nq_open/exact_match":             0.146,
    "piqa/acc_norm":                   0.810,
    "arc_challenge_mi/acc_mutual_info": 0.514,
    "arc_easy_mi/acc_mutual_info":     0.688,
    "coqa/f1":                         0.815,
}


def compute_averages(metrics):
    """Same aggregation logic as upload_wandb.py."""
    acc_norm_keys = [k for k in metrics
                     if ("acc_norm" in k or "acc_mutual_info" in k)
                     and "stderr" not in k]
    acc_raw_keys = [k for k in metrics
                    if k.endswith("/acc") and "norm" not in k
                    and "stderr" not in k]
    f1_keys = [k for k in metrics if k.endswith("/f1") and "stderr" not in k]
    em_keys = [k for k in metrics if k.endswith("/exact_match") and "stderr" not in k]

    avgs = {}
    if acc_norm_keys:
        avgs["avg_acc_norm"] = sum(metrics[k] for k in acc_norm_keys) / len(acc_norm_keys)
    if acc_raw_keys:
        avgs["avg_acc"] = sum(metrics[k] for k in acc_raw_keys) / len(acc_raw_keys)
    composite_keys = acc_norm_keys + f1_keys + em_keys
    if composite_keys:
        avgs["avg_score"] = sum(metrics[k] for k in composite_keys) / len(composite_keys)
    return avgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wandb_project", default="expedition44_test")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_id", default="gpt3-175b-baseline")
    p.add_argument("--wandb_run_name", default="GPT-3 175B baseline")
    p.add_argument("--processed_file",
                   default="/home/matin/Expedition44/.auto_eval_acc_processed_steps",
                   help="File listing the training steps already evaluated")
    p.add_argument("--extra_max_step", type=int, default=590_000,
                   help="Extend the baseline line out to this step")
    p.add_argument("--delete_existing", action="store_true",
                   help="Delete the existing wandb run first (needed to re-log at old steps)")
    args = p.parse_args()

    steps = set()
    with open(args.processed_file) as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                steps.add(int(line))

    if not steps:
        print("No steps found in processed file, nothing to log.")
        return

    min_step = 0
    max_step = args.extra_max_step
    all_steps = sorted(steps | {min_step, max_step})

    if args.delete_existing:
        api = wandb.Api()
        try:
            old_run = api.run(f"{args.wandb_project}/{args.wandb_run_id}")
            old_run.delete()
            print(f"Deleted existing run {args.wandb_run_id}")
        except wandb.errors.CommError:
            print(f"No existing run {args.wandb_run_id} found, creating fresh")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="allow",
        config={"model": "GPT-3 175B", "type": "baseline"},
        settings=wandb.Settings(init_timeout=300),
    )

    run.define_metric("*", step_metric="train_step")

    avgs = compute_averages(GPT3_BASELINES)
    print(f"Computed averages: { {k: f'{v*100:.2f}%' for k, v in avgs.items()} }")

    for step in all_steps:
        payload = dict(GPT3_BASELINES)
        payload.update(avgs)
        payload["train_step"] = step
        run.log(payload, step=step)

    run.finish()
    print(f"Logged GPT-3 175B baselines at {len(all_steps)} steps to run {run.id}")


if __name__ == "__main__":
    main()
