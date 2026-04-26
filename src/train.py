import torch
import torch.distributed.checkpoint
import torch.distributed.checkpoint.state_dict
import os
import time
import argparse
import data.utils_data
import models.utils_models
import models.transformer
import utils
import logging
torch._logging.set_logs(all=logging.ERROR)
import contextlib
import warnings
import nvtx
import datetime
warnings.filterwarnings("ignore", module="torch.distributed.checkpoint.state_dict_loader")
import json
import csvlogger
import microseconds_formatter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("NAME", help="Training log will be saved in NAME.csv", type=str)
parser.add_argument("--save_model", help="Save the model with the min validation loss in NAME.pt", type=utils.str_to_bool, default=True)
parser.add_argument("--resume", help="Continue training from the specified checkpoint. Can be the step number, 'latest', or an absolute path.", type=str, nargs="?", const="latest", default=None)
parser.add_argument("--checkpoint_freq", help="Every how many batches to save a checkpoint", type=int, default=utils.INF)
parser.add_argument("--info", help="Print information about the model", type=utils.str_to_bool, default=True)
parser.add_argument("--graph", help="Draw computational graph in NAME.pdf", type=utils.str_to_bool, default=False)
parser.add_argument("--test_parametrization", help="Print parametrization information", type=utils.str_to_bool, default=False)
parser.add_argument("--print_schedule", help="Print learning rate schedule", type=utils.str_to_bool, default=True)
parser.add_argument("--warning", type=utils.str_to_bool, default=True)
parser.add_argument("--extra_freq", help="Every how many batches to perform the extra evaluations", type=int, default=200)
parser.add_argument("--model_stats_freq", help="Every how many batches to log model stats", type=int, default=1000)
parser.add_argument("--rmse", help="As an extra, evaluate the train and validation Root Mean Squared Errors", type=utils.str_to_bool, default=False)
parser.add_argument("--nrmse", help="As an extra, evaluate the train and validation Normalized Root Mean Squared Errors", type=utils.str_to_bool, default=False)
parser.add_argument("--mae", help="As an extra, evaluate the train and validation Mean Absolute Errors", type=utils.str_to_bool, default=False)
parser.add_argument("--nmae", help="As an extra, evaluate the train and validation Normalized Mean Absolute Errors", type=utils.str_to_bool, default=False)
parser.add_argument("--r2", help="As an extra, evaluate the train and validation Coefficients of Determination", type=utils.str_to_bool, default=False)
parser.add_argument("--acc", help="As an extra, evaluate the train and validation accuracies", type=utils.str_to_bool, default=False)
parser.add_argument("--ppl", help="As an extra, evaluate the train and validation perplexities", type=utils.str_to_bool, default=False)
parser.add_argument("--lambada", help="As an extra, evaluate the LAMBADA (OpenAI version) accuracy", type=utils.str_to_bool, default=False)
parser.add_argument("--arc", help="As an extra, evaluate the ARC (easy) NORMALIZED accuracy", type=utils.str_to_bool, default=False)
parser.add_argument("--hellaswag", help="As an extra, evaluate the HellaSwag NORMALIZED accuracy", type=utils.str_to_bool, default=False)
parser.add_argument("--piqa", help="As an extra, evaluate the PIQA NORMALIZED accuracy", type=utils.str_to_bool, default=False)

parser.add_argument("--dataset", choices=data.utils_data.DATASETS, default="climbmix10m")
parser.add_argument("--vocab_size", type=int, default=32000)
parser.add_argument("--family", help="Model architecture", choices=models.utils_models.FAMILIES, default="transformer")
parser.add_argument("--parametrization", help="(a)bc parametrization, as defined in Tensor Programs IV (https://arxiv.org/abs/2011.14522). np (No Parametrization) means that the initialization is handled internally by the model.", choices=models.parametrizations.PARAMETRIZATIONS, default="np")
parser.add_argument("--zeta", help="Width scaling factor", type=int, default=16)
parser.add_argument("--scale_type", help="Scaling factor applied prior to softmax", choices=models.transformer.SCALE_TYPES, default="1/sqrt(d)")

parser.add_argument("--decoupling", help="Decouples c/k_input, c/k_hidden and c/k_output. If coupled, they are controlled by c/k_input.", type=utils.str_to_bool, default=False)
parser.add_argument("--c_input", type=float, default=0.02)
parser.add_argument("--c_hidden", type=float, default=0.5)
parser.add_argument("--c_output", type=float, default=0.5)
parser.add_argument("--opt", choices=models.parametrizations.OPTIMIZERS, default="adam")
parser.add_argument("--k_input", type=float, default=1e-3)
parser.add_argument("--k_hidden", type=float, default=1e-3)
parser.add_argument("--k_output", type=float, default=1e-3)
parser.add_argument("--scheduler", help="Learning rate schedule", choices=utils.SCHEDULERS, default="trapezoidal")
parser.add_argument("--warmup", help="Warmup steps; either a fraction or number of steps.", type=float, default=0.05)
parser.add_argument("--cooldown", help="Cooldown steps; either a fraction or number of steps.", type=float, default=0.2)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)
parser.add_argument("--beta3", type=float, default=0.98)
parser.add_argument("--alpha", type=float, default=5)
parser.add_argument("--gamma", type=float, default=0.025)
parser.add_argument("--eps", type=float, default=1e-8)

parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--label_smoothing", type=float, default=0)

parser.add_argument("--batch_size", help="Total batch size, over all GPUs and accumulations, for one gradient update", type=int, default=512)
parser.add_argument("--micro_batch_size", help="Batch size that fits in every GPU", type=int, default=32)
parser.add_argument("--context", type=int, default=1024)
parser.add_argument("--train_batches", help="The number of batches used during training", type=int, default=10_000)
parser.add_argument("--thresh", help="Keep the model that first crosses this threshold in NAME_thresh.pt", type=float, default=4.2)
parser.add_argument("--val_batches", help="The number of batches used during validation", type=int, default=20)

parser.add_argument("--model_device_index", help="CUDA device that stores the model", type=int, default=0)
parser.add_argument("--dataset_device_type", choices=["cpu", "cuda"], help="Device type that preloads the dataset", default="cpu")
parser.add_argument("--dtype", help="torch.dtype for Automatic Mixed Precision (AMP)", type=lambda x: getattr(torch, x), default="bfloat16")
parser.add_argument("--comp", help="Use torch.compile()", type=utils.str_to_bool, default=True)
parser.add_argument("--backend", help="Scaled Dot Product Attention (SDPA) backend", choices=models.transformer.BACKENDS, default="flash2")
parser.add_argument("--tokenizer_type", choices=data.utils_data.TOKENIZER_TYPES, help="Tokenizer library to use", default="tokenmonster")
parser.add_argument("--tokenizer", help="Name/URL/File of the tokenizer", default="https://huggingface.co/gvlassis/tokenmonster/resolve/main/englishcode-32000-strict-nocapcode-v1-eot%3D14199.vocab?download=true")
parser.add_argument("--eot_id", help="End-Of-Text token id", type=int, default=14199)

parser.add_argument("--quartet", help="quartet2.linear.Quartet_II_linear instead of torch.nn.Linear", type=utils.str_to_bool, default=True)
parser.add_argument("--fake_quartet", help="Fake (simulated) NVFP4 quantization", type=utils.str_to_bool, default=False)
parser.add_argument("--quartet_matmul_backend", help="Matmul backend for real Quartet-II linears", choices=["flashinfer", "qutlass", "dequantized"], default="flashinfer")
parser.add_argument("--quartet_weight_quantizer", help="Weight quantizer for real Quartet-II linears", choices=["four_six", "gridflip"], default="four_six")
parser.add_argument("--gridflip_shift", help="GridFlip shifted-grid correction value", type=float, default=0.25)
parser.add_argument("--wush", help="Apply blockwise WUSH transforms in Quartet-II forward quantization", type=utils.str_to_bool, default=False)
parser.add_argument("--wush_update_freq", help="Every how many optimizer steps to refresh WUSH transforms", type=int, default=200)
parser.add_argument("--wush_damp", help="Tikhonov damping for KFAC WUSH second-moment estimates", type=float, default=1e-3)
parser.add_argument("--wush_s_min", help="Singular value floor for KFAC WUSH transform updates", type=float, default=1e-2)
parser.add_argument("--wush_max_cond", help="Per-block condition-number fallback threshold for KFAC WUSH", type=float, default=1e4)
parser.add_argument("--wush_ema_decay", help="EMA decay for KFAC WUSH activation second moments", type=float, default=0.99)
parser.add_argument("--wush_group_size", help="Block size for KFAC WUSH transforms", type=int, default=128)
parser.add_argument("--wush_g_identity", help="Use identity output KFAC factor G when recomputing WUSH transforms", type=utils.str_to_bool, default=True)
parser.add_argument("--num_blocks", help="Number of Transformer blocks", type=int, default=4)
parser.add_argument("--heads", help="Number of Q heads in the MHSA", type=int, default=6)
parser.add_argument("--ratio", help="Ratio between Q heads and KV heads", type=int, default=3)
parser.add_argument("--tied_embeddings", help="Tie input and output embeddings", type=utils.str_to_bool, default=True)
parser.add_argument("--dataset_path", help="If passed, overrides where the dataset is loaded from", type=os.path.abspath, default=None)
parser.add_argument("--dataset_seed", help="Seed to use for dataset sampling.", type=int, default=-1)
parser.add_argument("--seed", help="Seed for model initialization and torch RNGs. Negative means do not set it.", type=int, default=-1)
parser.add_argument("--wandb_kwargs", help="Keyword arguments for wandb.init()", type=json.loads, default=None)
parser.add_argument("--val_fixed", help="As an extra, evaluate the loss on fixed validation batches", type=utils.str_to_bool, default=True)
args=parser.parse_args()

if args.quartet and args.fake_quartet:
    parser.error("--quartet and --fake_quartet are mutually exclusive")
if not args.quartet and args.quartet_matmul_backend != "flashinfer":
    parser.error("--quartet_matmul_backend applies only when --quartet true")
if not args.quartet and args.quartet_weight_quantizer != "four_six":
    parser.error("--quartet_weight_quantizer applies only when --quartet true")
if args.quartet_weight_quantizer == "gridflip" and args.quartet_matmul_backend == "flashinfer":
    parser.error("--quartet_weight_quantizer gridflip requires --quartet_matmul_backend qutlass or dequantized")
if args.gridflip_shift < 0:
    parser.error("--gridflip_shift must be non-negative")
if args.quartet_weight_quantizer == "gridflip" and args.comp:
    if "MASTER_ADDR" not in os.environ or int(os.getenv("RANK", 0)) == 0:
        print("📌 GridFlip's fused matmul adapter is not torch.compile-ready in this integration, so disabling --comp for this run.")
    args.comp = False
if args.wush and not args.quartet:
    parser.error("--wush currently applies to real Quartet-II linears; use --quartet true")
if args.wush and (args.wush_group_size & (args.wush_group_size - 1)):
    parser.error("--wush_group_size must be a power of two for the Hadamard transform")
if args.wush and args.comp:
    if "MASTER_ADDR" not in os.environ or int(os.getenv("RANK", 0)) == 0:
        print("📌 WUSH uses dynamic eigendecompositions, so disabling --comp for this run.")
    args.comp = False

if torch.distributed.is_torchelastic_launched():
    # Get environment variables set by torchrun
    WORLD_SIZE = int(os.getenv("WORLD_SIZE"))
    RANK = int(os.getenv("RANK"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK"))
    # Set global variables
    master = (RANK == 0)
    model_device_index = LOCAL_RANK
    model_device_type = "cuda"
    model_device = f"{model_device_type}:{model_device_index}"
    accumulation = args.batch_size//(WORLD_SIZE*args.micro_batch_size)

    # Set default "cuda" device (prevents weird bugs like with Triton, DistributedShampoo etc.) - BEFORE init()
    torch.cuda.set_device(model_device)

    # If the backend is not provided, then both a gloo and nccl backend will be created - AFTER set_device()
    torch.distributed.init_process_group(backend="nccl", device_id=torch.device(model_device), timeout=datetime.timedelta(minutes=5))
else:
    WORLD_SIZE = 1
    RANK = 0
    master = True
    model_device_index = args.model_device_index
    model_device_type = "cuda"
    model_device = f"{model_device_type}:{model_device_index}"
    accumulation = args.batch_size//args.micro_batch_size
    torch.cuda.set_device(model_device)

gpu_reservation_gib = max(0, int(os.getenv("CLOVERLM_GPU_RESERVATION_GIB", "0")))
if gpu_reservation_gib:
    # Optional keepalive for external GPU reservation monitors during CPU-side
    # dataset loading. The allocation is released before model initialization.
    _gpu_reservation_touch = torch.empty(gpu_reservation_gib * 1024 ** 3, device=model_device, dtype=torch.uint8)
else:
    _gpu_reservation_touch = None

subpath_dir = os.path.dirname(os.path.abspath(args.NAME))
if master: os.makedirs(subpath_dir, exist_ok=True)
checkpoint_path = args.NAME+"_checkpoint"
graph_path = args.NAME+".pdf"
log_path = args.NAME+".dat"
extra_path = args.NAME+"_extra.dat"
model_path = args.NAME+".pt"
thresh_path = args.NAME+"_thresh.pt"

resume_from = None

# can give absolute path to checkpoint, "latest", or a step number
if args.resume is not None and os.path.isabs(args.resume):
    resume_from = args.resume
elif args.resume == "latest":
    resume_from = utils.find_latest_checkpoint(checkpoint_path)
    if master and resume_from is None:
        print("📌 No checkpoints found to resume from. Starting training from scratch.")
elif args.resume is not None:
    resume_from = f"{checkpoint_path}/{args.resume}"

if resume_from is not None:
    if os.path.exists(resume_from):
        resume = True
        if master: print(f"📌 Resuming training from checkpoint {resume_from}")
    else:
        resume = False
        if master: print(f"📌 Checkpoint {resume_from} not found. Starting training from scratch.")
else:
    resume = False

checkpoint_dict = {"checkpoint": utils.Checkpoint()}

cols_extra = []
if args.rmse: cols_extra += ["train_rmse", "val_rmse"]
if args.nrmse: cols_extra += ["train_nrmse", "val_nrmse"]
if args.mae: cols_extra += ["train_mae", "val_mae"]
if args.nmae: cols_extra += ["train_nmae", "val_nmae"]
if args.r2: cols_extra += ["train_r2", "val_r2"]
if args.acc: cols_extra += ["train_acc", "val_acc"]
if args.ppl: cols_extra += ["train_ppl", "val_ppl"]

lm_eval_tasks = []
if args.lambada: lm_eval_tasks.append("lambada_openai")
if args.arc: lm_eval_tasks.append("arc_easy")
if args.hellaswag: lm_eval_tasks.append("hellaswag")
if args.piqa: lm_eval_tasks.append("piqa")
cols_extra += lm_eval_tasks

if args.val_fixed: cols_extra += ["val_loss_fixed"]

extra = cols_extra and args.extra_freq != utils.INF

if args.decoupling:
    c_input = args.c_input
    c_hidden = args.c_hidden
    c_output = args.c_output
    k_input = args.k_input
    k_hidden = args.k_hidden
    k_output = args.k_output
else:
    c_input = args.c_input
    c_hidden = args.c_input
    c_output = args.c_input
    k_input = args.k_input
    k_hidden = args.k_input
    k_output = args.k_input

if args.dataset_device_type == "cpu":
    dataset_device = "cpu"
elif args.dataset_device_type == "cuda":
    dataset_device = model_device

if args.seed >= 0:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

if master: print("💾 Loading dataset")
train_iterator = data.utils_data.get_iterator(args.dataset, "train", dataset_device, args.micro_batch_size, args.context, RANK, args.dataset_path, args.dataset_seed)
val_iterator = data.utils_data.get_iterator(args.dataset, "val", dataset_device, args.micro_batch_size, args.context, RANK, args.dataset_path, args.dataset_seed)
fixed_val_batch = next(val_iterator)
if _gpu_reservation_touch is not None:
    del _gpu_reservation_touch
    torch.cuda.empty_cache()

if args.quartet:
    import quartet2.linear
    quartet2.linear.set_fp4_mm_backend(args.quartet_matmul_backend)
    quartet2.linear.set_fp4_weight_quantizer(
        args.quartet_weight_quantizer,
        gridflip_shift=args.gridflip_shift,
    )

if master: print("🧠 Initializing model")
model_or_ddp, opts = models.utils_models.get_model_opts(
    args.vocab_size, args.family, args.parametrization, args.zeta,
    args.scale_type, c_input, c_hidden, c_output, k_input, k_hidden,
    k_output, args.opt, args.momentum, args.beta2, args.beta3, args.alpha,
    args.gamma, args.eps, args.weight_decay, args.context, args.test_parametrization and master,
    args.warning and master, args.backend, model_device, args.comp, args.quartet, args.fake_quartet,
    num_blocks=args.num_blocks, heads=args.heads, ratio=args.ratio, tied_embeddings=args.tied_embeddings)
model = model_or_ddp.module if torch.distributed.is_initialized() else model_or_ddp

if args.wush:
    import quartet2.linear
    quartet2.linear.configure_wush(model, True, args.wush_update_freq, args.wush_damp, args.wush_s_min,
                                   args.wush_max_cond, args.wush_ema_decay, args.wush_group_size,
                                   args.wush_g_identity)

checkpoint_dict["checkpoint"].model = model
checkpoint_dict["checkpoint"].opts = opts

if master and args.info:
    import fvcore.nn

    batch_X, _ = next(train_iterator)
    # Not having batch dimension can cause problems (e.g. BatchNorm)
    X = batch_X[:1]
    input_data = data.utils_data.transform(args.dataset, X.to(model_device))
    print(fvcore.nn.flop_count_table(fvcore.nn.FlopCountAnalysis(model, input_data), max_depth=3, show_param_shapes=False))

if master and args.graph:
    import torchview

    batch_X, _ = next(train_iterator)
    input_data = data.utils_data.transform(args.dataset, batch_X.to(model_device))
    torchview.draw_graph(model, input_data=input_data, depth=1, expand_nested=True, graph_dir="TB", show_shapes=True).visual_graph.render(cleanup=True, format="pdf", outfile=graph_path)

# Get the parameters' names before compile
cols_train_stats = models.utils_models.get_cols_train_stats(model)

# float16 (not bfloat16) needs scaling
scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype==torch.float16))
checkpoint_dict["checkpoint"].scaler = scaler

if resume:
    # as we replay the lr-scheduler later on, we need to make sure loading the checkpoint
    # does not change the configured learning rates, so we stash and restore
    initial_lrs = utils.extract_initial_lrs(opts)
    if master: print("⏳ Loading checkpoint")
    torch.distributed.checkpoint.load(checkpoint_dict, checkpoint_id=resume_from)
    utils.restore_initial_lrs(opts, initial_lrs)


schedulers = []
for opt in opts:
    scheduler = utils.get_scheduler(args.scheduler, opt, args.train_batches, args.warmup, args.cooldown)
    if args.print_schedule and master: utils.print_schedule(args.train_batches, scheduler)

    # NOTE: torch.optim.lr_scheduler has a start_batch __init__ argument, but for `SequentialLR` this
    # is buggy. So we resort to literally replaying the schedule until we arrive at the desired step.
    utils.advance_scheduler(scheduler, checkpoint_dict["checkpoint"].train_batch)
    schedulers.append(scheduler)

# register optimizer hooks for abs-max
if args.quartet:
    for opt in opts:
        import quartet2
        quartet2.linear.register_optimizer_hook(model, opt)

# Compile last
if args.comp:
    if master: print("⏳ Compiling")
    # mode=max-autotune gives NaN
    get_loss = torch.compile(data.utils_data.get_loss)
    x, y = next(train_iterator)
    x = x.to(model_device)
    y = y.to(model_device)
    with torch.autocast(device_type=model_device_type, dtype=args.dtype):
        get_loss(args.dataset, model_or_ddp, x, y, args.label_smoothing)
else:
    get_loss = data.utils_data.get_loss

# make sure all nodes have done their setup before we enter the main loop
if torch.distributed.is_initialized(): torch.distributed.barrier()
if master: print("🚀 Setup finished on all nodes")

if master:
    logger = csvlogger.Logger("train_batch", "tokens", "lr0", "train_loss", "train_batch_time", "train_opt_time", name=args.NAME, resume=resume,
                              stdout_track_min=["train_loss"], stdout_flag=False,
                              wandb_flag=bool(args.wandb_kwargs), wandb_kwargs=args.wandb_kwargs)

    if args.wandb_kwargs:
        import wandb
        wandb.define_metric("train_batch")
        wandb.define_metric("*", step_metric="train_batch")

    if extra:
        logger_extra = csvlogger.Logger(*(["train_batch", "tokens", "train_loss"] + cols_extra + ["toks/sec", "toks/sec/gpu", "ETA"]), name=args.NAME+"_extra", resume=resume,
                                        stdout_track_min=["train_loss", "val_loss_fixed"], stdout_init_flag=True,
                                        wandb_flag=bool(args.wandb_kwargs), wandb_kwargs=args.wandb_kwargs)

    logger_train_stats = csvlogger.Logger(*(["train_batch", "tokens"] + cols_train_stats), name=args.NAME+"_train_stats", resume=resume,
                                          stdout_flag=False)

loss_scale_acc = torch.tensor(1.0 / accumulation).to(model_device)

tokens_per_batch = args.batch_size * args.context
eta_estimator = utils.ProgressTracker(checkpoint_dict["checkpoint"].train_batch, args.train_batches, WORLD_SIZE, tokens_per_batch)

step_timer = utils.CudaEventTimer()
opt_timer = utils.CudaEventTimer()

while checkpoint_dict["checkpoint"].train_batch < args.train_batches:
    step_timer.start()
    model_or_ddp.train()
    train_loss = torch.tensor(0.0).to(model_device)
    for micro_train_batch in range(accumulation):
        with nvtx.annotate("forward", color="green"):
            batch_train_X, batch_train_Y = next(train_iterator)
            batch_train_X = batch_train_X.to(model_device)
            batch_train_Y = batch_train_Y.to(model_device)

            # Only sync gradients in the last micro_train_batch
            with (model_or_ddp.no_sync() if torch.distributed.is_initialized() and micro_train_batch<accumulation-1 else contextlib.nullcontext()):
                with torch.autocast(device_type=model_device_type, dtype=args.dtype):
                    micro_train_loss = get_loss(args.dataset, model_or_ddp, batch_train_X, batch_train_Y, args.label_smoothing)[1] * loss_scale_acc
                    train_loss += micro_train_loss.detach()
                retain_accumulation_graph = args.quartet and micro_train_batch < accumulation - 1
                scaler.scale(micro_train_loss).backward(retain_graph=retain_accumulation_graph)

    step_timer.end()
    
    if torch.distributed.is_initialized(): torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.AVG)
    train_loss = train_loss.item()
    last_batch = args.train_batches-1

    is_extra_log_step = checkpoint_dict["checkpoint"].train_batch > 1 and ((checkpoint_dict["checkpoint"].train_batch % args.extra_freq == 0) or (checkpoint_dict["checkpoint"].train_batch == last_batch))

    # if requested, calculate validation sample using all workers
    if is_extra_log_step and args.val_fixed:
        val_loss_fixed = data.utils_data.get_batch_loss(fixed_val_batch[0], fixed_val_batch[1], args.dataset, model, args.dtype)
        if torch.distributed.is_initialized(): torch.distributed.all_reduce(val_loss_fixed, op=torch.distributed.ReduceOp.AVG)
        val_loss_fixed = val_loss_fixed.item()

    # TODO actually keep track of this, in case we have batch-size schedules
    total_toks = (checkpoint_dict["checkpoint"].train_batch + 1) * tokens_per_batch

    if master:
        if is_extra_log_step:
            if extra:
                vals_extra = [checkpoint_dict["checkpoint"].train_batch, total_toks, train_loss]

                if args.rmse:
                    train_rmse = data.utils_data.approximate_rmse(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
                    val_rmse = data.utils_data.approximate_rmse(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
                    vals_extra += [train_rmse, val_rmse]

                if args.nrmse:
                    train_nrmse = data.utils_data.approximate_nrmse(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
                    val_nrmse = data.utils_data.approximate_nrmse(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
                    vals_extra += [train_nrmse, val_nrmse]

                if args.mae:
                    train_mae = data.utils_data.approximate_mae(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
                    val_mae = data.utils_data.approximate_mae(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
                    vals_extra += [train_mae, val_mae]

                if args.nmae:
                    train_nmae = data.utils_data.approximate_nmae(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
                    val_nmae = data.utils_data.approximate_nmae(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
                    vals_extra += [train_nmae, val_nmae]

                if args.r2:
                    train_r2 = data.utils_data.approximate_r2(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
                    val_r2 = data.utils_data.approximate_r2(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
                    vals_extra += [train_r2, val_r2]

                if args.acc:
                    train_acc = data.utils_data.approximate_acc(args.val_batches, train_iterator,  args.dataset, model, args.dtype)*100
                    val_acc = data.utils_data.approximate_acc(args.val_batches, val_iterator,  args.dataset, model, args.dtype)*100
                    vals_extra += [train_acc, val_acc]

                if args.ppl:
                    train_ppl = data.utils_data.approximate_ppl(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
                    val_ppl = data.utils_data.approximate_ppl(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
                    vals_extra += [train_ppl, val_ppl]

                if lm_eval_tasks:
                    import lm_eval

                    if args.tokenizer_type=="tokenizers":
                        import transformers
                        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(args.tokenizer).backend_tokenizer
                    elif args.tokenizer_type=="tokenmonster":
                        import tokenmonster
                        tokenizer = tokenmonster.load(args.tokenizer)


                    lm_eval_results = lm_eval.simple_evaluate(model = data.utils_data.lm_eval_wrapper(args.tokenizer_type, tokenizer, args.eot_id, model, args.dtype),
                                                              tasks = lm_eval_tasks,
                                                              num_fewshot = 0,
                                                              batch_size = 1, # Higher is not supported
                                                              device = model_device, # Defaults to cuda
                                                              use_cache = None, # Do NOT cache results
                                                              rewrite_requests_cache = True, # Dataset requests cache
                                                              limit = None, # Total samples. Be careful, some datasets are very noisy and/or not even shuffled
                                                              bootstrap_iters = 100, # Default=100_000, with bootstrap_iters=min(bootstrap_iters, 100)
                                                              log_samples = False,
                                                              verbosity = "ERROR",
                                                              random_seed = None, # Do NOT fix random seed
                                                              numpy_random_seed = None,
                                                              torch_random_seed = None,
                                                              fewshot_random_seed = None)

                if args.lambada:
                    lambada = lm_eval_results["results"]["lambada_openai"]["acc,none"]*100
                    vals_extra += [lambada]

                if args.arc:
                    arc = lm_eval_results["results"]["arc_easy"]["acc_norm,none"]*100
                    vals_extra += [arc]

                if args.hellaswag:
                    hellaswag = lm_eval_results["results"]["hellaswag"]["acc_norm,none"]*100
                    vals_extra += [hellaswag]

                if args.piqa:
                    piqa = lm_eval_results["results"]["piqa"]["acc_norm,none"]*100
                    vals_extra += [piqa]

                if args.val_fixed:
                    vals_extra += [val_loss_fixed]

                speed = eta_estimator.update(checkpoint_dict["checkpoint"].train_batch + 1)
                vals_extra += [round(speed.toks_sec), round(speed.toks_sec_gpu), microseconds_formatter.adaptive(speed.eta * 1e6)]
                logger_extra.log(*vals_extra)

        if checkpoint_dict["checkpoint"].train_batch % args.model_stats_freq == 0 or checkpoint_dict["checkpoint"].train_batch == last_batch:
            vals_train_stats = [checkpoint_dict["checkpoint"].train_batch, total_toks]
            vals_train_stats += models.utils_models.get_vals_train_stats(model)
            logger_train_stats.log(*vals_train_stats)

    opt_timer.start()
    with nvtx.annotate("optimizer", color="yellow"):
        for opt in opts:
            scaler.step(opt)

    for opt in opts:
        opt.zero_grad()
    scaler.update()
    opt_timer.end()

    next_train_batch = checkpoint_dict["checkpoint"].train_batch + 1
    if args.wush and next_train_batch > 0 and next_train_batch % args.wush_update_freq == 0:
        import quartet2.linear
        n_updated = quartet2.linear.update_wush_transforms(model, sync_distributed=torch.distributed.is_initialized())
        if master:
            print(f"📌 WUSH recomputed transforms for {n_updated} Quartet-II layers at step {next_train_batch}")

    lr = schedulers[0].get_last_lr()[0]
    for scheduler in schedulers:
        scheduler.step()

    train_batch_time = step_timer.elapsed()
    checkpoint_dict["checkpoint"].train_time += train_batch_time

    # note: delay logging until here, so that
    if master:
        logger.log(checkpoint_dict["checkpoint"].train_batch, total_toks, lr, train_loss, microseconds_formatter.adaptive(train_batch_time), microseconds_formatter.adaptive(opt_timer.elapsed()))

    checkpoint_dict["checkpoint"].train_batch += 1

    current_batch = checkpoint_dict["checkpoint"].train_batch
    if args.checkpoint_freq != utils.INF and current_batch > 1 and ((current_batch % args.checkpoint_freq == 0) or (current_batch == last_batch)):
        checkpoint_id = f"{checkpoint_path}/{current_batch}"
        torch.distributed.checkpoint.save(checkpoint_dict, checkpoint_id=checkpoint_id)
        if master:
            latest_checkpoint_id = f"{checkpoint_path}/latest"
            if os.path.islink(latest_checkpoint_id):
                os.unlink(latest_checkpoint_id)
            os.symlink(checkpoint_id, latest_checkpoint_id)

if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
exit(0)
