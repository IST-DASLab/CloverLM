from typing import List

import torch
import os
import re
import time
import unicodedata
import copy
import numpy
import scipy
import sklearn.metrics
import models.utils_models
import data.utils_data
import warnings
warnings.filterwarnings("ignore", module="torch.optim.lr_scheduler")
import torch.distributed.checkpoint
import torch.distributed.checkpoint.state_dict

from pathlib import Path
from collections import namedtuple

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)
root_path = os.path.dirname(src_path)
out_path = root_path + "/out"

SCHEDULERS = ["trapezoidal", "1cycle", "cos", "constant"]
INF = 10**15

def get_files(root):
    root = os.path.abspath(root)

    files_abs = []
    for directory, _, files_rel in os.walk(root):
        for file in files_rel:
            files_abs.append(directory+"/"+file)

    return files_abs

def match_list(_list, regex, group):
    list_matched = set()
    
    for elem in _list:
        match = re.search(regex, elem)
        if match:
            list_matched.add(match.group(group))

    return list(list_matched)

def get_subdir(path):
    return [child for child in os.listdir(path) if os.path.isdir(path+"/"+child)]

def get_subdat(path):
    return [child[:-4] for child in os.listdir(path) if child[-4:]==".dat"]

def numel(elem):
    if type(elem) == list:
        return sum(numel(subelem) for subelem in elem)
    else:
        return 1

def str_to_bool(string):
    if string in ["True", "true", "1"]:
        boolean = True
    elif string in ["False", "false", "0"]:
        boolean = False

    return boolean

# Time logging causes negligible performance impact (~3%)
def get_sync_time(device):
    torch.cuda.synchronize(device)

    return round(time.time()*1e6)

# Based on https://github.com/gvlassis/bashrc_utils/blob/main/src/utils.sh
def us_to_human_friendly(micros):
    ms = micros//1000

    s = ms//1000
    
    rem_s = s%60
    m = s//60
    
    rem_m = m%60
    h = m//60

    rem_h = h%24
    d = h//24

    if d > 0:
        human_friendly = "%dd%dh" % (d,rem_h)
    elif h > 0:
        human_friendly = "%dh%dm" % (h,rem_m)
    elif m > 0:
        human_friendly = "%dm%ds" % (m,rem_s)
    elif s > 0:
        # s=AB, ms=ABCDE
        if s > 10:
            human_friendly = "%d.%ss" % (s,str(ms)[2])
        # s=A, ms=ABCD
        else:
            human_friendly = "%d.%ss" % (s,str(ms)[1:3])
    elif ms > 0:
        # ms=ABC
        if ms > 100:
            human_friendly = "%dms" % ms
        # ms=AB, micros=ABCDE
        elif ms > 10:
            human_friendly = "%d.%sms" % (ms,str(micros)[2])
        # ms=A, micros=ABCD
        else:
            human_friendly = "%d.%sms" % (ms,str(micros)[1:3])
    else:
        human_friendly = "%dμs" % micros

    return human_friendly

def topp(A, P):
    A_values, A_indices = torch.sort(A, descending=True)

    A_cumsum = torch.cumsum(A_values, dim=0)

    indices = A_indices[A_cumsum < P]

    indices = A_indices[:len(indices)+1]

    return A[indices], indices

def intersection(A, B):
    uniques, counts = torch.cat((A, B)).unique(return_counts=True)
    
    return uniques[counts>1]

def generate_text(starting_string, tokenizer, unk_id, eot_id, model, context=128, max_tokens=128, T=1, K=50, P=0.95):
    string = starting_string

    device = next(model.parameters()).device

    ids = tokenizer.encode(starting_string).ids

    while ( ids[-1] != eot_id ) and ( len(ids) < max_tokens ):

        X = torch.tensor(ids[-context:])

        model.eval()
        with torch.no_grad():
            Y = model( X.to(device) )
        
        Y[-1][unk_id] = -float("inf")

        Y_ = torch.softmax(Y[-1]/T, dim=0)
        
        topk_indices = torch.topk(Y_, K).indices
        topp_indices = topp(Y_, P)[1]
        indices = intersection(topk_indices, topp_indices)
        
        Y__ = torch.zeros_like(Y_)
        Y__[indices] = Y_[indices]

        ids = ids + [torch.multinomial(Y__, num_samples=1).item()]
    
    string = tokenizer.decode(ids)

    string = unicodedata.normalize("NFKC", string)
    
    print(string)

def split_to_ints(total, ratios):
    # final or final-1
    floored = [int(ratio*total) for ratio in ratios]

    total_floored = sum(floored)
    remainder = total - total_floored

    # Distribute remainder to floored with biggest decimals
    decimals = [ratios[i]*total-floored[i] for i in range(len(ratios))]
    indices = numpy.argsort(decimals)[:(-remainder-1):-1]
    for i in indices:
        floored[i] += 1

    assert total==sum(floored)

    return floored

def get_scheduler(scheduler, optimizer, batches, warmup, cooldown):
    if scheduler == "trapezoidal":
        # steps instead of fraction
        if warmup > 1:
            assert warmup == int(warmup)
            warmup = warmup / batches

        if cooldown > 1:
            assert cooldown == int(cooldown)
            cooldown = cooldown / batches

        main = 1.0 - warmup - cooldown
        warmup_iters, constant_iters, cooldown_iters = split_to_ints(batches, [warmup, main, cooldown])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                          [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1, total_iters=warmup_iters),
                                                          torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=constant_iters),
                                                          torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=cooldown_iters)],
                                                          milestones=[warmup_iters, warmup_iters+constant_iters])
    elif scheduler == "1cycle":
        warmup_iters, cooldown_iters = split_to_ints(batches, [0.50, 0.50])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                          [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1, total_iters=warmup_iters),
                                                          torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=cooldown_iters)],
                                                          milestones=[warmup_iters])
    elif scheduler == "cos":
        warmup_iters, annealing_iters = split_to_ints(batches, [0.05, 0.95])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                          [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1, total_iters=warmup_iters),
                                                          torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=annealing_iters, eta_min=0)],
                                                          milestones=[warmup_iters])
    elif scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=batches)

    return scheduler


def advance_scheduler(scheduler, steps):
    for _ in range(steps):
        scheduler.step()


def print_schedule(train_batches, scheduler):
    import plotext

    scheduler = copy.deepcopy(scheduler)
    
    lrs = []
    for train_batch in range(train_batches):
        # optimizer.step()
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    plotext.plot_size(width=plotext.terminal_width()/2, height=plotext.terminal_height()/4)
    plotext.theme("pro")
    plotext.xlabel("train_batch")
    plotext.xaxes(lower=True, upper=False)
    plotext.yaxes(left=True, right=False)
    plotext.xticks([0,train_batches*0.25,train_batches*0.5,train_batches*0.75,train_batches])
    
    plotext.plot(range(train_batches), lrs, marker="braille", label="lr")
    plotext.show()
    plotext.clear_figure()

# (...*)rows*cols
def entropy(X, normalized=True, eps=1e-9):
    rows = X.shape[-2]

    # (...*)rows
    entropies = -torch.sum(X*torch.log(X+eps), dim=-1)

    maximums = torch.arange(1, rows+1).log()
    if normalized:
        denominators = maximums
        # Prevents 0/0
        denominators[...,0] = 1 
    else:
        denominators = 1

    return (entropies/denominators).mean()

# (...*)rows*cols
def l2(X):
    rows = X.shape[-2]

    # (...*)rows
    l2s = X.norm(dim=-1)

    return l2s.mean()

# (...*)rows*cols
def invsimp(X, normalized=True):
    rows = X.shape[-2]

    # (...*)rows
    invsimps = 1/torch.sum(X**2, dim=-1)

    maximums = torch.arange(1, rows+1)
    denominators = maximums if normalized else 1

    return (invsimps/denominators).mean()

# (...*)rows*cols
def thresh(X, thresh, normalized=True):
    rows = X.shape[-2]

    # (...*)rows
    threshs = torch.sum(X>thresh, dim=-1)

    maximums = torch.arange(1, rows+1)
    denominators = maximums if normalized else 1

    return (threshs/denominators).mean()

def inter(X, Y, balanced=False):
    sim = 0

    classes = numpy.unique(Y)
    pairs = 0
    for cls in classes:
        X_intra = X[Y == cls]
        X_inter = X[Y != cls]

        dists = sklearn.metrics.pairwise_distances(X_intra, X_inter, metric="cosine")
        sims = 1 - dists
        pairs_intra = sims.size
        pairs += pairs_intra

        sim += sims.mean() if balanced else sims.sum()

    return sim/len(classes) if balanced else sim/pairs

def intra(X, Y, balanced):
    sim = 0

    classes = numpy.unique(Y)
    pairs = 0
    for cls in classes:
        X_intra = X[Y == cls]

        dists = sklearn.metrics.pairwise_distances(X_intra, metric="cosine")
        sims = 1 - dists
        pairs_intra = (X_intra.shape[0]-1)*X_intra.shape[0] / 2
        pairs += pairs_intra

        sim += numpy.triu(sims, k=1).sum()/pairs_intra if balanced else numpy.triu(sims, k=1).sum()

    return sim/len(classes) if balanced else sim/pairs

def write_featsdist(vocab_size, family, parametrization, scale_type, ζ, context, arch, device, dataset, batch_X, block, bins, density):    
    featsdist_path = "%s/%s%dfeatsdist.dat" % (out_path, arch, ζ)
    with open(featsdist_path, "w") as file:
        file.write("x y\n")

    model, _ = models.utils_models.get_model_optimizer(vocab_size, family, parametrization, scale_type, ζ, 0.02, 0.5, 0.5, 0.001, 0.001, 0.001, "adam", 0, False, (0.9, 0.95), 0, context, False, True)
    model_path = "%s/%s%d.pt" % (out_path, arch, ζ)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(data.utils_data.transform(dataset, batch_X.to(device)))
    
    # (batches*)context*d
    feats = embeddings[...,block,:,:].abs()
    
    print("%2.2s %12.12s %8.8s %8.8s %10.10s" % (ζ, arch, "%.2f" % feats.mean(), "%.2f" % feats.std(), "%.2f" % feats.max()))

    feats = feats.log10()
    hist, bin_edges = numpy.histogram(feats, bins=bins, range=(-1,5), density=density)

    for x, y in zip(bin_edges, hist):
        with open(featsdist_path, "a") as file:
            file.write("%f %f\n" % (x, y))

def write_heat(vocab_size, family, parametrization, scale_type, ζ, context, arch, device, dataset, batch_X, block, run):    
    heat_path = "%s/%sheat.dat" % (out_path, run)
    with open(heat_path, "w") as file:
        file.write("x y z\n")

    model, _ = models.utils_models.get_model_opts(vocab_size, family, parametrization, ζ, scale_type, 0.02, 0.5, 0.5, 0.001, 0.001, 0.001, "adam", 0.9, 0.95, 0.98, 5, 0.025, 1e-8, 0, context, False, True, False, "pytorch", "cuda:0", False)
    model_path = "%s/%s.pt" % (out_path, run)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cuda:0"))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        Y, embeddings = model(data.utils_data.transform(dataset, batch_X.to(device)), return_emb=True)
    
    # context*d
    feats = embeddings[...,block,:,:].mean(dim=0)

    std1 = ((feats.mean()-1*feats.std()<feats) & (feats<feats.mean()+1*feats.std())).sum()*100/feats.numel()
    std2 = ((feats.mean()-2*feats.std()<feats) & (feats<feats.mean()+2*feats.std())).sum()*100/feats.numel()
    std3 = ((feats.mean()-3*feats.std()<feats) & (feats<feats.mean()+3*feats.std())).sum()*100/feats.numel()
    skew = scipy.stats.skew(feats.flatten().tolist())
    kurt = scipy.stats.kurtosis(feats.flatten().tolist())
    # See https://arxiv.org/abs/2405.19279
    s = (feats**2).mean(dim=0).sqrt()
    kurtrms = scipy.stats.kurtosis(s.tolist())
    agg = feats.abs().max(dim=1).values/feats.abs().median(dim=1).values
    mmr = agg.mean()

    print("%2.2s %8.8s %8.8s %12.12s %8.8s %8.8s %10.10s %10.10s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s" % (ζ, feats.shape[1], feats.shape[0], arch, "%.2f" % feats.mean(), "%.2f" % feats.std(), "%.2f" % feats.min(), "%.2f" % feats.max(), "%.2f%%" % std1, "%.2f%%" % std2, "%.2f%%" % std3, "%.2f" % skew, "%.2f" % kurt, "%.2f" % kurtrms, "%.2f" % mmr))
    
    # for feat in range(feats.shape[1]):
    #     for token in range(feats.shape[0]):
    #         with open(heat_path, "a") as file:
    #             file.write("%d %d %f\n" % (token, feat, feats[token,feat]))
    #
    #     with open(heat_path, "a") as file:
    #         file.write("\n")

def write_cumexpvar(vocab_size, family, parametrization, scale_type, ζ, context, arch, device, dataset, batch_X, block):    
    cumexpvar_path = "%s/%s%dcumexpvar.dat" % (out_path, arch, ζ)
    with open(cumexpvar_path, "w") as file:
        file.write("x y\n")

    model, _ = models.utils_models.get_model_optimizer(vocab_size, family, parametrization, scale_type, ζ, 0.02, 0.5, 0.5, 0.001, 0.001, 0.001, "adam", 0, False, (0.9, 0.95), 0, context, False, True)
    model_path = "%s/%s%d.pt" % (out_path, arch, ζ)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(data.utils_data.transform(dataset, batch_X.to(device)))
    
    # context*d
    feats = embeddings[...,block,:,:].mean(dim=0)

    sings = torch.linalg.svdvals(feats)
    eigs = sings**2
    cumexpvar = eigs.cumsum(dim=0)*100/eigs.sum()
    
    print("%2.2s %8.8s %12.12s %12.12s %12.12s %12.12s" % (ζ, feats.shape[1], arch, "%.2f%%" % cumexpvar[0], "%.2f%%" % cumexpvar[4], "%.2f%%" % cumexpvar[9]))

    for x, y in enumerate(cumexpvar):
        with open(cumexpvar_path, "a") as file:
            file.write("%d %f\n" % (x, y.item()))

def write_eigsdist(vocab_size, family, parametrization, scale_type, ζ, context, arch, device, dataset, batch_X, block, bins, density):    
    eigsdist_path = "%s/%s%deigsdist.dat" % (out_path, arch, ζ)
    with open(eigsdist_path, "w") as file:
        file.write("x y\n")

    model, _ = models.utils_models.get_model_optimizer(vocab_size, family, parametrization, scale_type, ζ, 0.02, 0.5, 0.5, 0.001, 0.001, 0.001, "adam", 0, False, (0.9, 0.95), 0, context, False, True)
    model_path = "%s/%s%d.pt" % (out_path, arch, ζ)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(data.utils_data.transform(dataset, batch_X.to(device)))
    
    # context*d
    feats = embeddings[...,block,:,:].mean(dim=0)

    sings = torch.linalg.svdvals(feats)
    eigs = sings**2
    
    print("%2.2s %12.12s %14.14s %14.14s %16.16s %16.16s" % (ζ, arch, "%.2f" % eigs.mean(), "%.2f" % eigs.std(), "%.2f" % eigs[-1], "%.2f" % eigs[0]))
    
    eigs = eigs.log10()
    hist, bin_edges = numpy.histogram(eigs, bins=bins, range=(1,10), density=density)

    for x, y in zip(bin_edges, hist):
        with open(eigsdist_path, "a") as file:
            file.write("%f %f\n" % (x, y))

# Our checkpoint does NOT include RNG seeds and dataset state.
class Checkpoint(torch.distributed.checkpoint.stateful.Stateful):

    # Built gradually
    def __init__(self):
        super().__init__()
        
        # Do not have state_dict()
        self.thresh_crossed = False
        self.train_batch = 0
        self.train_time = 0
        self.min_train_loss = INF
        self.min_val_loss = INF
        
        # Have state_dict()
        self.model = None
        self.opts = None
        self.scaler = None

    def state_dict(self):
        model_state_dict = torch.distributed.checkpoint.state_dict.get_model_state_dict(self.model)

        opts_state_dict = []
        for i in range(len(self.opts)):
            if self.opts[i].__class__.__name__ == "DistributedShampoo":
                opt_state_dict = self.opts[i].distributed_state_dict(key_to_param=self.model.named_parameters())
            else:
                opt_state_dict = torch.distributed.checkpoint.state_dict.get_optimizer_state_dict(self.model, self.opts[i])

            opts_state_dict.append(opt_state_dict)

        scaler_state_dict = self.scaler.state_dict()

        return {
            "thresh_crossed": self.thresh_crossed,
            "train_batch": self.train_batch,
            "train_time": self.train_time,
            "min_train_loss": self.min_train_loss,
            "min_val_loss": self.min_val_loss,
            "model": model_state_dict,
            "opts": opts_state_dict,
            "scaler": scaler_state_dict,
        }
    
    def load_state_dict(self, state_dict):
        self.thresh_crossed = state_dict["thresh_crossed"]
        self.train_batch = state_dict["train_batch"]
        self.train_time = state_dict["train_time"]
        self.min_train_loss = state_dict["min_train_loss"]
        self.min_val_loss = state_dict["min_val_loss"]
        
        torch.distributed.checkpoint.state_dict.set_model_state_dict(self.model, state_dict["model"])

        for i in range(len(self.opts)):
            if self.opts[i].__class__.__name__ == "DistributedShampoo":
                self.opts[i].load_distributed_state_dict(state_dict["opts"][i], key_to_param=self.model.named_parameters())
            else:
                torch.distributed.checkpoint.state_dict.set_optimizer_state_dict(self.model, self.opts[i], state_dict["opts"][i])

        self.scaler.load_state_dict(state_dict["scaler"])


def find_latest_checkpoint(path):
    path = Path(path)
    candidates = set()
    if path.exists():
        for subdir in path.iterdir():
            if not subdir.is_dir():
                continue
            try:
                candidates.add(int(subdir.name))
            except ValueError:
                continue

    return str(path / str(max(candidates)) if candidates else None)


TrainingSpeedEstimator = namedtuple("TrainingSpeedEstimator", ["toks_sec", "toks_sec_gpu", "eta"])

class ProgressTracker:
    def __init__(self, init_batch: int, last_batch: int, world_size: int, tokens_per_batch: int):
        self.last_batch = last_batch
        self.world_size = world_size
        self.tokens_per_batch = tokens_per_batch

        self._last_wall_time = time.perf_counter()
        self._last_batch_id = init_batch

    def update(self, current_batch):
        now = time.perf_counter()
        duration = now - self._last_wall_time
        num_batches = current_batch - self._last_batch_id

        self._last_wall_time = now
        self._last_batch_id = current_batch

        seconds_per_batch = duration / num_batches
        rem_batches = self.last_batch - current_batch
        eta = seconds_per_batch * rem_batches

        toks_sec = self.tokens_per_batch / seconds_per_batch
        toks_sec_gpu = toks_sec / self.world_size

        return TrainingSpeedEstimator(toks_sec, toks_sec_gpu, eta)


def extract_initial_lrs(opts: List[torch.optim.Optimizer]):
    initial_lr_values = []
    for opt in opts:
        initial_lr_list = []
        for group in opt.param_groups:
            initial_lr = group["lr"]
            if isinstance(initial_lr, torch.Tensor):
                initial_lr = initial_lr.clone()
            initial_lr_list.append(initial_lr)
        initial_lr_values.append(initial_lr_list)
    return initial_lr_values


def restore_initial_lrs(opts: List[torch.optim.Optimizer], initial_lr_values: List[List[float|torch.Tensor]]):
    for opt, initial_lr_list in zip(opts, initial_lr_values):
        if len(opt.param_groups) != len(initial_lr_list):
            raise ValueError(
                f"Mismatch between number of param groups ({len(opt.param_groups)}) and "
                f"initial learning rates ({len(initial_lr_list)}) for optimizer {opt}."
            )
        for group, initial_lr in zip(opt.param_groups, initial_lr_list):
            group["lr"] = initial_lr
            group.pop("initial_lr", None)


class CudaEventTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.synced = False

    def start(self):
        self.start_event.record()
        self.synced = False

    def end(self):
        self.end_event.record()

    def elapsed(self):
        if not self.synced:
            torch.cuda.synchronize()
            self.synced = True
        # cuda reports milliseconds, but rest of the code uses microseconds
        return self.start_event.elapsed_time(self.end_event) * 1000
