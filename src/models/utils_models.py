import torch
import warnings
from . import mlp
from . import transformer
from . import parametrizations

FAMILIES=["transformer"]

def get_model_opts(vocab_size=32000, family="transformer", parametrization="np", zeta=16, scale_type="1/sqrt(d)",
                   c_input=0.02, c_hidden=0.02, c_output=0.02, k_input=1e-3, k_hidden=1e-3, k_output=1e-3, opt="adam",
                   momentum=0.9, beta2=0.95, beta3=0.98, alpha=5, gamma=0.025, eps=1e-8, weight_decay=0, max_context=1024,
                   test_parametrization=False, warning=True, backend="pytorch", device="cuda:0", comp=False, quartet=True, fake_quartet=False,
                   num_blocks=4, heads=6, ratio=3, tied_embeddings: bool = True):
    if warning and ((parametrization != "mup" and scale_type == "1/d") or (parametrization == "mup" and scale_type == "1/sqrt(d)")): warnings.warn(f"You use {scale_type} attention scaling even though the parametrization is {parametrization}", UserWarning)
    
    if family=="transformer":
        d_head0 = 8
        kwargs = {
            "vocab_size": vocab_size,
            "num_blocks": num_blocks,
            "heads": heads,
            "scale_type": scale_type,
            "ratio": ratio,
            "backend": backend,
            "max_context": max_context,
            "std": c_input,
            "quartet": quartet,
            "fake_quartet": fake_quartet,
            "weight_tying": tied_embeddings,
        }
        model0 = transformer.Transformer(d_head=d_head0, **kwargs, test=False)
        model = transformer.Transformer(d_head=zeta*d_head0, **kwargs, test=test_parametrization)
        model_ = transformer.Transformer(d_head=2*d_head0, **kwargs, test=False)
    
    model = model.to(device)
    # AFTER .to()
    model_or_ddp = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False) if torch.distributed.is_initialized() else model
    
    # AFTER DDP()
    opts = parametrizations.parametrize(model0, model_or_ddp, model_, parametrization, c_input, c_hidden, c_output, k_input, k_hidden, k_output, opt, momentum, beta2, beta3, alpha, gamma, eps, weight_decay, test_parametrization, warning, comp)

    return model_or_ddp, opts

def weight_norm(model):
    for parameter_name, parameter in model.named_parameters():
        parent_name, _, suffix = parameter_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        
        if parent_name.endswith(".lo") and suffix=="weight":
            parameter.data = ngpt.sphere_norm(parameter.data, dim=0)
        elif parent_name.endswith(".l2") and suffix=="weight":
            parameter.data = ngpt.sphere_norm(parameter.data, dim=0)
        elif suffix=="weight":
            parameter.data = ngpt.sphere_norm(parameter.data, dim=1)

    return model

def get_train_stats_header(model):
    train_stats_header = ""

    for name, _ in model.named_parameters():
        train_stats_header += f"{name}.grad_mean {name}.grad_top {name}.grad_bot {name}.grad_max {name}.data_mean {name}.data_top {name}.data_bot {name}.data_max "

    # Remove last space
    train_stats_header = train_stats_header[:-1]

    return train_stats_header

def get_cols_train_stats(model):
    cols_train_stats = []

    for name, _ in model.named_parameters():
        cols_train_stats += [f"{name}.grad_mean", f"{name}.grad_top", f"{name}.grad_bot", f"{name}.grad_max", f"{name}.data_mean", f"{name}.data_top", f"{name}.data_bot", f"{name}.data_max"]

    return cols_train_stats

def get_stats_abs(tensor):
    mean = tensor.mean().item()

    # https://github.com/pytorch/pytorch/issues/29372
    std = 0 if tensor.numel()==1 else tensor.std().item()

    top = mean+std
    # Absolute value cannot be negative
    bot = max(mean-std,0)
    _max = tensor.max().item()

    return mean, top, bot, _max

def get_train_stats(model):
    train_stats = ""

    for parameter in model.parameters():
        grad_mean, grad_top, grad_bot, grad_max = get_stats_abs(parameter.grad.abs())
        
        data_mean, data_top, data_bot, data_max = get_stats_abs(parameter.data.abs())

        train_stats += f"{grad_mean} {grad_top} {grad_bot} {grad_max} {data_mean} {data_top} {data_bot} {data_max} "
    
    # Remove last space
    train_stats = train_stats[:-1]

    return train_stats

def get_vals_train_stats(model):
    vals_train_stats = []

    for parameter in model.parameters():
        grad_mean, grad_top, grad_bot, grad_max = get_stats_abs(parameter.grad.abs())
        
        data_mean, data_top, data_bot, data_max = get_stats_abs(parameter.data.abs())

        vals_train_stats += [grad_mean, grad_top, grad_bot, grad_max, data_mean, data_top, data_bot, data_max]

    return vals_train_stats

def get_batch_stats(family, model, batch_Y_):
    out = batch_Y_.abs().mean().item()

    if family == "mlp":
        grad_mean = model.l2.weight.grad.abs().mean().item()
        data_mean = model.l2.weight.data.abs().mean().item()

    return out, grad_mean, data_mean
