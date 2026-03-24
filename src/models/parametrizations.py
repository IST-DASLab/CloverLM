from math import sqrt
import torch
import os
import warnings
# More compact and discreet
def get_formatwarning(message, category, filename, lineno, line=None):
    return f"\x1b[90;3m[{category.__name__}] {os.path.basename(filename)} ({lineno}L): {message}\x1b[0m\n"
warnings.formatwarning = get_formatwarning

PARAMETRIZATIONS = ["np", "sp", "ntk", "mup", "mf"]
OPTIMIZERS = ["sgd", "adam", "kron", "pro", "shampoo", "laprop", "lion", "ademamix", "soap", "adopt", "marsadam", "cadam", "muon", "scion"]

def lookup_table1(parametrization, layer, fanin0, fanin, fanout0, fanout):
    if parametrization == "sp":
        if layer == "input":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1
            C2_adam = 1
        elif layer == "hidden":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1
            C2_adam = 1
        elif layer == "output":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1
            C2_adam = 1

    elif parametrization == "ntk":
        if layer == "input":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = sqrt(fanin0)
            C2_adam = 1 / sqrt(fanin)
        elif layer == "hidden":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = sqrt(fanin0)
            C2_adam = 1 / sqrt(fanin)
        elif layer == "output":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = sqrt(fanin0)
            C2_adam = 1 / sqrt(fanin)

    elif parametrization == "mup":
        if layer == "input":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1 / fanout0
            C2_sgd = fanout
            C1_adam = 1
            C2_adam = 1
        elif layer == "hidden":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = fanin0
            C2_adam = 1 / fanin
        elif layer == "output":
            B1 = sqrt(fanin0)
            B2 = 1 / fanin
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = fanin0
            C2_adam = 1 / fanin

    elif parametrization == "mf":
        if layer == "input":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1 / sqrt(fanin0)
            C2_adam = sqrt(fanin)
        elif layer == "hidden":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1 / sqrt(fanin0)
            C2_adam = sqrt(fanin)
        elif layer == "output":
            B1 = sqrt(fanin0)
            B2 = 1 / fanin
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = 1
            C2_adam = 1

    return B1, B2, C1_sgd, C2_sgd, C1_adam, C2_adam

def test_table1(fanin0, fanin, fanout0, fanout):
    print("\x1b[1m%16.16s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\x1b[0m" % ("parametrization", "layer", "B1", "B2", "C1_sgd", "C2_sgd", "C1_adam", "C2_adam"))
    for parametrization in ["sp", "ntk", "mup", "mf"]:
        for layer in ["input", "hidden", "output"]:
            B1, B2, C1_sgd, C2_sgd, C1_adam, C2_adam = lookup_table1(parametrization, layer, fanin0, fanin, fanout0, fanout)

            B1_decorated = "%8.8s" % ("%f" % B1)
            B2_decorated = "%8.8s" % ("%f" % B2)
            C1_sgd_decorated = "%8.8s" % ("%f" % C1_sgd)
            C2_sgd_decorated = "%8.8s" % ("%f" % C2_sgd)
            C1_adam_decorated = "%8.8s" % ("%f" % C1_adam)
            C2_adam_decorated = "%8.8s" % ("%f" % C2_adam)

            print("%16.16s %8.8s %s %s %s %s %s %s" % (parametrization, layer, B1_decorated, B2_decorated, C1_sgd_decorated, C2_sgd_decorated, C1_adam_decorated, C2_adam_decorated))
        print("━"*80)

def lookup_table2(fanin, parameter_type, layer, c_input, c_hidden, c_output, k_input, k_hidden, k_output):
    if fanin == 1:
        if parameter_type == "bias":
            μ = 0
            B0 = 0
            C0 = k_input
        elif parameter_type == "Norm.weight":
            μ = 1
            B0 = 0
            C0 = k_input
        elif parameter_type == "class":
            μ = 0
            B0 = c_input
            C0 = k_input

    elif fanin > 1:
        if parameter_type == "Linear/Conv.weight":
            if layer == "input":
                μ = 0
                B0 = c_input * sqrt(fanin)
                C0 = k_input
            elif layer == "hidden":
                μ = 0
                B0 = c_hidden
                C0 = k_hidden
            elif layer == "output":
                μ = 0
                B0 = c_output
                C0 = k_output

        elif parameter_type == "emb/pos":
            μ = 0
            B0 = c_input * sqrt(fanin)
            C0 = k_input

    return μ, B0, C0

def test_table2(c_input, c_hidden, c_output, k_input, k_hidden, k_output):
    print("\x1b[1m%12.12s %18.18s %8.8s %8.8s %8.8s %8.8s\x1b[0m" % ("fanin", "parameter_type", "layer", "μ", "B0", "C0"))

    μ, B0, C0 = lookup_table2(1, "bias", "input", c_input, c_hidden, c_output, k_input, k_hidden, k_output)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s %8.8s" % ("fanin=1", "bias", "input", "%f" % μ, "%f" % B0, "%f" % C0))
    μ, B0, C0 = lookup_table2(1, "Norm.weight", "input", c_input, c_hidden, c_output, k_input, k_hidden, k_output)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s %8.8s" % ("fanin=1", "Norm.weight", "input", "%f" % μ, "%f" % B0, "%f" % C0))
    μ, B0, C0 = lookup_table2(1, "class", "input", c_input, c_hidden, c_output, k_input, k_hidden, k_output)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s %8.8s" % ("fanin=1", "class", "input", "%f" % μ, "%f" % B0, "%f" % C0))
    
    print("━"*70)

    μ, B0, C0 = lookup_table2(4, "Linear/Conv.weight", "input", c_input, c_hidden, c_output, k_input, k_hidden, k_output)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s %8.8s" % ("fanin=4", "Linear/Conv.weight", "input", "%f" % μ, "%f" % B0, "%f" % C0))
    μ, B0, C0 = lookup_table2(4, "Linear/Conv.weight", "hidden", c_input, c_hidden, c_output, k_input, k_hidden, k_output)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s %8.8s" % ("fanin=4", "Linear/Conv.weight", "hidden", "%f" % μ, "%f" % B0, "%f" % C0))
    μ, B0, C0 = lookup_table2(4, "Linear/Conv.weight", "output", c_input, c_hidden, c_output, k_input, k_hidden, k_output)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s %8.8s" % ("fanin=4", "Linear/Conv.weight", "output", "%f" % μ, "%f" % B0, "%f" % C0))

    print("━"*70)

    μ, B0, C0 = lookup_table2(4, "emb/pos", "input", c_input, c_hidden, c_output, k_input, k_hidden, k_output)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s %8.8s" % ("fanin=4", "emb/pos", "input", "%f" % μ, "%f" % B0, "%f" % C0))

    print("━"*70)

def get_fan(parameter, suffix, parent):
    if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and suffix=="weight":
        fanin = parameter.shape[1]
        fanout = parameter.shape[0]
    elif isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and suffix=="bias":
        fanin = 1
        fanout = len(parameter)
    elif isinstance(parent, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.RMSNorm)) and (suffix=="weight" or suffix=="bias"):
        fanin = 1
        fanout = len(parameter)
    else:
        # class
        if parameter.ndim == 1:
            fanin = 1
            fanout = len(parameter)
        # emb, pos
        elif parameter.ndim == 2:
            fanin = parameter.shape[0]
            fanout = parameter.shape[1]

    return fanin, fanout

# model: base
# model_: A scaled (up or down) version of model
def get_layers(model, model_, warning=True):
    layers = {}

    for parameter_name, parameter in model.named_parameters():
        parameter_ = model_.get_parameter(parameter_name)

        parent_name, _, suffix = parameter_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        
        fanin, fanout = get_fan(parameter, suffix, parent)
        fanin_, fanout_ = get_fan(parameter_, suffix, parent)
        
        if fanin == fanin_ and fanout != fanout_:
            layers[parameter_name] = "input"
        elif fanin != fanin_ and fanout != fanout_:
            layers[parameter_name] = "hidden"
        elif fanin != fanin_ and fanout == fanout_:
            layers[parameter_name] = "output"
        else:
            if warning: warnings.warn(f"{parameter_name} is not \"input\", \"hidden\" or \"output\". This means that you are either not scaling the WHOLE model_ (in which case parametrizations do not work), or that {parameter_name} is an output bias, and hence an \"input\" layer. We are going to assume the latter.", UserWarning)
            layers[parameter_name] = "input"

    return layers

def get_parameter_type(parameter, suffix, parent):
    if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and suffix=="weight":
        parameter_type = "Linear/Conv.weight"
    elif isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)) and suffix=="bias":
        parameter_type = "bias"
    elif isinstance(parent, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.RMSNorm)) and suffix=="weight":
        parameter_type = "Norm.weight"
    else:
        # class
        if parameter.ndim == 1:
            parameter_type = "class"
        # emb, pos
        elif parameter.ndim == 2:
            parameter_type = "emb/pos"

    return parameter_type

# model0: proxy
# model: target
# model_: A scaled (up or down) version of model0
def parametrize(model0, model_or_ddp, model_, parametrization, c_input, c_hidden, c_output, k_input, k_hidden, k_output, opt, momentum, beta2, beta3, alpha, gamma, eps, weight_decay, test, warning, comp):
    if parametrization == "np":
        model = model_or_ddp.module if torch.distributed.is_initialized() else model_or_ddp
        input_params = list(model.emb.parameters())
        vector_params = [parameter for parameter in model.parameters() if (parameter.ndim < 2 or parameter.ndim > 3)]
        hidden_params = [parameter for parameter in model.blocks.parameters() if parameter.ndim == 2]
        # output_params = list(model.linear.parameters())
    else:
        layers = get_layers(model0, model_, warning)
        
        params = []
        
        if test: print("\x1b[1m%36.36s %8.8s %20.20s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\x1b[0m" % ("parameter_name", "layer", "parameter_type", "fanin0", "fanin", "fanout0", "fanout", "mean", "B0", "B1", "B2", "std", "C0", "C1", "C2", "lr"))
        for parameter_name, parameter in model.named_parameters():
            parameter0 = model0.get_parameter(parameter_name)

            parent_name, _, suffix = parameter_name.rpartition(".")
            parent = model.get_submodule(parent_name)
            
            layer = layers[parameter_name]

            fanin0, fanout0 = get_fan(parameter0, suffix, parent)
            fanin, fanout = get_fan(parameter, suffix, parent)

            B1, B2, C1_sgd, C2_sgd, C1_adam, C2_adam = lookup_table1(parametrization, layer, fanin0, fanin, fanout0, fanout)
            if opt == "sgd":
                C1 = C1_sgd
                C2 = C2_sgd
            elif opt == "adam":
                C1 = C1_adam
                C2 = C2_adam
            
            parameter_type = get_parameter_type(parameter, suffix, parent)
            
            μ, B0, C0 = lookup_table2(fanin, parameter_type, layer, c_input, c_hidden, c_output, k_input, k_hidden, k_output)

            mean = μ
            std = B0*B1*B2
            torch.nn.init.normal_(parameter, mean, std)

            lr = C0*C1*C2
            params.append({"params": parameter, "lr": lr})

            if test: print("%36.36s %8.8s %20.20s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\x1b[0m" % (parameter_name, layer, parameter_type, fanin0, fanin, fanout0, fanout, "%f" % mean, "%f" % B0, "%f" % B1, "%f" % B2, "%f" % std, "%f" % C0, "%f" % C1, "%f" % C2, "%f" % lr))
    
    # Sep, 1951
    if opt=="sgd":
        opts = [
            torch.optim.SGD(input_params+vector_params, lr=k_input, momentum=momentum, weight_decay=weight_decay, nesterov=True, fused=True),
            torch.optim.SGD(hidden_params, lr=k_hidden, momentum=momentum, weight_decay=weight_decay, nesterov=True, fused=True),
            # torch.optim.SGD(output_params, lr=k_output, momentum=momentum, weight_decay=weight_decay, nesterov=True, fused=True)
        ]
    
    # Dec, 2014
    elif opt=="adam":
        opts = [
            torch.optim.AdamW(input_params+vector_params, lr=k_input, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, fused=True),
            torch.optim.AdamW(hidden_params, lr=k_hidden, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, fused=True),
            # torch.optim.AdamW(output_params, lr=k_output, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, fused=True)
        ]
    
    # Dec, 2015
    elif opt=="kron":
        import pytorch_optimizer

        opts = [
            pytorch_optimizer.Kron(input_params+vector_params, lr=k_input, momentum=momentum, weight_decay=weight_decay, weight_decouple=True, max_size_triangular=12288, min_ndim_triangular=2, memory_save_mode=None),
            pytorch_optimizer.Kron(hidden_params, lr=k_hidden, momentum=momentum, weight_decay=weight_decay, weight_decouple=True, max_size_triangular=12288, min_ndim_triangular=2, memory_save_mode=None),
            # pytorch_optimizer.Kron(output_params, lr=k_output, momentum=momentum, weight_decay=weight_decay, weight_decouple=True, max_size_triangular=12288, min_ndim_triangular=2, memory_save_mode=None)
        ]
    
    # FIND OUT
    elif opt=="pro":
        import quad_torch

        opts = [
            quad_torch.Procrustes(input_params+vector_params, lr=k_input, momentum=momentum, weight_decay=weight_decay, lr_style="adam", max_size_dense=12288, max_skew_dense=1.0),
            quad_torch.Procrustes(hidden_params, lr=k_hidden, momentum=momentum, weight_decay=weight_decay, lr_style="adam", max_size_dense=12288, max_skew_dense=1.0),
            # quad_torch.Procrustes(output_params, lr=k_output, momentum=momentum, weight_decay=weight_decay, lr_style="adam", max_size_dense=12288, max_skew_dense=1.0)
        ]
    
    # Feb, 2018
    elif opt=="shampoo":
        import distributed_shampoo
        
        distributed_config = distributed_shampoo.DDPShampooConfig() if torch.distributed.is_initialized() else None

        shampoo_pt2_compile_config = distributed_shampoo.ShampooPT2CompileConfig() if comp else None
        
        opts = [
            distributed_shampoo.DistributedShampoo(input_params+vector_params, lr=k_input, betas=(momentum, beta2), epsilon=eps, weight_decay=weight_decay, use_decoupled_weight_decay=True, grafting_config=distributed_shampoo.AdamGraftingConfig(beta2=beta2, epsilon=eps), distributed_config=distributed_config, shampoo_pt2_compile_config=shampoo_pt2_compile_config, precondition_frequency=20, max_preconditioner_dim=12288, start_preconditioning_step=-1, use_bias_correction=True),
            distributed_shampoo.DistributedShampoo(hidden_params, lr=k_hidden, betas=(momentum, beta2), epsilon=eps, weight_decay=weight_decay, use_decoupled_weight_decay=True, grafting_config=distributed_shampoo.AdamGraftingConfig(beta2=beta2, epsilon=eps), distributed_config=distributed_config, shampoo_pt2_compile_config=shampoo_pt2_compile_config, precondition_frequency=20, max_preconditioner_dim=12288, start_preconditioning_step=-1, use_bias_correction=True),
            # distributed_shampoo.DistributedShampoo(output_params, lr=k_output, betas=(momentum, beta2), epsilon=eps, weight_decay=weight_decay, use_decoupled_weight_decay=True, grafting_config=distributed_shampoo.AdamGraftingConfig(beta2=beta2, epsilon=eps), distributed_config=distributed_config, shampoo_pt2_compile_config=shampoo_pt2_compile_config, precondition_frequency=20, max_preconditioner_dim=12288, start_preconditioning_step=-1, use_bias_correction=True)
        ]

    # Feb, 2020
    elif opt=="laprop":
        import pytorch_optimizer

        opts = [
            pytorch_optimizer.LaProp(input_params+vector_params, lr=k_input, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, weight_decouple=True),
            pytorch_optimizer.LaProp(hidden_params, lr=k_hidden, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, weight_decouple=True),
            # pytorch_optimizer.LaProp(output_params, lr=k_output, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, weight_decouple=True)
        ]

    # Feb, 2023
    elif opt=="lion":
        import pytorch_optimizer

        opts = [
            pytorch_optimizer.Lion(input_params+vector_params, lr=k_input, betas=(momentum, beta2), weight_decay=weight_decay, weight_decouple=True),
            pytorch_optimizer.Lion(hidden_params, lr=k_hidden, betas=(momentum, beta2), weight_decay=weight_decay, weight_decouple=True),
            # pytorch_optimizer.Lion(output_params, lr=k_output, betas=(momentum, beta2), weight_decay=weight_decay, weight_decouple=True)
        ]
    
    # Sep, 2024
    elif opt=="ademamix":
        import pytorch_optimizer

        opts = [
            pytorch_optimizer.AdEMAMix(input_params+vector_params, lr=k_input, betas=(momentum, beta2, beta3), alpha=alpha, eps=eps, weight_decay=weight_decay, weight_decouple=True),
            pytorch_optimizer.AdEMAMix(hidden_params, lr=k_hidden, betas=(momentum, beta2, beta3), alpha=alpha, eps=eps, weight_decay=weight_decay, weight_decouple=True),
            # pytorch_optimizer.AdEMAMix(output_params, lr=k_output, betas=(momentum, beta2, beta3), alpha=alpha, eps=eps, weight_decay=weight_decay, weight_decouple=True)
        ]

    # Sep, 2024
    elif opt=="soap":
        import pytorch_optimizer

        opts = [
            pytorch_optimizer.SOAP(input_params+vector_params, lr=k_input, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, precondition_frequency=10, max_precondition_dim=12288, precondition_1d=False, correct_bias=True),
            pytorch_optimizer.SOAP(hidden_params, lr=k_hidden, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, precondition_frequency=10, max_precondition_dim=12288, precondition_1d=False, correct_bias=True),
            # pytorch_optimizer.SOAP(output_params, lr=k_output, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, precondition_frequency=10, max_precondition_dim=12288, precondition_1d=False, correct_bias=True)
        ]

    # Nov, 2024
    elif opt=="adopt":
        import pytorch_optimizer

        opts = [
            pytorch_optimizer.ADOPT(input_params+vector_params, lr=k_input, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, weight_decouple=True, clip_lambda=lambda step: step**(1/4)),
            pytorch_optimizer.ADOPT(hidden_params, lr=k_hidden, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, weight_decouple=True, clip_lambda=lambda step: step**(1/4)),
            # pytorch_optimizer.ADOPT(output_params, lr=k_output, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, weight_decouple=True, clip_lambda=lambda step: step**(1/4))
        ]

    # Nov, 2024
    elif opt=="marsadam":
        import pytorch_optimizer

        opts = [
            pytorch_optimizer.MARS(input_params+vector_params, lr=k_input, betas=(momentum, beta2), gamma=gamma, eps=eps, weight_decay=weight_decay, weight_decouple=True, mars_type="adamw", optimize_1d=True),
            pytorch_optimizer.MARS(hidden_params, lr=k_hidden, betas=(momentum, beta2), gamma=gamma, eps=eps, weight_decay=weight_decay, weight_decouple=True, mars_type="adamw", optimize_1d=True),
            # pytorch_optimizer.MARS(output_params, lr=k_output, betas=(momentum, beta2), gamma=gamma, eps=eps, weight_decay=weight_decay, weight_decouple=True, mars_type="adamw", optimize_1d=True)
        ]

    # Nov, 2024
    elif opt=="cadam":
        import heavyball

        opts = [
            heavyball.ForeachAdamW(input_params+vector_params, lr=k_input, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, caution=True, foreach=True),
            heavyball.ForeachAdamW(hidden_params, lr=k_hidden, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, caution=True, foreach=True),
            # heavyball.ForeachAdamW(output_params, lr=k_output, betas=(momentum, beta2), eps=eps, weight_decay=weight_decay, caution=True, foreach=True)
        ]
    
    # Dec, 2024
    elif opt=="muon":
        # Distributed Muon does NOT work with DCP, is less robust and more complicated 
        import muon

        opts = [
            torch.optim.AdamW(input_params+vector_params, lr=3e-3, betas=(0.9,0.95), eps=1e-6, weight_decay=weight_decay, fused=True),
            muon.SingleDeviceMuon(hidden_params, lr=k_hidden, momentum=momentum, weight_decay=weight_decay),
            # torch.optim.AdamW(output_params, lr=3e-3, betas=(0.9,0.95), eps=1e-6, weight_decay=weight_decay, fused=True)
        ]
    
    # Feb, 2025
    elif opt=="scion":
        import pytorch_optimizer

        opts = [
            pytorch_optimizer.SCION(input_params, lr=1, momentum=1-momentum, weight_decay=weight_decay, weight_decouple=True, constraint=False, norm_type=4, scale=1),
            pytorch_optimizer.SCION(vector_params, lr=3e-3, momentum=1-momentum, weight_decay=weight_decay, weight_decouple=True, constraint=False, norm_type=8, scale=1),
            pytorch_optimizer.SCION(hidden_params, lr=k_hidden, momentum=1-momentum, weight_decay=weight_decay, weight_decouple=True, constraint=False, norm_type=2, scale=1),
            # pytorch_optimizer.SCION(output_params, lr=k_output, momentum=1-momentum, weight_decay=weight_decay, weight_decouple=True, constraint=False, norm_type=4, scale=1)
        ]

    return opts
