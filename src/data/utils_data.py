import os
from typing import Optional

import datasets
datasets.logging.set_verbosity_error()
import torch
import torchvision.transforms.v2
import sys
from math import sqrt
import collections
import numpy

script_path = os.path.abspath(__file__)
data_path = os.path.dirname(script_path)
src_path = os.path.dirname(data_path)
root_path = os.path.dirname(src_path)

DATASETS_TABULAR = ["california_housing"]
DATASETS_IMAGE = ["mnist", "cifar10", "tinyimagenet"]
DATASETS_TEXT = ["shakespearefirstfolio", "wikitext", "minipile", "openwebtext", "fineweb", "finewebedu", "climbmix10m", "climbmix100m", "ancient_greek_theatre", "culturay_el"]
DATASETS = DATASETS_TABULAR + DATASETS_IMAGE + DATASETS_TEXT
TOKENIZER_TYPES = ["tokenizers", "tokenmonster"]

def get_splits(dataset, keep_in_memory=False):
    if dataset=="california_housing":
        train_dataset = datasets.load_dataset("gvlassis/california_housing", split="train", keep_in_memory=keep_in_memory)
        val_dataset = datasets.load_dataset("gvlassis/california_housing", split="validation", keep_in_memory=keep_in_memory)
        test_dataset = datasets.load_dataset("gvlassis/california_housing", split="test", keep_in_memory=keep_in_memory)
    elif dataset=="mnist":
        dataset = datasets.load_dataset("ylecun/mnist", split="train", keep_in_memory=keep_in_memory)
        dataset = dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True, keep_in_memory=keep_in_memory)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        test_dataset = datasets.load_dataset("ylecun/mnist", split="test", keep_in_memory=keep_in_memory)
    elif dataset=="cifar10":
        dataset = datasets.load_dataset("uoft-cs/cifar10", split="train", keep_in_memory=keep_in_memory)
        dataset = dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True, keep_in_memory=keep_in_memory)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        test_dataset = datasets.load_dataset("uoft-cs/cifar10", split="test", keep_in_memory=keep_in_memory)
    elif dataset=="tinyimagenet":
        dataset = datasets.load_dataset("zh-plus/tiny-imagenet", split="train", keep_in_memory=keep_in_memory)
        dataset = dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True, keep_in_memory=keep_in_memory)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        val_dataset = datasets.load_dataset("zh-plus/tiny-imagenet", split="valid", keep_in_memory=keep_in_memory)
    elif dataset=="shakespearefirstfolio":
        train_dataset = datasets.load_dataset("gvlassis/shakespearefirstfolio", split="train", keep_in_memory=keep_in_memory)
        val_dataset = datasets.load_dataset("gvlassis/shakespearefirstfolio", split="validation", keep_in_memory=keep_in_memory)
        test_dataset = datasets.load_dataset("gvlassis/shakespearefirstfolio", split="test", keep_in_memory=keep_in_memory)
    elif dataset=="wikitext":
        train_dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-103-v1", split="train", keep_in_memory=keep_in_memory)
        val_dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-103-v1", split="validation", keep_in_memory=keep_in_memory)
        test_dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-103-v1", split="test", keep_in_memory=keep_in_memory)
    elif dataset=="minipile":
        train_dataset = datasets.load_dataset("JeanKaddour/minipile", split="train", keep_in_memory=keep_in_memory)
        val_dataset = datasets.load_dataset("JeanKaddour/minipile", split="validation", keep_in_memory=keep_in_memory)
        test_dataset = datasets.load_dataset("JeanKaddour/minipile", split="test", keep_in_memory=keep_in_memory)
    elif dataset=="openwebtext":
        dataset = datasets.load_dataset("Skylion007/openwebtext", split="train", keep_in_memory=keep_in_memory)
        dataset = dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True, keep_in_memory=keep_in_memory)
        test_dataset = dataset["test"]
        dataset = dataset["train"]
        dataset = dataset.train_test_split(train_size=None, test_size=500, shuffle=True, keep_in_memory=keep_in_memory)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    elif dataset=="fineweb":
        dataset = datasets.load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", keep_in_memory=keep_in_memory)
        dataset = dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True, keep_in_memory=keep_in_memory)
        test_dataset = dataset["test"]
        dataset = dataset["train"]
        dataset = dataset.train_test_split(train_size=None, test_size=500, shuffle=True, keep_in_memory=keep_in_memory)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    elif dataset=="finewebedu":
        dataset = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", keep_in_memory=keep_in_memory)
        dataset = dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True, keep_in_memory=keep_in_memory)
        test_dataset = dataset["test"]
        dataset = dataset["train"]
        dataset = dataset.train_test_split(train_size=None, test_size=500, shuffle=True, keep_in_memory=keep_in_memory)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    elif dataset=="climbmix10m":
        dataset = datasets.load_dataset("gvlassis/ClimbMix10m", split="train", keep_in_memory=keep_in_memory)
        dataset = dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True, keep_in_memory=keep_in_memory)
        test_dataset = dataset["test"]
        dataset = dataset["train"]
        dataset = dataset.train_test_split(train_size=None, test_size=500, shuffle=True, keep_in_memory=keep_in_memory)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    elif dataset=="ancient_greek_theatre":
        train_dataset = datasets.load_dataset("gvlassis/ancient_greek_theatre", split="train", keep_in_memory=keep_in_memory)
        val_dataset = datasets.load_dataset("gvlassis/ancient_greek_theatre", split="validation", keep_in_memory=keep_in_memory)
        test_dataset = datasets.load_dataset("gvlassis/ancient_greek_theatre", split="test", keep_in_memory=keep_in_memory)
    elif dataset=="culturay_el":
        dataset = datasets.load_dataset("ontocord/CulturaY", name="el", split="train", keep_in_memory=keep_in_memory)
        dataset = dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True, keep_in_memory=keep_in_memory)
        test_dataset = dataset["test"]
        dataset = dataset["train"]
        dataset = dataset.train_test_split(train_size=None, test_size=500, shuffle=True, keep_in_memory=keep_in_memory)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]

    return train_dataset, val_dataset, test_dataset

class TabularDataset(torch.utils.data.Dataset):
    # device is only used for initialization
    def __init__(self, dataset_path, split="train", device=None):
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split

        X_path = "%s/%s_X.pt" % (dataset_path, split)
        Y_path = "%s/%s_Y.pt" % (dataset_path, split)

        self.X = torch.load(X_path, map_location=device, weights_only=True)
        self.Y = torch.load(Y_path, map_location=device, weights_only=True)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return len(self.X)

def tabular_dataset_to_tensors(dataset):
    dataset = torch.tensor(dataset.to_pandas().values)
    
    # Target is last column
    X = dataset[:,:-1]
    Y = dataset[:,-1]

    print("\x1b[36m%d\x1b[0m samples, %d feats" % (X.shape[0], X.shape[1]))

    return X, Y

class ImageDataset(torch.utils.data.Dataset):
    # device is only used for initialization
    def __init__(self, dataset_path, split="train", device=None):
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split

        X_path = "%s/%s_X.pt" % (dataset_path, split)
        Y_path = "%s/%s_Y.pt" % (dataset_path, split)

        self.X = torch.load(X_path, map_location=device, weights_only=True)
        self.Y = torch.load(Y_path, map_location=device, weights_only=True)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return len(self.X)

def image_dataset_to_tensors(dataset, cores):
    # Unpacking set to single element
    name, = set(["img", "image"]) & set(dataset.column_names)
    
    # Image datasets may contain mixed modes (e.g. ImageNet contains greyscale images)
    print("🌄 Modes")
    modes = [ image.mode for image in dataset[name] ]
    modes = collections.Counter(modes)
    modes = dict(sorted(modes.items(), key=lambda item: item[1], reverse=True))
    for mode in modes.keys():
        print("%6.6s: %10.10s" % (mode,modes[mode]))
    dataset = dataset.filter(lambda sample: sample[name].mode==next(iter(modes)))
    
    X_dataset = dataset.map(lambda sample: {"tensor": torchvision.transforms.v2.functional.pil_to_tensor(sample[name])}, remove_columns=["label"], num_proc=cores)
    X = torch.tensor(X_dataset["tensor"], dtype=torch.uint8)
    Y = torch.tensor(dataset["label"])

    print("\x1b[36m%d\x1b[0m images, %d×%d×%d" % (X.shape[0], X.shape[1], X.shape[2], X.shape[3]))

    return X, Y

class TextDataset(torch.utils.data.Dataset):
    # device is only used for initialization
    def __init__(self, dataset_path, split="train", device: Optional[str] = None, context=1024):
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split
        self.context = context

        X_path = "%s/%s_X.pt" % (dataset_path, split)
        self.mmap = os.path.getsize(X_path) > 10 * 1024 ** 3
        if self.mmap and (device is not None and device != "cpu"):
            raise ValueError("Attempting to load TextDataset > 10 GiB onto GPU")

        self.X = torch.load(X_path, map_location=device, weights_only=True, mmap=self.mmap)

    # i is the first index of the context
    def __getitem__(self, i):
        X = self.X[i:i+self.context]
        Y = self.X[i+1:i+1+self.context]

        return X, Y

    def __len__(self):
        return self.X.numel()-self.context

def text_dataset_to_tensor(dataset, tokenizer_type, tokenizer, eot_id, batch_size=1000, cores=1):

    if tokenizer_type=="tokenizers":
        import tokenizers

        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer).backend_tokenizer

        def tokenize(batch):
            batch = tokenizer.encode_batch_fast(batch["text"], add_special_tokens=False)
            batch = [numpy.array(sample.ids, dtype=numpy.uint16) for sample in batch]
            return {"ids": batch}

    elif tokenizer_type=="tokenmonster":
        import tokenmonster
        
        tokenizer = tokenmonster.load(tokenizer) # i) load_multiprocess_safe() hangs, ii) https://github.com/alasdairforsythe/tokenmonster/issues/33

        def tokenize(batch):
            batch = tokenizer.tokenize(batch["text"]) # Does NOT have add_special_tokens=False
            return {"ids": batch}

        cores = 1 # map(num_proc=cores) crushes with TokenMonster if num_proc>1
    
    dataset = dataset.map(tokenize, remove_columns=["text"], batched=True, batch_size=batch_size, num_proc=cores)

    dataset.set_format("torch", dtype=torch.uint16)
    dataset = dataset["ids"]

    eot = torch.tensor([eot_id], dtype=torch.uint16)
    dataset = [elem for tensor in dataset for elem in (tensor, eot)]

    dataset = torch.cat(dataset)

    print("\x1b[36m%d\x1b[0m tokens" % dataset.numel())

    return dataset

def dataset_to_tensors(dataset, tokenizer_type, tokenizer, eot_id, batch_size, cores):

    # Auto-detection
    if set(["img", "image"]) & set(dataset.column_names):
        tensors = image_dataset_to_tensors(dataset, cores)
    elif "text" in dataset.column_names:
        tensors = (text_dataset_to_tensor(dataset, tokenizer_type, tokenizer, eot_id, batch_size, cores), None)
    else:
        tensors = tabular_dataset_to_tensors(dataset)

    return tensors

def get_iterator(dataset, split, device, batch_size, context, rank, dataset_path=None, seed=-1):
    dataset_path = dataset_path if dataset_path else "%s/%s" % (root_path, dataset)

    if dataset in DATASETS_TABULAR:
        dataset = TabularDataset(dataset_path, split=split, device=device)
    elif dataset in DATASETS_IMAGE:
        dataset = ImageDataset(dataset_path, split=split, device=device)
    elif dataset in DATASETS_TEXT:
        dataset = TextDataset(dataset_path, split=split, device=device, context=context)
    
    # Has to be on the same device as the dataset
    generator = torch.Generator(device)
    # For DDP, it must have different seeds on different ranks
    if seed <= 0:
        generator.seed()
    else:
        generator.manual_seed(seed + rank * 2097593)

    # num_samples: Maximum samples (default: len(dataset))
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=sys.maxsize, generator=generator)
    # shuffle=True hangs

    is_mmap = isinstance(dataset, TextDataset) and dataset.mmap
    num_workers = 1 if is_mmap else 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, pin_memory=True, num_workers=num_workers, persistent_workers=num_workers > 0)
    iterator = iter(dataloader)

    return iterator

def transform(dataset, x):
    if dataset in DATASETS_TABULAR:
        x = x.to(torch.float32)
    elif dataset in DATASETS_IMAGE:
        x = x.to(torch.float32)
    elif dataset in DATASETS_TEXT:
        x = x.to(torch.int32)
        
    return x

def get_loss(dataset, model, batch_X, batch_Y, label_smoothing=0):
    batch_Y_ = model(transform(dataset, batch_X))

    # huggingface
    # batch_Y_ = model(transform(dataset, batch_X)).logits

    if dataset in DATASETS_TABULAR:
        loss = torch.nn.functional.mse_loss(
            batch_Y_.flatten(),
            batch_Y.to(torch.float32)
        )

    elif dataset in DATASETS_IMAGE:
        loss = torch.nn.functional.cross_entropy(
            batch_Y_,
            batch_Y.to(torch.int64),
            label_smoothing = label_smoothing
        )

    elif dataset in DATASETS_TEXT:
        loss = torch.nn.functional.cross_entropy(
            torch.reshape(batch_Y_, (-1,batch_Y_.shape[-1])),
            torch.flatten(batch_Y.to(torch.int64)),
            label_smoothing = label_smoothing
        )
        
    return batch_Y_, loss

@torch.no_grad()
def approximate_loss(batches, iterator, dataset, model, dtype):
    model.eval()

    device = next(model.parameters()).device
    
    loss = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        with torch.autocast(device_type=device.type, dtype=dtype):
            loss += get_loss(dataset, model, batch_X.to(device), batch_Y.to(device), 0)[1].item()
    loss = loss/batches

    return loss

@torch.no_grad()
def get_batch_loss(batch_X, batch_Y, dataset, model, dtype) -> torch.Tensor:
    model.eval()

    device = next(model.parameters()).device
    
    with torch.autocast(device_type=device.type, dtype=dtype):
        loss = get_loss(dataset, model, batch_X.to(device), batch_Y.to(device), 0)[1]

    return loss

@torch.no_grad()
def approximate_rmse(batches, iterator, dataset, model, dtype):
    model.eval()

    device = next(model.parameters()).device
    
    mse = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        mse += ((batch_Y.to(device) - batch_Y_)**2).sum().item()
    batch_size = batch_X.shape[0]
    mse = mse/(batches*batch_size)
    rmse = sqrt(mse)

    return rmse

@torch.no_grad()
def approximate_nrmse(batches, iterator, dataset, model, dtype):
    model.eval()

    device = next(model.parameters()).device
    
    mse = 0
    minimum = float("+inf")
    maximum = float("-inf")
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        minimum = min(minimum, batch_Y.min().item())
        maximum = max(maximum, batch_Y.max().item())
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        mse += ((batch_Y.to(device) - batch_Y_)**2).sum().item()
    batch_size = batch_X.shape[0]
    mse = mse/(batches*batch_size)
    rmse = sqrt(mse)
    nrmse = rmse/(maximum-minimum)

    return nrmse

@torch.no_grad()
def approximate_mae(batches, iterator, dataset, model, dtype):
    model.eval()

    device = next(model.parameters()).device
    
    mae = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        mae += (batch_Y.to(device) - batch_Y_).abs().sum().item()
    batch_size = batch_X.shape[0]
    mae = mae/(batches*batch_size)

    return mae

@torch.no_grad()
def approximate_nmae(batches, iterator, dataset, model, dtype):
    model.eval()

    device = next(model.parameters()).device
    
    mae = 0
    minimum = float("+inf")
    maximum = float("-inf")
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        minimum = min(minimum, batch_Y.min().item())
        maximum = max(maximum, batch_Y.max().item())
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        mae += (batch_Y.to(device) - batch_Y_).abs().sum().item()
    batch_size = batch_X.shape[0]
    mae = mae/(batches*batch_size)
    nmae = mae/(maximum-minimum)

    return nmae

@torch.no_grad()
def approximate_r2(batches, iterator, dataset, model, dtype):
    model.eval()

    device = next(model.parameters()).device
    
    rss = 0
    ss = 0
    mean = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        rss += ((batch_Y.to(device) - batch_Y_)**2).sum().item()
        ss += (batch_Y.to(device)**2).sum().item()
        mean += batch_Y.to(device).sum().item()
    batch_size = batch_X.shape[0]
    mean = mean/(batches*batch_size)
    tss = ss - batches*batch_size*(mean**2)
    r2 = 1 - rss/tss
    
    return r2

@torch.no_grad()
def approximate_acc(batches, iterator, dataset, model, dtype):
    model.eval()

    device = next(model.parameters()).device
    
    correct = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            batch_Y_ = model(transform(dataset, batch_X.to(device))).argmax(dim=-1)
        
        correct += (batch_Y_ == batch_Y.to(device)).sum().item()
    batch_size = batch_X.shape[0]
    acc = correct/(batches*batch_size)

    return acc

# @torch.no_grad()
# def approximate_ppl(batches, iterator, dataset, model, dtype):
#     model.eval()
#
#     device = next(model.parameters()).device
#
#     ce = 0
#     for batch in range(batches):
#         batch_X, batch_Y = next(iterator)
#
#         with torch.autocast(device_type=device.type, dtype=dtype):
#             batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()
#
#         ce += ((batch_Y.to(device) - batch_Y_)**2).sum().item()
#     batch_size = batch_X.shape[0]
#     mse = mse/(batches*batch_size)
#     ppl = sqrt(ce)
#
#     return ppl
#
#     batch_Y.to(torch.int64)

def lm_eval_wrapper(tokenizer_type, tokenizer, eot_id, model, dtype):
    import lm_eval
    
    # requests=[lm_eval.api.instance.Instance,...] (e.g. multiple-choice answers)
    class LMEvalWrapper(lm_eval.api.model.LM):
        def __init__(self, tokenizer_type, tokenizer, eot_id, model, dtype):
            super().__init__()
            
            model.eval()

            self.tokenizer_type = tokenizer_type
            self.tokenizer = tokenizer
            self.eot_id = eot_id
            self.model = model
            self.device = next(self.model.parameters()).device
            self.dtype = dtype
        
        # lm_eval.api.instance.Instance.args=(condition, target)
        # Returns [(LL, is_greedy)], where:
        # 1) logP(T₀...Tₘ₋₁|C₀...Cₙ₋₁)=logP(T₀|C₀...Cₙ₋₁)+...+logP(Tₘ₋₁|C₀...Tₘ₋₂)
        # 2) is_greedy (0/1): If the target would be generated by greedy sampling
        @torch.no_grad()
        def loglikelihood(self, requests):
            responses = []
            for condition, target in [req.args for req in requests]:
                string = condition + target

                if self.tokenizer_type=="tokenizers":
                    condition_ids = self.tokenizer.encode(condition, add_special_tokens=False).ids
                    ids = self.tokenizer.encode(string, add_special_tokens=False).ids
                elif self.tokenizer_type=="tokenmonster":
                    # Does NOT have add_special_tokens=False
                    condition_ids = self.tokenizer.tokenize(condition).tolist()
                    ids = self.tokenizer.tokenize(string).tolist()
                n = len(condition_ids)
                
                # In token merging, the target starts from the first merged token
                token_merging = False
                for boundary in range(n):
                    if condition_ids[boundary]!=ids[boundary]:
                        token_merging = True
                        break
                if not token_merging: boundary = n
                
                X = torch.tensor(ids)

                self.model.eval()
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    # (1*)context*vocab_size
                    Y = self.model( X.to(self.device) )
                    # context*vocab_size
                    Y = Y.reshape(Y.shape[-2:])

                # P(T₀|C₀...Cₙ₋₁) comes from Cₙ₋₁ (boundary-1)
                # m*vocab_size
                logP = torch.nn.functional.log_softmax(Y[boundary-1:-1], dim=-1)
                
                target_ids = ids[boundary:]
                m = len(target_ids)
                # Advanced indexing
                LL = logP[range(m), target_ids].sum().item()
                
                greedy_ids = logP.argmax(dim=-1)
                is_greedy = (greedy_ids.tolist() == target_ids)
                is_greedy = int(is_greedy)

                responses.append((LL, is_greedy))

            return responses
        
        @torch.no_grad()
        def loglikelihood_rolling(self, requests):
            raise NotImplementedError
        
        @torch.no_grad()
        def generate_until(self, requests):
            raise NotImplementedError

    return LMEvalWrapper(tokenizer_type, tokenizer, eot_id, model, dtype)
