import os
import argparse
import utils_data
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", choices=utils_data.DATASETS, default="climbmix10m")
parser.add_argument("--keep_in_memory", help="Keep datasets in RAM", type=bool, default=False)
parser.add_argument("--tokenizer_type", choices=utils_data.TOKENIZER_TYPES, help="Tokenizer library to use", default="tokenmonster")
parser.add_argument("--tokenizer", help="Name/URL/File of the tokenizer", default="https://huggingface.co/gvlassis/tokenmonster/resolve/main/englishcode-32000-strict-nocapcode-v1-eot%3D14199.vocab?download=true")
parser.add_argument("--eot_id", help="End-Of-Text token id", type=int, default=14199)
parser.add_argument("--batch_size", help="Size of each batch of datasets.Dataset.map(batch_size=)", type=int, default=1000)
parser.add_argument("--cores", help="CPU cores used by datasets.Dataset.map(num_proc=).", type=int, default=os.cpu_count()//2)
args=parser.parse_args()

script_path = os.path.abspath(__file__)
data_path = os.path.dirname(script_path)
src_path = os.path.dirname(data_path)
root_path = os.path.dirname(src_path)
dataset_path = f"{root_path}/{args.dataset}"
os.makedirs(dataset_path, exist_ok=True)

print("💾 Loading splits")
train_dataset, val_dataset, test_dataset = utils_data.get_splits(args.dataset, args.keep_in_memory)

print("✏️ Preprocessing")
train_X, train_Y = utils_data.dataset_to_tensors(train_dataset, args.tokenizer_type, args.tokenizer, args.eot_id, args.batch_size, args.cores)
test_X, test_Y = utils_data.dataset_to_tensors(test_dataset, args.tokenizer_type, args.tokenizer, args.eot_id, args.batch_size, args.cores)
val_X, val_Y = utils_data.dataset_to_tensors(val_dataset, args.tokenizer_type, args.tokenizer, args.eot_id, args.batch_size, args.cores)

print("💾 Saving splits")
torch.save(train_X, dataset_path+"/train_X.pt")
if train_Y is not None: torch.save(train_Y, dataset_path+"/train_Y.pt")
torch.save(val_X, dataset_path+"/val_X.pt")
if val_Y is not None: torch.save(val_Y, dataset_path+"/val_Y.pt")
torch.save(test_X, dataset_path+"/test_X.pt")
if test_Y is not None: torch.save(test_Y, dataset_path+"/test_Y.pt")

print("🍾 Done!")
