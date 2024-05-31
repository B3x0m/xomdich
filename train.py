import os
import torch
import torch.nn as nn
import lightning as L
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from model import Transformer
from config import ModelArgs
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------
device = "cpu"
hidden_dim: int = 4096 # 832 # 1024
n_layers: int = 32 # 8 # 16
n_heads: int = 32 # 8 # 16
n_kv_heads: int = 2 
vocab_size: int = 20000 
ff_dim: int = 1024 # 256 # 512
norm_eps: float = 1e-5
batch_size: int = 12
seq_len: int = 512
dropout: float = 0.1

n_epochs = 3
n_steps = 10000
learning_rate = 1e-3
n_workers=4
ckpts_after=10

train_tokenizer = False
detect_anomaly = False
model_path = "./ckpts"
model_name = "Xomdich"
dataset_path = "./sample.txt" # "./sample.txt"
tokenizer_path = "./tokenizer_files"
exec(open('configurator.py').read())
# ------------------------------------------------------------------

params = ModelArgs(vocab_size, n_layers, n_heads,
                   n_kv_heads, hidden_dim, 
                   ff_dim, norm_eps,
                   batch_size, seq_len,
                   dropout, device)

if train_tokenizer:
    print("Training tokenizer...")
    tk_tokenizer = ByteLevelBPETokenizer()
    tokenizer_path = "tokenizer_files"
    tk_tokenizer.train(files=dataset_path, vocab_size=20000, min_frequency=2, special_tokens=[
        "<unk>",
        "<s>",
        "</s>",
        "<mask>",
        "<vi>",
        "<cv>",
        "<zh>"
    ])
    if not os.path.isdir(tokenizer_path):
        os.mkdir(tokenizer_path)
    tk_tokenizer.save_model(tokenizer_path)
    tokenizer = ByteLevelBPETokenizer(
        f"{tokenizer_path}/vocab.json",
        f"{tokenizer_path}/merges.txt",)

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.enable_truncation(max_length=512)
else:
    print("Loading saved tokenizer...")
    tokenizer = ByteLevelBPETokenizer(
        f"{tokenizer_path}/vocab.json",
        f"{tokenizer_path}/merges.txt",)

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)

if os.path.isfile(f"{dataset_path}.pt"):
    print("Have preprocessed dataset, loading....")
    dataset = torch.load(f'{dataset_path}.pt', map_location=torch.device(device)).to(device)
else:
    print("Preprocessing dataset...")
    def read_file_in_chunks(dataset_path, chunk_size=1024):
        with open(dataset_path, 'r', encoding='utf-8') as file:
            while True:
                data = file.read(chunk_size)
                if not data:
                    break
                yield data
    all_data = []
    for chunk in read_file_in_chunks(dataset_path):
        encoded_chunk = tokenizer.encode(chunk).ids
        tensor_chunk = torch.tensor(encoded_chunk, dtype=torch.long).to(device)
        all_data.append(tensor_chunk)
    dataset = torch.cat(all_data).to(device)
    torch.save(dataset, f'{dataset_path}.pt')
print("Total data tokens: " + str(dataset.shape))

class CustomDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.data = data
        self.max_seq_len = max_seq_len
    def __len__(self):
        return len(self.data) - self.max_seq_len
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.max_seq_len]
        y = self.data[idx+1:idx+self.max_seq_len+1]
        return x, y

n = int(0.9 * len(dataset))
class PrepareData(L.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        self.train_data = CustomDataset(dataset[:n], params.seq_len)
        self.val_data = CustomDataset(dataset[n:], params.seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=n_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=n_workers,
            drop_last=False,
            pin_memory=True,
        )

class CheckpointEveryNSteps(L.Callback):
    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: L.Trainer, _):
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

class LightningModel(L.LightningModule):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.model = Transformer(args)
        
    def forward(self, x, y):
        return self.model(x, y)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def calc_loss(self, preds, labels):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(preds.contiguous().view(-1, self.vocab_size), labels.contiguous().view(-1))
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        preds = self.model(x, y)
        loss = self.calc_loss(preds, y)

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        preds = self.model(x, y)
        loss = self.calc_loss(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)


model = LightningModel(params)
data = PrepareData()
trainer = L.Trainer(max_steps=n_steps, default_root_dir=model_path,
                    check_val_every_n_epoch=1000,
                    callbacks=[CheckpointEveryNSteps(ckpts_after)],
                    precision=16)
trainer.fit(model, data)
