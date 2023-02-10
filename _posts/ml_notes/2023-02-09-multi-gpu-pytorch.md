---
title: Multi-GPU Pytorch training
blog_type: ml_notes
excerpt: Going from Andrej Karpathy's GPT video to multi-GPU dataparallel training.
layout: post_with_toc_lvl3
last_modified_at: 2023-02-09
---

### [Optional] Backstory
The last time I trained NN's was in 2017 where I was training RNNs for text. We
simply used a massive machine with a single GPU for training and it worked just
fine for our case. The hope of this blog is to break the shackles and do multi gpu
training.

### Prereqs
I am going to be taking off from where Andrek Karpathy's GPT lecture left off, that
is training a character level GPT decoder model on a single GPU training over the
Shakespeare dataset to write Shakespeare like text.

See [video here](https://www.youtube.com/watch?v=kCc8FmEb1nY).

### Goals for this doc
The goal here is to mod his code to do multi-gpu training (data-parallel training). His
repo [NanoGPT](https://github.com/karpathy/nanoGPT/blob/master/train.py) makes this
possible already but I found the code in the training script hard to follow. I love
his idea of brevity in code, but personally I found the code too terse to parse easily.

### High level approach
Main idea iss to use Pytorch Distributed Data Parallel API (DDP). With this system we
will make copies of the model one per GPU, in parallel train across GPUs and synchronise
the weights.

1. Refactor code in the lecture into functions. You can see an initial version of the
   refactored code [here](https://github.com/psvishnu91/andrej_lectures/tree/fe9cf11f52184ebcc5d7e1f8c4124cfcd2952a18/gpt).
   No logic changes but just moving the script into functions.
2. Move the `get_batch` function into a pytorch `Dataset` and then a pytorch
   `Dataloader`. This will help us do dataloading in parallel and also to use the
   `DistributedSampler`. The `DistributedSampler` class will guarantee each multiprocess
   working with a specific gpu gets a different batch of training data.
3. Define rank and global_rank for this process ie., for this gpu.
4. Wrap the model DDP and run.

### Steps
#### 1: Refactors
* We move all the constants or hyperparameters of the model into a hyperparam dataclass

{: .code title="Get batch code in Andrej's video" .x}
``` python
@dataclass(frozen=True)
class TrainConfig:

    batch_sz: int
    save_every: int
    learning_rate: float
    eval_freq: int
    max_iters: int
    eval_sz: int = 2000
    train_frac: float = 0.9
    checkpoint_folder: str = f"logging/checkpoints/{RUN_ID}/"


@dataclass(frozen=True)
class TransformerConfig:
    block_sz: int
    # if you update this also update mlp_hidden_dim
    embed_dim: int
    # embed_dim * 4
    mlp_hidden_dim: int
    num_attn_heads: int
    dropout_frac: float
    # number of decoder transformer blocks stacked one on top of another
    num_blocks: int


@dataclass(frozen=True)
class HyperParams:
    transformer_cfg: TransformerConfig
    train_cfg: TrainConfig


FULL_MODEL_HYPERPARAMS = HyperParams(
    train_cfg=TrainConfig(
        batch_sz=128,
        save_every=300,
        learning_rate=3e-4,
        eval_freq=300,
        max_iters=1_001,
    ),
    transformer_cfg=TransformerConfig(
        block_sz=256,
        embed_dim=384,
        mlp_hidden_dim=384 * 4,
        num_attn_heads=6,
        dropout_frac=0.2,
        num_blocks=6,
    ),
)
HYPERPARAMS = FULL_MODEL_HYPERPARAMS
```
* We checkpoint the model time to time. We only checkpoint if we are better than the
  previous validation score.

{: .code title="Code to checkpoint the model" .x}
``` python
# We want to always checkpoint in the last iteration
if ((i == num_iters - 1) or (i % save_every == 0)) and (
    eval_losses["val"] < prev_val_loss
):
    _checkpoint(
        model=model,
        checkpoint_folder=checkpoint_folder,
        it=i + START_IT,
        use_multigpu=use_multigpu,
        device=device,
    )

def _checkpoint(model, checkpoint_folder, it, use_multigpu, device):
    if device not in {"cpu", 0}:
        # Don't save unless you're CPU or the first GPU.
        # All GPUs possess an identical copy and we don't want each process to
        # save a checkpoint.
        return
    fl = os.path.join(checkpoint_folder, f"{it}.pt")
    msg = f"Iteration {it}: Checkpointing model at {fl}"
    print(msg)
    state_dict = model.module.state_dict() if use_multigpu else model.state_dict()
    torch.save(state_dict, f=fl)
```
* We all the script logic into functions. Essentially we create a `wrapper` function
  as below

{: .code title="Skeleton of the wrapper function" .x}
``` python
def wrapper(...):
    train_data, val_data, vocab_sz, ixtoc = load_input_file(input_fl)
    train_data, val_data = get_data_split(train_data, val_data, configs)
    model = build_model(vocab_sz, load_model_ckpt_path, configs)
    print(f"Examples BEFORE training the model")
    gpt.print_examples(model)  # type: ignore
    train_model(
        model=model,
        train_dl=train_dl,
        get_batch_fn=ft.partial(get_batch, train_data=train_data, val_data=val_data),
        optimizer=torch.optim.AdamW(params=model.parameters(), lr=learning_rate),
        eval_loss_fn=ft.partial(
            eval_loss, train_dl=train_dl, val_dl=val_dl, eval_sz=eval_sz
        ),
        configs=configs,
    )
    print(f"Examples AFTER training the model")
    gpt.print_examples(**print_example_kwargs)  # type: ignore
```

#### 2: Moving `get_batch` to Dataloader

We need to move this simple batch code in the video.

{: .code title="Get batch code in Andrej's video" .x}
``` python
def get_batch_helper(split, batch_sz, train_data, val_data, block_sz, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_sz, (batch_sz,))
    x = torch.stack([data[i : i + block_sz] for i in ix])
    y = torch.stack([data[i + 1 : i + block_sz + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

{: .code title="Get batch moved into a `Dataset` and a `Dataloader` class" .x}
``` python
@dataclass
class DecoderDataset(data_utils.Dataset):

    block_sz: int
    device: str
    data: torch.Tensor

    def __post_init__(self):
        super().__init__()
        self.len = len(self.data) - self.block_sz - 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index
        x = self.data[i : i + self.block_sz]
        y = self.data[i + 1 : i + self.block_sz + 1]
        return x.to(self.device), y.to(self.device)

def get_dataloaders(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_sz: int,
    batch_sz: int,
    device: str,
):
    train_dl = data_utils.DataLoader(
        dataset=DecoderDataset(data=train_data, block_sz=block_sz, device=device),
        batch_size=batch_sz, shuffle=False, sampler=None,
    )
    val_dl = data_utils.DataLoader(
        dataset=DecoderDataset(data=val_data, block_sz=block_sz, device=device),
        batch_size=batch_sz, shuffle=False, sampler=None,
    )
    return train_dl, val_dl
```

Our evaluation function now needs to use the dataloaders instead of the `get_batch`
function.

{: .code title="**Old** eval function using 'get_batch' function" .x}
``` python
@torch.no_grad()
def eval_loss(model, get_batch_fn, eval_sz):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        Xbatch, Ybatch = get_batch_fn(split=split, batch_sz=eval_sz)
        # fwd, loss = model(X, target)
        out[split] = eval_model_loss(model=model, X=Xbatch, Ytrue=Ybatch).item()
    model.train()
    return out
```

{: .code title="**New** eval function using 'dataloaders'" .x}
``` python
@torch.no_grad()
def eval_loss(model, train_dl, val_dl, eval_sz):
    model.eval()
    out = {}
    # Dataloader issues out x, y of length batch_sz but we want eval_sz. Find the
    #   number of batch_sz we need and vstack them.
    # next(iter(train_dl)) -> x, y
    batch_sz = next(iter(train_dl))[0].shape[0]
    # slightly more than eval sz but 🤷
    stacks = (eval_sz // batch_sz) + 1
    for split, dl in [("train", train_dl), ("val", val_dl)]:
        Xbatches, Ybatches = [], []
        for _ in range(stacks):
            Xb, Yb = next(iter(dl))
            Xbatches.append(Xb)
            Ybatches.append(Yb)
        Xeval, Yeval = torch.vstack(Xbatches), torch.vstack(Ybatches)
        out[split] = eval_model_loss(model=model, X=Xeval, Ytrue=Yeval).item()
    model.train()
    return out
```
<div class="panel panel-default">
<div class="panel-body">
Gotcha: To get the next batch from `dataloader`, we need to first wrap it in `iter` and
do `next(iter(dataloader))`.
</div>
</div>

#### 2: Wrap with `DistributedDataParallel`

1. We will first define a global parameter called use `USE_MULTIGPU` which will be passed
into our main function.
2. We will setup `DistributedDataParallel` environment variables which will be the
  master node's address and port. Since we are using only one machine this will be
  `localhost` and any random port.
3. We now create a process group one per GPU with `init_process_group`
  setting the backend to nvidia's `nccl` backend.
4. We will then wrap our model with `DDP` in the `build_model` function.
5. We will also have to update `save_ckpt` function because now `model.state_dict`
  will need to become `model.module.state_dict`. Also we only checkpoint the model from
  one GPU as they all possess an identical copy and we don't want each process to
  save a checkpoint.
6. We need to update our `main` function to take in `rank` and `world_size` where `world_size`
  is the number of GPUs (`world_size = torch.cuda.device_count()`).
7. We will now need to use `torch.multiprocessing.spawn` to spawn our `main` function
  with `world_size` and other necessary args passed in through `args` param of the
  spawn method. We will not pass in `rank` as this will populated by the `spawn`
  method. We will also need to pass in `nprocs` to `spawn` which will be the number of
  gpus ie., `world_size`.


<div class="panel panel-default">
<div class="panel-body">
Gotcha: While creating a `Dataloader`, make sure to set `pin_memory=False` if we are
moving the data to GPU within the `Dataset` class.
</div>
</div>

#### 3: Run and hopefully everything will just work
[Here's](https://github.com/psvishnu91/andrej_lectures/blob/683bcce7cfa3b53dcf66aa2c1db6741c75123a62/gpt/train_gpt.py)
the final version of the code that worked with multi-gpu dataparallel training.
