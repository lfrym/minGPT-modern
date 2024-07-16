import datasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import wandb

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, batch_end_callback

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up data
dataset = datasets.load_dataset('Skylion007/openwebtext', trust_remote_code=True)
train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# create a GPT instance
model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257 # openai's model vocabulary
model_config.block_size = 1024  # openai's model block_size
model = GPT(model_config)

# train the model
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-5
train_config.max_iters = 2000
train_config.num_workers = 0
train_config.scheduler = 'cosine'

trainer = Trainer(train_config, model, train_dataset)
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

# evaluate the model
test_loss = trainer.evaluate(test_dataset)