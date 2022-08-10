from main import train, get_data, eval
from utils import set_params
import wandb

args = set_params()
wandb.init(config=args)

data = get_data(wandb.config)
embeds = train(wandb.config, data)
metrics = eval(wandb.config, data, embeds)
wandb.log(metrics)
