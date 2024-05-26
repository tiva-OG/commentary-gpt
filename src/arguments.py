from argparse import ArgumentParser

parser = ArgumentParser(
    prog="commentary-gpt train",
    description="trains the commentary-gpt model"
)

# MODEL
model_keys = ["batch_size", "block_size", "dropout", "n_embed", "n_head", "n_layer", "vocab_size"]
model_group = parser.add_argument_group("model group")
model_group.add_argument("--batch_size", nargs="?", type=int, default=8, help="take the batch size")
model_group.add_argument("--block_size", nargs="?", type=int, default=32, help="take the block size")
model_group.add_argument("--dropout", nargs="?", type=float, default=0.2, help="take the dropout rate")
model_group.add_argument("--n_embed", nargs="?", type=int, default=64, help="take the embedding size")
model_group.add_argument("--n_head", nargs="?", type=int, default=6, help="take the head size")
model_group.add_argument("--n_layer", nargs="?", type=int, default=6, help="take the layer size")
model_group.add_argument("--vocab_size", nargs="?", type=int, default=102, help="take the vocab size")

# OPTIMIZER
optim_keys = ["learning_rate"]
optim_group = parser.add_argument_group("optimizer group")
optim_group.add_argument("-lr", "--learning_rate", nargs="?", type=float, default=3e-4, help="optimizer learning rate")

# TRAIN
train_keys = ["always_save_ckpt", "eval_interval", "eval_iters", "log_interval", "max_iters", "output_dir", "resume"]
train_group = parser.add_argument_group("train group")
train_group.add_argument("--always_save_ckpt", action="store_true", help="always save checkpoint after each eval")
train_group.add_argument("--eval_interval", nargs="?", type=int, default=200)
train_group.add_argument("--eval_iters", nargs="?", type=float, default=200)
train_group.add_argument("--log_interval", nargs="?", type=int, default=20)
train_group.add_argument("--max_iters", nargs="?", type=int, default=6000, help="total number of training iterations")
train_group.add_argument("--output_dir", nargs="?", type=str, default="./artifacts/checkpoint")
train_group.add_argument("-r", "--resume", action="store_true",
                         help="begin training from scratch or resume from checkpoint")

args = vars(parser.parse_args())

model_args = {k: args[k] for k in model_keys}
optim_args = {k: args[k] for k in optim_keys}
train_args = {k: args[k] for k in train_keys}
