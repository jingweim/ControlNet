import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from share import *
from my_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, required=True, 
                        help='config file path')

    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="name of experiment, e.g. fill_run0, tgbh_run1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="set seed for reproducing experiments",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default='./models/control_sd15_ini.ckpt',
        help="path to model we resume training from.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="for a40, the max is 10",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="same as number of cpus",
    )
    parser.add_argument(
        "--logger_freq",
        type=int,
        required=True,
        help="for every this number of steps, save images and model checkpoints; \
            if this number bigger than data_size/batch_size, save at the end of epoch",
    )
    parser.add_argument(
        "--save_ckpt_starting_epoch",
        type=int,
        default=0,
        help="start saving model checkpoint at this epoch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="learning rate for control net parameters",
    )
    parser.add_argument(
        "--sd_locked",
        action="store_true",
        default=True,
        help="sd weights frozen if true",
    )
    parser.add_argument(
        "--only_mid_control",
        action="store_true",
        default=False,
        help="no decoder connection if True",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="path to prompt file",
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="directory to input conditions (e.g. edge map)",
    )
    parser.add_argument(
        "--tgt_dir",
        type=str,
        required=True,
        help="directory to output images",
    )

    args = parser.parse_args()
    return args


# load configs and set seed
args = config_parser()
seed_everything(args.seed)


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
model.learning_rate = args.learning_rate
model.sd_locked = args.sd_locked
model.only_mid_control = args.only_mid_control
model.save_ckpt_starting_epoch = args.save_ckpt_starting_epoch


# Misc
dataset = MyDataset(args.exp_name, args.prompt_path, args.src_dir, args.tgt_dir)
dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
mc = ModelCheckpoint(every_n_train_steps=args.logger_freq, save_top_k=-1)
logger = ImageLogger(batch_frequency=args.logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[mc, logger])


# save configs
os.makedirs(trainer.logger.log_dir)
f = os.path.join(trainer.logger.log_dir, 'args.txt')
with open(f, 'w') as file:
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        file.write('{} = {}\n'.format(arg, attr))
if args.config is not None:
    f = os.path.join(trainer.logger.log_dir, 'config.txt')
    with open(f, 'w') as file:
        file.write(open(args.config, 'r').read())


# Train!
trainer.fit(model, dataloader)
