from __future__ import absolute_import, division, print_function

from options import MonodepthOptions
from trainer_monopcc import Trainer


options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
