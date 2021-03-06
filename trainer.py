import functools
import logging
import os

import torch
from tqdm import tqdm
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup

from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, model, dataloader, num_train_steps, writer, step_callback=None):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.num_train_steps = num_train_steps
        # self.num_train_steps = 100
        self.writer = writer
        self.step_callback = step_callback

        self.optimizer = self._create_optimizer(model)
        self.scheduler = self._create_scheduler(self.optimizer)

    def train(self):
        model = self.model
        optimizer = self.optimizer
        scaler = GradScaler()

        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        epoch = 0
        global_step = 0
        tr_loss = 0.0

        num_workers = torch.cuda.device_count()

        model.train()
        with tqdm(total=self.num_train_steps, disable=self.args.local_rank not in (-1, 0)) as pbar:
            while True:
                for step, batch in enumerate(self.dataloader):
                    inputs = {k: v.to(self.args.device) for k, v in self._create_model_arguments(batch).items()}

                    with autocast():
                        outputs = model(**inputs)
                        loss = outputs[0]

                    # loss.backward()
                    scaler.scale(loss).backward() # fp16

                    tr_loss += loss.item()
                    self.writer.report_loss(loss.item(), global_step)

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # self.optimizer.step()
                    # self.scheduler.step()
                    scaler.step(optimizer)
                    model.zero_grad()
                    scale = scaler.get_scale() # fp16
                    scaler.update() # fp16
                    skip_lr_sched = (scale != scaler.get_scale()) # fp16
                    if not skip_lr_sched: self.scheduler.step() # fp16

                    pbar.set_description("epoch: %d loss: %.7f" % (epoch, loss.item()))
                    pbar.update()
                    global_step += 1

                    if self.step_callback is not None:
                        self.step_callback(model, global_step)

                    if (
                        self.args.local_rank in (-1, 0)
                        and self.args.output_dir
                        and self.args.save_steps > 0
                        and global_step % self.args.save_steps == 0
                    ):
                        output_dir = os.path.join(self.args.output_dir, "checkpoint-{}".format(global_step))

                        if hasattr(model, "module"):
                            # torch.save(model.module.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
                            model.save_pretrained(output_dir)
                        else:
                            # torch.save(model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
                            model.save_pretrained(output_dir)
                    if global_step == self.num_train_steps:
                        break

                if global_step == self.num_train_steps:
                    break
                epoch += 1

        logger.info("global_step = %s, average loss = %s", global_step, tr_loss / global_step)

        return model, global_step, tr_loss / global_step

    def _create_optimizer(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(
            optimizer_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )

    def _create_scheduler(self, optimizer):
        warmup_steps = int(self.num_train_steps * 0.06)
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, self.num_train_steps)

    def _create_model_arguments(self, batch):
        return batch