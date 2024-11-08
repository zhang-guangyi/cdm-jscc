import torch
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter

from .utils import cycle
from torch.optim.lr_scheduler import LambdaLR
from ema_pytorch import EMA
from torch.cuda.amp import GradScaler
from collections import defaultdict 
from timm.utils import AverageMeter
import numpy as np
import lpips

def batch_psnr(imgs1, imgs2):
    with torch.no_grad():
        batch_mse = torch.mean((imgs1 - imgs2) ** 2, axis=(1, 2, 3))
        batch_psnr = 20 * torch.log10(1.0 / torch.sqrt(batch_mse))
        return torch.mean(batch_psnr)

# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        sample_steps,
        diffusion_model,
        train_dl,
        val_dl, 
        scheduler_function,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        save_and_sample_every=1000,
        results_folder="./results",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        ema_decay=0.999,
        ema_update_interval=10,
        ema_step_start=100,
        use_mixed_precision=False,
        finetune=False
    ):
        super().__init__()
        self.model = diffusion_model
        self.val_num_of_batch = val_num_of_batch
        self.sample_steps = sample_steps
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps

        self.train_dl_class = train_dl
        self.val_dl_class = val_dl
        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)
        self.finetune = finetune
        if optimizer == "adam":
            if self.finetune:
                self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=train_lr)
            else:
                self.opt = Adam(self.model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_interval, power=0.75, update_after_step=ema_step_start)
        if use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name = model_name


        self.fn_alex = lpips.LPIPS(net="alex", eval_mode=True).to(rank)

    def save(self):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
        }
        idx = (self.step // self.save_and_sample_every) % 3
        torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))

    def save111(self):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
        }
        idx = (self.step // self.save_and_sample_every) // 20
        torch.save(data, str(self.results_folder / f"{self.model_name}_line_{idx}.pt"))

    def load(self, idx=0, load_step=True, ckpt=None):
        # load_weights(self.ema, ckpt)
        data = torch.load(
            ckpt,
            map_location=lambda storage, loc: storage,
        )
        all_params = data["model"].keys()
        poped_params = []
        for key in all_params:
            if "train_" in key or 'attn_mask' in key or 'rate_adaption.mask' in key or 'loss_fn_vgg' in key or 'attn.rope.rotations' in key:
            # if "train_" in key:
                poped_params.append(key)
        for key in poped_params:
            data["model"].pop(key)
        if "ema" in data.keys():
            all_params = data["ema"].keys()
            poped_params = []
            for key in all_params: 
                # if "train_" in key:
                if "train_" in key or 'attn_mask' in key or 'rate_adaption.mask' in key or 'loss_fn_vgg' in key or 'attn.rope.rotations' in key:
                    poped_params.append(key)
            for key in poped_params:
                data["ema"].pop(key)
        print(load_step)
        if load_step:
            self.step = data["step"]
            print(load_step)
        self.model.load_state_dict(data["model"], strict=False)
        if "ema" not in data.keys():
            self.ema.ema_model.load_state_dict(data["model"], strict=False)
        else:
            self.ema.load_state_dict(data["ema"], strict=False)

    def train(self,logger):
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info('The number of parameters is %s.' % n_parameters)
         
        if self.finetune:
            for i, (name, param) in enumerate(self.model.named_parameters()):
                # print(name)
                if name.startswith('context_fn.fe.') or name.startswith('context_fn.fd.'):
                    print(i, name)
                else:
                    param.requires_grad = False
                

        while self.step < self.train_num_steps:
            # for i,p in enumerate(self.model.parameters()):
            #     print(i, p)

            self.opt.zero_grad()
            if (self.step >= self.scheduler_checkpoint_step) and (self.step != 0):
                self.scheduler.step()
            data = next(self.train_dl).to(self.device)
            self.model.train()
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, aloss = self.model(data * 2.0 - 1.)
                self.scaler.scale(loss).backward()
                self.scaler.scale(aloss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss, aloss = self.model(data * 2.0 - 1.)
                loss.backward()
                if self.finetune==False:
                    aloss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

            # logger.info('======Loss %s ======' % loss)
            self.ema.update()

            all_image_stats = defaultdict(AverageMeter)
            if (self.step % self.save_and_sample_every == 0): 
                logger.info('======Current step %s ======' % self.step)
                # milestone = self.step // self.save_and_sample_every
                for i, batch in enumerate(self.val_dl):
                    if i >= self.val_num_of_batch:
                        break
                    self.ema.ema_model.eval()
                    compressed, bpp, cbr_y = self.ema.ema_model.compress(
                        batch.to(self.device) * 2.0 - 1.0, self.sample_steps 
                    )
                    compressed = (compressed + 1.0) * 0.5
                    psnr = batch_psnr(compressed.clamp(0.0, 1.0).to('cpu'), batch[0])
                        # capacity-achieving channel code
                    llpips = self.fn_alex(batch.to(self.device), compressed)
                    cbr_sideinfo = np.log2(16) / (16 * 16 * 3) / np.log2(
        1 + 10 ** (10.0 / 10))
                    log = (' | '.join([
                f'bpp {bpp:.4f}',
                f'cbr_y {cbr_y}',
                f'cbr {cbr_y+cbr_sideinfo}',
                f'PSNR {psnr}',
                f'LPIPS {llpips}',
            ]))
                    # logger.info(log)

                    stats = {
            'bpp':  float(bpp),
            'cbr':  float(cbr_y + cbr_sideinfo),
            'psnr': float(psnr),
            'lpips': float(llpips)
        }
                    # accumulate stats
                    for k,v in stats.items():
                            all_image_stats[k].update(v)
                results = {k: meter.avg for k,meter in all_image_stats.items()}
                # print(results)
                logger.info(results)


                self.save()
                # self.save111()

            self.step += 1
        self.save()
        print("training completed")
