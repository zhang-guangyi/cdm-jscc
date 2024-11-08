import argparse
import os
import torch
import torchvision 
import numpy as np
import pathlib
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.compress_modules import ResnetCompressor
from ema_pytorch import EMA

import torchvision.transforms.functional as tvf
import math
from PIL import Image
from collections import defaultdict 
from timm.utils import AverageMeter
import lpips
from pytorch_fid import fid_score
from torchvision import transforms
import logging
from modules.utils import logger_configuration, load_weights
from datetime import datetime

parser = argparse.ArgumentParser(description="values from bash script")

parser.add_argument("--ckpt", type=str, 
    default=None) # ckpt path
parser.add_argument("--gamma", type=float, default=0.8) # noise intensity for decoding
parser.add_argument("--n_denoise_step", type=int, default=17) # 1: CDM-JSCC-D; 17: CDM-JSCC-P
parser.add_argument("--device", type=int, default=0) # gpu device index
parser.add_argument("--snr", type=float, default=10) # snr
parser.add_argument("--img_dir", type=str, default=None) # img path
parser.add_argument("--cropped_input_dir", type=str, default='/home/yangpujing/SharedData/coco/cropped') # cropped_img path
parser.add_argument("--lpips_weight", type=float, default=0.5) # either 0.5 or 0.1, note that this must match the ckpt you use.
parser.add_argument("--root", type=str, default=None, help="root of the project")
parser.add_argument("--mask_ratio", type=float, default=0.13) # 0.13: cbr=1/48, eta0.5; 0.15: cbr=1/48, eta0.1

config = parser.parse_args()

def seed_initial(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_dirs(workdir):
    out_dir = workdir + '/samples_step' + str(config.n_denoise_step)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cropped_dir = workdir + '/cropped_step' + str(config.n_denoise_step)
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir, exist_ok=True)    
    return out_dir, cropped_dir

def main(rank):
    seed_initial(0)
    workdir, logger = logger_configuration(config.root, datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), phase='test')
    sampledir = workdir + '/samples'
    config.out_dir, cropped_dir = make_dirs(sampledir)
    logger.info(config.__dict__)

    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=64,
        dim_mults=[1,2,3,4,5,6],
        context_dim_mults=[1,2,3,4],
        embd_type="01",
    )

    context_model = ResnetCompressor(
        dim=64,
        dim_mults=[1,2,3,4],
        reverse_dim_mults=[4,3,2,1],
        hyper_dims_mults=[4,4,4],
        channels=3,
        out_channels=64,
        snr = config.snr,
        mask_ratio = config.mask_ratio,
    ) 

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        ae_fn=None,
        num_timesteps=8193,
        loss_type="l2",
        lagrangian=0.0032,
        pred_mode="x",
        aux_loss_weight=config.lpips_weight,
        aux_loss_type="lpips",
        var_schedule="cosine",
        use_loss_weight=True,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False,
    )
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)

    load_weights(ema, config.ckpt) 
    step = torch.load(config.ckpt)["step"]
    logger.info('The Step is %s.' % step)
    diffusion = ema.ema_model
    diffusion.to(rank)
    diffusion.eval()
    fn_vgg = lpips.LPIPS(net="alex", eval_mode=True).to(rank)


    all_image_stats = defaultdict(AverageMeter)
    i=0 
    for img in os.listdir(config.img_dir):
        i+=1
        if img.endswith(".png") or img.endswith(".jpg"):
            to_be_compressed = torchvision.io.read_image(os.path.join(config.img_dir, img)).unsqueeze(0).float().to(rank) / 255.0
            
            # input image size must be (128*m, 128*n), int m, n
            target_size = (to_be_compressed.shape[2]-to_be_compressed.shape[2]%128,  to_be_compressed.shape[3]-to_be_compressed.shape[3]%128)
            center_crop = transforms.CenterCrop(target_size)
            to_be_compressed = center_crop(to_be_compressed)
 
            compressed, bpp, cbr_y = diffusion.compress(
                to_be_compressed * 2.0 - 1.0,
                sample_steps=config.n_denoise_step,
                bpp_return_mean=True,
                init=torch.randn_like(to_be_compressed) * config.gamma
            )
            compressed = compressed.clamp(-1, 1) / 2.0 + 0.5
            pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(compressed.cpu(), os.path.join(config.out_dir, img))

            # compute psnr
            device=compressed.device
            real = to_be_compressed
            mse = (real - compressed).square().mean().item()
            psnr = -10 * math.log10(mse) 

            # compute lpips 
            llpips = fn_vgg(real, compressed)

            # sidelink cbr for transmitting y
            cbr_sideinfo = np.log2(16) / (16 * 16 * 3) / np.log2(1 + 10 ** (config.snr / 10))

        stats = {
            'cbr': float(cbr_y+cbr_sideinfo),
            'psnr': float(psnr),
            'lpips': float(llpips)
        }
        log = (' | '.join([
                f'cbr {cbr_y+cbr_sideinfo}',
                f'PSNR {psnr}',
                f'LPIPS {llpips.item()}',
            ]))
        logger.info(log)


        # accumulate stats
        for k,v in stats.items():
            all_image_stats[k].update(v)
    
    results = {k: meter.avg for k,meter in all_image_stats.items()}

    # ************************************Compute FID************************************* #
    # crop output imgs to 256*256
    for img_name in os.listdir(config.out_dir):
        img_path = os.path.join(config.out_dir, img_name)
        if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width < 256 or height < 256:
                        print(f"Skipping {img_name}, image size {width}x{height} is smaller than 256x256.")
                        continue
                    top=0
                    img_name, img_ext = os.path.splitext(img_name)
                    while (top+256) <= height:
                        bottom = top + 256
                        left=0
                        while (left+256) <= width:
                            right = left + 256
                            cropped_img = img.crop((left, top, right, bottom))
                            output_path = os.path.join(cropped_dir, img_name+'_left'+str(left//256)+'_top'+str(top//256)+img_ext)
                            cropped_img.save(output_path)
                            print(f"Saved cropped image: {output_path}")
                            left = right
                        top = bottom
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # *********************************calculate FID score***************************************  
    fid_value = fid_score.calculate_fid_given_paths(paths=[config.cropped_input_dir, cropped_dir], batch_size=12, device = device, dims = 2048)
    
    results['fid'] = fid_value
    logger.info(results)


if __name__ == "__main__":
    main(config.device)
