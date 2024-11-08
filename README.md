# PyTorch implementation of[Rate-Adaptive Generative Semantic Communication Using Conditional Diffusion Models](https://arxiv.org/pdf/2409.02597) 

This repository is built upon [CDC_compression](https://github.com/buggyyang/CDC_compression) and [NTSCC](https://github.com/wsxtyrdd/NTSCC_JSAC22), thanks very much!

We would gradually upload the full-version of the implementation.

## Citation (Preprint Version)
``` bash
@article{yang2024rate,
  title={Rate-Adaptive Generative Semantic Communication Using Conditional Diffusion Models},
  author={Yang, Pujing and Zhang, Guangyi and Cai, Yunlong},
  journal={arXiv preprint arXiv:2409.02597},
  year={2024}
}
```


## Usage
### Clone
Clone this repository and enter the directory using the commands below:
```bash
git clone https://github.com/zhang-guangyi/cdm-jscc.git
cd cdm-jscc/
```

### Requirements
`Python 3.9.12` is recommended.

Install the required packages with:
```bash
pip install -r requirements.txt (Not provided yet)
```
If you're having issues with installing PyTorch compatible with your CUDA version, we strongly recommend related documentation page](https://pytorch.org/get-started/previous-versions/).

## Pretrained Models
They are provided in the  ./ckpt folder.

## Usage
Example of test the CDM-JSCC model:
1. When evaluating FID metric, images are cropped to non-overlapping patches of $256 \times 256$, for example, Kodak images are cropped to 144 patches of $256 \times 256$:
```bash
python crop.py
```
2. Run test.py
- Evaluating CDM-JSCC-P model at an average CBR of $1/48$ across SNR of $10$dB:
```bash
python test.py --img_dir path_to_testimgdir --cropped_input_dir path_to_croppedimgdir --snr 10 --root path_to_cdm-jscc --ckpt ckpt/cbr1_48-eta0.5-snr10.pt --mask_ratio 0.13 --n_denoise_step 17
``` 

- Evaluating CDM-JSCC-D model at an average CBR of $1/48$ across SNR of $10$dB:
```bash
python test.py --img_dir path_to_testimgdir --cropped_input_dir path_to_croppedimgdir --snr 10 --root path_to_cdm-jscc --ckpt ckpt/cbr1_48-eta0.1-snr10.pt --mask_ratio 0.15 --n_denoise_step 1
``` 

- Evaluating CDM-JSCC-P model at an average CBR of $1/24$ across SNR of $10$dB:
```bash
python test.py --img_dir path_to_testimgdir --cropped_input_dir path_to_croppedimgdir --snr 10 --root path_to_cdm-jscc --ckpt ckpt/cbr1_24-eta0.5-snr10.pt --mask_ratio 0.316 --n_denoise_step 17
``` 

- Evaluating CDM-JSCC-D model at an average CBR of $1/24$ across SNR of $10$dB:
```bash
python test.py --img_dir path_to_testimgdir --cropped_input_dir path_to_croppedimgdir --snr 10 --root path_to_cdm-jscc --ckpt ckpt/cbr1_24-eta0.1-snr10.pt --mask_ratio 0.41 --n_denoise_step 1
```

