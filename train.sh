python train.py --device 1 --batch_size 4 --log_checkpoint_step 100 --lr 5e-5 \
--sample_steps 1  --aux_weight 0.5 --n_step 400100 --load_model --finetune --snr 10 \
--train_data_dir SharedData/Kodak \
--test_data_dir SharedData/Kodak \
--root ./ \
--ckpt ckpt/cbr1_24-eta0.1-snr10.pt