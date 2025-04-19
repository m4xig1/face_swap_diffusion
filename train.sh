CUDA_VISIBLE_DEVICES=0 python -u main_swap.py \
--logdir checkpoints/REFace/ \
--pretrained_model pretrained_models/sd1.4-paint-by-example.ckpt \
--base configs/train_id_only.yaml \
--scale_lr False \
--debug False \
--train_from_scratch True # we load unet cpt from diffusers