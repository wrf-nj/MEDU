CUDA_VISIBLE_DEVICES="2" \
python train.py \
--hiera_path "../downloadckpt/sam2_hiera_large.pt" \
--save_path "../log" \
--expname "abd_ours" \
--fseed 1 \
--epoch 1500 \
--lr 0.0003 \
--batch_size 20 \
--num_classes 5 \
--dataset "ABDOMINAL" \
--tr_domain 'CHAOST2' \
--per 0.2 \
--aug_type 'augseg_two_concat' \
--model_type 'few_two_concat_linear' \
--kl_w 15.0 \
--con_f 1 \
--dropoutflag 1
