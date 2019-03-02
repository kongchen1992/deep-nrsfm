subject=23
root="/dataset/chenk/release"
if [ ! -d "$root" ]; then
  mkdir "$root"
fi
root+="/$subject"
if [ ! -d "$root" ]; then
  mkdir "$root"
fi

nice -n 10 python3 train.py \
--gpu=3 \
--model_dir=$root \
--save_checkpoints_steps=1000 \
--keep_checkpoint_max=0 \
--max_steps=200000 \
--subject=$subject \
--buffer_size=1000 \
--throttle_secs=60 \
--hparams="num_atoms=125,num_points=31,learning_rate=0.001,loss_norm=2,
batch_size=32,num_dictionaries=12,num_atoms_bottleneck=10"
