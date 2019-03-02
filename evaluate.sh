subject=23
step=184000
root="/dataset/chenk/release/$subject"
output="/dataset/chenk/release/"

nice -n 10 python3 evaluate.py \
--gpu=3 \
--model_dir=$root \
--subject=$subject \
--checkpoint=$step \
--error_metrics_dir="$output/$subject.csv" \
--predictions_dir="$output/$subject.npz" \
--hparams="num_atoms=125,num_points=31,learning_rate=0.001,loss_norm=2,
batch_size=32,num_dictionaries=12,num_atoms_bottleneck=10"
