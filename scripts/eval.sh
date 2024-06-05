export MODEL=$1;
export DATASET=$2;
export PYTHONPATH=`pwd`:`pwd`/fairseq:$PYTHONPATH;
export TASK=$3;
export ADAPTER=$4;
export DEVICE=$5;
export TIME=$6;

CUDA_VISIBLE_DEVICES=$5 python scripts/evaluate_kilt_dataset.py \
    models/$1 \
    datasets/kilt++-test/$2/Q$6.jsonl \
    predictions/$1_$2.jsonl \
    --batch_size 32 \
    --beams 10 \
    --max_len_a 384 \
    --max_len_b 15 \
    --trie trie/D0-$6.pkl \
    --checkpoint_file checkpoint_best.pt;
     --adapter_path_list $4 \

