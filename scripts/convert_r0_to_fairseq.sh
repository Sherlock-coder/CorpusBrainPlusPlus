for data in `ls datasets/kilt++-train/*/R0.jsonl`; do
    shuf -n 500 $data >> $data.dev;
    python utils/convert_kilt_to_fairseq.py $data.dev data/R0/;
    python utils/convert_kilt_to_fairseq.py $data data/R0/;
    echo $data finished;
done
mv data/R0/R0.source data/R0/train.source;
mv data/R0/R0.target data/R0/train.target;
mv data/R0/R0.jsonl.dev.source data/R0/dev.source;
mv data/R0/R0.jsonl.dev.target data/R0/dev.target;
