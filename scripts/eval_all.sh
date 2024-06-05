export adapter=$1;
export time=$2;
export device=$3;

echo $adapter;



for i in aidayago2-dev-kilt  cweb-dev-kilt wned-dev-kilt; do
  echo $i;
  bash scripts/eval.sh hf-D4 $i el $adapter $device $time;
done

for i in structured_zeroshot-dev-kilt  trex-dev-kilt; do
  echo $i;
  bash scripts/eval.sh hf-D4 $i sf $adapter $device $time;
done

for i in hotpotqa-dev-kilt nq-dev-kilt triviaqa-dev-kilt eli5-dev-kilt;do
  echo $i;
  bash scripts/eval.sh hf-D4 $i qa $adapter $device $time;
done


for i in fever-dev-kilt;do
  echo $i;
  bash scripts/eval.sh hf-D4 $i fc $adapter $device $time;
done

for i in wow-dev-kilt; do
  echo $i;
  bash scripts/eval.sh hf-D4 $i dlg $adapter $device $time;
done

