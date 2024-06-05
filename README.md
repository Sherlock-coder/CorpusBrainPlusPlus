# CorpusBrain++: A Continual Generative Pre-Training Framework for Knowledge-Intensive Language Tasks

![illustration](.\assets\illustration.png)

![model](.\assets\model.png)

## Data construction

Construct KILT++ with Wikipedia knowledge source and the original KILT dataset.

```
python utils/construct_dataset.py
```

## Construct prefix tree

Construct prefix tree for `D0` to `D4`.

```
python utils/construct_trie.py
```

## Train backbone model

Train the backbone model in the initial phase with `D0` and `R0`. 

For this procedure, please refer the [CorpusBrain](https://github.com/ict-bigdatalab/CorpusBrain.git) repository. 

Note that the backbone is trained with `fairseq` to facilitate efficiency, use the following script to convert the `fairseq` checkpoint to `huggingface` version.

```
python utils/convert_fairseq_huggingface.py [fairseq_path]
```

## Revisit old documents

```
python replay/kmeans.py
```

## Pre-training tasks

Generate query-document pairs for each specific task.

```
python tasks/[specific_task]/generate.py
```

## Continual learning

Continually pre-train the adapters with the backbone parameters frozen.

```
python train_adapter.py --task [task] --batch_size [batch_size] --config_file [config_file] --save_name [save_name] --lr [learning_rate] --max_steps [max_steps] -grad_acc [grad_acc] --eval_steps [eval_steps] --load_adapter_path [load_adapter_path] 
```

## Evaluation

```
python scripts/eval_all.sh
```

## Citation

```
@article{li2024matching,
  title={From Matching to Generation: A Survey on Generative Information 	    Retrieval},
  author={Li, Xiaoxi and Jin, Jiajie and Zhou, Yujia and Zhang, Yuyao and Zhang, Peitian and Zhu, Yutao and Dou, Zhicheng},
  journal={arXiv preprint arXiv:2404.14851},
  year={2024}
}
```

