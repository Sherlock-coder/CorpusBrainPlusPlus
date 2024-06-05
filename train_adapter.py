import jsonlines
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqAdapterTrainer
from datasets import load_dataset, load_from_disk
from transformers import Seq2SeqTrainingArguments
import random
import argparse
import sys
import os

model_path = './models/hf-R04D0/'
if_load_from_disk = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="sf",
        help="Task",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./",
        help="Config file",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="sf",
        help="Save adapter name",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--rehearsal",
        default=False,
        action='store_true',
        help="Revisit old documents",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action='store_true',
        help="Continually train with resume",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30000,
        help="Max steps",
    )

    parser.add_argument(
        "--grad_acc",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="evaluation steps",
    )

    parser.add_argument(
        "--load_adapter_path",
        type=str,
        nargs='*',
        help="path to load a pre-trained adapter",
    )

    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    task = args.task
    batch_size = args.batch_size
    save_name = args.save_name
    save_adapter_path = f'./adapters/{task}/{task}-{save_name}/'
    load_adapter_path = args.load_adapter_path

    if not os.path.exists(save_adapter_path):
        os.mkdir(save_adapter_path)
    sys.stdout = open(save_adapter_path + 'stdout.log', 'a')
    sys.stderr = open(save_adapter_path + 'stderr.log', 'a')

    print(args)

    train_dataset_path = './pretrain_tasks/' + task + '/generated_data.jsonl' 
    eval_dataset_path = './pretrain_tasks/' + task + '/generated_data.jsonl.dev'
    if args.rehearsal:
        print("rehearsal！")
        train_dataset_path = './pretrain_tasks/all/' + task + '/all_generated_data.jsonl'
        eval_dataset_path = './pretrain_tasks/all/' + task + '/all_generated_data.jsonl.dev'
    disk_train_path = './adapters/' + task+ '/tokenized_train_dataset/'
    disk_eval_path = './adapters/' + task+ '/tokenized_eval_dataset/'   



    training_args = Seq2SeqTrainingArguments(
        output_dir= save_adapter_path,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        metric_for_best_model='eval_loss',
        weight_decay=0.01,
        save_total_limit=1,
        max_steps=args.max_steps,
        # eval_delay=30000,
        load_best_model_at_end=True,
        warmup_ratio=0.1,
        label_smoothing_factor=0.1,
        gradient_accumulation_steps=args.grad_acc,
        fp16=True
    )


    if not if_load_from_disk:
        train_dataset = load_dataset("json", data_files=train_dataset_path)
        print(train_dataset)
        train_dataset = train_dataset.shuffle(seed=args.seed)
        eval_dataset = load_dataset("json", data_files=eval_dataset_path)
        print(eval_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["source"], max_length=512, truncation=True, add_special_tokens=True)

        labels = tokenizer(text=examples["target"], max_length=64, truncation=True, add_special_tokens=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    if not if_load_from_disk:
        tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=8)
        # tokenized_train_dataset.save_to_disk(disk_train_path)
        tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=8)
        # tokenized_eval_dataset.save_to_disk(disk_eval_path)

    if if_load_from_disk:
        tokenized_train_dataset = load_from_disk(disk_train_path, keep_in_memory=False)
        tokenized_eval_dataset = load_from_disk(disk_eval_path, keep_in_memory=False)

    name = task
    if load_adapter_path:
        for path in load_adapter_path:
            name = model.load_adapter(path)
            model.set_active_adapters(name)
            print(f'load adapter checkpoint from {path}')
    else:
        model.add_adapter(name, config=args.config_file)
        model.set_active_adapters(name)
    model.train_adapter(name)

    trainer = Seq2SeqAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset['train'],
        eval_dataset=tokenized_eval_dataset['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None
    )

    trainer.train(resume_from_checkpoint=args.resume)

    model.save_all_adapters(save_adapter_path, with_head=False)