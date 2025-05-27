import torch
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
import pandas as pd
import os
from dotenv import load_dotenv
import boto3
from datasets import Dataset
import pandas as pd
# from clearml import Task, TaskTypes
from transformers import TrainerCallback
import random
import re
from jinja2 import Template


# TRAIN_PROMPT_TEMPLATE_V1 = """
# ## Overview
# You are a assistant that accepts a table of students grades.
# Your goal is to analyze this table and fill in the gaps in the estimates based on the available

# ## Grade Table
# =========
# {source_file}
# =========

# ## Сompleted table with grades
# =========
# """


TRAIN_TARGET_TEMPLATE_V1 = """{test_file}
"""


# If we want download checkpoints to s3 bucket
class S3UploadCallback(TrainerCallback):
    def __init__(self, s3_client, bucket_name, s3_prefix=''):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}")
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, args.output_dir)
                s3_path = os.path.join(self.s3_prefix, relative_path)
                self.s3_client.upload_file(
                    local_path, self.bucket_name, s3_path)


def load_model(model_path: str, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # load_in_4bit=True,
        # TODO check parametr
        # bnb_4bit_compute_dtype=torch.float16
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_and_split_dataset(dataset_path: str, tokenizer, max_seq_length=4096, train_proportion=0.85):
    df = pd.read_parquet(dataset_path)
    # df["focal_file_content"] = df["focal_file_info"].str["content"]
    # dataset = Dataset.from_pandas(df[['focal_file_content','test_file_content_no_bad_annot']])
    dataset = Dataset.from_pandas(df)

    def formatting_prompts_func(examples):  # extra function
        return {
            "text": tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": TRAIN_PROMPT_TEMPLATE_V1.format(
                            source_file=examples["focal_file_content"]
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": TRAIN_TARGET_TEMPLATE_V1.format(
                            test_file=examples["test_file_content_no_bad_annot"]
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(
        formatting_prompts_func,
        # batched=True,
    )

    random.seed(42)

    prompt_lens = pd.Series([len(tokenizer.encode(x))
                            for x in dataset['text']])
    dataset = dataset.select(prompt_lens.index[prompt_lens < max_seq_length])
    split_dataset = dataset.train_test_split(
        train_size=train_proportion, seed=42)

    # print('LENGHT OF DATASET = ', len(dataset))

    train_dataset, val_dataset = split_dataset['train'], split_dataset['test']

    # print('LENGHT OF TRAIN DATASET = ', len(train_dataset))
    # print('LENGHT OF VALID DATASET = ', len(val_dataset))

    return train_dataset, val_dataset


def initialize_s3client():
    S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
    S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY')

    s3_client = boto3.client(
        's3',
        # region_name = '',
        # endpoint_url = ,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

    return s3_client


# Find and download last model checkpoint
def download_checkpoint(s3_client, prefix='', bucket_name='', local_dir='data/model/lora_adapters'):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(
        Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    checkpoints = []
    for page in pages:
        if "CommonPrefixes" in page:
            for cp in page["CommonPrefixes"]:
                folder = cp["Prefix"]
                checkpoint_name = folder[len(prefix):].rstrip('/')
                match = re.match(r'checkpoint-(\d+)', checkpoint_name)
                if match:
                    num = int(match.group(1))
                    checkpoints.append((num, checkpoint_name))

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
        print('\nLAST CHECKPOINT = ', latest_checkpoint, '\n')
        checkpoint_folder = prefix + latest_checkpoint + '/'
        response = s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix=checkpoint_folder)

        for obj in response["Contents"]:
            key = obj["Key"]
            relative_path = os.path.relpath(key, checkpoint_folder)
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3_client.download_file(bucket_name, key, local_file_path)


def is_bfloat16_supported():
    return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False


def main():
    project_name = "educational-trajectories"
    task = Task.get_task(task_name=task_name, project_name=project_name)
    task = Task.init(
        task_name=task_name,
        project_name=project_name,
        reuse_last_task_id=task.id if task else False,
        continue_last_task=0 if task else False,
        task_type=TaskTypes.training,
        auto_resource_monitoring=True,
        auto_connect_streams=True,
        auto_connect_frameworks=False
    )

    max_seq_length = 4600  # 4096
    model, tokenizer = load_model(model_path="/data/model")

    # TODO check param
    # tokenizer.padding_side = 'left'

    # model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.01,
        use_rslora=False,
        target_modules=["q_proj", "k_proj", "v_proj",
                        "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        # use_dora=True,
    )
    model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    model = get_peft_model(model, peft_config)

    task.connect_configuration(peft_config.to_dict(), "peft_config")

    train_dataset, val_dataset = load_and_split_dataset(
        "/data/finetuning-data/", tokenizer, max_seq_length=max_seq_length)

    print('\nLEN_TRAIN_DATASET', len(train_dataset), '\n')
    print('\nLEN_VAL_DATASET', len(val_dataset), '\n')

    # 3_folder = os.getenv("name", "lora-dataset")

    s3_client = initialize_s3client()
    s3_callback = S3UploadCallback(
        s3_client=s3_client,
        bucket_name='',
        s3_prefix='weights/'+s3_folder
    )

    resume_checkpoint = None
    local_adapter_path = '/data/lora-checkpoint'
    try:
        download_checkpoint(s3_client, prefix='weights/' +
                            s3_folder+'/', local_dir=local_adapter_path)
        model.load_adapter(local_adapter_path, adapter_name='lora_adapter')
        resume_checkpoint = local_adapter_path
        print('\Continue training!\n')
    except:
        print('\nStarting training!\n')

    # train_steps =
    # eval_steps =
    gradient_accumulation_steps = 2
    # num_train_epochs = 15
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer),
        args=SFTConfig(
            dataset_num_proc=32,
            packing=False,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_checkpointing=True,
            # Fixed major bug in latest Unsloth
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps = 25,
            warmup_ratio=0.1,
            dataset_text_field="text",
            max_seq_length=None,
            # num_train_epochs = num_train_epochs, # Set this for 1 full training run.
            max_steps=train_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            logging_steps=1,
            dataloader_num_workers=32,
            # load_best_model_at_end=True,
            # metric_for_best_model="loss",
            # greater_is_better=False,
            learning_rate=5e-7,  # was 2e-4
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="paged_adamw_8bit",  # Save more memory
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir='./results',
            report_to="clearml",
            # neftune_noise_alpha=5,
            use_liger=True,
            # group_by_length=True,
        ),
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=6)], optimal
        callbacks=[s3_callback],
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    main()
