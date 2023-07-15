import shutil

import torch
from transformers import GPT2Tokenizer, \
    GPT2LMHeadModel, \
    AutoConfig, \
    DataCollatorForLanguageModeling, \
    Trainer, \
    TrainingArguments, LlamaForCausalLM, GPT2Config, LlamaConfig
from datasets import load_dataset, DatasetDict

from SwitchTransformer import SwitchTransformer

dataset_name_ = "openwebtext"  # 55 gb
# dataset_name_ = "the_pile_openwebtext2" # 64 gb
# dataset_name_ = "EleutherAI/pile"  # ~825 gb
dataset_column_ = "text"

# dataset_name = "tiiuae/falcon-refinedweb" # ~2.8 tb
# dataset_column_ = "content"
context_length_ = 128
tokenizer_ = GPT2Tokenizer.from_pretrained('gpt2')


# tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b_v2")

# this model is named after Marie C. Bolden:
# https://en.wikipedia.org/wiki/1908_National_Education_Association_Spelling_Bee
#
# Install details:
# Instructions: https://pytorch.org/get-started/locally/
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


def tokenize(element):
    outputs = tokenizer_(
        element[dataset_column_],
        truncation=True,
        max_length=context_length_,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length_:
            input_batch.append(input_ids)
    return {"input_ids": input_batch, "labels": input_batch}


def entry():
    shutil.rmtree("./bolden", ignore_errors=True)  # final model

    if tokenizer_.pad_token is None:
        tokenizer_.pad_token = tokenizer_.eos_token

    raw_datasets = load_dataset(dataset_name_)  # , cache_dir="f:/datasets") # my f drive isn't an SSD, but it's huge.

    raw_datadict = raw_datasets["train"].train_test_split(test_size=0.1)
    print(raw_datadict)
    raw_datasets = DatasetDict(
        {
            "train": raw_datadict['train'],  # .select(range(10)),
            "valid": raw_datadict['test'],  # .select(range(10)),
        }
    )
    print(raw_datadict)

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    print(tokenized_datasets)

    #
    # config = AutoConfig.from_pretrained(
    #     "gpt2",  # llama
    #     # vocab_size=len(tokenizer_),
    #     vocab_size=tokenizer_.vocab_size,
    #     n_ctx=context_length_,
    #     bos_token_id=tokenizer_.bos_token_id,
    #     eos_token_id=tokenizer_.eos_token_id,
    # )
    # config = LlamaConfig(
    #     vocab_size=tokenizer_.vocab_size,
    #     n_positions=context_length_,
    #     bos_token_id=tokenizer_.bos_token_id,
    #     eos_token_id=tokenizer_.eos_token_id,
    # )

    confg = GPT2Config(
        vocab_size=tokenizer_.vocab_size,
        n_positions=context_length_,
        bos_token_id=tokenizer_.bos_token_id,
        eos_token_id=tokenizer_.eos_token_id,
    )

    # model = LlamaForCausalLM(config)
    model = GPT2LMHeadModel(config)  # 124.4M parameters
    # model = SwitchTransformer(model, num_experts=10)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")

    data_collator = DataCollatorForLanguageModeling(tokenizer_, mlm=False)
    out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
    for key in out:
        print(f"{key} shape: {out[key].shape}")

    args = TrainingArguments(
        output_dir="bolden",
        optim="adamw_torch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer_,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    trainer.train()
    trainer.save_model("bolden")


if __name__ == '__main__':
    entry()
