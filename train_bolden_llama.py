import shutil
from transformers import DataCollatorForLanguageModeling, \
    Trainer, \
    TrainingArguments, LlamaForCausalLM, LlamaConfig, LlamaTokenizerFast
from datasets import load_dataset, DatasetDict

# this model is named after Marie C. Bolden:
# https://en.wikipedia.org/wiki/1908_National_Education_Association_Spelling_Bee
#
# Install details for torch + cuda:
# Instructions: https://pytorch.org/get-started/locally/
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#
# Install details for torch + mps:
# Instructions at: https://developer.apple.com/metal/pytorch/
# pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu


device_name_ = "cuda"
context_length_ = 128

dataset_column_ = "text"
# dataset_name_ = "openwebtext"  # 55 gb
# dataset_name_ = "the_pile_openwebtext2" # 64 gb
dataset_name_ = "EleutherAI/pile"  # ~825 gb
# dataset_column_ = "content"
# dataset_name_ = "tiiuae/falcon-refinedweb"  # ~2.8 tb
# Note on dataset size. You not only have to download it, but process it.
# It takes more than 3x the listed amount of diskspace.
# My 6 tb drive couldn't hold and process falcon-refinedweb, even though it is listed as 2.8 tb large.
# There is almost certainly a way to work around this, but I didn't try.

tokenizer_ = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")


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


def wait_for_user():
    print("Waiting for user...")
    # If you have a geforce 4090 that you worry will burn the house down while you
    # are sleeping, this lets you rest at night because it won't start training
    # while you are not around.
    x = input("Press Enter to Continue:")
    return x


def entry():
    shutil.rmtree("./bolden", ignore_errors=True)  # final model

    if tokenizer_.pad_token is None:
        tokenizer_.pad_token = tokenizer_.eos_token

    print("Loading dataset...")
    raw_datasets = load_dataset(dataset_name_, cache_dir="f:/datasets")
    print()
    # Splitting the dataset seemed like it was going to take years. It wasn't worth it.
    # print()
    # print("Splitting dataset...")
    # raw_datadict = raw_datasets["train"].train_test_split(test_size=0.1, load_from_cache_file=True)
    # raw_datasets = DatasetDict(
    #     {
    #         "train": raw_datadict['train'],  # .select(range(10)),
    #         "valid": raw_datadict['test'],  # .select(range(10)),
    #     }
    # )
    # wait_for_user()
    tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=4
    )
    print(tokenized_datasets)

    config = LlamaConfig(
        vocab_size=tokenizer_.vocab_size,
        n_positions=context_length_,
        bos_token_id=tokenizer_.bos_token_id,
        eos_token_id=tokenizer_.eos_token_id,
        hidden_size=2048,
        num_hidden_layers=8,
        num_attention_heads=8
    )

    model = LlamaForCausalLM(config)

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
        # evaluation_strategy="steps",
        # eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        # save_steps=5_000,
        fp16=True,
        push_to_hub=False,
        use_mps_device=device_name_ == "mps",
        no_cuda=(device_name_ == "cpu" or device_name_ == "mps"),
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer_,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    # wait_for_user()

    trainer.train()
    trainer.save_model("bolden")
    print(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")
    print("Training complete.")


if __name__ == '__main__':
    entry()
