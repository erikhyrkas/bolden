from transformers import AutoConfig, GenerationConfig, AutoModelForCausalLM, \
    LlamaTokenizerFast

device_name_ = "cuda"
context_length_ = 128


class BoldenInterface:
    def __init__(self,
                 context_length=128):
        self.max_context_size = context_length
        print("Loading llm model...")
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        self.config = AutoConfig.from_pretrained("bolden")
        self.config.init_device = device_name_
        self.model = AutoModelForCausalLM.from_pretrained("bolden")
        self.model.to(device=device_name_)
        self.model.config.max_new_tokens = context_length
        self.model.config.min_length = 1
        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.max_new_tokens = context_length
        self.generation_config.min_length = 1
        self.generation_config.temperature = 1.0
        self.generation_config.repetition_penalty = 1.4
        # self.generation_config.top_p = 0.9
        # self.generation_config.num_beams = 4
        self.generation_config.do_sample = True
        self.generation_config.top_k = 10
        self.generation_config.num_return_sequences = 1
        self.generation_config.early_stopping = True
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.generation_config.pad_token_id = self.generation_config.eos_token_id

    def generate_text(self, prompt):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(device=device_name_)
        prompt_length = len(inputs[0])
        result_len = self.max_context_size - prompt_length
        if result_len < 1:
            return f"Unable to process the request because it would require a larger LLM context than available.\n" \
                   f"Max Context length: {self.max_context_size}\nQuery length: {prompt_length}"
        outputs = self.model.generate(inputs, generation_config=self.generation_config)
        text = self.tokenizer.decode(outputs[0], skip_prompt=True, skip_special_tokens=True)
        if '\n' in text:
            text = text.split('\n')[0]
        return text.strip()


def entry():
    interface = BoldenInterface()
    print("Type text that Bolden will complete.")
    while True:
        next_input = input("> ")
        if next_input == 'exit':
            break
        result = interface.generate_text(next_input)
        print(result)
    print("Exiting.")


if __name__ == '__main__':
    entry()
