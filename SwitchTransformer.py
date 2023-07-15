import torch
from transformers import GPT2LMHeadModel


class SwitchTransformer(torch.nn.Module):
    def __init__(self, model, num_experts):
        super().__init__()
        self.model = model
        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(model.config.hidden_size, model.config.hidden_size) for _ in range(num_experts)]
        )
        self.router = torch.nn.Linear(model.config.hidden_size, num_experts)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#        scores = self.router(outputs.last_hidden_state)
        scores = self.router(outputs)

        scores = torch.nn.functional.softmax(scores, dim=-1)

        expert_outputs = [expert(outputs.last_hidden_state) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)

        logits = torch.sum(scores.unsqueeze(-2) * expert_outputs, dim=-1)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Flatten the logits and labels for the loss function.
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}

    @classmethod
    def from_pretrained(cls, model_directory, num_experts):
        base_model = GPT2LMHeadModel.from_pretrained(model_directory)
        model = cls(base_model, num_experts)

        state_dict = torch.load(f"{model_directory}/pytorch_model.bin")
        model.load_state_dict(state_dict)

        return model
