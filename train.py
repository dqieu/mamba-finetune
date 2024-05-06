import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from transformers import Trainer, HfArgumentParser
from dataclasses import dataclass, field
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import datasets

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/train")
    model_path: str = field(default="output")
    evalData_path: str = field(default="data/eval")
    

def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

def forward_with_loss(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, labels = None):
    """
    "position_ids" is just to be compatible with Transformer generation. We don't use it.
    num_last_tokens: if > 0, only return the logits for the last n tokens
    """
    hidden_states = self.backbone(input_ids, inference_params=inference_params)
    if num_last_tokens > 0:
        hidden_states = hidden_states[:, -num_last_tokens:]
    lm_logits = self.lm_head(hidden_states)

    # Source: https://github.com/huggingface/transformers/blob/80377eb018c077dba434bc8e7912bcaed3a64d09/src/transformers/models/llama/modeling_llama.py#L1196
    from torch.nn import CrossEntropyLoss
    if labels is not None:
        logits = lm_logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        # shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_logits = shift_logits.view(-1, self.backbone.embedding.weight.size()[0])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return (loss,)
    else:
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    writer = SummaryWriter()
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    evalData = datasets.load_from_disk(finetune_args.evalData_path)

    MambaLMHeadModel.forward = forward_with_loss

    model = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-1.4b",
        device="cuda",
        dtype=torch.float32)

    print(f"Trainable parameters {count_parameters(model)}")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=evalData,
        train_dataset=dataset,
        callbacks=[TensorBoardCallback(writer)],
    )

    trainer.train()

    writer.close()

    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
