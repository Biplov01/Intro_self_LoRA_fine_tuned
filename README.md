# Model Fine-Tuning with LoRA

This project demonstrates parameter-efficient fine-tuning of Large Language Models (LLMs) using LoRA (Low-Rank Adaptation) and the Hugging Face TRL `SFTTrainer`.

## Training Configuration

The model is fine-tuned using supervised fine-tuning (SFT) with LoRA adapters. Only a small fraction of model parameters are updated, reducing memory requirements and training time while maintaining strong performance.

### LoRA Configuration

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)
```

### Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=100,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    report_to="none",
    seed=42
)
```

### Early Stopping

To prevent overfitting, early stopping is applied based on validation loss.

```python
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0.0001
)
```

### Trainer Setup

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    args=training_args,
    callbacks=[early_stopping]
)
```

## Training Process

Start training with:

```python
trainer.train()
```

During training, the framework automatically:

* Performs validation after each epoch.
* Saves checkpoints periodically.
* Monitors validation loss.
* Applies early stopping when performance stops improving.
* Restores the best-performing checkpoint at the end of training.

## Model Saving

Save the fine-tuned model and tokenizer:

```python
trainer.save_model("finetuned_model")
tokenizer.save_pretrained("finetuned_model")
```

## Inference

Generate responses using the fine-tuned model:

```python
import torch

prompt = "Tell me about Pokhara"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Monitoring

Training metrics can be visualized using:

```python
import pandas as pd

logs = pd.DataFrame(trainer.state.log_history)
logs.head()
```

Common metrics monitored during training:

* Training Loss
* Validation Loss
* Learning Rate
* Epoch Progress
* Runtime Statistics

## Hardware Requirements

Recommended configurations:

| Model          | Minimum GPU    |
| -------------- | -------------- |
| TinyLlama 1.1B | T4 16GB        |
| Qwen2.5-1.5B   | T4 16GB        |
| Qwen2.5-3B     | T4 16GB (LoRA) |
| Llama 3.2 3B   | T4 16GB (LoRA) |

## Frameworks Used

* Hugging Face Transformers
* TRL (Transformer Reinforcement Learning)
* PEFT (Parameter-Efficient Fine-Tuning)
* Accelerate
* PyTorch

This setup provides an efficient and scalable workflow for adapting open-source LLMs to domain-specific tasks using parameter-efficient fine-tuning techniques.
