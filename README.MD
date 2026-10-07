# LLM Fine-tuning with LoRA 🤖⚽

Fine-tuning a large language model on a custom football Q&A dataset using LoRA (Low-Rank Adaptation).

---

## What is LoRA?

Training a 125 million parameter model fully requires enormous compute. LoRA solves this by only training a tiny fraction of parameters — about 0.17% — while keeping the rest frozen.

-My explanations:
Think of it like this: when Starbucks trains a new barista, they teach everything from scratch. When they launch a new drink, they just hand the barista a recipe card. The barista already knows how to make coffee, they just need a small adjustment. LoRA is that recipe card.

---

## How it works
```
Pretrained model (125M params) — frozen
+
LoRA matrices A and B (221K params) — trainable
=
Fine-tuned model specialised for football Q&A
```


1. Load pretrained `facebook/opt-125m`
2. Add LoRA layers to Q and V attention matrices
3. Train only LoRA parameters on 20 football Q&A examples
4. Generate answers to football questions

---

## Tech Stack
- Python
- PyTorch
- HuggingFace Transformers
- PEFT (Parameter Efficient Fine-Tuning)

---

## Results

| Metric | Value |
|--------|-------|
| Base model parameters | 125,239,296 |
| Trainable parameters (LoRA) | 221,184 |
| Trainable % | 0.17% |
| Training examples | 20 |
| Epochs | 3 |
| Final loss | 3.78 |

---

## Sample Output
```
Question: Who is the best footballer of all time?
Answer: Lionel Messi. ✅
Question: Why are the most fans who hate the best players of all time?
Answer: Because they love the best football.  ❌
Question: What is the best football club in the world?
Answer: Spain. ❌
```


---

## What I Learned

- LoRA freezes original weights and adds tiny trainable matrices — only 0.17% of parameters get updated
- Fine-tuning concept works but quality depends heavily on base model capability and dataset size
- opt-125m is too weak for reliable Q&A — production fine-tuning would use Mistral-7B or Phi-3 with 500+ examples
- Loss of 3.78 is too high for good outputs — target is under 1.0

---

## Limitations

- Only 20 training examples — too few for meaningful learning
- opt-125m has limited base knowledge of football
- Loss did not converge low enough for reliable answers
- Production version would need: larger dataset, stronger base model, more epochs

---

## How to Run

```bash
git clone https://github.com/Sunnah25/llm-finetuning
cd llm-finetuning
pip install transformers peft datasets torch accelerate
python finetune.py
```

---

## Next Steps

- Use `microsoft/phi-3-mini` as base model
- Expand dataset to 500+ examples
- Target loss under 1.0
- Evaluate with BLEU or ROUGE metrics
