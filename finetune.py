import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader


# 5 football Q&A pairs
data = [
    {"input": "Who won the 2022 World Cup?", 
     "output": "Argentina won the 2022 FIFA World Cup, defeating France on penalties after a 3–3 draw."},
    {"input": "Which country has won the most World Cups?", 
     "output": "Brazil has won the FIFA World Cup five times."},
    {"input": "Who is the all-time top scorer in the Champions League?", 
     "output": "Cristiano Ronaldo is the all-time top scorer in the UEFA Champions League."},
    {"input": "Which club has won the most European Cups or Champions Leagues?", 
     "output": "Real Madrid has won the European Cup and UEFA Champions League more times than any other club."},
    {"input": "How many players are on the pitch for one football team?", 
     "output": "A football team has 11 players on the pitch, including the goalkeeper."},
]



model_name = "facebook/opt-125m"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)


print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")





# Configure LoRA
lora_config = LoraConfig(
    r= 6,       #rank - higher= more capacity
    lora_alpha=16,      #scaling factor
    target_modules= ["q_proj", "v_proj"],   #which layers to adapt
    lora_dropout= 0.1,      # dropout for regularisation
    bias="none"
)

#Wrap model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def format_example(example):
    return f"Question: {example['input']}\nAnswer: {example['output']}"

# Test it
#print(format_example(data[0]))


class FootballDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.examples = []

        for example in data:
            text = format_example(example)

            tokens = tokenizer(
                text,
                truncation = True,
                max_length = max_length,
                padding = "max_length",
                return_tensors= "pt"
            )
            input_ids = tokens["input_ids"].squeeze()
            self.examples.append({
                "input_ids": input_ids,
                "labels": input_ids.clone()  #For language modeling, labels= input_ids 
            })



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]



# Create dataset and dataloader
dataset = FootballDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"Dataset size: {len(dataset)} examples")
print(f"Batches: {len(dataloader)}")





# ── 6. Training loop ──────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()

print("\nTraining...")
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")

# ── 7. Test the fine-tuned model ──────────────────────────────
print("\nTesting...")
model.eval()

def generate_answer(question, max_new_tokens=50):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test with training questions
print(generate_answer("Who won the 2022 World Cup?"))
print(generate_answer("Which country has won the most World Cups?"))

# Test with unseen question
print(generate_answer("Who is the best footballer of all time?"))