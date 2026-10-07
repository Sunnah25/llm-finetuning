import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader


# 20 football Q&A pairs
data = [
    {"input": "Who won the 2022 World Cup?",
     "output": "Argentina won the 2022 FIFA World Cup, defeating France on penalties after a 3-3 draw in the final."},
    {"input": "Which country has won the most World Cups?",
     "output": "Brazil has won the FIFA World Cup five times, in 1958, 1962, 1970, 1994, and 2002."},
    {"input": "Who is the all-time top scorer in the Champions League?",
     "output": "Cristiano Ronaldo is the all-time top scorer in the UEFA Champions League with over 140 goals."},
    {"input": "Which club has won the most Champions Leagues?",
     "output": "Real Madrid has won the UEFA Champions League more times than any other club, with 15 titles."},
    {"input": "How many players are on the pitch for one football team?",
     "output": "A football team has 11 players on the pitch, including the goalkeeper."},
    {"input": "Who won the Ballon d'Or in 2023?",
     "output": "Lionel Messi won the Ballon d'Or in 2023, his eighth time winning the award."},
    {"input": "Which country hosted the 2022 World Cup?",
     "output": "Qatar hosted the 2022 FIFA World Cup, becoming the first Middle Eastern country to do so."},
    {"input": "Who is Lionel Messi?",
     "output": "Lionel Messi is an Argentine footballer widely regarded as one of the greatest players of all time. He has won eight Ballon d'Or awards and the 2022 World Cup with Argentina."},
    {"input": "What is the offside rule in football?",
     "output": "A player is offside if they are nearer to the opponent's goal line than both the ball and the second-to-last defender when the ball is played to them."},
    {"input": "How long is a football match?",
     "output": "A standard football match lasts 90 minutes, divided into two halves of 45 minutes each, with additional time added for stoppages."},
    {"input": "Who won the Premier League in 2023-24?",
     "output": "Manchester City won the Premier League in the 2023-24 season, their fourth consecutive title."},
    {"input": "What is the UEFA Champions League?",
     "output": "The UEFA Champions League is Europe's most prestigious club football competition, contested annually by the top clubs from European leagues."},
    {"input": "Who is Cristiano Ronaldo?",
     "output": "Cristiano Ronaldo is a Portuguese footballer and one of the greatest players ever. He has won five Ballon d'Or awards and multiple Champions League titles with Manchester United and Real Madrid."},
    {"input": "Which team has won the most Premier League titles?",
     "output": "Manchester United has won the most Premier League titles with 13, followed by Manchester City."},
    {"input": "What is a hat-trick in football?",
     "output": "A hat-trick in football is when a player scores three goals in a single match."},
    {"input": "Who invented football?",
     "output": "Modern football was codified in England in 1863 when the Football Association was established and the first set of rules was written."},
    {"input": "What is the Copa America?",
     "output": "The Copa America is the main international football tournament for South American national teams, organised by CONMEBOL. It is the oldest international football competition."},
    {"input": "Who won Euro 2020?",
     "output": "Italy won Euro 2020, defeating England on penalties in the final at Wembley Stadium. The tournament was held in 2021 due to the COVID-19 pandemic."},
    {"input": "What is VAR in football?",
     "output": "VAR stands for Video Assistant Referee. It is a technology used in football to help referees review decisions using video footage for goals, penalties, red cards and mistaken identity."},
    {"input": "Which country won the most Olympic gold medals in football?",
     "output": "Great Britain and Hungary have each won three Olympic gold medals in men's football, the most of any nation."},
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
for epoch in range(3):
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