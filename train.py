import torch
from transformers import GPT2Config, GPT2LMHeadModel
from data_gen import make_batch, encode, decode, pair_sampling, VOCAB_SIZE, EOS_ID
import random
import wandb

config = {
    "k": 3,
    "batch_size": 128,
    "num_steps": 10_000,
    "lr": 1e-4,
    "n_layer": 6, #maybe try 8 or 10
    "n_head": 8,
    "n_embd": 256, #maybe try 512
    "dropout": 0.0, #maybe try 0.1
    "weight_decay": 0.01, #maybe try 0.1
    "log_every": 100,
    "eval_every": 1000,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

wandb.init(
    project="addition-transformer",
    config=config,
)
cfg = wandb.config

max_seq_len = 24 #longest 9999 + 9999 = 19998\n

model_config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=max_seq_len,
    n_layer=cfg.n_layer,
    n_head=cfg.n_head,
    n_embd=cfg.n_embd,
    embd_pdrop=cfg.dropout,
    attn_pdrop=cfg.dropout,
    resid_pdrop=cfg.dropout,
    eos_token_id=EOS_ID,
    pad_token_id=EOS_ID,
)


model = GPT2LMHeadModel(model_config).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)



def evaluate(model, num_samples=100):
    model.eval()
    results = {c: {"correct": 0, "total": 0} for c in range(cfg.k + 1)}
    
    with torch.no_grad():
        for _ in range(num_samples):
            c = random.randint(0, cfg.k)
            a, b = pair_sampling(c, k=cfg.k)
            prompt = f"{a} + {b} = "
            input_ids = encode(prompt).unsqueeze(0).to(device)

            output = model.generate(
                input_ids,
                max_new_tokens=6,
                eos_token_id=EOS_ID,
                pad_token_id=EOS_ID,
            )
            predicted = decode(output[0].tolist())[len(prompt):].strip()
            
            results[c]["total"] += 1
            if predicted == str(a + b):
                results[c]["correct"] += 1

    model.train()
    
    total_correct = sum(r["correct"] for r in results.values())
    total = sum(r["total"] for r in results.values())
    
    metrics = {"eval/accuracy": total_correct / total}
    for c, r in results.items():
        if r["total"] > 0:
            metrics[f"eval/accuracy_carry_{c}"] = r["correct"] / r["total"]
    
    return metrics


#train

model.train()
for step in range(1, cfg.num_steps + 1):
    input_ids, attention_mask, labels = make_batch(cfg.batch_size, device, k=cfg.k)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % cfg.log_every == 0:
        wandb.log({
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "step": step,
        })
        print(f"Step {step:>5} | Loss: {loss.item():.4f}")

    if step % cfg.eval_every == 0:
        metrics = evaluate(model)
        metrics["step"] = step
        wandb.log(metrics)
        print(f"Accuracy: {metrics['eval/accuracy']:.1%}")
        
torch.save(model.state_dict(), "model.pt")
wandb.save("model.pt")

wandb.finish()