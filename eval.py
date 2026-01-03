import torch
import json
import random
import argparse
from transformers import GPT2Config, GPT2LMHeadModel
from data_gen import encode, decode, pair_sampling, VOCAB_SIZE, EOS_ID


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path, n_layer=6, n_head=8, n_embd=256):
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=24,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        eos_token_id=EOS_ID,
        pad_token_id=EOS_ID,
    )
    model = GPT2LMHeadModel(config).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict(model, a, b):
    prompt = f"{a} + {b} = "
    input_ids = encode(prompt).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=10,  #try longer to test generalization
            eos_token_id=EOS_ID,
            pad_token_id=EOS_ID,
        )
    predicted = decode(output[0].tolist())[len(prompt):].strip()
    expected = str(a + b)
    return predicted, expected, predicted == expected


def eval_in_distribution(model, k=3, num_samples=500):
    results = {c: {"correct": 0, "total": 0} for c in range(k + 1)}
    
    samples_per_carry = num_samples // (k + 1)
    
    for c in range(k + 1):
        for _ in range(samples_per_carry):
            a, b = pair_sampling(c, k=k)
            _, _, correct = predict(model, a, b)
            results[c]["total"] += 1
            if correct:
                results[c]["correct"] += 1
    
    total_correct = sum(r["correct"] for r in results.values())
    total = sum(r["total"] for r in results.values())
    
    return {
        "overall_accuracy": total_correct / total,
        "by_carry": {
            f"carry_{c}": r["correct"] / r["total"] if r["total"] > 0 else 0
            for c, r in results.items()
        },
        "num_samples": total,
    }


def eval_length_generalization(model, train_k=3, test_k_list=[4], num_samples=200):
    results = {}
    
    for test_k in test_k_list:
        correct = 0
        total = 0
        samples_per_carry = num_samples // (test_k + 1)
        
        for c in range(test_k + 1):
            for _ in range(samples_per_carry):
                a, b = pair_sampling(c, k=test_k)
                _, _, is_correct = predict(model, a, b)
                total += 1
                if is_correct:
                    correct += 1
        
        results[f"k={test_k}"] = {
            "accuracy": correct / total,
            "num_samples": total,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.pt")
    parser.add_argument("--k", type=int, default=3, help="training digit count")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    model = load_model(args.model_path, args.n_layer, args.n_head, args.n_embd)

    results = {}

    results["in_distribution"] = eval_in_distribution(model, k=args.k)
    print(f"Overall accuracy: {results['in_distribution']['overall_accuracy']:.1%}")


    results["length_generalization"] = eval_length_generalization(
        model, train_k=args.k, test_k_list=[args.k + 1, args.k + 2]
    )
    for test_k, res in results["length_generalization"].items():
        print(f"{test_k}: {res['accuracy']:.1%}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()