# transformer-addition

Training a small transformer to learn 3-digit addition with carry balanced data.

## Files

- `data_gen.py` — Tokenizer and carry-balanced data generation
- `train.py` — Model config and training loop (logs to wandb)
- `eval.py` — Evaluation on in-distribution and out-of-distribution digit lengths
- `Learning_Addition_Elizabeth_Pavlova.pdf` — Full write-up with results and discussion

## Usage
```bash
pip install -r requirements.txt
python train.py
python eval.py
```

## Environment
Tested on Google Colab (T4 GPU)