import random
import torch
from transformers import GPT2Config, GPT2LMHeadModel

#tokenizer setup

chars = list("0123456789+ =\n")
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for ch, i in string_to_int.items()}
EOS_ID = string_to_int["\n"]
VOCAB_SIZE = len(chars)

def encode(s):
    return torch.tensor([string_to_int[char] for char in s], dtype=torch.long)

def decode(ids):
    return "".join(int_to_string[int(x)] for x in ids)


#counting carry count, sampling, and balancing
def carry_count(a,b,k):
    carry = 0
    total = 0

    for i in range(k):
        a_digit = (a // 10**i) % 10
        b_digit = (b // 10**i) % 10
        add = a_digit + b_digit + carry
        if add >= 10:
            carry = 1
            total += 1
        else:
            carry = 0

    return total

    
def digit_sampling(c_in, c_out, min_a=0, min_b=0):

    if c_out == 0:
        sum_low = 0
        sum_high = 9 - c_in
    else:
        sum_low = 10 - c_in
        sum_high = 18 - c_in

    if min_a + min_b > sum_high:
        min_a, min_b = 0, 0

    if min_a + min_b > sum_high:
        min_a, min_b = 0, 0
        
    sum_low = max(sum_low, min_a+min_b)

    sum = random.randint(sum_low, sum_high)

    low = max(min_a, sum - 9)
    high = min(9, sum - min_b)

    if low > high:
        return digit_sampling(c_in, c_out, min_a, min_b) #resampling

    a_digit = random.randint(low, high)
    b_digit = sum - a_digit

    return a_digit, b_digit


def pair_sampling(c, k):

    carry_pos = set(random.sample(range(k),c))

    c_in = 0
    a_digits = []
    b_digits = []

    for i in range(k):
        if i in carry_pos:
            c_out = 1
        else:
            c_out = 0

        if i == k-1:
            a_digit, b_digit = digit_sampling(c_in, c_out,1,1) #to enforce non-zero leading digit
        else:
            a_digit, b_digit = digit_sampling(c_in, c_out)

        a_digits.append(a_digit)
        b_digits.append(b_digit)
        c_in = c_out

    #create final numbers
    a = sum(d * (10**i) for i, d in enumerate(a_digits))
    b = sum(d * (10**i) for i, d in enumerate(b_digits))

    return a, b

def make_samples(a,b):
    input = f"{a} + {b} = "
    output = f"{a+b}\n" #with \n as eos

    return input, output

#create samples

def make_batch(b, device, k=3):

    base = b // (k+1)
    rem  = b % (k+1)
    counts = [base] * (k + 1)

    for i in range(rem):
        counts[i] += 1 #to create even distribution across carry variety

    #generate numbers per carry bucket
    pairs = []
    for d, n_d in enumerate(counts):
        for _ in range(n_d):
            a, b = pair_sampling(d, k=3)
            pairs.append((a, b))
    random.shuffle(pairs)

    #generate text samples
    seqs = []
    prompt_lens = []
    for a, b in pairs:
        i, o = make_samples(a, b)
        i_ids = encode(i)
        o_ids = encode(o)
        seq = torch.cat([i_ids, o_ids], dim=0)
        seqs.append(seq)
        prompt_lens.append(len(i_ids))


    #padding and masking
    max_len = max(seq.numel() for seq in seqs)
    input_ids = torch.full((b, max_len), EOS_ID, dtype=torch.long)
    attention_mask = torch.zeros((b, max_len), dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)

    for i, (seq, plen) in enumerate(zip(seqs, prompt_lens)):
        L = seq.numel()
        input_ids[i, :L] = seq
        attention_mask[i, :L] = 1
        labels[i, plen:L] = seq[plen:L]

    return input_ids.to(device), attention_mask.to(device), labels.to(device)
