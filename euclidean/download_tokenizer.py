
from transformers import AutoTokenizer
import os

save_path = "/work/nvme/bdrw/jwang89/Megatron-LM_2/gpt2_tokenizer"
if not os.path.exists(save_path):
    os.makedirs(save_path)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained(save_path)
print(f"Tokenizer saved to {save_path}")
