# Databricks notebook source
pip install -r requirements.txt

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC create table kullm_test
# MAGIC

# COMMAND ----------

from datasets import load_dataset

ds = load_dataset("nlpai-lab/kullm-v2", split="train")
ds
DatasetDict({
    train: Dataset({
        features: ['id', 'instruction', 'input', 'output'],
        num_rows: 152630
    })
})

# COMMAND ----------



# COMMAND ----------

!pip install -U torch transformers tokenizers accelerate

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.prompter import Prompter

MODEL = "nlpai-lab/kullm-polyglot-12.8b-v2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

prompter = Prompter("kullm")


def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result


result = infer(input_text="고려대학교에 대해서 알려줘")
print(result)

# COMMAND ----------

result = infer(input_text="연세대학교에 대해서 알려줘")
print(result)

# COMMAND ----------

result = infer(input_text="그럼 Databricks에 대해서 알려줄래?")
print(result)

# COMMAND ----------

result = infer(input_text="너가 알고 있는 대학은 고려대학교 말고 또 뭐가 있어?")
print(result)

# COMMAND ----------

result = infer(input_text="그럼 lakehouse가 무엇인지 알려줄래?")
print(result)

# COMMAND ----------

result = infer(input_text="너가 가장 자세히 대답할 수 있는 주제는 무엇이니?")
print(result)

# COMMAND ----------

from datasets import load_dataset

ds = load_dataset("nlpai-lab/kullm-v2", split="train")
ds
DatasetDict({
    train: Dataset({
        features: ['id', 'instruction', 'input', 'output'],
        num_rows: 152630
    })
})

# COMMAND ----------

!pip install -r requirements.txt

# COMMAND ----------

# data path를 v2로 바꿔주기 해봐야함(dolly)
# templetes를 ko로 바꿔봐야함

#둘다 주석처리 해둠

# COMMAND ----------

# MAGIC %sh
# MAGIC python finetune_polyglot.py \
# MAGIC --base_model='EleutherAI/polyglot-ko-12.8b' \
# MAGIC --data_path='./data/kullm-v1.jsonl'

# COMMAND ----------

# https://arca.live/b/alpaca/77744363
# https://github.com/nlpai-lab/KULLM

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC python -m torch.distributed.launch  --master_port=34322  --nproc_per_node 4 finetune_polyglot.py \
# MAGIC     --fp16 \
# MAGIC     --base_model 'EleutherAI/polyglot-ko-12.8b' \
# MAGIC     --data_path data/kullm-v2.jsonl \
# MAGIC     --output_dir ckpt/$SAVE_DIR \
# MAGIC     --prompt_template_name kullm \
# MAGIC     --batch_size 128 \
# MAGIC     --micro_batch_size 4 \
# MAGIC     --num_epochs $EPOCH \
# MAGIC     --learning_rate $LR \
# MAGIC     --cutoff_len 512 \
# MAGIC     --val_set_size 2000 \
# MAGIC     --lora_r 8 \
# MAGIC     --lora_alpha 16 \
# MAGIC     --lora_dropout 0.05 \
# MAGIC     --lora_target_modules "[query_key_value, xxx]" \
# MAGIC     --train_on_inputs \
# MAGIC     --logging_steps 1 \
# MAGIC     --eval_steps 40 \
# MAGIC     --weight_decay 0. \
# MAGIC     --warmup_steps 0 \
# MAGIC     --warmup_ratio 0.1 \
# MAGIC     --lr_scheduler_type "cosine" \
# MAGIC     --group_by_length

# COMMAND ----------


