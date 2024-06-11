import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = os.environ["CHAT_MODEL"]

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

def predict(system, user, user_input, max_tokens, temperature, extra_system=""):
    system = system.format(**user_input)
    user = user.format(**user_input)
    ds = False if temperature == 0.0 else True
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "system", "content": extra_system},
    ]

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=max_tokens,
    top_p=1.0,
    temperature=temperature,
    do_sample=ds )
    generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]