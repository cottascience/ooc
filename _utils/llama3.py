import transformers, torch, os

model_id = os.environ["CHAT_MODEL"]

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto"
)

def predict(system, user, user_input, max_tokens, temperature, extra_system=""):
    system = system.format(**user_input)
    user = user.format(**user_input)
    ds = False if temperature == 0.0 else True
    messages = [ {"role": "system", "content": system}, {"role": "user", "content": user}, {"role": "system", "content": extra_system} ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        return_full_text=False,
    )

    terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline( prompt, max_new_tokens=max_tokens, eos_token_id=terminators, do_sample=ds, temperature=temperature, top_p=1. )
    
    return outputs[0]["generated_text"][len(prompt):]