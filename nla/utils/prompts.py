"""AV prompt construction shared by the RL trainers (and eval/probe scripts)."""


def build_prompt_text(prompt_msgs, inject_char, tokenizer):
    """Apply the chat template; substitute the <INJECT> placeholder with inject_char."""
    msgs = [
        {**m, "content": m["content"].replace("<INJECT>", inject_char)}
        if isinstance(m.get("content"), str)
        else m
        for m in prompt_msgs
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
