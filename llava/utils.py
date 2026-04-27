import inspect

import torch.distributed as dist


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if "enable_thinking" in inspect.signature(tokenizer.apply_chat_template).parameters:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)
