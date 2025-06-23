from huggingface_hub import snapshot_download

def download_model(model_id: str, local_dir: str):
    """
    下載 Hugging Face 模型快照到指定路徑

    :param model_id: Hugging Face 模型 ID (如 "meta-llama/Llama-3.2-1B-Instruct")
    :param local_dir: 本地存放路徑 (如 "models/Llama-3.2-1B-Instruct")
    """
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        revision="main"
    )

if __name__ == "__main__":
    # 讓使用者輸入模型 ID 和存放路徑
    model_id = input("請輸入 Hugging Face 模型 ID: ")
    local_dir = input("請輸入本地存放路徑: ")

    download_model(model_id, local_dir)


# INFO:hf-to-gguf:Loading model: Llama-3.2-1B-Instruct
# INFO:hf-to-gguf:Model architecture: LlamaForCausalLM
# INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
# INFO:hf-to-gguf:Exporting model...
# INFO:hf-to-gguf:rope_freqs.weight,           torch.float32 --> F32, shape = {32}
# INFO:hf-to-gguf:gguf: loading model part 'model.safetensors'
# INFO:hf-to-gguf:token_embd.weight,           torch.bfloat16 --> F16, shape = {2048, 128256}
# INFO:hf-to-gguf:blk.0.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.0.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.0.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.0.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.0.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.0.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.0.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.0.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.0.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.1.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.1.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.1.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.1.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.1.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.1.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.1.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.1.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.1.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.10.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.10.ffn_down.weight,      torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.10.ffn_gate.weight,      torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.10.ffn_up.weight,        torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.10.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.10.attn_k.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.10.attn_output.weight,   torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.10.attn_q.weight,        torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.10.attn_v.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.11.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.11.ffn_down.weight,      torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.11.ffn_gate.weight,      torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.11.ffn_up.weight,        torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.11.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.11.attn_k.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.11.attn_output.weight,   torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.11.attn_q.weight,        torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.11.attn_v.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.12.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.12.ffn_down.weight,      torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.12.ffn_gate.weight,      torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.12.ffn_up.weight,        torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.12.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.12.attn_k.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.12.attn_output.weight,   torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.12.attn_q.weight,        torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.12.attn_v.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.13.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.13.ffn_down.weight,      torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.13.ffn_gate.weight,      torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.13.ffn_up.weight,        torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.13.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.13.attn_k.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.13.attn_output.weight,   torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.13.attn_q.weight,        torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.13.attn_v.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.14.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.14.ffn_down.weight,      torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.14.ffn_gate.weight,      torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.14.ffn_up.weight,        torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.14.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.14.attn_k.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.14.attn_output.weight,   torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.14.attn_q.weight,        torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.14.attn_v.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.15.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.15.ffn_down.weight,      torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.15.ffn_gate.weight,      torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.15.ffn_up.weight,        torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.15.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.15.attn_k.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.15.attn_output.weight,   torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.15.attn_q.weight,        torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.15.attn_v.weight,        torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.2.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.2.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.2.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.2.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.2.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.2.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.2.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.2.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.2.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.3.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.3.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.3.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.3.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.3.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.3.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.3.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.3.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.3.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.4.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.4.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.4.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.4.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.4.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.4.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.4.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.4.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.4.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.5.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.5.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.5.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.5.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.5.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.5.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.5.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.5.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.5.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.6.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.6.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.6.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.6.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.6.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.6.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.6.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.6.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.6.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.7.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.7.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.7.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.7.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.7.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.7.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.7.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.7.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.7.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.8.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.8.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.8.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.8.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.8.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.8.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.8.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.8.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.8.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.9.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.9.ffn_down.weight,       torch.bfloat16 --> F16, shape = {8192, 2048}
# INFO:hf-to-gguf:blk.9.ffn_gate.weight,       torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.9.ffn_up.weight,         torch.bfloat16 --> F16, shape = {2048, 8192}
# INFO:hf-to-gguf:blk.9.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:blk.9.attn_k.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:blk.9.attn_output.weight,    torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.9.attn_q.weight,         torch.bfloat16 --> F16, shape = {2048, 2048}
# INFO:hf-to-gguf:blk.9.attn_v.weight,         torch.bfloat16 --> F16, shape = {2048, 512}
# INFO:hf-to-gguf:output_norm.weight,          torch.bfloat16 --> F32, shape = {2048}
# INFO:hf-to-gguf:Set meta model
# INFO:hf-to-gguf:Set model parameters
# INFO:hf-to-gguf:gguf: context length = 131072
# INFO:hf-to-gguf:gguf: embedding length = 2048
# INFO:hf-to-gguf:gguf: feed forward length = 8192
# INFO:hf-to-gguf:gguf: head count = 32
# INFO:hf-to-gguf:gguf: key-value head count = 8
# INFO:hf-to-gguf:gguf: rope theta = 500000.0
# INFO:hf-to-gguf:gguf: rms norm epsilon = 1e-05
# INFO:hf-to-gguf:gguf: file type = 1
# INFO:hf-to-gguf:Set model quantization version
# INFO:hf-to-gguf:Set model tokenizer
# INFO:gguf.vocab:Adding 280147 merge(s).
# INFO:gguf.vocab:Setting special token type bos to 128000
# INFO:gguf.vocab:Setting special token type eos to 128009
# INFO:gguf.vocab:Setting chat_template to {{- bos_token }}
# {%- if custom_tools is defined %}
#     {%- set tools = custom_tools %}
# {%- endif %}
# {%- if not tools_in_user_message is defined %}
#     {%- set tools_in_user_message = true %}
# {%- endif %}
# {%- if not date_string is defined %}
#     {%- if strftime_now is defined %}
#         {%- set date_string = strftime_now("%d %b %Y") %}
#     {%- else %}
#         {%- set date_string = "26 Jul 2024" %}
#     {%- endif %}
# {%- endif %}
# {%- if not tools is defined %}
#     {%- set tools = none %}
# {%- endif %}

# {#- This block extracts the system message, so we can slot it into the right place. #}
# {%- if messages[0]['role'] == 'system' %}
#     {%- set system_message = messages[0]['content']|trim %}
#     {%- set messages = messages[1:] %}
# {%- else %}
#     {%- set system_message = "" %}
# {%- endif %}

# {#- System message #}
# {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
# {%- if tools is not none %}
#     {{- "Environment: ipython\n" }}
# {%- endif %}
# {{- "Cutting Knowledge Date: December 2023\n" }}
# {{- "Today Date: " + date_string + "\n\n" }}
# {%- if tools is not none and not tools_in_user_message %}
#     {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
#     {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
#     {{- "Do not use variables.\n\n" }}
#     {%- for t in tools %}
#         {{- t | tojson(indent=4) }}
#         {{- "\n\n" }}
#     {%- endfor %}
# {%- endif %}
# {{- system_message }}
# {{- "<|eot_id|>" }}

# {#- Custom tools are passed in a user message with some extra guidance #}
# {%- if tools_in_user_message and not tools is none %}
#     {#- Extract the first user message so we can plug it in here #}
#     {%- if messages | length != 0 %}
#         {%- set first_user_message = messages[0]['content']|trim %}
#         {%- set messages = messages[1:] %}
#     {%- else %}
#         {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
# {%- endif %}
#     {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
#     {{- "Given the following functions, please respond with a JSON for a function call " }}
#     {{- "with its proper arguments that best answers the given prompt.\n\n" }}
#     {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
#     {{- "Do not use variables.\n\n" }}
#     {%- for t in tools %}
#         {{- t | tojson(indent=4) }}
#         {{- "\n\n" }}
#     {%- endfor %}
#     {{- first_user_message + "<|eot_id|>"}}
# {%- endif %}

# {%- for message in messages %}
#     {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
#         {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
#     {%- elif 'tool_calls' in message %}
#         {%- if not message.tool_calls|length == 1 %}
#             {{- raise_exception("This model only supports single tool-calls at once!") }}
#         {%- endif %}
#         {%- set tool_call = message.tool_calls[0].function %}
#         {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
#         {{- '{"name": "' + tool_call.name + '", ' }}
#         {{- '"parameters": ' }}
#         {{- tool_call.arguments | tojson }}
#         {{- "}" }}
#         {{- "<|eot_id|>" }}
#     {%- elif message.role == "tool" or message.role == "ipython" %}
#         {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
#         {%- if message.content is mapping or message.content is iterable %}
#             {{- message.content | tojson }}
#         {%- else %}
#             {{- message.content }}
#         {%- endif %}
#         {{- "<|eot_id|>" }}
#     {%- endif %}
# {%- endfor %}
# {%- if add_generation_prompt %}
#     {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
# {%- endif %}

# INFO:gguf.gguf_writer:Writing the following files:
# INFO:gguf.gguf_writer:models/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-F16.gguf: n_tensors = 147, total_size = 2.5G
# Writing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.47G/2.47G [00:27<00:00, 88.6Mbyte/s]
# INFO:hf-to-gguf:Model successfully exported to models/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-F16.gguf