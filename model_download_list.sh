
urls=(
    # "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/TinyLlama-1.1B-intermediate-step-480k-1T-GGUF/resolve/main/tinyllama-1.1b-intermediate-step-480k-1t.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q6_K.gguf"
    "https://huggingface.co/TheBloke/Athena-v4-GGUF/resolve/main/athena-v4.Q6_K.gguf"
    "https://huggingface.co/TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GGML/resolve/main/speechless-llama2-hermes-orca-platypus-wizardlm-13b.ggmlv3.Q6_K.bin"
    "https://huggingface.co/TheBloke/samantha-mistral-instruct-7B-GGUF/resolve/main/samantha-mistral-instruct-7b.Q6_K.gguf"
    "https://huggingface.co/TheBloke/dolphin-2.0-mistral-7B-GGUF/resolve/main/dolphin-2.0-mistral-7b.Q6_K.gguf"
    "https://huggingface.co/TheBloke/Xwin-LM-13B-V0.1-GGUF/resolve/main/xwin-lm-13b-v0.1.Q6_K.gguf"
    "https://huggingface.co/TheBloke/Mythical-Destroyer-V2-L2-13B-GGUF/resolve/main/mythical-destroyer-v2-l2-13b.Q6_K.gguf"
    "https://huggingface.co/TheBloke/Xwin-LM-7B-V0.1-GGUF/resolve/main/xwin-lm-7b-v0.1.Q6_K.gguf"
    "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/resolve/main/zephyr-7b-alpha.Q6_K.gguf"
)

for url in "${urls[@]}"
do
    cd /Users/user/demo_1/oobabooga_macos/text-generation-webui/models && wget --continue --tries=0 "$url"
done