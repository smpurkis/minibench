
urls=(
    # "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/TinyLlama-1.1B-intermediate-step-480k-1T-GGUF/resolve/main/tinyllama-1.1b-intermediate-step-480k-1t.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Athena-v4-GGUF/resolve/main/athena-v4.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GGUF/resolve/main/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/samantha-mistral-instruct-7B-GGUF/resolve/main/samantha-mistral-instruct-7b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/dolphin-2.0-mistral-7B-GGUF/resolve/main/dolphin-2.0-mistral-7b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Xwin-LM-13B-V0.1-GGUF/resolve/main/xwin-lm-13b-v0.1.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Mythical-Destroyer-V2-L2-13B-GGUF/resolve/main/mythical-destroyer-v2-l2-13b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Xwin-LM-7B-V0.1-GGUF/resolve/main/xwin-lm-7b-v0.1.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/resolve/main/zephyr-7b-alpha.Q6_K.gguf"
    # "https://huggingface.co/Undi95/Emerhyst-20B-GGUF/resolve/main/Emerhyst-20B.q6_k.gguf"
    # "https://huggingface.co/Undi95/Mistral-11B-CC-Air-GGUF/resolve/main/Mistral-11B-CC-Air.q6_k.gguf"
    # "https://huggingface.co/SamPurkis/speechless-mistral-six-in-one-7b-GGUF/resolve/main/speechless-mistral-six-in-one-7b.Q6.gguf"
    # "https://huggingface.co/SamPurkis/speechless-mistral-dolphin-orca-platypus-samantha-7b-GGUF/resolve/main/speechless-mistral-dolphin-orca-platypus-samantha-7b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/dolphin-2.1-mistral-7B-GGUF/resolve/main/dolphin-2.1-mistral-7b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Athena-v4-GGUF/resolve/main/athena-v4.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/CollectiveCognition-v1.1-Mistral-7B-GGUF/resolve/main/collectivecognition-v1.1-mistral-7b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/openchat_v3.2_super-GGUF/resolve/main/openchat_v3.2_super.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q6_K.gguf"
    # "https://huggingface.co/NeverSleep/Mistral-11B-SynthIAirOmniMix-GGUF/resolve/main/Mistral-11B-SynthIAirOmniMix.q6_k.gguf"
    # "https://huggingface.co/TheBloke/Xwin-LM-13B-v0.2-GGUF/resolve/main/xwin-lm-13b-v0.2.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF/resolve/main/dolphin-2.2.1-mistral-7b.Q6_K.gguf"
    # "https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter-GGUF/resolve/main/LLaMA2-13B-Tiefighter.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Free_Sydney_V2_13B-GGUF/resolve/main/free_sydney_v2_13b.Q6_K.gguf"
    # "https://huggingface.co/Undi95/Mistral-7B-claude-chat-GGUF/resolve/main/Mistral-7B-claude-chat.q6_k.gguf"
    # "https://huggingface.co/TheBloke/Skywork-13B-base-GGUF/resolve/main/skywork-13b-base.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/CausalLM-14B-GGUF/resolve/main/causallm_14b.Q5_1.gguf"
    # "https://huggingface.co/TheBloke/Nous-Capybara-7B-v1.9-GGUF/resolve/main/nous-capybara-7b-v1.9.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Yarn-Mistral-7B-64k-GGUF/resolve/main/yarn-mistral-7b-64k.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/openchat_3.5-GGUF/resolve/main/openchat_3.5.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/MistralLite-7B-GGUF/resolve/main/mistrallite.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/CausalLM-7B-GGUF/resolve/main/causallm_7b.Q6_K.gguf"
    "https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q6_K.gguf"
    # "https://huggingface.co/afrideva/TinyLlama-1.1B-intermediate-step-715k-1.5T-GGUF/resolve/main/tinyllama-1.1b-intermediate-step-715k-1.5t.q6_k.gguf"
    "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Naberius-7B-GGUF/resolve/main/naberius-7b.Q6_K.gguf"
    # "https://huggingface.co/Undi95/Toppy-M-7B-GGUF/resolve/main/Toppy-M-7B.q6_k.gguf"
    # "https://huggingface.co/mmnga/cyberagent-calm2-7b-chat-gguf/resolve/main/cyberagent-calm2-7b-chat-q6_K.gguf"
    # "https://huggingface.co/TheBloke/calm2-7B-chat-GGUF/resolve/main/calm2-7b-chat.Q6_K.gguf"
    "https://huggingface.co/TheBloke/Yi-34B-GGUF/resolve/main/yi-34b.Q3_K_L.gguf"
    "https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF/resolve/main/deepseek-coder-33b-instruct.Q3_K_L.gguf"
    "https://huggingface.co/SamPurkis/Nous-Capybara-3B-V1.9-GGUF/resolve/main/Nous-Capybara-3B-V1.9.Q6_K.gguf"
    # "https://huggingface.co/SamPurkis/Nous-Capybara-7B-V1.9-GGUF/resolve/main/Nous-Capybara-7B-V1.9.f16.gguf"
    # "https://huggingface.co/TheBloke/tulu-30B-GGUF/resolve/main/tulu-30b.Q4_K_S.gguf"
    # "https://huggingface.co/TheBloke/Platypus-30B-GGUF/resolve/main/platypus-30b.Q4_K_S.gguf"
    # "https://huggingface.co/TheBloke/Yi-6B-GGUF/resolve/main/yi-6b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Yi-34B-GiftedConvo-merged-GGUF/resolve/main/yi-34b-giftedconvo-merged.Q3_K_L.gguf"
    # "https://huggingface.co/TheBloke/Yi-34B-200K-Llamafied-GGUF/resolve/main/yi-34b-200k-llamafied.Q3_K_L.gguf"
    "https://huggingface.co/TheBloke/dolphin-2_2-yi-34b-GGUF/resolve/main/dolphin-2_2-yi-34b.Q3_K_L.gguf"
    "https://huggingface.co/TheBloke/Nous-Capybara-34B-GGUF/resolve/main/nous-capybara-34b.Q3_K_L.gguf"
    # "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-oasst_top1_2023-08-25-v1-GGUF/resolve/main/mistral-7b-openorca-oasst_top1_2023-08-25-v1.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/cat-v1.0-13B-GGUF/resolve/main/cat-v1.0-13b.Q6_K.gguf"
    # "https://huggingface.co/TheBloke/Augmental-Unholy-13B-GGUF/resolve/main/augmental-unholy-13b.Q6_K.gguf"
    "https://huggingface.co/fakezeta/neural-chat-7b-v3-1/resolve/main/neural-chat-7b-v3-1_Q5_K_M.gguf"
    "https://huggingface.co/TheBloke/ShiningValiantXS-GGUF/resolve/main/shiningvaliantxs.Q6_K.gguf"
) 

for url in "${urls[@]}"
do
    cd /Users/user/demo_1/oobabooga_macos/text-generation-webui/models && wget --continue --tries=0 "$url"
done