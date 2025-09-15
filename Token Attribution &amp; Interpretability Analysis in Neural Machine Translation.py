!pip install inseq==0.5.0
!pip install transformers==4.40.2
!pip install bitsandbytes==0.43.1
!pip install -U accelerate==0.28.0

# Machine Translation Task

import inseq

# List of attribution methods to be used
attribution_methods = ['saliency', 'attention']

for method in attribution_methods:
    print(f"======= Attribution Method: {method} =======")
    # Load the Chinese-to-English translation model and set up the attribution method
    model = inseq.load_model("Helsinki-NLP/opus-mt-zh-en", method)

    # Apply attribution to the input text using the specified method
    attribution_result = model.attribute(
        input_texts="我喜欢人工智慧。",
        step_scores=["probability"],
    )

    # Remove '▁' from the tokenizer in the prefix to avoid confusion (You can ignore this part of code)
    for attr in attribution_result.sequence_attributions:
        for item in attr.source:
            item.token = item.token.replace('▁', '')
        for item in attr.target:
            item.token = item.token.replace('▁', '')

    # Display the attribution results
    attribution_result.show()


# Sentence Completion Task

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2-xl", load_in_8bit=True, device_map="auto")

for method in attribution_methods:
    # Load the model with the specified attribution method
    inseq_model = inseq.load_model(model, method)

    # Apply attribution to the input text using the specified method
    attribution_result = inseq_model.attribute(
        input_texts="The first president of America is",
        step_scores=["probability"],
    )

    # Remove 'Ġ' from GPT2's BPE tokenizer in the prefix to avoid confusion (You can ignore this part of code)
    for attr in attribution_result.sequence_attributions:
        for item in attr.source:
            item.token = item.token.replace('Ġ', '')
        for item in attr.target:
            item.token = item.token.replace('Ġ', '')

    # Display the attribution results
    attribution_result.show()
