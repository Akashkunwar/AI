from unsloth import FastLanguageModel
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instruction = "Create a function to find largest number in a array."
input = "[1, 2, 3, 4, 5]"

# 2. Generate output using the base model (unsloth/Phi-3-mini-4k-instruct)
base_model_name = "unsloth/Phi-3-mini-4k-instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = os.getenv("HF_TOKEN")
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction, # instruction
            input, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt"
).to("cuda")

text_streamer = TextStreamer(tokenizer)
print("Base model output:")
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1000)

# 3. Generate output using the fine-tuned model
fine_tuned_model_dir = "lora_model"
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_dir)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)

inputs_fine_tuned = fine_tuned_tokenizer(
    alpaca_prompt.format(
        instruction, # instruction
        input, # input
        "", # output - leave this blank for generation!
    ), 
    return_tensors="pt"
).to("cuda")

text_streamer_fine_tuned = TextStreamer(fine_tuned_tokenizer)
print("\nFine-tuned model output:")
output_fine_tuned = fine_tuned_model.generate(**inputs_fine_tuned, streamer=text_streamer_fine_tuned, max_new_tokens=1000)

# Print the generated output from the fine-tuned model
print(fine_tuned_tokenizer.decode(output_fine_tuned[0], skip_special_tokens=True))