import os
import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_project import DoLa  # Import your student model here (make sure to adjust the import path)

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute the distillation loss (KL divergence).
    :param student_logits: Logits from the student model.
    :param teacher_logits: Logits from the teacher model.
    :param temperature: Temperature for scaling logits.
    :return: Computed distillation loss.
    """
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction="batchmean")

def build_prompt(sample):
    # Define how you build prompts from your sample data
    # Example: Concatenate the question with a prompt template.
    return f"Question: {sample['question']}\nAnswer:"

def download_url(url, save_path):
    # This function downloads the file from the URL
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    # Add your file download logic here (using requests or another method)
    pass

def load_csv(file_path):
    # Example function to load a CSV file into a dictionary format
    # You should adapt this function based on your CSV format
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            question, answer = line.strip().split(',')
            data.append({"question": question, "answer": answer})
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./tfqa")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--use-knowledge-distillation", action="store_true", help="Enable KD")
    args = parser.parse_args()

    model_name = args.model_name
    device = args.device

    # Load the student model (your model for fine-tuning)
    student_model = DoLa(model_name, device, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)

    # Load the teacher model for KD
    teacher_model_name = "gpt2"  # Use a larger teacher model like GPT-3 if possible
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model.eval()

    # Get test file
    fp = os.path.join(args.data_path, 'TruthfulQA.csv')
    if not os.path.exists(fp):
        download_url('https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv', args.data_path)
    list_data_dict = load_csv(fp)

    if args.debug:
        list_data_dict = list_data_dict[:10]

    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]

    stop_word_list = ["Q:"]
    student_model.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    
    answers = []
    result_dict = {'question': [], 'model_completion': []}
    
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample)
        
        # Generate output using the student model
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty)
        student_model_completion = student_model.generate(input_text, **generate_kwargs)

        # Use the teacher model to guide generation (if KD is enabled)
        if args.use_knowledge_distillation:
            # Prepare the input for the teacher model
            teacher_input = teacher_tokenizer.encode(input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                teacher_logits = teacher_model(teacher_input).logits

            # Compute distillation loss
            student_input = teacher_tokenizer.encode(input_text, return_tensors="pt").to(device)
            student_logits = student_model(student_input).logits
            loss = distillation_loss(student_logits, teacher_logits)
            
            print(f"Distillation Loss: {loss.item()}")

        # Store the result
        result_dict['model_completion'].append(student_model_completion)
        result_dict['question'].append(sample)

        print(f'Question: {sample}\nModel Completion: {student_model_completion}\n')

    # Save results to a JSON file
    output_file = args.output_path if args.shard_id is None else (args.output_path + "_" + str(args.shard_id) + ".jsonl")
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
