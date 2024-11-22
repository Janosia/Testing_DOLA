import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dola import DoLa
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute the distillation loss (KL divergence) between student and teacher logits.
    Ensures that the logits match in shape for KL divergence computation.
    """
    # Apply softmax on teacher's logits to get probabilities
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    print(f"Teacher probs shape: {teacher_probs.shape}")  # Print shape of teacher probs
    
    # Apply log_softmax on student's logits
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    print(f"Student probs shape: {student_probs.shape}")  # Print shape of student probs
    
    # KL Divergence loss (average across the batch)
    loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    print(f"Loss value: {loss.item()}")  # Print the loss value
    
    return loss

def build_prompt(sample):
    """
    Build input prompt from a sample (e.g., question-answer pair).
    """
    return f"Question: {sample[0]}\nAnswer:"

# Initialize teacher and student models
teacher_model_name = "BEE-spoke-data/smol_llama-101M-GQA-python"
student_model_name = "BEE-spoke-data/smol_llama-101M-GQA-python"
# student_model_name = "PY007/TinyLlama-1.1B-step-50K-105b"

# Load teacher with DoLa
teacher_model = DoLa(teacher_model_name, device=device, num_gpus=1)
teacher_model.model.eval()  # Teacher is frozen during KD
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Move the student model to the correct device
student_model = AutoModelForCausalLM.from_pretrained(student_model_name).to(device)
student_model.train()

# Define parameters for DoLa (early exit layers)
early_exit_layers = [2, 6]  # Example: Layer 2 (premature) and Layer 6 (mature)
mode = "dola"  # Specify DoLa mode

# Input data
samples = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
]  # Add more question-answer pairs for training
optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)

for sample in samples:
    input_text = f"Question: {sample[0]}\nAnswer:"
    input_ids = teacher_tokenizer(input_text, return_tensors="pt").input_ids.to(device)  # Ensure input_ids are on the correct device
    
    student_logits = student_model(input_ids).logits.to(device) 
    print(f"Student logits shape: {student_logits.shape}")
    with torch.no_grad():
      
        teacher_logits = teacher_model.model(input_ids).logits.to(device) 
        
        print(f"Teacher logits shape: {teacher_logits.shape}")
            
        # Project teacher logits to match the student model's vocabulary size
        # projection_layer = nn.Linear(teacher_logits.shape[-1], student_model.config.vocab_size).to(device)
        # teacher_logits = projection_layer(teacher_logits)
        teacher_logits = teacher_logits[..., :student_logits.size(-1)]
        
        print(f"Adjusted teacher logits shape: {teacher_logits.shape}")
    print(f"Student logits shape: {student_logits.shape}, device: {student_logits.device}")
    print(f"Teacher logits shape: {teacher_logits.shape}, device: {teacher_logits.device}")

    loss = torch.nn.functional.mse_loss(student_logits, teacher_logits)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")
# # Hyperparameters
# temperature = 2.0
# optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)
# epochs = 3

# # Training loop
# for epoch in range(epochs):
#     total_loss = 0
#     print(f"Epoch {epoch + 1}/{epochs}")
#     for sample in samples:
#         input_text = build_prompt(sample)
#         print(f"\nInput text: {input_text}")

#         # Tokenize input
#         input_ids = teacher_tokenizer(input_text, return_tensors="pt").input_ids.to(device)
#         print(f"Tokenized input_ids shape: {input_ids.shape}")

#         # Apply DoLa on the teacher model
#         with torch.no_grad():
#             # Generate teacher's raw output logits
#             teacher_outputs = teacher_model.model(input_ids)
#             teacher_logits = teacher_outputs.logits
#             print(f"Teacher logits shape: {teacher_logits.shape}")
            
#             # Project teacher logits to match the student model's vocabulary size
#             projection_layer = nn.Linear(teacher_logits.shape[-1], student_model.config.vocab_size).to(device)
#             teacher_logits = projection_layer(teacher_logits)
#             print(f"Adjusted teacher logits shape: {teacher_logits.shape}")


#         # Student model forward pass
#         student_outputs = student_model(input_ids, labels=input_ids)
#         student_logits = student_outputs.logits
#         print(f"Student logits shape: {student_logits.shape}")
#         print(f"Adjusted teacher logits shape: {teacher_logits.shape}")
#         # Compute KD loss
#         loss = distillation_loss(student_logits, teacher_logits, temperature=temperature)
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Total Loss for Epoch {epoch + 1}: {total_loss:.4f}")

# print("Training Complete.")
