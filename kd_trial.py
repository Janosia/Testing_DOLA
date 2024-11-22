import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dola import DoLa

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute the distillation loss (KL divergence) between student and teacher logits.
    """
    # Ensure both logits are the same size (vocabulary alignment)
    if student_logits.size(-1) != teacher_logits.size(-1):
        # Adjust the size of the logits by truncating or padding (in practice, truncating may be preferred)
        min_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[:, :, :min_vocab_size]
        teacher_logits = teacher_logits[:, :, :min_vocab_size]
    
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

def build_prompt(sample):
    """
    Build input prompt from a sample (e.g., question-answer pair).
    """
    return f"Question: {sample[0]}\nAnswer:"

# Initialize teacher and student models
teacher_model_name = "BEE-spoke-data/smol_llama-101M-GQA-python"
student_model_name = "PY007/TinyLlama-1.1B-step-50K-105b"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load teacher with DoLa
teacher_model = DoLa(teacher_model_name, device=device, num_gpus=1)

teacher_model.model.eval()   # Teacher is frozen during KD
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Load student model
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

# Hyperparameters
temperature = 2.0
optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)
epochs = 3

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for sample in samples:
        input_text = build_prompt(sample)

        # Tokenize input
        input_ids = teacher_tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Apply DoLa on the teacher model
        with torch.no_grad():
            # Access the actual model inside the DoLa wrapper
            teacher_outputs = teacher_model.model(input_ids=input_ids)
            teacher_logits = teacher_outputs[0]  # Get logits from the teacher model output
            
        # Student model forward pass
        student_outputs = student_model(input_ids, labels=input_ids)
        student_logits = student_outputs.logits  # Get logits from the student model

        # Compute KD loss
        loss = distillation_loss(student_logits, teacher_logits, temperature=temperature)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}: Total Loss = {total_loss:.4f}")

print("Training Complete.")
