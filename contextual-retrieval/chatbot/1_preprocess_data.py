import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load your Q&A dataset
try:
    with open('jepco_chatbot_dataset.json', 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
except FileNotFoundError:
    print("Error: jepco_chatbot_dataset.json not found. Make sure it's in the same directory.")
    exit()

print(f"Loaded {len(qa_data)} Q&A pairs.")

# Load a small multilingual instruct model (Qwen2-0.5B-Instruct)
model_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Force model onto GPU only
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # use float16 for faster GPU inference
).to("cuda")

# Force pipeline to use GPU
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,
    device=0  # CUDA:0
)

contextualized_chunks = []
for i, item in enumerate(qa_data):
    # Use the first question as the representative question
    question_text = item['questions'][0] if item['questions'] else ""
    answer_text = item['answer']
    category = item.get('category', '')

    original_content = f"Category: {category}\nQuestion: {question_text}\nAnswer: {answer_text}"

    print(f"Processing item {i+1}/{len(qa_data)}")

    # Prompt for the local model
    prompt = f"""Here is a Question/Answer pair from a knowledge base:
<chunk>
{original_content}
</chunk>
Please give a short, succinct context to situate this chunk for the purposes of improving search retrieval. This context should summarize what the chunk is about in a single sentence. Answer only with the succinct context and nothing else."""

    try:
        result = generator(prompt)
        generated_context = result[0]['generated_text'].split("</chunk>")[-1].strip().split("\n")[0]

        final_chunk_for_indexing = f"Context: {generated_context}\n{original_content}"

        contextualized_chunks.append({
            "category": category,
            "questions": item['questions'],
            "answer": answer_text,
            "original_content": original_content,
            "contextualized_content": final_chunk_for_indexing
        })

    except Exception as e:
        print(f"An error occurred with item {i+1}: {e}")
        continue

with open('contextualized_data.json', 'w', encoding='utf-8') as f:
    json.dump(contextualized_chunks, f, ensure_ascii=False, indent=4)

print("\nPreprocessing complete. Data saved to contextualized_data.json.")
