import warnings
import gc
from typing import Dict

# LangChain core components for memory and chains
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Your existing imports
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama
from langchain_huggingface import HuggingFaceEmbeddings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
DB_CHROMA_PATH = 'vectorstore/db_chroma'
OLLAMA_MODEL = "qwen3:8b-q4_K_M" 

# --- Session History Store ---
# This dictionary will store chat histories for different sessions.
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves the chat history for a given session_id.
    If the session doesn't exist, it creates a new one.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# --- Stream Response Function ---
def stream_response(chain, input_data, config):
    """
    Streams the response from a chain and prints it.
    This function is updated to handle different chunk formats.
    """
    full_response = ""
    for chunk in chain.stream(input_data, config=config):
        # The 'answer' key is common for retrieval chains
        if 'answer' in chunk:
            text = chunk['answer']
        # The 'content' attribute is common for direct LLM calls
        elif hasattr(chunk, 'content'):
            text = chunk.content
        else:
            # Handle other potential formats or skip
            continue
        
        print(text, end="", flush=True)
        full_response += text
    return full_response

# --- Query Classification ---
def classify_query_type(query):
    """Classify if query is about JEPCO services or general conversation"""
    service_keywords = [
        "دفع", "فاتورة", "سداد", "كهرباء", "طاقة", "حساب", "مبلغ",
        "اشتراك", "خدمة", "شكوى", "عطل", "انقطاع", "قراءة", "عداد",
        "جيبكو", "JEPCO", "شركة الكهرباء", "فني", "صيانة", "تركيب",
        "رسوم", "تعرفة", "فصل", "وصل", "طلب", "موظف", "مكتب", "فرع"
    ]
    general_keywords = [
        "من انت", "اسمك", "مرحبا", "شكرا", "وداعا", "كيف حالك",
        "ما اسمك", "من هذا", "اهلا", "السلام عليكم"
    ]

    query_lower = query.lower().strip()
    for keyword in general_keywords:
        if keyword in query_lower:
            return "general"
    for keyword in service_keywords:
        if keyword in query_lower:
            return "service"
    return "service"  # Default to service for unknown queries

# --- Chain Creation ---
def create_chains():
    """Create both RAG and General Conversation chains."""
    #print("Initializing chains...")

    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.2,
        num_predict=2048,
        top_k=20,
        top_p=0.5,
        reasoning= False
    )

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        model_kwargs={'device':'cuda'}
    )

    db = Chroma(
        persist_directory=DB_CHROMA_PATH,
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={'k': 25})

    # --- RAG Chain for Service Questions (More Strict Prompt) ---
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """أنت نظام آلي للإجابة المباشرة. مهمتك هي استخراج الإجابة من السياق وتقديمها فوراً.

**قواعد صارمة للغاية:**
1. أجب بطريقة مناسبة لتحويلها لكلام
2. ممنوع منعاً باتاً طباعة أي نص تمهيدي، أو أفكار، أو أي شيء يشبه <think>.
3. يجب أن تكون إجابتك هي النص الفعلي للسؤال فقط.
4. إذا كانت المعلومات غير موجودة في السياق، أجب فقط بـ: "المعلومات المطلوبة غير متوفرة حالياً."
5. dont print emojis.
6. لا تستخدم علامات الاقتباس أو أي تنسيق آخر.
7. dont use any markdown formatting or qotes or # or *.

**السياق:**
---
{context}
---
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # --- General Conversation Chain (More Strict Prompt) ---
    general_prompt = ChatPromptTemplate.from_messages([
        ("system", """أنت مساعد آلي لشركة جيبكو. أجب على المستخدم مباشرةً وبإيجاز.

**قواعد صارمة للغاية:**
1. ممنوع منعاً باتاً طباعة أي نص تمهيدي، أو أفكار، أو أي شيء يشبه <think>.
2. ابدأ بالإجابة النهائية مباشرةً.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    general_chain = general_prompt | llm
    
    return rag_chain, general_chain

def main():
    """Main interactive chat with memory and smart routing"""
    base_rag_chain, base_general_chain = create_chains()

    # --- Wrap chains with memory ---
    conversational_rag_chain = RunnableWithMessageHistory(
        base_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    conversational_general_chain = RunnableWithMessageHistory(
        base_general_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    #print("Smart RAG system with memory is ready!")
    #print("\n" + "=" * 60)
    print(" مرحباً بك في خدمة عملاء شركة الكهرباء الأردنية (جيبكو)")
    #print(f" النموذج المستخدم: {OLLAMA_MODEL}")
    print(" اكتب سؤالك أو 'خروج' للإنهاء")
    #print("=" * 60 + "\n")

    session_id = "main_chat_session"

    while True:
        try:
            user_input = input("أنت: ").strip()
            if user_input.lower() in ["خروج", "exit", ""]:
                print("شكراً لاستخدام خدمة عملاء جيبكو! نتطلع لخدمتك مرة أخرى.")
                break

            config = {"configurable": {"session_id": session_id}}
            
            query_type = classify_query_type(user_input)
            #print(f"نوع السؤال: {'محادثة عامة' if query_type == 'general' else 'خدمات الشركة'}")

            if query_type == "general":
                #print("المساعد: ", end="")
                stream_response(conversational_general_chain, {"input": user_input}, config)
            else:
                #print("جاري البحث في قاعدة البيانات...")
                #print("المساعد: ", end="")
                stream_response(conversational_rag_chain, {"input": user_input}, config)
            
            #print("\n" + "-" * 40)

        except KeyboardInterrupt:
            print("\n\nتم إنهاء الجلسة بواسطة المستخدم.")
            break
        except Exception as e:
            print(f"\nحدث خطأ: {str(e)}")
            print("يرجى المحاولة مرة أخرى.\n")
        finally:
            gc.collect()

if __name__ == "__main__":
    main()