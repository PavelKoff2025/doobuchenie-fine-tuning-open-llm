"""
Скрипт для запуска модели в терминале для общения
"""
import os
import re
import math
from dataclasses import dataclass
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
import json

def load_model_and_tokenizer(base_model_name, lora_model_path=None):
    """
    Загружает модель и токенизатор
    
    Args:
        base_model_name: имя базовой модели с HuggingFace
        lora_model_path: путь к дообученной LoRA модели (опционально)
    """
    print(f"Загрузка модели {base_model_name}...")
    
    # Проверка наличия GPU
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"\n{'='*60}")
        print(f"ИСПОЛЬЗОВАНИЕ GPU")
        print(f"{'='*60}")
        print(f"GPU устройство: {device_name}")
        print(f"Количество GPU: {device_count}")
        print(f"Текущий GPU: {current_device}")
        
        # Показываем информацию о памяти GPU
        for i in range(device_count):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i} память: {total_memory:.2f} GB")
        print(f"{'='*60}\n")
        
        torch_dtype = torch.float16
        device_map = "auto"
    else:
        device = "cpu"
        torch_dtype = torch.float32
        device_map = None
        print(f"\n⚠ ВНИМАНИЕ: GPU не обнаружен, используется CPU")
        print(f"Генерация на CPU будет медленной!\n")
    
    # Определяем директорию для базовой модели (в проекте)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_cache_dir = os.path.join(project_root, "models", base_model_name.replace("/", "_"))
    os.makedirs(base_model_cache_dir, exist_ok=True)
    
    config_path = os.path.join(base_model_cache_dir, "config.json")
    has_local_model = os.path.exists(config_path)
    
    # Проверяем, есть ли локальная копия модели
    if has_local_model:
        print(f"Использование локальной копии модели из {base_model_cache_dir}")
        model_path = base_model_cache_dir
    else:
        print(f"Модель будет скачана и сохранена в {base_model_cache_dir}")
        model_path = base_model_name
    
    # Загрузка токенизатора
    tokenizer_config_path = os.path.join(base_model_cache_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        print(f"Загрузка токенизатора из локальной директории...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_cache_dir)
    else:
        print(f"Загрузка токенизатора из HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(base_model_cache_dir)
        print(f"Токенизатор сохранен в {base_model_cache_dir}")
    
    # Установка pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Загрузка базовой модели
    print(f"Загрузка модели из {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        use_safetensors=True,
    )
    
    # Сохраняем модель локально, если она была скачана
    if not has_local_model:
        print(f"Сохранение модели в {base_model_cache_dir}...")
        model.save_pretrained(base_model_cache_dir)
        print(f"Модель сохранена!")
    
    # Показываем информацию о памяти GPU после загрузки
    if torch.cuda.is_available():
        print(f"\nСостояние памяти GPU после загрузки:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = total - reserved
            print(f"  GPU {i}: {allocated:.2f}GB / {reserved:.2f}GB (свободно: {free:.2f}GB)")
    
    # Загрузка LoRA весов если указан путь
    if lora_model_path and os.path.exists(lora_model_path):
        print(f"Загрузка LoRA весов из {lora_model_path}...")
        model = PeftModel.from_pretrained(model, lora_model_path)
        model = model.merge_and_unload()  # Объединяем LoRA веса с базовой моделью
        print("LoRA веса успешно загружены и объединены!")
    elif lora_model_path:
        print(f"Предупреждение: Путь {lora_model_path} не найден. Используется базовая модель.")
    
    # Переводим модель в режим оценки
    model.eval()
    
    return model, tokenizer

def _tokenize_for_retrieval(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    # Оставляем буквы/цифры/пробелы. Подходит и для кириллицы.
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t]

@dataclass
class RetrievalDoc:
    query_text: str
    answer_text: str
    tokens: list[str]

class BM25Index:
    def __init__(self, docs: list[RetrievalDoc], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b

        self.doc_lens = [len(d.tokens) for d in docs]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0

        df: dict[str, int] = {}
        for d in docs:
            for tok in set(d.tokens):
                df[tok] = df.get(tok, 0) + 1
        self.df = df
        self.N = len(docs)

    def idf(self, term: str) -> float:
        # BM25+ style idf with smoothing
        n_q = self.df.get(term, 0)
        return math.log(1.0 + (self.N - n_q + 0.5) / (n_q + 0.5)) if self.N else 0.0

    def score(self, query_tokens: list[str], doc: RetrievalDoc, dl: int) -> float:
        if not query_tokens or not doc.tokens or self.avgdl <= 0:
            return 0.0
        tf: dict[str, int] = {}
        for t in doc.tokens:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for q in query_tokens:
            f = tf.get(q, 0)
            if f <= 0:
                continue
            idf = self.idf(q)
            denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            score += idf * (f * (self.k1 + 1.0)) / denom
        return score

    def search(self, query: str, k: int = 3) -> list[tuple[float, RetrievalDoc]]:
        q_toks = _tokenize_for_retrieval(query)
        scored: list[tuple[float, RetrievalDoc]] = []
        for doc, dl in zip(self.docs, self.doc_lens):
            s = self.score(q_toks, doc, dl)
            if s > 0:
                scored.append((s, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

def load_dataset_examples(dataset_path: str) -> list[dict[str, Any]]:
    if not dataset_path:
        return []
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл датасета не найден: {dataset_path}")

    if dataset_path.endswith(".jsonl"):
        items: list[dict[str, Any]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items

    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())[0] if data else []
        return []

    raise ValueError("Поддерживаются только .json и .jsonl файлы")

def build_retrieval_index(examples: list[dict[str, Any]]) -> BM25Index:
    docs: list[RetrievalDoc] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue

        # Формируем "вопрос" (то, по чему ищем)
        q = (
            ex.get("instruction")
            or ex.get("prompt")
            or ex.get("input")
            or ex.get("question")
            or ex.get("text")
            or ""
        )
        # Формируем "ответ" (что подставляем как знание)
        a = ex.get("output") or ex.get("completion") or ex.get("answer") or ex.get("text") or ""

        q = str(q).strip()
        a = str(a).strip()
        if not q or not a:
            continue

        toks = _tokenize_for_retrieval(q)
        if not toks:
            continue

        docs.append(RetrievalDoc(query_text=q, answer_text=a, tokens=toks))

    return BM25Index(docs)

def format_retrieved_context(hits: list[tuple[float, RetrievalDoc]]) -> str:
    if not hits:
        return ""
    lines = ["### Контекст из датасета (похожие примеры)"]
    for i, (score, doc) in enumerate(hits, start=1):
        lines.append(f"{i}) Вопрос: {doc.query_text}")
        lines.append(f"   Ответ: {doc.answer_text}")
    return "\n".join(lines)

def generate_response(
    model,
    tokenizer,
    prompt,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_new_tokens=128,
):
    """
    Генерирует ответ модели на промпт
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: входной промпт
        max_length: максимальная длина генерируемого текста
        temperature: температура для генерации (чем выше, тем более случайно)
        top_p: nucleus sampling параметр
        top_k: top-k sampling параметр
    """
    # Токенизация промпта (важно: attention_mask, чтобы не было предупреждений)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    
    # Перемещение на устройство модели (автоматически определяется device_map)
    # Если device_map="auto", модель сама определяет устройство
    if hasattr(model, 'device'):
        input_ids = input_ids.to(model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
    elif torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cuda")
    
    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            # Генерируем только max_new_tokens, не ограничивая длину входа,
            # чтобы избежать предупреждений о превышении max_length
            max_new_tokens=max_new_tokens,
            # Более "строгая" генерация: меньше галлюцинаций, но ответы суше
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Декодирование ответа
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Удаляем промпт из ответа, оставляем только сгенерированный текст
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

def chat_loop(model, tokenizer, system_prompt="", max_length=512, temperature=0.7):
    """
    Основной цикл чата
    
    Args:
        model: модель
        tokenizer: токенизатор
        system_prompt: системный промпт (опционально)
        max_length: максимальная длина ответа
        temperature: температура генерации
    """
    print("\n" + "="*50)
    print("Чат с моделью запущен!")
    print("Введите 'quit', 'exit' или 'q' для выхода")
    print("Введите 'clear' для очистки истории")
    print("="*50 + "\n")
    
    conversation_history = []
    
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})

    # Retrieval index (optional)
    retrieval_index = None
    retrieval_k = 0
    retrieval_min_score = 0.0
    answer_from_dataset = False
    if getattr(chat_loop, "_retrieval_config", None):
        cfg = chat_loop._retrieval_config
        retrieval_index = cfg.get("index")
        retrieval_k = int(cfg.get("k", 0))
        retrieval_min_score = float(cfg.get("min_score", 0.0))
        answer_from_dataset = bool(cfg.get("answer_from_dataset", False))
    
    while True:
        try:
            # Получение ввода пользователя
            user_input = input("Вы: ").strip()
            
            if not user_input:
                continue
            
            # Команды выхода
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("До свидания!")
                break
            
            # Очистка истории
            if user_input.lower() == 'clear':
                conversation_history = []
                if system_prompt:
                    conversation_history.append({"role": "system", "content": system_prompt})
                print("История очищена.\n")
                continue
            
            # Формирование промпта из истории разговора
            retrieved_context = ""
            if retrieval_index is not None and retrieval_k > 0:
                hits = retrieval_index.search(user_input, k=retrieval_k)
                hits = [(s, d) for (s, d) in hits if s >= retrieval_min_score]
                retrieved_context = format_retrieved_context(hits)
                if answer_from_dataset and hits:
                    # Самый "жесткий" режим: отвечаем как в датасете
                    print("\nМодель: " + hits[0][1].answer_text + "\n")
                    conversation_history.append({"role": "user", "content": user_input})
                    conversation_history.append({"role": "assistant", "content": hits[0][1].answer_text})
                    continue

            if conversation_history:
                # Форматируем историю для модели
                prompt_parts = []
                for msg in conversation_history:
                    if msg["role"] == "system":
                        prompt_parts.append(f"System: {msg['content']}")
                    elif msg["role"] == "user":
                        prompt_parts.append(f"User: {msg['content']}")
                    elif msg["role"] == "assistant":
                        prompt_parts.append(f"Assistant: {msg['content']}")
                
                prompt_parts.append(f"User: {user_input}")
                prompt_parts.append("Assistant:")
                if retrieved_context:
                    prompt_parts.insert(0, retrieved_context)
                prompt = "\n".join(prompt_parts)
            else:
                if retrieved_context:
                    prompt = f"{retrieved_context}\n\nUser: {user_input}\nAssistant:"
                else:
                    prompt = f"User: {user_input}\nAssistant:"
            
            # Генерация ответа
            print("Модель генерирует ответ...")
            response = generate_response(
                model, 
                tokenizer, 
                prompt, 
                max_length=max_length,
                temperature=0.3,
                top_p=0.8,
                top_k=40,
            )
            
            # Вывод ответа
            print(f"\nМодель: {response}\n")
            
            # Сохранение в историю
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            
            # Ограничение истории (оставляем последние 10 сообщений)
            if len(conversation_history) > 20:
                if system_prompt:
                    conversation_history = [conversation_history[0]] + conversation_history[-19:]
                else:
                    conversation_history = conversation_history[-20:]
        
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")
            continue

def main():
    parser = argparse.ArgumentParser(description="Чат с моделью в терминале")
    parser.add_argument("--base_model", type=str, required=True, help="Имя базовой модели с HuggingFace")
    parser.add_argument("--lora_model", type=str, default=None, help="Путь к дообученной LoRA модели")
    parser.add_argument("--system_prompt", type=str, default="", help="Системный промпт")
    parser.add_argument("--max_length", type=int, default=512, help="Максимальная длина ответа")
    parser.add_argument("--temperature", type=float, default=0.7, help="Температура генерации")
    parser.add_argument("--dataset_path", type=str, default="", help="Путь к датасету (.json/.jsonl) для поиска (RAG)")
    parser.add_argument("--retrieval_k", type=int, default=3, help="Сколько примеров доставать из датасета")
    parser.add_argument("--retrieval_min_score", type=float, default=0.0, help="Минимальный score для контекста")
    parser.add_argument("--answer_from_dataset", action="store_true", help="Отвечать напрямую лучшим ответом из датасета")
    
    args = parser.parse_args()
    
    # Загрузка модели
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.lora_model)
    
    # Готовим retrieval (если задан датасет)
    if args.dataset_path:
        examples = load_dataset_examples(args.dataset_path)
        index = build_retrieval_index(examples)
        chat_loop._retrieval_config = {
            "index": index,
            "k": args.retrieval_k,
            "min_score": args.retrieval_min_score,
            "answer_from_dataset": args.answer_from_dataset,
        }

    # Запуск чата
    chat_loop(
        model, 
        tokenizer, 
        system_prompt=args.system_prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()
