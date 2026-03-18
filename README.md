# Fine-tuning и запуск моделей с LoRA

Проект содержит:
- скрипт дообучения модели LoRA: `fine_tuning/train.py`
- скрипт запуска чата: `inference/chat.py`
- пример датасета: `example_dataset.json`

## Структура

```text
.
├── fine_tuning/
│   ├── train.py
│   └── README.md
├── inference/
│   ├── chat.py
│   └── README.md
├── example_dataset.json
├── requirements.txt
└── README.md
```

## Установка

Если вы работаете в уже созданном `venv`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Для RTX 5060 можно поставить PyTorch с CUDA 12.8:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Проверка:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Датасет

В репозитории уже есть готовый пример: `example_dataset.json`.

Рекомендации по расширению датасета (чтобы меньше галлюцинаций): `DATASET_GUIDE.md`.

Поддерживаются `.json` и `.jsonl` в форматах:
- `{"text": "..."}`
- `{"instruction": "...", "output": "..."}`
- `{"prompt": "...", "completion": "..."}`
- `{"input": "...", "output": "..."}`

## Быстрый старт

### CPU

```powershell
python .\fine_tuning\train.py --model_name "microsoft/DialoGPT-small" --dataset_path ".\example_dataset.json" --output_dir ".\fine_tuning\lora_model_cpu" --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --device cpu
```

### GPU

```powershell
python .\fine_tuning\train.py --model_name "microsoft/DialoGPT-small" --dataset_path ".\example_dataset.json" --output_dir ".\fine_tuning\lora_model_gpu" --use_4bit --num_train_epochs 3 --device cuda
```

Примечания:
- на CPU не используйте `--use_4bit`
- на слабом железе уменьшайте `--per_device_train_batch_size`
- базовая модель скачивается автоматически при первом запуске

## Проверочный запуск на маленькой модели

```powershell
python .\fine_tuning\train.py --model_name "sshleifer/tiny-gpt2" --dataset_path ".\example_dataset.json" --output_dir ".\fine_tuning\tmp_cpu_test" --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_length 32 --device cpu
```

## Запуск чата после обучения

```powershell
python .\inference\chat.py --base_model "microsoft/DialoGPT-small" --lora_model ".\fine_tuning\lora_model_cpu"
```

## Рекомендуемый запуск (RAG + консультантский промпт)

```bash
python3 inference/chat.py \
  --base_model "ai-forever/rugpt3small_based_on_gpt2" \
  --lora_model "fine_tuning/lora_rugpt3small_v2" \
  --dataset_path "example_dataset.json" \
  --retrieval_k 5 \
  --system_prompt "Ты инженер-консультант по инженерным системам частных домов. Если не хватает данных — задай 2-5 уточняющих вопросов. Не выдумывай адреса/телефоны/точные цены/сроки — если данных нет, так и скажи и попроси вводные."
```

## Что важно знать

- `fine_tuning/train.py` теперь поддерживает `--device auto|cpu|cuda`
- если указать `--device cpu`, скрипт автоматически отключит 4-bit quantization
- на Windows/Powershell убраны проблемные Unicode-символы из логов

## Полезные модели для старта

- `microsoft/DialoGPT-small`
- `distilgpt2`
- `gpt2`
- `sshleifer/tiny-gpt2` для smoke-test
