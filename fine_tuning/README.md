# Fine-tuning модели с LoRA

`train.py` обучает LoRA-адаптер поверх базовой модели Hugging Face.

## Что поддерживается

- устройство: `--device auto|cpu|cuda`
- датасеты: `.json` и `.jsonl`
- форматы примеров:
  - `text`
  - `instruction/output`
  - `prompt/completion`
  - `input/output`

## Быстрый запуск

### CPU

```powershell
python .\fine_tuning\train.py --model_name "microsoft/DialoGPT-small" --dataset_path ".\example_dataset.json" --output_dir ".\fine_tuning\lora_model_cpu" --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --device cpu
```

### GPU

```powershell
python .\fine_tuning\train.py --model_name "microsoft/DialoGPT-small" --dataset_path ".\example_dataset.json" --output_dir ".\fine_tuning\lora_model_gpu" --use_4bit --num_train_epochs 3 --device cuda
```

## Полезные параметры

- `--model_name` - базовая модель с Hugging Face
- `--dataset_path` - путь к `.json` или `.jsonl`
- `--output_dir` - папка, куда сохранить LoRA-веса
- `--num_train_epochs` - число эпох
- `--per_device_train_batch_size` - размер батча
- `--gradient_accumulation_steps` - накопление градиента
- `--max_length` - максимальная длина токенизированного примера
- `--use_4bit` - квантование, только для GPU
- `--device` - `auto`, `cpu` или `cuda`

## Рекомендации

- Для CPU начинайте с `batch_size=1`
- Для быстрого smoke-test используйте `sshleifer/tiny-gpt2`
- Для GPU с маленькой VRAM используйте `--use_4bit`
- Если на CPU случайно передан `--use_4bit`, скрипт сам его отключит

## Пример датасета

В проекте уже есть файл `example_dataset.json`, его можно использовать сразу.
