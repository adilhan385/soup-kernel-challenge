# Soup Kernel Challenge — решение (Decentrathon 6.0)

**Кратко.** Ускоряем `soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct в 4-bit NF4, LoRA r=16 на `q_proj`/`v_proj`, batch 4, длина 512, 64 шага, fp16, Kaggle Tesla T4). В `solution/` семь собственных `@triton.jit`-кернелов: распаковка NF4, онлайн cross-entropy (прямой и обратный проход), RMSNorm (прямой/обратный), RoPE (прямой/обратный). Вокруг них собственные autograd-функции, которые не хранят лишние активации, и удаление pad-токенов внутри батча.
Последний полный замер на Kaggle T4: **speedup 1.355x**, **время 58.4 s**, **пиковая память в 5.199 раза ниже базы** (2759 MiB против 14345 MiB), eval loss **+0.007 %** к базе, пройдены **все 10 проверок** `check_rules.py`. Замер один (`--repeats 1`).

**English summary.** We speed up `soup train` on a Kaggle T4 for the fixed task (Qwen2.5-1.5B NF4 + LoRA r=16 on q/v, batch 4, seq 512, 64 steps). The solution has 7 hand-written Triton kernels: NF4 dequantization, online cross-entropy with entropy and accuracy (fwd/bwd), RMSNorm (fwd/bwd) and RoPE (fwd/bwd). Around them are custom autograd functions that drop activations nobody needs, plus within-batch unpadding. Last full run (1 repeat): 79.1 s → 58.4 s (**1.355x**), 14345 → 2759 MiB peak (**5.199x**), eval loss +0.007 %, all 10 rule checks passed. Kernel code is our own. FlashAttention, Liger-Kernel and Unsloth are cited only as sources of ideas; no code was copied from them.

---

## 1. Что сделано и почему

### 1.1 Куда уходит время и память в базе

Мы опирались на профиль базового Soup из условия задачи: около 44 % времени уходит на поэлементные операции и приведения типов, около 7 % на softmax по словарю из 152k токенов, около 5 % на распаковку NF4, а память занята на 14 из 15 ГБ. Отдельно мы разобрали код Soup, `transformers`, `peft` и `bitsandbytes` и нашли такие источники затрат:

1. **fp32 там, где хватает fp16.** `peft.prepare_model_for_kbit_training` на GPU без bf16 переводит замороженные не квантованные веса в fp32: эмбеддинги `[151936, 1536]`, веса RMSNorm, bias проекций. Из-за этого весь residual-поток идёт в fp32, а вокруг каждого 4-битного слоя появляются лишние приведения fp32↔fp16.
2. **Логиты по словарю 152k.** База целиком строит логиты `[4, 512, 151936]`, переводит их в fp32 и отдельно считает `log_softmax`, чтобы получить entropy. Это несколько буферов по ~1.2 ГБ.
3. **Лишние сохранённые активации.** `MatMul4Bit` из bitsandbytes сохраняет вход даже для замороженных весов. Обычный `Qwen2MLP` хранит несколько тензоров `[B·S, 8960]` на каждый из 28 слоёв, хотя LoRA на MLP нет.
4. **Мелкие раздельные кернелы** в RMSNorm, RoPE и cross-entropy.
5. **Pad-токены.** Внутри батча из 4 строк разной длины часть позиций занята паддингом, и на них тоже идут вычисления.

### 1.2 Собственные Triton-кернелы (всё в `solution/`)

Вместе они занимают **10.6 % времени обучения** (пункт 4 `check_rules.py`, шаги 3+, при пороге 5 %). Доля каждого кернела по отдельности: [УТОЧНИТЬ: `own_kernel_ms` в `runs/solution_1/metrics.json`].

| Кернел | Файл | Что делает |
|---|---|---|
| `qwen2_nf4_dequant_kernel` | `nf4_mlp.py` | Распаковывает 4-битные веса NF4 в fp16. Из каждого байта берёт два 4-битных индекса и находит их значения в таблице из 16 уровней NF4. Масштаб блока восстанавливает в два уровня (double quantization: `state2.code`, `state2.absmax`, `offset`), считает в fp32 и записывает fp16 в переиспользуемый буфер. Используется в прямом и обратном проходе всех 4-битных линейных слоёв. |
| `qwen2_online_ce_entropy_fwd_kernel` | `fused_ce.py` | Одна программа на одну строку логитов. Проходит словарь тайлами по 2048 и за один проход считает log-sum-exp (онлайн-softmax), loss по целевому токену, точную энтропию `H = lse − Σe^(z−m)·z / Σe^(z−m)` и argmax для `mean_token_accuracy`. Полный `log_softmax` в fp32 не нужен. |
| `qwen2_online_ce_bwd_kernel` | `fused_ce.py` | Записывает градиент `(softmax(z) − onehot(y)) / N` прямо на место fp16-логитов чанка. Сетка `(строки, словарь/2048)`. Для строк без метки пишет нули. |
| `qwen2_rmsnorm_fwd_kernel` | `rmsnorm.py` | Одна программа на строку: средний квадрат и `rstd` считаются в fp32, выход пишется в fp16. Для обратного прохода сохраняется только вектор `rstd`. |
| `qwen2_rmsnorm_bwd_kernel` | `rmsnorm.py` | `dx = (w·dy − x̂·mean(x̂·w·dy))·rstd` за один проход по строке. |
| `qwen2_rope_fwd_kernel` | `rope.py` | Поворачивает q (12 голов) и k (2 KV-головы) одним запуском, сетка `(B·S, Hq+Hk)`. Работает по страйдам без промежуточного `rotate_half`. Для обратного прохода сохраняются только `cos`/`sin`. |
| `qwen2_rope_bwd_kernel` | `rope.py` | Обратный поворот градиентов dq и dk тем же способом. |

### 1.3 Обвязка вокруг кернелов: PyTorch и cuBLAS, не Triton

Перемножения матриц делает cuBLAS (`torch.matmul`/`addmm_`) на fp16 Tensor Cores. Своего GEMM на Triton у нас нет.

- **`TritonChunkedHeadLossFn` (`fused_ce.py`).** Подключается через штатный интерфейс Soup `soup_cli.utils.stream_loss_chunk.ChunkedHeadLoss` (`default_chunk=64`, чанк до 256 строк). Логиты чанка `[≤256, 151936]` считаются в один общий fp16-буфер (74.2 MiB), полный тензор логитов не создаётся. Обратный проход считается только для токенов с меткой. Токены промпта проходят только прямой путь, потому что Soup логирует для них entropy. Градиент по скрытому состоянию считается сразу в forward, поэтому в backward логиты не пересчитываются. Soup по-прежнему пишет `loss`, `entropy`, `num_tokens`, `mean_token_accuracy` и `rewind.jsonl` (пункт 5 проверок пройден).
- **fp16 вместо fp32 для замороженных весов на sm<80 (`fused_ce.py`).** Патч `prepare_model_for_kbit_training` и `_setup_transformers` оставляет замороженные не квантованные веса (эмбеддинги, нормы, bias) в fp16. Обучаемые LoRA-веса остаются в fp32.
- **`TritonLinear4bitFn` (`nf4_mlp.py`)** для замороженных `k_proj`/`o_proj`. Для backward не сохраняет вход, сохраняет только ссылку на 4-битный вес. В backward вес распаковывается заново.
- **`TritonLoRALinear4bitFn` (`nf4_mlp.py`)** для `q_proj`/`v_proj` с LoRA. Базовый слой и LoRA (`x·Aᵀ`, затем `addmm_` с `Bᵀ·scaling`) считаются в fp16. Сохраняются `x` и `x·Aᵀ`. Градиенты `lora_A`/`lora_B` возвращаются в dtype параметров (fp32). Работает только для обычного случая: одна активная адаптер-группа, без DoRA, без dropout, без bias. В остальных случаях вызывается оригинальный forward PEFT.
- **`TritonCompactNF4MLPFn` (`nf4_mlp.py`).** `gate_proj` и `up_proj` распаковываются в один буфер `[2·8960, 1536]` и считаются одним GEMM. SwiGLU считается штатными `F.silu` и `torch.ops.aten.silu_backward` на месте с переиспользованием тензора градиента (на один буфер `N×8960` меньше). Отдельный Triton-кернел для SwiGLU мы **сознательно не пишем**: по правилам SwiGLU из примера не засчитывается.
- **Удаление pad-токенов внутри батча (`rope.py`).** Патч `Qwen2Model.forward` убирает pad-позиции и склеивает 4 последовательности в одну `[1, ΣLᵢ]` с корректными `position_ids`. Attention считается штатным `F.scaled_dot_product_attention` (не наш кернел) отдельно для каждой последовательности с `is_causal=True`. После модели скрытые состояния возвращаются в форму `[B, S, H]` (`_ScatterUnpaddedFn`). Батчи, их порядок и число токенов не меняются: `num_tokens` и строки `rewind.jsonl` совпадают с базой (пункт 5).
- **`fast_repeat_kv`.** Повтор KV-голов сделан так, что срезы по последовательности остаются плотными в памяти.
- **Кэш метаданных NF4 и общие scratch-буферы.** Метаданные квантования (указатели, blocksize, nested) готовятся один раз на каждый вес. Распакованные веса пишутся в два переиспользуемых буфера (слот 0 и слот 1) и не выделяются заново на каждый слой.

### 1.4 Собственный пересчёт в MLP (это не встроенный checkpointing Soup)

`TritonCompactNF4MLPFn` сам решает, что сохранить для backward. В финальной версии мы сохраняем промежуточный выход `gu = x·[W_gate; W_up]ᵀ` для всех 28 слоёв (страховочный порог `4096 MiB`), что полностью устраняет повторные тяжелые GEMM `gate+up` в backward. Режим `gradient_checkpointing` в Soup и HF не включается, `torch.utils.checkpoint` не вызывается.

История: в коммите `b28e41f` был собственный checkpointing подблока attention через `torch.utils.checkpoint`. В `f0aedf0` мы его **убрали**, чтобы обменять запас памяти на скорость. Сейчас его в коде нет.

### 1.5 Источники идей (код не копировался)

- Онлайн-нормализация softmax и log-sum-exp: Milakov & Gimelshein, 2018, *Online normalizer calculation for softmax*. Dao et al., 2022, *FlashAttention*.
- Идея считать loss по чанкам, не создавая полные логиты: Liger-Kernel (fused linear cross-entropy). Только идея, код свой.
- Идея отдельной autograd-функции для QLoRA-слоя без лишних сохранённых тензоров: Unsloth. Только идея, код свой.
- NF4 и double quantization: Dettmers et al., 2023, *QLoRA*, и формат `QuantState` из bitsandbytes.
- Шаблон построчной нормализации на Triton: официальный Triton tutorial *Layer Normalization*.
- Удаление паддинга перед attention (unpadding / varlen): идея из FlashAttention varlen. У нас это реализовано циклом по последовательностям поверх штатного PyTorch SDPA.

Ни одна из запрещённых библиотек (unsloth, liger-kernel, xformers, flash-attn, torchao и т. п.) не импортируется. Это подтверждает пункт 2 проверок: `loaded: none; imported in solution/: none`.

---

## 2. Как запустить

**Публичный ноутбук:** https://www.kaggle.com/code/lolkek4ik/notebook90b0a93b32  
**Репозиторий:** https://github.com/adilhan385/soup-kernel-challenge  
**Финальный коммит решения:** `26512f3` (ветка `main`).

### Kaggle (так же запускает жюри)

1. Открыть `templates/kaggle_notebook.ipynb` на Kaggle.
2. Справа выставить **Accelerator: GPU T4 x2** и **Internet: on** (нужен аккаунт с подтверждённым телефоном).
3. В первой ячейке указать:
   ```python
   REPO_URL = "https://github.com/adilhan385/soup-kernel-challenge.git"
   COMMIT = "26512f3"   # или оставить пустым "" для последнего main
   ```
4. **Run All**. Полный прогон занимает около 10 минут (только шаг `run_task.py` в нашем запуске: warm-up + база 79.1 s + решение 58.4 s + оценка обоих адаптеров).

Ноутбук ставит такие версии:
```
soup-cli[train]==0.77.0  transformers==5.16.1  peft==0.20.0  trl==0.29.1
bitsandbytes==0.50.2     accelerate==1.14.0    pytest
```
Версии torch, triton и CUDA взяты из образа Kaggle: [УТОЧНИТЬ].

### Команды

```bash
cd /kaggle/working/repo
python task/prepare_data.py                 # data/train.jsonl, data/eval_public.jsonl (256 + 256 строк)
python -m pytest solution -q                # юнит-тесты наших кернелов против эталона PyTorch / bitsandbytes
python harness/run_task.py                  # база, затем решение -> runs/result.json
python harness/check_rules.py               # 10 проверок правил
python harness/score.py runs/result.json
# быстрая итерация только решения, с профилем:
python harness/run_task.py --only solution --out runs_try --profile
```

Решение подключается само: harness вызывает `solution.apply()`, который ставит четыре патча (`rmsnorm`, `rope` + unpadding, `nf4_mlp`, `fused_ce`). Файлы `harness/`, `tests/` и `task/` не менялись с коммита стартового набора `c7c2fcd`.

---

## 3. Результаты и ограничения

### 3.1 Последний полный замер (коммит `26512f3`, Kaggle T4, `--repeats 1`)

| | База (Soup 0.77.0) | Решение | Отношение |
|---|---:|---:|---:|
| Время `soup train` целиком | 79.1 s | 58.4 s | **speedup 1.355x** |
| Пик памяти GPU (nvidia-smi) | 14345 MiB | 2759 MiB | **memory 5.199x** |
| Eval loss (`eval_adapter.py`) | 2.04966 | 2.04980 | **+0.007 %** |
| Доля наших кернелов | — | 10.6 % | порог ≥ 5 % |

`check_rules.py`, **ALL CHECKS PASSED**:

| # | Проверка | Результат |
|---|---|---|
| 1 | все прогоны завершились | PASS, 1 база + 1 решение |
| 2 | нет готовых библиотек кернелов | PASS, loaded: none; imported: none; packages changed: none |
| 3 | свои `@triton.jit` работают в обучении | PASS, все 7 кернелов |
| 4 | доля своих кернелов ≥ 5 % | PASS, 10.6 % |
| 5 | те же задача и выходы Soup | PASS, identical within tolerance |
| 6 | кривая loss повторяет базу | PASS, первый шаг 0.03 %, среднее 0.29 % (лимиты 1 % и 3 %) |
| 7 | eval loss не выше базы больше чем на 0.5 % | PASS, +0.007 % |
| 8 | не хуже базы по времени и памяти | PASS, time x0.738, memory x0.192 |
| 9 | чистый замер | PASS, clean |
| 10 | регрессионные тесты Soup с нашими патчами | PASS, 79 passed, 14 warnings |

`pytest solution -q`: **5 passed**.

### 3.2 Как менялся результат (все замеры на Kaggle T4, по одному прогону)

| Коммит | Что изменили | База, s | Решение, s | Speedup | Пик, MiB | Memory | Eval loss |
|---|---|---:|---:|---:|---:|---:|---:|
| `b28e41f` | unpadding, CE только по токенам с меткой, кэш метаданных NF4, checkpointing attention | 81.3 | 70.2 | 1.158x | 2037 | 7.042x | +0.004 % |
| `f0aedf0` | убрали checkpointing attention, сохраняем `gu` в MLP, единые чанки CE | 90.1 | 69.1 | 1.303x | 3119 | 4.599x | +0.010 % |
| `cd1b98d` | `TritonLoRALinear4bitFn`, `silu_backward` на месте, плотные `repeat_kv`/SDPA, чанк CE 256 | 84.8 | 62.9 | 1.348x | 2779 | 5.162x | +0.005 % |
| `26512f3` | сохранение `gu` без пересчета в MLP, оптимизация буферов SwiGLU backward | 79.1 | 58.4 | 1.355x | 2759 | 5.199x | +0.007 % |

### 3.3 Честно об ограничениях

- **Каждый замер сделан один раз** (`--repeats 1`). Жюри запускает `--repeats 2` на скрытых строках. Время базы между сессиями Kaggle колеблется от 79 до 90 секунд (разброс до 10-12%), поэтому относительное ускорение держится около ~1.35–1.40x.
- **Время по-прежнему в основном уходит на GEMM в cuBLAS и PyTorch SDPA.** Это не наши кернелы, наша доля 10.6 %. Своего fused-GEMM с распаковкой NF4 внутри и своего attention на Triton у нас нет.
- **Распаковка NF4 идёт заново в каждом forward и backward** (веса не кэшируются постоянно в VRAM ради сохранения сверхнизкого пика памяти в 2759 MiB).
- **Не мерили:** работу на других GPU (A100/L4, bf16), другие размеры батча и длины, `--repeats 2` у себя.

### 3.4 Что можно сделать дальше

1. Triton GEMM с распаковкой NF4 прямо в тайле, чтобы не писать fp16-вес в память.
2. Свой varlen-attention на Triton вместо цикла SDPA по последовательностям.
3. Слить residual add и RMSNorm в один кернел.
4. Замерить `--repeats 2` и проверять стабильность по медиане.

---

Лицензия: см. `LICENSE`. Тесты в `tests/` скопированы из Soup v0.77.0 (Apache-2.0, см. `NOTICE`).
