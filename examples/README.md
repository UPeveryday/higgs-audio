# Examples

> [!NOTE]  
> If you do not like the audio you get, you can generate multiple times with different seeds. In addition, you may need to apply text normalization to get the best performance, e.g. converting 70 °F to "seventy degrees Fahrenheit", and converting "1 2 3 4" to "one two three four". The model also performs better in longer sentences. Right now, the model has not been post-trained, we will release the post-trained model in the future.

## generation.py 用法与参数

> 本节介绍 `examples/generation.py` 的命令行用法与全部参数说明。

### 基本用法

```bash
python3 generation.py [OPTIONS]
# Windows PowerShell
python .\generation.py [OPTIONS]
```

常见最小示例：

```bash
# 随机音色朗读英文
python3 generation.py \
  --transcript transcript/single_speaker/en_dl.txt \
  --out_path generation.wav

# 克隆指定音色朗读
python3 generation.py \
  --transcript transcript/single_speaker/en_dl.txt \
  --ref_audio belinda \
  --seed 12345 \
  --out_path generation.wav
```

### 参数说明（含默认值）

- **--model_path (str, 默认: `bosonai/higgs-audio-v2-generation-3B-base`)**: 模型路径或名称（Hugging Face）。
- **--audio_tokenizer (str, 默认: `bosonai/higgs-audio-v2-tokenizer`)**: 音频分词器路径或名称。
- **--max_new_tokens (int, 默认: 2048)**: 生成的最大新 token 数（影响最大时长）。
- **--transcript (str, 默认: `transcript/single_speaker/en_dl.txt`)**: 文本路径或直接文本；若该路径存在则读取文件内容。
- **--scene_prompt (str, 默认: `scene_prompts/quiet_indoor.txt`)**: 场景描述提示词；设为 `empty` 或不存在则为空。
- **--temperature (float, 默认: 1.0)**: 采样温度（低=稳定，高=多样）。
- **--top_k (int, 默认: 50)**: Top-K 采样阈值。
- **--top_p (float, 默认: 0.95)**: Top-P（核采样）阈值。
- **--ras_win_len (int, 默认: 7)**: RAS 采样窗口长度；≤0 则禁用 RAS。
- **--ras_win_max_num_repeat (int, 默认: 2)**: RAS 窗口最大重复次数（仅在启用 RAS 时有效）。
- **--ref_audio (str, 默认: None)**: 参考音色；可用 `voice_prompts` 下基名（如 `belinda`），或多音色 `a,b` 对应 `SPEAKER0/1`；也支持 `profile:<name>` 从 `voice_prompts/profile.yaml` 以文字特征设定音色。
- **--ref_audio_in_system_message (flag, 默认: False)**: 是否将参考音色描述放入系统消息（多说话人/占位更方便）。
- **--chunk_method (None|speaker|word, 默认: None)**: 文本分段策略：
  - None：整段一次生成；
  - speaker：按 `[SPEAKER*]` 轮次切分；
  - word：按词数/字数切分（中文按字、英文按空格）。
- **--chunk_max_word_num (int, 默认: 200)**: `word` 分段时，单段最大词/字数。
- **--chunk_max_num_turns (int, 默认: 1)**: `speaker` 分段时，合并的轮次数。
- **--generation_chunk_buffer_size (int, 默认: None)**: 生成时上下文保留的最近音频段数（除参考音频外），提升长文连贯性；建议 2–4。
- **--seed (int, 默认: None)**: 随机种子（固定可复现）。
- **--device_id (int, 默认: None)**: 指定 CUDA 设备号（隐含使用 GPU）。
- **--device (auto|cuda|mps|none, 默认: auto)**: 设备选择：auto 优先 CUDA→MPS→CPU；`none` 为 CPU。指定 `device_id` 时强制 CUDA。
- **--use_static_kv_cache (int, 默认: 1)**: 开启静态 KV cache 与 CUDA Graphs 加速（仅 GPU；MPS 自动禁用）。
- **--out_path (str, 默认: `generation.wav`)**: 输出 wav 路径（采样率 24kHz）。

### 小贴士

- 文本会进行基础规范化：中英标点统一、去多余空白、替换若干标签（如 `[laugh] → <SE>[Laughter]</SE>`）。
- 多说话人：脚本会读取文本中的 `[SPEAKER0]`、`[SPEAKER1]` 标签；若未显式给参考音色且存在 2+ 说话人，会在系统消息中交替分派“feminine/masculine”提示。
- 稳定性 vs 多样性：追求稳定可用 `--temperature 0.3 --ras_win_len 0`；追求多样可提高 temperature/top_p。

## Single-speaker Audio Generation

### Voice clone

```bash
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--ref_audio broom_salesman \
--seed 12345 \
--out_path generation.wav
```

The model will read the transcript with the same voice as in the [reference audio](./voice_prompts/broom_salesman.wav). The technique is also called shallow voice clone.

We have some example audio prompts stored in [voice_prompts](./voice_prompts/). Feel free to pick one in the folder and try out the model. Here's another example that uses the voice of `belinda`. You can also add new own favorite voice in the folder and clone the voice.

```bash
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--ref_audio belinda \
--seed 12345 \
--out_path generation.wav
```

#### (Experimental) Cross-lingual voice clone

This example demonstrates voice cloning with a Chinese prompt, where the synthesized speech is in English.

```bash
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--scene_prompt empty \
--ref_audio zh_man_sichuan \
--temperature 0.3 \
--seed 12345 \
--out_path generation.wav
```

### Smart voice

The model supports reading the transcript with a random voice.

```bash
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--seed 12345 \
--out_path generation.wav
```

It also works for other languages like Chinese.

```bash
python3 generation.py \
--transcript transcript/single_speaker/zh_ai.txt \
--seed 12345 \
--out_path generation.wav
```

### Describe speaker characteristics with text

The model allows you to describe the speaker via text. See [voice_prompts/profile.yaml](voice_prompts/profile.yaml) for examples. You can run the following two examples that try to specify male / female British accent for the speakers. Also, try to remove the `--seed 12345` flag to see how the model is generating different voices.

```bash
# Male British Accent
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--ref_audio profile:male_en_british \
--seed 12345 \
--out_path generation.wav

# Female British Accent
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--ref_audio profile:female_en_british \
--seed 12345 \
--out_path generation.wav
```

### Chunking for long-form audio generation

To generate long-form audios, you can chunk the text and render each chunk one by one while putting the previous generated audio and the reference audio in the prompt. Here's an example that generates the first five paragraphs of Higgs Audio v1 release blog. See [text](./transcript/single_speaker/en_higgs_audio_blog.md).

```bash
python3 generation.py \
--scene_prompt scene_prompts/reading_blog.txt \
--transcript transcript/single_speaker/en_higgs_audio_blog.md \
--ref_audio en_man \
--chunk_method word \
--temperature 0.3 \
--generation_chunk_buffer_size 2 \
--seed 12345 \
--out_path generation.wav
```

### Experimental and Emergent Capabilities

As shown in our demo, the pretrained model is demonstrating emergent features. We prepared some samples to help you explore these experimental prompts. We will enhance the stability of these experimental prompts in the future version of HiggsAudio.

#### (Experimental) Hum a tune with the cloned voice
The model is able to hum a tune with the cloned voice.

```bash
python3 generation.py \
--transcript transcript/single_speaker/experimental/en_humming.txt \
--ref_audio en_woman \
--ras_win_len 0 \
--seed 12345 \
--out_path generation.wav
```

#### (Experimental) Read the sentence while adding background music (BGM)

```bash
python3 generation.py \
--transcript transcript/single_speaker/experimental/en_bgm.txt \
--ref_audio en_woman \
--ras_win_len 0 \
--ref_audio_in_system_message \
--seed 123456 \
--out_path generation.wav
```

## Multi-speaker Audio Generation


### Smart voice

To get started to explore HiggsAudio's capability in generating multi-speaker audios. Let's try to generate a multi-speaker dialog from transcript in the zero-shot fashion. See the transcript in [transcript/multi_speaker/en_argument.txt](transcript/multi_speaker/en_argument.txt). The speakers are annotated with `[SPEAKER0]` and `[SPEAKER1]`.

```bash
python3 generation.py \
--transcript transcript/multi_speaker/en_argument.txt \
--seed 12345 \
--out_path generation.wav
```

### Multi-voice clone
You can also try to clone the voices from multiple people simultaneously and generate audio about the transcript. Here's an example that puts reference audios in the system message and prompt the model iteratively. You can hear "Belinda" arguing with "Broom Salesman".

```bash
python3 generation.py \
--transcript transcript/multi_speaker/en_argument.txt \
--ref_audio belinda,broom_salesman \
--ref_audio_in_system_message \
--chunk_method speaker \
--seed 12345 \
--out_path generation.wav
```

You can also let "Broom Salesman" talking to "Belinda", who recently trained HiggsAudio.

```bash
python3 generation.py \
--transcript transcript/multi_speaker/en_higgs.txt \
--ref_audio broom_salesman,belinda \
--ref_audio_in_system_message \
--chunk_method speaker \
--chunk_max_num_turns 2 \
--seed 12345 \
--out_path generation.wav
```
