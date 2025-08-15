"""Gradio demo for HiggsAudio

Features:
- Random voice TTS
- Cloned voice TTS (single speaker)

Usage:
  python examples/gradio_app.py
"""

import os
import re
import yaml
import torch
import gradio as gr
import importlib.util
import numpy as np
from typing import Dict, Optional, Tuple, List
from boson_multimodal.data_types import Message, AudioContent


CURR_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_generation_module():
    gen_path = os.path.join(CURR_DIR, "generation.py")
    spec = importlib.util.spec_from_file_location("examples_generation", gen_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to load generation.py spec"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


GENERATION = _load_generation_module()


def _read_text_file(path: str) -> Optional[str]:
    if not path:
        return None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def _list_voice_prompts() -> Tuple[List[str], List[str]]:
    """Return (audio_names, profile_names) without extensions.

    audio_names: available wav names under voice_prompts with matching .txt
    profile_names: `profile:<name>` keys from profile.yaml
    """
    vp_dir = os.path.join(CURR_DIR, "voice_prompts")
    audio_names = []
    profile_names: List[str] = []
    if os.path.isdir(vp_dir):
        wavs = [f for f in os.listdir(vp_dir) if f.endswith(".wav")]
        for w in sorted(wavs):
            name = os.path.splitext(w)[0]
            txt_path = os.path.join(vp_dir, f"{name}.txt")
            if os.path.exists(txt_path):
                audio_names.append(name)
        profile_yaml = os.path.join(vp_dir, "profile.yaml")
        if os.path.exists(profile_yaml):
            try:
                with open(profile_yaml, "r", encoding="utf-8") as f:
                    prof = yaml.safe_load(f)
                for k in sorted(list(prof.get("profiles", {}).keys())):
                    profile_names.append(f"profile:{k}")
            except Exception:
                pass
    return audio_names, profile_names


def _load_profiles_map() -> Dict[str, str]:
    """Return mapping: profile name -> description text from profile.yaml"""
    vp_dir = os.path.join(CURR_DIR, "voice_prompts")
    profile_yaml = os.path.join(vp_dir, "profile.yaml")
    profiles: Dict[str, str] = {}
    if os.path.exists(profile_yaml):
        try:
            with open(profile_yaml, "r", encoding="utf-8") as f:
                prof = yaml.safe_load(f)
            raw = prof.get("profiles", {}) or {}
            for k, v in raw.items():
                if isinstance(v, str):
                    profiles[k] = v
        except Exception:
            pass
    return profiles


def _choose_device(device_choice: str, device_id: Optional[int]) -> Tuple[str, Optional[int]]:
    """Mimic device resolution in examples/generation.py"""
    if device_id is None:
        if device_choice == "auto":
            if torch.cuda.is_available():
                return "cuda:0", 0
            elif torch.backends.mps.is_available():
                return "mps", None
            else:
                return "cpu", None
        elif device_choice == "cuda":
            return "cuda:0", 0
        elif device_choice == "mps":
            return "mps", None
        else:
            return "cpu", None
    else:
        return f"cuda:{device_id}", device_id


class ModelState:
    def __init__(self):
        self.client = None
        self.params: Dict[str, object] = {}

    def get_client(
        self,
        model_path: str,
        audio_tokenizer_path: str,
        max_new_tokens: int,
        device_choice: str,
        device_id: Optional[int],
        use_static_kv_cache: bool,
    ):
        # Resolve device similarly to generation.main
        device, resolved_device_id = _choose_device(device_choice, device_id)
        # Disable static KV cache on MPS
        if device == "mps" and use_static_kv_cache:
            use_static_kv_cache = False

        key = {
            "model_path": model_path,
            "audio_tokenizer_path": audio_tokenizer_path,
            "max_new_tokens": max_new_tokens,
            "device": device,
            "device_id": resolved_device_id,
            "use_static_kv_cache": use_static_kv_cache,
        }
        if key != self.params or self.client is None:
            self.client = GENERATION.HiggsAudioModelClient(
                model_path=model_path,
                audio_tokenizer=audio_tokenizer_path,
                device=device,
                device_id=resolved_device_id,
                max_new_tokens=max_new_tokens,
                use_static_kv_cache=use_static_kv_cache,
            )
            self.params = key
        return self.client


MODEL_STATE = ModelState()


def _preprocess_transcript(text: str) -> str:
    # normalize punctuation
    text = GENERATION.normalize_chinese_punctuation(text)
    # Replace known tags (same mapping as generation.main)
    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        text = text.replace(tag, replacement)
    lines = text.split("\n")
    text = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    text = text.strip()
    # Basic replacements
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
    # Ensure ending punctuation
    if not any([text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        text += "."
    return text


def _generate_internal(
    text: str,
    scene_prompt_text: Optional[str],
    ref_audio: Optional[str],
    sampling: Dict[str, object],
    model_cfg: Dict[str, object],
):
    # Device/config
    client = MODEL_STATE.get_client(
        model_path=model_cfg["model_path"],
        audio_tokenizer_path=model_cfg["audio_tokenizer"],
        max_new_tokens=int(model_cfg["max_new_tokens"]),
        device_choice=model_cfg["device"],
        device_id=model_cfg["device_id"],
        use_static_kv_cache=bool(model_cfg["use_static_kv_cache"]),
    )

    pattern = re.compile(r"\[(SPEAKER\d+)\]")
    speaker_tags = sorted(set(pattern.findall(text)))
    text = _preprocess_transcript(text)

    # Scene prompt
    scene_prompt = scene_prompt_text.strip() if scene_prompt_text else None
    if scene_prompt == "":
        scene_prompt = None

    messages, audio_ids = GENERATION.prepare_generation_context(
        scene_prompt=scene_prompt,
        ref_audio=ref_audio,
        ref_audio_in_system_message=False,
        audio_tokenizer=client._audio_tokenizer,
        speaker_tags=speaker_tags,
    )
    chunked_text = GENERATION.prepare_chunk_text(text, chunk_method=None)

    concat_wv, sr, text_out = client.generate(
        messages=messages,
        audio_ids=audio_ids,
        chunked_text=chunked_text,
        generation_chunk_buffer_size=None,
        temperature=float(sampling["temperature"]),
        top_k=int(sampling["top_k"]),
        top_p=float(sampling["top_p"]),
        ras_win_len=int(sampling["ras_win_len"]),
        ras_win_max_num_repeat=int(sampling["ras_win_max_num_repeat"]),
        seed=None if sampling["seed"] in (None, "") else int(sampling["seed"]),
    )

    # Convert to numpy for Gradio
    if hasattr(concat_wv, "detach"):
        arr = concat_wv.detach().cpu().numpy()
    else:
        arr = np.asarray(concat_wv)
    return (sr, arr), text_out


def ui_app():
    default_text = _read_text_file(os.path.join(CURR_DIR, "transcript", "single_speaker", "en_dl.txt")) or "Hello, this is HiggsAudio."
    audio_names, profile_names = _list_voice_prompts()
    profiles_map = _load_profiles_map()

    with gr.Blocks(title="HiggsAudio Gradio Demo") as demo:
        with gr.Accordion("高级设置（模型 / 设备 / 最大 tokens / 采样参数）", open=False):
            with gr.Row():
                with gr.Column():
                    model_path = gr.Textbox(label="Model Path", value="bosonai/higgs-audio-v2-generation-3B-base")
                    audio_tokenizer = gr.Textbox(label="Audio Tokenizer", value="bosonai/higgs-audio-v2-tokenizer")
                    max_new_tokens = gr.Slider(256, 4096, value=2048, step=64, label="Max New Tokens")
                with gr.Column():
                    device = gr.Dropdown(["auto", "cuda", "mps", "none"], value="auto", label="Device")
                    device_id = gr.Number(value=None, label="CUDA Device ID (optional)")
                    use_static_kv_cache = gr.Checkbox(value=True, label="Use Static KV Cache (GPU only)")
            with gr.Row():
                temperature_g = gr.Slider(0.1, 1.5, value=1.0, step=0.05, label="Temperature")
                top_p_g = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-p")
                top_k_g = gr.Slider(1, 100, value=50, step=1, label="Top-k")
            with gr.Row():
                ras_win_len_g = gr.Slider(0, 16, value=7, step=1, label="RAS Window Length (0 disables)")
                ras_win_max_num_repeat_g = gr.Slider(1, 5, value=2, step=1, label="RAS Max Repeat")
                # 移除全局 Seed，改为在各模式 TAB 内单独提供

        with gr.Tabs():
            with gr.Tab("随机音色朗读"):
                txt_input = gr.Textbox(label="Transcript", value=default_text, lines=3)
                scene_prompt = gr.Textbox(label="Scene Prompt (optional)", value="Audio is recorded from a quiet room.")

                with gr.Row():
                    seed_random = gr.Number(value=None, label="Seed (optional)")
                    generate_btn = gr.Button("Generate")
                audio_out = gr.Audio(label="Output Audio", type="numpy")
                text_out = gr.Textbox(label="Generated Text")

                def on_generate_random(transcript, scene, temperature, top_k, top_p, ras_win_len, ras_win_max_num_repeat, seed,
                                       model_path, audio_tokenizer, max_new_tokens, device, device_id, use_static_kv_cache):
                    sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p, ras_win_len=ras_win_len,
                                    ras_win_max_num_repeat=ras_win_max_num_repeat, seed=seed)
                    model_cfg = dict(model_path=model_path, audio_tokenizer=audio_tokenizer, max_new_tokens=int(max_new_tokens),
                                     device=device, device_id=None if device_id in (None, "") else int(device_id),
                                     use_static_kv_cache=bool(use_static_kv_cache))
                    return _generate_internal(transcript, scene, None, sampling, model_cfg)

                generate_btn.click(
                    on_generate_random,
                    inputs=[txt_input, scene_prompt, temperature_g, top_k_g, top_p_g, ras_win_len_g, ras_win_max_num_repeat_g, seed_random,
                            model_path, audio_tokenizer, max_new_tokens, device, device_id, use_static_kv_cache],
                    outputs=[audio_out, text_out],
                )

            with gr.Tab("克隆音色朗读"):
                with gr.Tabs():
                    # Profile 描述模式（不上传音频）
                    with gr.Tab("角色描述模式"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=5):
                                txt_input_profile = gr.Textbox(label="Transcript", lines=1)
                                # Scene Prompt 单独一行
                                scene_prompt_profile = gr.Textbox(label="Scene Prompt (optional)", value="Audio is recorded from a quiet room.", lines=1)
                                # Profile 选择与描述同一行
                                with gr.Row():
                                    profile_select = gr.Dropdown(
                                        choices=sorted(list(profiles_map.keys())),
                                        value=None,
                                        label="提示音色选择",
                                        visible=len(profiles_map) > 0,
                                    )
                                    profile_desc = gr.Textbox(label="说话人描述", value="", lines=1)
                                # Seed 单独一行
                                seed_profile = gr.Number(value="12345", label="Seed (optional)")
                                # 生成按钮单独一行且位于最下方
                                generate_btn_profile = gr.Button("Generate (Profile Mode)")
                            with gr.Column(scale=2):
                                audio_out_profile = gr.Audio(label="Output Audio", type="numpy")
                                text_out_profile = gr.Textbox(label="Generated Text", lines=2)

                        def on_profile_selected(key: Optional[str]):
                            if key and key in profiles_map:
                                return gr.update(value=profiles_map[key])
                            return gr.update()

                        profile_select.change(on_profile_selected, inputs=[profile_select], outputs=[profile_desc])

                        def on_generate_clone_profile(transcript, scene, profile_text,
                                                      temperature, top_k, top_p, ras_win_len, ras_win_max_num_repeat, seed,
                                                      model_path, audio_tokenizer, max_new_tokens, device, device_id, use_static_kv_cache):
                            sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p, ras_win_len=ras_win_len,
                                            ras_win_max_num_repeat=ras_win_max_num_repeat, seed=seed)
                            model_cfg = dict(model_path=model_path, audio_tokenizer=audio_tokenizer, max_new_tokens=int(max_new_tokens),
                                             device=device, device_id=None if device_id in (None, "") else int(device_id),
                                             use_static_kv_cache=bool(use_static_kv_cache))

                            text = transcript or ""
                            text = _preprocess_transcript(text)
                            scene_text = (scene or "").strip() or None

                            system_parts = ["Generate audio following instruction."]
                            scene_block_lines = []
                            if scene_text:
                                scene_block_lines.append(scene_text)
                            if profile_text and str(profile_text).strip():
                                scene_block_lines.append(f"SPEAKER0: {str(profile_text).strip()}")
                            if scene_block_lines:
                                system_parts.append("<|scene_desc_start|>\n" + "\n\n".join(scene_block_lines) + "\n<|scene_desc_end|>")
                            messages: List[Message] = []
                            if system_parts:
                                messages.append(Message(role="system", content="\n\n".join(system_parts)))

                            audio_ids: List[List[int]] = []
                            client = MODEL_STATE.get_client(
                                model_path=model_cfg["model_path"],
                                audio_tokenizer_path=model_cfg["audio_tokenizer"],
                                max_new_tokens=int(model_cfg["max_new_tokens"]),
                                device_choice=model_cfg["device"],
                                device_id=model_cfg["device_id"],
                                use_static_kv_cache=bool(model_cfg["use_static_kv_cache"]),
                            )

                            chunked_text = GENERATION.prepare_chunk_text(text, chunk_method=None)
                            concat_wv, sr, text_out = client.generate(
                                messages=messages,
                                audio_ids=audio_ids,
                                chunked_text=chunked_text,
                                generation_chunk_buffer_size=None,
                                temperature=float(sampling["temperature"]),
                                top_k=int(sampling["top_k"]),
                                top_p=float(sampling["top_p"]),
                                ras_win_len=int(sampling["ras_win_len"]),
                                ras_win_max_num_repeat=int(sampling["ras_win_max_num_repeat"]),
                                seed=None if sampling["seed"] in (None, "") else int(sampling["seed"]),
                            )
                            if hasattr(concat_wv, "detach"):
                                arr = concat_wv.detach().cpu().numpy()
                            else:
                                import numpy as _np
                                arr = _np.asarray(concat_wv)
                            return (sr, arr), text_out

                        generate_btn_profile.click(
                            on_generate_clone_profile,
                            inputs=[txt_input_profile, scene_prompt_profile, profile_desc,
                                    temperature_g, top_k_g, top_p_g, ras_win_len_g, ras_win_max_num_repeat_g, seed_profile,
                                    model_path, audio_tokenizer, max_new_tokens, device, device_id, use_static_kv_cache],
                            outputs=[audio_out_profile, text_out_profile],
                        )

                    # 上传参考音频模式（不使用 Profile 描述）
                    with gr.Tab("参考音频"):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=5):
                                txt_input_audio = gr.Textbox(label="Transcript", lines=1)
                                # Scene Prompt 单独一行
                                scene_prompt_audio = gr.Textbox(label="Scene Prompt (optional)", value="Audio is recorded from a quiet room.", lines=1)
                                with gr.Row():
                                    ref_audio_file = gr.Audio(sources=["upload"], type="filepath", label="Reference Audio", elem_id="clone_ref_audio_file")
                                    ref_audio_text = gr.Textbox(label="Reference Audio Transcript", value="", lines=1)
                                # Seed 单独一行
                                seed_audio = gr.Number(value=None, label="Seed (optional)")
                                # 生成按钮单独一行且位于最下方
                                generate_btn_audio = gr.Button("Generate (Audio Mode)")
                            with gr.Column(scale=2):
                                audio_out_audio = gr.Audio(label="Output Audio", type="numpy")
                                text_out_audio = gr.Textbox(label="Generated Text", lines=2)

                        def on_generate_clone_audio(transcript, scene, ref_audio_path, ref_audio_transcript,
                                                     temperature, top_k, top_p, ras_win_len, ras_win_max_num_repeat, seed,
                                                     model_path, audio_tokenizer, max_new_tokens, device, device_id, use_static_kv_cache):
                            sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p, ras_win_len=ras_win_len,
                                            ras_win_max_num_repeat=ras_win_max_num_repeat, seed=seed)
                            model_cfg = dict(model_path=model_path, audio_tokenizer=audio_tokenizer, max_new_tokens=int(max_new_tokens),
                                             device=device, device_id=None if device_id in (None, "") else int(device_id),
                                             use_static_kv_cache=bool(use_static_kv_cache))
                            text = transcript or ""
                            text = _preprocess_transcript(text)
                            scene_text = (scene or "").strip() or None

                            system_parts = ["Generate audio following instruction."]
                            if scene_text:
                                system_parts.append("<|scene_desc_start|>\n" + scene_text + "\n<|scene_desc_end|>")
                            messages: List[Message] = []
                            if system_parts:
                                messages.append(Message(role="system", content="\n\n".join(system_parts)))

                            audio_ids = []
                            if ref_audio_path in (None, ""):
                                raise gr.Error("请上传提示音频")
                            if not ref_audio_transcript or str(ref_audio_transcript).strip() == "":
                                raise gr.Error("请提供提示音频的文本内容")

                            client = MODEL_STATE.get_client(
                                model_path=model_cfg["model_path"],
                                audio_tokenizer_path=model_cfg["audio_tokenizer"],
                                max_new_tokens=int(model_cfg["max_new_tokens"]),
                                device_choice=model_cfg["device"],
                                device_id=model_cfg["device_id"],
                                use_static_kv_cache=bool(model_cfg["use_static_kv_cache"]),
                            )
                            ref_tokens = client._audio_tokenizer.encode(ref_audio_path)
                            audio_ids.append(ref_tokens)
                            messages.append(Message(role="user", content=str(ref_audio_transcript).strip()))
                            messages.append(Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)))

                            chunked_text = GENERATION.prepare_chunk_text(text, chunk_method=None)
                            concat_wv, sr, text_out = client.generate(
                                messages=messages,
                                audio_ids=audio_ids,
                                chunked_text=chunked_text,
                                generation_chunk_buffer_size=None,
                                temperature=float(sampling["temperature"]),
                                top_k=int(sampling["top_k"]),
                                top_p=float(sampling["top_p"]),
                                ras_win_len=int(sampling["ras_win_len"]),
                                ras_win_max_num_repeat=int(sampling["ras_win_max_num_repeat"]),
                                seed=None if sampling["seed"] in (None, "") else int(sampling["seed"]),
                            )
                            if hasattr(concat_wv, "detach"):
                                arr = concat_wv.detach().cpu().numpy()
                            else:
                                import numpy as _np
                                arr = _np.asarray(concat_wv)
                            return (sr, arr), text_out

                        generate_btn_audio.click(
                            on_generate_clone_audio,
                            inputs=[txt_input_audio, scene_prompt_audio, ref_audio_file, ref_audio_text,
                                    temperature_g, top_k_g, top_p_g, ras_win_len_g, ras_win_max_num_repeat_g, seed_audio,
                                    model_path, audio_tokenizer, max_new_tokens, device, device_id, use_static_kv_cache],
                            outputs=[audio_out_audio, text_out_audio],
                        )

            # 长格式音频的分块功能模块
            with gr.Tab("长文本/分块生成"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5):
                        txt_input_long = gr.Textbox(
                            label="Transcript",
                            value=_read_text_file(os.path.join(CURR_DIR, "transcript", "single_speaker", "en_higgs_audio_blog.md"))
                            or default_text,
                            lines=8,
                        )
                        # Scene Prompt 单独一行
                        scene_prompt_long = gr.Textbox(
                            label="Scene Prompt (optional)",
                            value=_read_text_file(os.path.join(CURR_DIR, "scene_prompts", "reading_blog.txt")) or "",
                            lines=1,
                        )
                        with gr.Row():
                            ref_audio_long = gr.Dropdown(
                                choices=[None] + audio_names + profile_names,
                                value=None,
                                label="参考音色 (可选)",
                            )
                            ref_in_sys_long = gr.Checkbox(value=False, label="将参考音色放入系统消息")
                        with gr.Row():
                            chunk_method_long = gr.Dropdown(
                                choices=[None, "speaker", "word"], value="word", label="分块方式"
                            )
                            chunk_max_word_num_long = gr.Number(value=200, label="每段最大词/字数 (word)")
                            chunk_max_num_turns_long = gr.Number(value=1, label="每段合并轮次 (speaker)")
                        generation_chunk_buffer_long = gr.Number(value=2, label="上下文保留生成段数 (可选)")
                        # Seed 单独一行
                        seed_long = gr.Number(value=12345, label="Seed (optional)")
                        # 生成按钮单独一行且位于最下方
                        generate_btn_long = gr.Button("Generate (Long-form)")
                    with gr.Column(scale=2):
                        audio_out_long = gr.Audio(label="Output Audio", type="numpy")
                        text_out_long = gr.Textbox(label="Generated Text", lines=4)

                def on_generate_longform(
                    transcript,
                    scene,
                    ref_audio_sel,
                    ref_in_sys,
                    chunk_method,
                    chunk_max_word_num,
                    chunk_max_num_turns,
                    generation_chunk_buffer,
                    temperature,
                    top_k,
                    top_p,
                    ras_win_len,
                    ras_win_max_num_repeat,
                    seed,
                    model_path,
                    audio_tokenizer,
                    max_new_tokens,
                    device,
                    device_id,
                    use_static_kv_cache,
                ):
                    sampling = dict(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        ras_win_len=ras_win_len,
                        ras_win_max_num_repeat=ras_win_max_num_repeat,
                        seed=seed,
                    )
                    model_cfg = dict(
                        model_path=model_path,
                        audio_tokenizer=audio_tokenizer,
                        max_new_tokens=int(max_new_tokens),
                        device=device,
                        device_id=None if device_id in (None, "") else int(device_id),
                        use_static_kv_cache=bool(use_static_kv_cache),
                    )

                    text = transcript or ""
                    # 提取说话人标签并预处理文本
                    pattern = re.compile(r"\[(SPEAKER\d+)\]")
                    speaker_tags = sorted(set(pattern.findall(text)))
                    text = _preprocess_transcript(text)
                    scene_text = (scene or "").strip() or None

                    client = MODEL_STATE.get_client(
                        model_path=model_cfg["model_path"],
                        audio_tokenizer_path=model_cfg["audio_tokenizer"],
                        max_new_tokens=int(model_cfg["max_new_tokens"]),
                        device_choice=model_cfg["device"],
                        device_id=model_cfg["device_id"],
                        use_static_kv_cache=bool(model_cfg["use_static_kv_cache"]),
                    )

                    messages, audio_ids = GENERATION.prepare_generation_context(
                        scene_prompt=scene_text,
                        ref_audio=ref_audio_sel if ref_audio_sel not in (None, "") else None,
                        ref_audio_in_system_message=bool(ref_in_sys),
                        audio_tokenizer=client._audio_tokenizer,
                        speaker_tags=speaker_tags,
                    )

                    chunked_text = GENERATION.prepare_chunk_text(
                        text,
                        chunk_method=None if chunk_method in (None, "None", "") else chunk_method,
                        chunk_max_word_num=int(chunk_max_word_num) if chunk_max_word_num not in (None, "") else 200,
                        chunk_max_num_turns=int(chunk_max_num_turns) if chunk_max_num_turns not in (None, "") else 1,
                    )

                    gen_buf = (
                        None if generation_chunk_buffer in (None, "") else int(generation_chunk_buffer)
                    )

                    concat_wv, sr, text_out = client.generate(
                        messages=messages,
                        audio_ids=audio_ids,
                        chunked_text=chunked_text,
                        generation_chunk_buffer_size=gen_buf,
                        temperature=float(sampling["temperature"]),
                        top_k=int(sampling["top_k"]),
                        top_p=float(sampling["top_p"]),
                        ras_win_len=int(sampling["ras_win_len"]),
                        ras_win_max_num_repeat=int(sampling["ras_win_max_num_repeat"]),
                        seed=None if sampling["seed"] in (None, "") else int(sampling["seed"]),
                    )
                    if hasattr(concat_wv, "detach"):
                        arr = concat_wv.detach().cpu().numpy()
                    else:
                        import numpy as _np
                        arr = _np.asarray(concat_wv)
                    return (sr, arr), text_out

                generate_btn_long.click(
                    on_generate_longform,
                    inputs=[
                        txt_input_long,
                        scene_prompt_long,
                        ref_audio_long,
                        ref_in_sys_long,
                        chunk_method_long,
                        chunk_max_word_num_long,
                        chunk_max_num_turns_long,
                        generation_chunk_buffer_long,
                        temperature_g,
                        top_k_g,
                        top_p_g,
                        ras_win_len_g,
                        ras_win_max_num_repeat_g,
                        seed_long,
                        model_path,
                        audio_tokenizer,
                        max_new_tokens,
                        device,
                        device_id,
                        use_static_kv_cache,
                    ],
                    outputs=[audio_out_long, text_out_long],
                )

            # 用克隆的声音哼一首曲子模块（实验）
            with gr.Tab("哼一首曲子（克隆音色）"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5):
                        txt_input_hum = gr.Textbox(
                            label="Transcript",
                            value=_read_text_file(os.path.join(CURR_DIR, "transcript", "single_speaker", "experimental", "en_humming.txt"))
                            or "Hum a tune with the cloned voice.",
                            lines=3,
                        )
                        # Scene Prompt 单独一行
                        scene_prompt_hum = gr.Textbox(label="Scene Prompt (optional)", value="", lines=1)
                        with gr.Row():
                            ref_audio_hum = gr.Dropdown(
                                choices=audio_names,
                                value="en_woman" if "en_woman" in audio_names else None,
                                label="参考音色",
                            )
                            ras_win_len_hum = gr.Number(value=0, label="RAS Window Length (本模块默认 0)")
                        # Seed 单独一行
                        seed_hum = gr.Number(value=12345, label="Seed (optional)")
                        # 生成按钮单独一行且位于最下方
                        generate_btn_hum = gr.Button("Generate (Humming)")
                    with gr.Column(scale=2):
                        audio_out_hum = gr.Audio(label="Output Audio", type="numpy")
                        text_out_hum = gr.Textbox(label="Generated Text", lines=2)

                def on_generate_hum(
                    transcript,
                    scene,
                    ref_audio_sel,
                    ras_win_len_local,
                    temperature,
                    top_k,
                    top_p,
                    ras_win_max_num_repeat,
                    seed,
                    model_path,
                    audio_tokenizer,
                    max_new_tokens,
                    device,
                    device_id,
                    use_static_kv_cache,
                ):
                    sampling = dict(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        ras_win_len=ras_win_len_local,
                        ras_win_max_num_repeat=ras_win_max_num_repeat,
                        seed=seed,
                    )
                    model_cfg = dict(
                        model_path=model_path,
                        audio_tokenizer=audio_tokenizer,
                        max_new_tokens=int(max_new_tokens),
                        device=device,
                        device_id=None if device_id in (None, "") else int(device_id),
                        use_static_kv_cache=bool(use_static_kv_cache),
                    )

                    text = transcript or ""
                    text = _preprocess_transcript(text)
                    scene_text = (scene or "").strip() or None

                    client = MODEL_STATE.get_client(
                        model_path=model_cfg["model_path"],
                        audio_tokenizer_path=model_cfg["audio_tokenizer"],
                        max_new_tokens=int(model_cfg["max_new_tokens"]),
                        device_choice=model_cfg["device"],
                        device_id=model_cfg["device_id"],
                        use_static_kv_cache=bool(model_cfg["use_static_kv_cache"]),
                    )
                    # 单说话人场景，无需显式 speaker 标签
                    messages, audio_ids = GENERATION.prepare_generation_context(
                        scene_prompt=scene_text,
                        ref_audio=ref_audio_sel,
                        ref_audio_in_system_message=False,
                        audio_tokenizer=client._audio_tokenizer,
                        speaker_tags=[],
                    )
                    chunked_text = GENERATION.prepare_chunk_text(text, chunk_method=None)
                    concat_wv, sr, text_out = client.generate(
                        messages=messages,
                        audio_ids=audio_ids,
                        chunked_text=chunked_text,
                        generation_chunk_buffer_size=None,
                        temperature=float(sampling["temperature"]),
                        top_k=int(sampling["top_k"]),
                        top_p=float(sampling["top_p"]),
                        ras_win_len=int(sampling["ras_win_len"]),
                        ras_win_max_num_repeat=int(sampling["ras_win_max_num_repeat"]),
                        seed=None if sampling["seed"] in (None, "") else int(sampling["seed"]),
                    )
                    if hasattr(concat_wv, "detach"):
                        arr = concat_wv.detach().cpu().numpy()
                    else:
                        import numpy as _np
                        arr = _np.asarray(concat_wv)
                    return (sr, arr), text_out

                generate_btn_hum.click(
                    on_generate_hum,
                    inputs=[
                        txt_input_hum,
                        scene_prompt_hum,
                        ref_audio_hum,
                        ras_win_len_hum,
                        temperature_g,
                        top_k_g,
                        top_p_g,
                        ras_win_max_num_repeat_g,
                        seed_hum,
                        model_path,
                        audio_tokenizer,
                        max_new_tokens,
                        device,
                        device_id,
                        use_static_kv_cache,
                    ],
                    outputs=[audio_out_hum, text_out_hum],
                )

            # 阅读句子时添加背景音乐模块（实验）
            with gr.Tab("阅读 + 背景音乐（实验）"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5):
                        txt_input_bgm = gr.Textbox(
                            label="Transcript",
                            value=_read_text_file(os.path.join(CURR_DIR, "transcript", "single_speaker", "experimental", "en_bgm.txt"))
                            or "Read the sentence while adding background music.",
                            lines=3,
                        )
                        # Scene Prompt 单独一行
                        scene_prompt_bgm = gr.Textbox(label="Scene Prompt (optional)", value="", lines=1)
                        with gr.Row():
                            ref_audio_bgm = gr.Dropdown(
                                choices=audio_names,
                                value="en_woman" if "en_woman" in audio_names else None,
                                label="参考音色",
                            )
                            ref_in_sys_bgm = gr.Checkbox(value=True, label="将参考音色放入系统消息 (默认开启)")
                        # RAS 长度本模块默认 0
                        ras_win_len_bgm = gr.Number(value=0, label="RAS Window Length (本模块默认 0)")
                        # Seed 单独一行
                        seed_bgm = gr.Number(value=123456, label="Seed (optional)")
                        # 生成按钮单独一行且位于最下方
                        generate_btn_bgm = gr.Button("Generate (BGM)")
                    with gr.Column(scale=2):
                        audio_out_bgm = gr.Audio(label="Output Audio", type="numpy")
                        text_out_bgm = gr.Textbox(label="Generated Text", lines=2)

                def on_generate_bgm(
                    transcript,
                    scene,
                    ref_audio_sel,
                    ref_in_sys,
                    ras_win_len_local,
                    temperature,
                    top_k,
                    top_p,
                    ras_win_max_num_repeat,
                    seed,
                    model_path,
                    audio_tokenizer,
                    max_new_tokens,
                    device,
                    device_id,
                    use_static_kv_cache,
                ):
                    sampling = dict(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        ras_win_len=ras_win_len_local,
                        ras_win_max_num_repeat=ras_win_max_num_repeat,
                        seed=seed,
                    )
                    model_cfg = dict(
                        model_path=model_path,
                        audio_tokenizer=audio_tokenizer,
                        max_new_tokens=int(max_new_tokens),
                        device=device,
                        device_id=None if device_id in (None, "") else int(device_id),
                        use_static_kv_cache=bool(use_static_kv_cache),
                    )

                    text = transcript or ""
                    text = _preprocess_transcript(text)
                    scene_text = (scene or "").strip() or None

                    client = MODEL_STATE.get_client(
                        model_path=model_cfg["model_path"],
                        audio_tokenizer_path=model_cfg["audio_tokenizer"],
                        max_new_tokens=int(model_cfg["max_new_tokens"]),
                        device_choice=model_cfg["device"],
                        device_id=model_cfg["device_id"],
                        use_static_kv_cache=bool(model_cfg["use_static_kv_cache"]),
                    )
                    messages, audio_ids = GENERATION.prepare_generation_context(
                        scene_prompt=scene_text,
                        ref_audio=ref_audio_sel,
                        ref_audio_in_system_message=bool(ref_in_sys),
                        audio_tokenizer=client._audio_tokenizer,
                        speaker_tags=[],
                    )
                    chunked_text = GENERATION.prepare_chunk_text(text, chunk_method=None)
                    concat_wv, sr, text_out = client.generate(
                        messages=messages,
                        audio_ids=audio_ids,
                        chunked_text=chunked_text,
                        generation_chunk_buffer_size=None,
                        temperature=float(sampling["temperature"]),
                        top_k=int(sampling["top_k"]),
                        top_p=float(sampling["top_p"]),
                        ras_win_len=int(sampling["ras_win_len"]),
                        ras_win_max_num_repeat=int(sampling["ras_win_max_num_repeat"]),
                        seed=None if sampling["seed"] in (None, "") else int(sampling["seed"]),
                    )
                    if hasattr(concat_wv, "detach"):
                        arr = concat_wv.detach().cpu().numpy()
                    else:
                        import numpy as _np
                        arr = _np.asarray(concat_wv)
                    return (sr, arr), text_out

                generate_btn_bgm.click(
                    on_generate_bgm,
                    inputs=[
                        txt_input_bgm,
                        scene_prompt_bgm,
                        ref_audio_bgm,
                        ref_in_sys_bgm,
                        ras_win_len_bgm,
                        temperature_g,
                        top_k_g,
                        top_p_g,
                        ras_win_max_num_repeat_g,
                        seed_bgm,
                        model_path,
                        audio_tokenizer,
                        max_new_tokens,
                        device,
                        device_id,
                        use_static_kv_cache,
                    ],
                    outputs=[audio_out_bgm, text_out_bgm],
                )

    return demo


if __name__ == "__main__":
    app = ui_app()
    app.launch()


