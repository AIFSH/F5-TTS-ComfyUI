import os,sys
import os.path as osp
now_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(now_dir)
from f5_model import CFM, UNetT, DiT
from f5_model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
    save_spectrogram,
)
from LangSegment import LangSegment
from zh_normalization import text_normalize
import re
import torch
import tempfile
import shutil
import torchaudio
from transformers import pipeline
from pydub import AudioSegment,silence
import folder_paths
from tqdm import tqdm
from einops import rearrange
from vocos import Vocos
import numpy as np
from PIL import Image
import soundfile as sf
from comfy.utils import ProgressBar
from huggingface_hub import  snapshot_download
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")
pretrained_dir = osp.join(aifsh_dir,"F5-TTS")
openai_dir = osp.join(aifsh_dir,"whisper-large-v3-turbo")
LangSegment.setfilters(["zh", "en"])
SPLIT_WORDS = [
    "but", "however", "nevertheless", "yet", "still",
    "therefore", "thus", "hence", "consequently",
    "moreover", "furthermore", "additionally",
    "meanwhile", "alternatively", "otherwise",
    "namely", "specifically", "for example", "such as",
    "in fact", "indeed", "notably",
    "in contrast", "on the other hand", "conversely",
    "in conclusion", "to summarize", "finally"
]
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
tmp_dir = osp.join(now_dir, "tmp")

target_sample_rate = 24000
n_mel_channels = 100
target_rms = 0.1
hop_length = 256
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0

class F5TTSNode:
    def __init__(self):
        if not osp.exists(osp.join(pretrained_dir,"F5TTS_Base/model_1200000.safetensors")):
            snapshot_download(repo_id="SWivid/F5-TTS",
                            local_dir=pretrained_dir,
                            allow_patterns=["*.safetensors"])
        if not osp.exists(osp.join(openai_dir,"model.safetensors")):
            snapshot_download(repo_id="openai/whisper-large-v3-turbo",
                              local_dir=openai_dir)
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "gen_text":("TEXT",),
                "ref_audio":("AUDIO",),
                "model_choice":(["F5-TTS", "E2-TTS"],),
                "speed":("FLOAT",{
                    "default":1.0,
                    "min":0.5,
                    "max":2.0,
                    "step":0.05,
                    "round":0.001,
                    "display":"slider"
                }),
                "remove_silence":("BOOLEAN",{
                    "default": True,
                    "tooltip":"The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time."
                }),
                "split_words":("STRING",{
                    "default":",".join(SPLIT_WORDS),
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip":"Enter custom words to split on, separated by commas. Leave blank to use default list.",
                })
            },
            "optional":{
                "ref_text":("TEXT",)
            }
        }
    
    RETURN_TYPES = ("AUDIO","IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_audio"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_F5-TTS"

    def gen_audio(self,gen_text,ref_audio,model_choice,speed,
                  remove_silence,split_words,ref_text=None):
        os.makedirs(tmp_dir,exist_ok=True)
        if not split_words.strip():
            custom_words = [word.strip() for word in split_words.split(',')]
            global SPLIT_WORDS
            SPLIT_WORDS = custom_words
        
        print(gen_text)
        
        print("Converting audio...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav",dir=tmp_dir) as f:
            ref_audio_orig = osp.join(tmp_dir,"tmp_ref_audio.wav")
            waveform = ref_audio["waveform"].squeeze(0)

            torchaudio.save(ref_audio_orig,waveform,ref_audio["sample_rate"])
            aseg = AudioSegment.from_file(ref_audio_orig)
            # os.remove(ref_audio_orig)

            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave

            audio_duration = len(aseg)
            if audio_duration > 15000:
                print("Audio is over 15s, clipping to only first 15s.")
                aseg = aseg[:15000]
            aseg.export(f.name, format="wav")
            ref_audio = f.name
        
        if ref_text is None:
            print("No reference text provided, transcribing reference audio...")
            pipe = pipeline(
                "automatic-speech-recognition",
                model=openai_dir,
                torch_dtype=torch.float16,
                device=device,
            )
            ref_text = pipe(
                ref_audio,
                chunk_length_s=30,
                batch_size=128,
                generate_kwargs={"task": "transcribe"},
                return_timestamps=False,
            )["text"].strip()
            print("Finished transcription")
        else:
           print("Using custom reference text...")
        
        # Split the input text into batches
        if len(ref_text.encode('utf-8')) == len(ref_text) and len(gen_text.encode('utf-8')) == len(gen_text):
            max_chars = 400-len(ref_text.encode('utf-8'))
        else:
            max_chars = 300-len(ref_text.encode('utf-8'))
        gen_text_batches = split_text_into_batches(gen_text, max_chars=max_chars)
        print('ref_text', ref_text)
        gen_text_batches = text_list_normalize(gen_text_batches)
        for i, gen_text in enumerate(gen_text_batches):
            print(f'gen_text {i}', gen_text)
        print(f"Generating audio using {model_choice} in {len(gen_text_batches)} batches")
        (target_sr, waveform), img_path= infer_batch(ref_audio, ref_text, gen_text_batches, model_choice, remove_silence,speed)
        # print(waveform.shape)
        res_audio = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0),
            "sample_rate": target_sr
        }
        res_img = torch.from_numpy(np.array(Image.open(img_path))/255.0).unsqueeze(0)
        # print(res_img.shape)
        shutil.rmtree(tmp_dir)
        return (res_audio, res_img,)

NODE_CLASS_MAPPINGS = {
    "F5TTSNode": F5TTSNode
}

def text_list_normalize(texts):
    text_list = []
    for text in texts:
        for tmp in LangSegment.getTexts(text):
            normalize = text_normalize(tmp.get("text"))
            if normalize != "" and tmp.get("lang") == "en" and normalize not in ["."]:
                if len(text_list) > 0:
                    text_list[-1] += normalize
                else:
                    text_list.append(normalize)
            elif tmp.get("lang") == "zh":
                text_list.append(normalize)
            else:
                if len(text_list) > 0:
                    text_list[-1] += tmp.get("text")
                else:
                    text_list.append(tmp.get("text"))
    return text_list

def load_model(exp_name, model_cls, model_cfg, ckpt_step):

    ckpt_path = osp.join(pretrained_dir,f"{exp_name}/model_{ckpt_step}.safetensors")
    # ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema = True)

    return model



def infer_batch(ref_audio, ref_text, gen_text_batches, exp_name, remove_silence, speed):
    
    if exp_name == "F5-TTS":
        F5TTS_model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        ema_model = load_model(
            "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000
        )
    elif exp_name == "E2-TTS":
        E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        ema_model = load_model(
            "E2TTS_Base", UNetT, E2TTS_model_cfg, 1200000
        )

    audio, sr = torchaudio.load(ref_audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []
    comfybar = ProgressBar(len(gen_text_batches))
    for i, gen_text in enumerate(tqdm(gen_text_batches)):
        # Prepare the text
        if len(ref_text[-1].encode('utf-8')) == 1:
            ref_text = ref_text + " "
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        # Calculate duration
        ref_audio_len = audio.shape[-1] // hop_length
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            generated, _ = ema_model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
        
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()
        
        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())
        comfybar.update(1)

    # Combine all generated waves
    final_wave = np.concatenate(generated_waves)

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav",dir=tmp_dir) as f:
            sf.write(f.name, final_wave, target_sample_rate)
            aseg = AudioSegment.from_file(f.name)
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
            aseg.export(f.name, format="wav")
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False,dir=tmp_dir) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (target_sample_rate, final_wave), spectrogram_path


def split_text_into_batches(text, max_chars=200, split_words=SPLIT_WORDS):
    if len(text.encode('utf-8')) <= max_chars:
        return [text]
    if text[-1] not in ['。', '.', '!', '！', '?', '？']:
        text += '.'
        
    sentences = re.split('([。.!?！？])', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    
    batches = []
    current_batch = ""
    
    def split_by_words(text):
        words = text.split()
        current_word_part = ""
        word_batches = []
        for word in words:
            if len(current_word_part.encode('utf-8')) + len(word.encode('utf-8')) + 1 <= max_chars:
                current_word_part += word + ' '
            else:
                if current_word_part:
                    # Try to find a suitable split word
                    for split_word in split_words:
                        split_index = current_word_part.rfind(' ' + split_word + ' ')
                        if split_index != -1:
                            word_batches.append(current_word_part[:split_index].strip())
                            current_word_part = current_word_part[split_index:].strip() + ' '
                            break
                    else:
                        # If no suitable split word found, just append the current part
                        word_batches.append(current_word_part.strip())
                        current_word_part = ""
                current_word_part += word + ' '
        if current_word_part:
            word_batches.append(current_word_part.strip())
        return word_batches

    for sentence in sentences:
        if len(current_batch.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_batch += sentence
        else:
            # If adding this sentence would exceed the limit
            if current_batch:
                batches.append(current_batch)
                current_batch = ""
            
            # If the sentence itself is longer than max_chars, split it
            if len(sentence.encode('utf-8')) > max_chars:
                # First, try to split by colon
                colon_parts = sentence.split(':')
                if len(colon_parts) > 1:
                    for part in colon_parts:
                        if len(part.encode('utf-8')) <= max_chars:
                            batches.append(part)
                        else:
                            # If colon part is still too long, split by comma
                            comma_parts = re.split('[,，]', part)
                            if len(comma_parts) > 1:
                                current_comma_part = ""
                                for comma_part in comma_parts:
                                    if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                        current_comma_part += comma_part + ','
                                    else:
                                        if current_comma_part:
                                            batches.append(current_comma_part.rstrip(','))
                                        current_comma_part = comma_part + ','
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                            else:
                                # If no comma, split by words
                                batches.extend(split_by_words(part))
                else:
                    # If no colon, split by comma
                    comma_parts = re.split('[,，]', sentence)
                    if len(comma_parts) > 1:
                        current_comma_part = ""
                        for comma_part in comma_parts:
                            if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                current_comma_part += comma_part + ','
                            else:
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                                current_comma_part = comma_part + ','
                        if current_comma_part:
                            batches.append(current_comma_part.rstrip(','))
                    else:
                        # If no comma, split by words
                        batches.extend(split_by_words(sentence))
            else:
                current_batch = sentence
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

