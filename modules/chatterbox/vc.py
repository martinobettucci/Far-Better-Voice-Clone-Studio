from contextlib import nullcontext
from pathlib import Path

import librosa
import numpy as np
import torch
try:
    import perth
except ImportError:
    perth = None
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen


REPO_ID = "ResembleAI/chatterbox"


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR
    TARGET_SELECTION_STEP_SECONDS = 1.0

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker() if perth else None
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)
        
        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"
            
        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    @staticmethod
    def _longest_true_run(mask):
        best = 0
        current = 0
        for value in mask:
            if value:
                current += 1
                if current > best:
                    best = current
            else:
                current = 0
        return best

    def _trim_target_activity(self, wav, sr):
        intervals = librosa.effects.split(wav, top_db=35, frame_length=2048, hop_length=512)
        if len(intervals) <= 0:
            return wav
        start = int(intervals[0][0])
        end = int(intervals[-1][1])
        if end <= start:
            return wav
        return wav[start:end]

    def _score_target_window(self, wav, sr):
        eps = 1.0e-6
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size <= 0:
            return float("-inf")

        peak = float(np.max(np.abs(wav)))
        if peak <= eps:
            return float("-inf")

        clip_ratio = float(np.mean(np.abs(wav) >= 0.995))
        dc_offset = float(np.abs(np.mean(wav)))

        frame_length = min(2048, max(512, 1 << (wav.size.bit_length() - 1)))
        hop_length = max(128, frame_length // 4)

        rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
        if rms.size <= 0:
            return float("-inf")

        rms_ref = max(float(np.max(rms)), eps)
        rms_db = librosa.amplitude_to_db(np.maximum(rms, eps), ref=rms_ref)
        speech_mask = rms_db > -35.0
        speech_ratio = float(np.mean(speech_mask))
        longest_run_ratio = float(self._longest_true_run(speech_mask) / max(1, speech_mask.size))

        active_rms = rms[speech_mask] if speech_mask.any() else rms
        inactive_rms = rms[~speech_mask] if (~speech_mask).any() else rms[:1]
        active_level = float(np.percentile(active_rms, 75))
        noise_level = float(np.percentile(inactive_rms, 50))
        snr_db = 20.0 * np.log10((active_level + eps) / (noise_level + eps))
        snr_score = float(np.clip((snr_db - 3.0) / 18.0, 0.0, 1.0))
        loudness_score = float(np.clip((20.0 * np.log10(active_level + eps) + 36.0) / 24.0, 0.0, 1.0))
        rms_dynamic_db = float(np.percentile(rms_db, 90) - np.percentile(rms_db, 15))
        dynamics_score = float(np.clip((rms_dynamic_db - 8.0) / 18.0, 0.0, 1.0))

        zcr = librosa.feature.zero_crossing_rate(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
        mean_zcr = float(np.mean(zcr[speech_mask])) if speech_mask.any() else float(np.mean(zcr))
        zcr_penalty = float(np.clip((mean_zcr - 0.18) / 0.18, 0.0, 1.0))

        spectrum = np.abs(librosa.stft(wav, n_fft=frame_length, hop_length=hop_length))
        if spectrum.size <= 0:
            return float("-inf")
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        voice_band = (freqs >= 80.0) & (freqs <= 4500.0)
        band_energy = float(np.sum(spectrum[voice_band]))
        total_energy = float(np.sum(spectrum)) + eps
        voice_band_ratio = band_energy / total_energy

        flatness = librosa.feature.spectral_flatness(S=np.maximum(spectrum, eps))[0]
        flatness_penalty = float(np.mean(flatness[speech_mask])) if speech_mask.any() else float(np.mean(flatness))

        score = 0.0
        score += 4.5 * speech_ratio
        score += 2.0 * longest_run_ratio
        score += 1.5 * snr_score
        score += 1.0 * loudness_score
        score += 1.5 * dynamics_score
        score += 1.0 * float(np.clip((voice_band_ratio - 0.45) / 0.35, 0.0, 1.0))
        score -= 5.0 * flatness_penalty
        score -= 3.0 * zcr_penalty
        score -= 8.0 * clip_ratio
        score -= 2.0 * dc_offset
        return score

    def _select_target_segment(self, wav, sr):
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = librosa.to_mono(wav)

        wav = self._trim_target_activity(wav, sr)
        if wav.size <= self.DEC_COND_LEN:
            return wav[:self.DEC_COND_LEN]

        window_samples = int(self.DEC_COND_LEN)
        step_samples = max(1, int(round(self.TARGET_SELECTION_STEP_SECONDS * sr)))
        last_start = max(0, wav.size - window_samples)
        candidate_starts = list(range(0, last_start + 1, step_samples))
        if not candidate_starts or candidate_starts[-1] != last_start:
            candidate_starts.append(last_start)

        best_start = 0
        best_score = float("-inf")
        for start in candidate_starts:
            window = wav[start:start + window_samples]
            score = self._score_target_window(window, sr)
            if score > best_score:
                best_score = score
                best_start = int(start)

        return wav[best_start:best_start + window_samples]

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = self._select_target_segment(s3gen_ref_wav, S3GEN_SR)
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def _source_autocast_context(self):
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=autocast_dtype)
        return nullcontext()

    def _prepare_source_waveform(self, audio, source_sr):
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().float().numpy()

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 0:
            audio = np.expand_dims(audio, axis=0)
        elif audio.ndim > 1:
            # Match librosa.load(..., mono=True): support both (frames, channels) and (channels, frames).
            if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[-1]:
                audio = audio.mean(axis=0)
            else:
                audio = audio.mean(axis=-1)

        if int(source_sr) != S3_SR:
            audio = librosa.resample(audio, orig_sr=int(source_sr), target_sr=S3_SR)

        return torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(self.device)[None, :]

    def generate_from_waveform(
        self,
        audio,
        source_sr=S3_SR,
        target_voice_path=None,
        n_cfm_timesteps=None,
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice_path`"

        with torch.inference_mode():
            audio_16 = self._prepare_source_waveform(audio, source_sr=source_sr)
            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            with self._source_autocast_context():
                wav, _ = self.s3gen.inference(
                    speech_tokens=s3_tokens,
                    ref_dict=self.ref_dict,
                    n_cfm_timesteps=n_cfm_timesteps,
                )
            wav = wav.squeeze(0).detach().cpu().numpy()
            if self.watermarker:
                wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(wav).unsqueeze(0)

    def generate(
        self,
        audio,
        target_voice_path=None,
        n_cfm_timesteps=None,
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice_path`"

        audio_16, _ = librosa.load(audio, sr=S3_SR)
        return self.generate_from_waveform(
            audio_16,
            source_sr=S3_SR,
            target_voice_path=None,
            n_cfm_timesteps=n_cfm_timesteps,
        )
