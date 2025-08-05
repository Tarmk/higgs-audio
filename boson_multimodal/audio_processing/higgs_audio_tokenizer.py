# Based on code from: https://github.com/zhenye234/xcodec
# Licensed under MIT License
# Modifications by BosonAI

"""
Higgs Audio Tokenizer: A neural audio codec that converts raw audio waveforms into 
discrete tokens and reconstructs audio from those tokens.

Key Features:
- Combines acoustic (DAC encoder) and semantic (HuBERT/WavLM) features
- Uses vector quantization (RVQ/RFSQ) to create discrete tokens
- Operates at 50Hz token rate (320x compression from 16kHz audio)
- Supports multiple pre-trained semantic models
- Includes loudness normalization and audio preprocessing
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence
import numpy as np
from transformers import AutoModel
import torchaudio
import json
import librosa
from huggingface_hub import snapshot_download

from vector_quantize_pytorch import ResidualFSQ
from .descriptaudiocodec.dac.model import dac as dac2
from .quantization.vq import ResidualVectorQuantizer
from .semantic_module import Encoder, Decoder


class EncodedResult:
    """
    Container class to hold encoded audio codes/tokens.
    
    This simple wrapper is used to return the discrete audio tokens
    after encoding raw audio through the tokenizer.
    """
    def __init__(self, audio_codes):
        self.audio_codes = audio_codes


class HiggsAudioFeatureExtractor(nn.Module):
    """
    Audio preprocessing module that converts raw audio into properly formatted tensors.
    
    This class handles the conversion from numpy arrays or other formats into
    PyTorch tensors with the correct dimensions for the tokenizer.
    """
    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, raw_audio, sampling_rate=16000, return_tensors="pt"):
        """
        Convert raw audio to tensor format expected by the tokenizer.
        
        Args:
            raw_audio: Audio waveform as numpy array or similar
            sampling_rate: Sample rate of the audio (default: 16000)
            return_tensors: Format to return (default: "pt" for PyTorch)
            
        Returns:
            Dict with "input_values" key containing formatted audio tensor
        """
        # Convert from librosa/numpy to torch tensor
        audio_signal = torch.tensor(raw_audio)
        # Add batch dimension if missing
        audio_signal = audio_signal.unsqueeze(0)
        # Add channel dimension if missing (expecting [batch, channels, time])
        if len(audio_signal.shape) < 3:
            audio_signal = audio_signal.unsqueeze(0)
        return {"input_values": audio_signal}


class HiggsAudioTokenizer(nn.Module):
    """
    Main audio tokenizer that combines acoustic and semantic features to create discrete audio tokens.
    
    Architecture Overview:
    1. Acoustic Path: Raw audio → DAC Encoder → Acoustic Features (128D)
    2. Semantic Path: Raw audio → Pre-trained Model → Semantic Features (768D) 
    3. Feature Fusion: Concatenate features → Project → Vector Quantization
    4. Decoding: Tokens → Dequantize → Separate paths → Reconstruct audio
    
    The tokenizer operates at 50Hz (320x compression from 16kHz input).
    """
    def __init__(
        self,
        n_filters: int = 32,                    # Number of filters in encoder
        D: int = 128,                          # Acoustic feature dimension
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],  # VQ bandwidth options
        ratios: Sequence[int] = [8, 5, 4, 2],  # Downsampling ratios (total: 320x)
        sample_rate: int = 16000,              # Input audio sample rate
        bins: int = 1024,                      # Codebook size for quantization
        n_q: int = 8,                         # Number of quantization codebooks
        codebook_dim: int = None,             # Codebook embedding dimension
        normalize: bool = False,               # Whether to normalize features
        causal: bool = False,                 # Whether to use causal convolutions
        semantic_techer: str = "hubert_base_general",  # Pre-trained semantic model
        last_layer_semantic: bool = True,      # Whether to use only last layer features
        merge_mode: str = "concat",           # How to merge acoustic/semantic features
        downsample_mode: str = "step_down",   # How to downsample semantic features
        semantic_mode: str = "classic",       # Semantic processing mode
        vq_scale: int = 1,                    # Scale factor for quantizer dimension
        semantic_sample_rate: int = None,     # Override semantic model sample rate
        device: str = "cuda",                 # Device to run on
    ):
        super().__init__()
        
        # Calculate key parameters from downsampling ratios
        self.hop_length = np.prod(ratios)  # Total downsampling factor (320)
        self.semantic_techer = semantic_techer  # Store semantic model name
        
        # Frame rate: 16000 / 320 = 50 Hz (tokens per second)
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))
        
        # Store configuration parameters
        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate
        
        # Acoustic processing components
        # DAC encoder: converts raw audio to acoustic features
        self.encoder = dac2.Encoder(64, ratios, D)
        # DAC decoder: reconstructs audio from acoustic features  
        self.decoder_2 = dac2.Decoder(D, 1024, ratios)
        
        self.last_layer_semantic = last_layer_semantic
        self.device = device
        
        # Initialize semantic model based on configuration
        if semantic_techer == "hubert_base":
            self.semantic_model = AutoModel.from_pretrained("facebook/hubert-base-ls960")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768           # HuBERT output dimension
            self.encoder_semantic_dim = 768   # Encoder target dimension
            
        elif semantic_techer == "wavlm_base_plus":
            self.semantic_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768
            
        elif semantic_techer == "hubert_base_general":
            self.semantic_model = AutoModel.from_pretrained("bosonai/hubert_base", trust_remote_code=True)
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        # Override semantic sample rate if specified
        if semantic_sample_rate is not None:
            self.semantic_sample_rate = semantic_sample_rate

        # Set semantic model to evaluation mode and freeze parameters
        self.semantic_model.eval()
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        # Calculate downsampling factor to align semantic features with acoustic features
        # This ensures both feature streams have the same temporal resolution
        self.semantic_downsample_factor = int(self.hop_length / (self.sample_rate / self.semantic_sample_rate) / 320)

        # Quantizer input dimension: combined acoustic + semantic features
        self.quantizer_dim = int((D + self.encoder_semantic_dim) // vq_scale)
        
        # Semantic feature processing modules
        self.encoder_semantic = Encoder(input_channels=self.semantic_dim, encode_channels=self.encoder_semantic_dim)
        self.decoder_semantic = Decoder(
            code_dim=self.encoder_semantic_dim, output_channels=self.semantic_dim, decode_channels=self.semantic_dim
        )

        # Vector quantization setup - supports two types
        if isinstance(bins, int):  # Residual Vector Quantization (RVQ)
            self.quantizer = ResidualVectorQuantizer(
                dimension=self.quantizer_dim, codebook_dim=codebook_dim, n_q=n_q, bins=bins
            )
            self.quantizer_type = "RVQ"
        else:  # Residual Finite Scalar Quantization (RFSQ)
            self.quantizer = ResidualFSQ(dim=self.quantizer_dim, levels=bins, num_quantizers=n_q)
            self.quantizer_type = "RFSQ"

        # Feature fusion and separation layers
        self.fc_prior = nn.Linear(D + self.encoder_semantic_dim, self.quantizer_dim)  # Combine features before quantization
        self.fc_post1 = nn.Linear(self.quantizer_dim, self.encoder_semantic_dim)     # Extract semantic features after quantization
        self.fc_post2 = nn.Linear(self.quantizer_dim, D)                            # Extract acoustic features after quantization

        # Semantic downsampling configuration
        self.downsample_mode = downsample_mode
        if downsample_mode == "avg":
            self.semantic_pooling = nn.AvgPool1d(
                kernel_size=self.semantic_downsample_factor, stride=self.semantic_downsample_factor
            )

        # Audio preprocessing component
        self.audio_tokenizer_feature_extractor = HiggsAudioFeatureExtractor(sampling_rate=self.sample_rate)

    @property
    def tps(self):
        """Tokens Per Second - the temporal resolution of the tokenizer (50 Hz)"""
        return self.frame_rate

    @property
    def sampling_rate(self):
        """Audio sampling rate used by the tokenizer (16000 Hz)"""
        return self.sample_rate

    @property
    def num_codebooks(self):
        """Number of quantization codebooks used for vector quantization"""
        return self.n_q

    @property
    def codebook_size(self):
        """Size/dimension of the quantizer codebooks"""
        return self.quantizer_dim

    def get_last_layer(self):
        """Get the last layer weights for potential gradient-based analysis"""
        return self.decoder.layers[-1].weight

    def calculate_rec_loss(self, rec, target):
        """
        Calculate reconstruction loss using normalized cosine similarity.
        
        This loss function measures how well the reconstructed features match
        the target features using cosine similarity (better for high-dimensional features).
        
        Args:
            rec: Reconstructed features
            target: Target features
            
        Returns:
            Reconstruction loss (lower is better)
        """
        # L2 normalize both tensors
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        # Cosine similarity loss: 1 - cosine_similarity
        rec_loss = (1 - (target * rec).sum(-1)).mean()
        return rec_loss

    @torch.no_grad()
    def get_regress_target(self, x):
        """
        Extract semantic features from raw audio using pre-trained models.
        
        This is a crucial method that processes audio through models like HuBERT
        to extract semantic representations that capture linguistic content.
        
        Args:
            x: Raw audio tensor [batch, channels, time]
            
        Returns:
            Semantic features aligned with acoustic feature timing
        """
        # Resample audio to match semantic model's expected sample rate
        x = torchaudio.functional.resample(x, self.sample_rate, self.semantic_sample_rate)

        # Process based on semantic model type
        if (
            self.semantic_techer == "hubert_base"
            or self.semantic_techer == "hubert_base_general"
            or self.semantic_techer == "wavlm_base_plus"
        ):
            # For HuBERT/WavLM: expect mono audio and add padding
            x = x[:, 0, :]  # Convert to mono
            x = F.pad(x, (160, 160))  # Add padding for stable processing
            
            # Extract features from all hidden layers
            target = self.semantic_model(x, output_hidden_states=True).hidden_states
            target = torch.stack(target, dim=1)
            
            # Average across all layers for richer representation
            target = target.mean(1)
            
        elif self.semantic_techer == "w2v_bert2":
            target = self.semantic_model(x)
            
        elif self.semantic_techer.startswith("whisper"):
            if self.last_layer_semantic:
                target = self.semantic_model(x, avg_layers=False)
            else:
                target = self.semantic_model(x, avg_layers=True)
                
        elif self.semantic_techer.startswith("mert_music"):
            if self.last_layer_semantic:
                target = self.semantic_model(x, avg_layers=False)
            else:
                target = self.semantic_model(x, avg_layers=True)
                
        elif self.semantic_techer.startswith("qwen_audio_omni"):
            target = self.semantic_model(x)

        # Downsample semantic features to match acoustic feature rate
        if self.downsample_mode == "step_down":
            if self.semantic_downsample_factor > 1:
                # Simple step downsampling: take every Nth frame
                target = target[:, :: self.semantic_downsample_factor, :]
        elif self.downsample_mode == "avg":
            # Average pooling downsampling: smoother but more computation
            target = self.semantic_pooling(target.transpose(1, 2)).transpose(1, 2)
            
        return target

    def forward(self, x: torch.Tensor, bw: int):
        """
        Forward pass for training - processes audio through full encode/decode pipeline.
        
        Args:
            x: Input audio tensor [batch, channels, time]
            bw: Target bandwidth for quantization
            
        Returns:
            Tuple of (reconstructed_audio, commit_loss, semantic_recon_loss, None)
        """
        # Extract semantic features from input audio (no gradients)
        e_semantic_input = self.get_regress_target(x).detach()

        # Process semantic features through custom encoder
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        
        # Process acoustic features through DAC encoder
        e_acoustic = self.encoder(x)

        # Concatenate acoustic and semantic features
        e = torch.cat([e_acoustic, e_semantic], dim=1)

        # Project combined features to quantizer dimension
        e = self.fc_prior(e.transpose(1, 2))

        # Apply vector quantization
        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            quantized = quantized.transpose(1, 2)
        else:  # RFSQ
            quantized, codes = self.quantizer(e)
            commit_loss = torch.tensor(0.0)

        # Separate quantized features back into semantic and acoustic components
        quantized_semantic = self.fc_post1(quantized).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        # Decode acoustic features to audio
        o = self.decoder_2(quantized_acoustic)

        # Decode semantic features and calculate reconstruction loss
        o_semantic = self.decoder_semantic(quantized_semantic)
        semantic_recon_loss = F.mse_loss(e_semantic_input.transpose(1, 2).detach(), o_semantic)

        return o, commit_loss, semantic_recon_loss, None

    def encode(self, audio_path_or_wv, sr=None, loudness_normalize=False, loudness_threshold=-23.0):
        """
        Encode raw audio or audio file into discrete tokens.
        
        This is the main interface for converting audio into tokens for generation tasks.
        
        Args:
            audio_path_or_wv: Either file path (str) or raw waveform (array)
            sr: Sample rate if providing raw waveform
            loudness_normalize: Whether to normalize audio loudness
            loudness_threshold: Target loudness in LUFS (default: -23.0)
            
        Returns:
            Tensor of discrete audio tokens [n_codebooks, sequence_length]
        """
        # Load audio from file or use provided waveform
        if isinstance(audio_path_or_wv, str):
            wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
        else:
            wv = audio_path_or_wv
            assert sr is not None
            
        # Optional loudness normalization for consistent audio levels
        if loudness_normalize:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            l = meter.integrated_loudness(wv)
            wv = pyln.normalize.loudness(wv, l, loudness_threshold)
            
        # Resample to target sample rate if needed
        if sr != self.sampling_rate:
            wv = librosa.resample(wv, orig_sr=sr, target_sr=self.sampling_rate)
            
        # Format audio for processing
        if self.audio_tokenizer_feature_extractor is not None:
            inputs = self.audio_tokenizer_feature_extractor(
                raw_audio=wv, sampling_rate=self.audio_tokenizer_feature_extractor.sampling_rate, return_tensors="pt"
            )
            input_values = inputs["input_values"].to(self.device)
        else:
            input_values = torch.from_numpy(wv).float().unsqueeze(0)
            
        # Encode to discrete tokens
        with torch.no_grad():
            encoder_outputs = self._xcodec_encode(input_values)
            vq_code = encoder_outputs.audio_codes[0]
        return vq_code

    def _xcodec_encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
        """
        Internal encoding method that handles the actual tokenization process.
        
        This method performs the same operations as forward() but returns tokens
        instead of reconstructed audio.
        
        Args:
            x: Input audio tensor
            target_bw: Target bandwidth for quantization
            
        Returns:
            EncodedResult containing discrete audio tokens
        """
        bw = target_bw

        # Extract semantic features
        e_semantic_input = self.get_regress_target(x).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        
        # Extract acoustic features
        e_acoustic = self.encoder(x)

        # Handle dimension mismatches between acoustic and semantic features
        if e_acoustic.shape[2] != e_semantic.shape[2]:
            # Add padding to acoustic input and re-encode
            pad_size = 160 * self.semantic_downsample_factor
            e_acoustic = self.encoder(F.pad(x[:, 0, :], (pad_size, pad_size)).unsqueeze(0))

        # Ensure features have matching sequence length
        if e_acoustic.shape[2] != e_semantic.shape[2]:
            if e_acoustic.shape[2] > e_semantic.shape[2]:
                e_acoustic = e_acoustic[:, :, : e_semantic.shape[2]]
            else:
                e_semantic = e_semantic[:, :, : e_acoustic.shape[2]]

        # Combine and quantize features
        e = torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2))

        # Apply quantization and format output codes
        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            codes = codes.permute(1, 0, 2)  # Rearrange dimensions for output format
        else:  # RFSQ
            quantized, codes = self.quantizer(e)
            codes = codes.permute(0, 2, 1)  # Different arrangement for RFSQ

        return EncodedResult(codes)

    def decode(self, vq_code: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete tokens back into audio waveform.
        
        This method reconstructs audio from the discrete tokens produced by encode().
        
        Args:
            vq_code: Discrete audio tokens [n_codebooks, sequence_length]
            
        Returns:
            Reconstructed audio waveform as numpy array
        """
        vq_code = vq_code.to(self.device)

        # Dequantize tokens back to continuous features
        if self.quantizer_type == "RVQ":
            vq_code = vq_code.permute(1, 0, 2)
            quantized = self.quantizer.decode(vq_code)
            quantized = quantized.transpose(1, 2)
        else:  # RFSQ
            vq_code = vq_code.permute(0, 2, 1)
            quantized = self.quantizer.get_output_from_indices(vq_code)
            
        # Extract acoustic component and decode to audio
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)
        o = self.decoder_2(quantized_acoustic)
        
        return o.detach().cpu().numpy()


def load_higgs_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
    """
    Convenience function to load a pre-trained Higgs Audio Tokenizer.
    
    This function handles downloading from HuggingFace Hub, loading configuration,
    and initializing the model with the correct parameters.
    
    Args:
        tokenizer_name_or_path: HuggingFace model name or local path
        device: Device to load the model on (default: "cuda")
        
    Returns:
        Initialized and loaded HiggsAudioTokenizer model ready for inference
    """
    # Check if path exists locally, otherwise download from HuggingFace
    is_local = os.path.exists(tokenizer_name_or_path)
    if not is_local:
        tokenizer_path = snapshot_download(tokenizer_name_or_path)
    else:
        tokenizer_path = tokenizer_name_or_path
        
    # Load configuration and model weights
    config_path = os.path.join(tokenizer_path, "config.json")
    model_path = os.path.join(tokenizer_path, "model.pth")
    config = json.load(open(config_path))
    
    # Initialize model with loaded configuration
    model = HiggsAudioTokenizer(
        **config,
        device=device,
    )
    
    # Load pre-trained weights
    parameter_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(parameter_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model
