"""
Music Conversion Module

This module contains the MusicConversion class for transforming vocal tracks into various instruments.
"""
import os
import time
from typing import Dict, Optional, Tuple, Any

import numpy as np
import librosa
import tensorflow as tf
import ddsp
import ddsp.training
import gin
import pickle
from scipy.io import wavfile
from ddsp.training.postprocessing import detect_notes

# Default sample rate for audio processing
DEFAULT_SAMPLE_RATE = 16000


def get_tuning_factor(f0_midi: np.ndarray, f0_confidence: np.ndarray, mask_on: np.ndarray) -> float:
    """Get an offset in cents to achieve the most consistent set of chromatic intervals.
    
    Args:
        f0_midi: Array of MIDI note values
        f0_confidence: Array of confidence values for each note
        mask_on: Boolean mask indicating which notes are active
        
    Returns:
        float: Tuning factor in cents
    """
    # Difference from midi offset by different tuning_factors
    tuning_factors = np.linspace(-0.5, 0.5, 101)  # 1 cent divisions
    midi_diffs = (f0_midi[mask_on][:, np.newaxis] - tuning_factors[np.newaxis, :]) % 1.0
    midi_diffs[midi_diffs > 0.5] -= 1.0
    weights = f0_confidence[mask_on][:, np.newaxis]

    # Compute minimum adjustment distance
    cost_diffs = np.abs(midi_diffs)
    cost_diffs = np.mean(weights * cost_diffs, axis=0)

    # Compute minimum "note" transitions
    f0_at = f0_midi[mask_on][:, np.newaxis] - midi_diffs
    f0_at_diffs = np.diff(f0_at, axis=0)
    deltas = (f0_at_diffs != 0.0).astype(float)
    cost_deltas = np.mean(weights[:-1] * deltas, axis=0)

    # Tuning factor is minimum cost
    norm = lambda x: (x - np.mean(x)) / np.std(x)
    cost = norm(cost_deltas) + norm(cost_diffs)
    return tuning_factors[np.argmin(cost)]


def auto_tune(f0_midi: np.ndarray, tuning_factor: float, mask_on: np.ndarray, 
              amount: float = 0.0, chromatic: bool = False) -> np.ndarray:
    """Reduce variance of f0 from the chromatic or scale intervals.
    
    Args:
        f0_midi: Array of MIDI note values
        tuning_factor: Tuning factor in cents
        mask_on: Boolean mask indicating which notes are active
        amount: Amount of auto-tune to apply (0.0 to 1.0)
        chromatic: Whether to use chromatic scale or infer major scale
        
    Returns:
        np.ndarray: Auto-tuned MIDI note values
    """
    if chromatic:
        midi_diff = (f0_midi - tuning_factor) % 1.0
        midi_diff[midi_diff > 0.5] -= 1.0
    else:
        major_scale = np.ravel(
            [np.array([0, 2, 4, 5, 7, 9, 11]) + 12 * i for i in range(10)])
        all_scales = np.stack([major_scale + i for i in range(12)])

        f0_on = f0_midi[mask_on]
        # [time, scale, note]
        f0_diff_tsn = (
            f0_on[:, np.newaxis, np.newaxis] - all_scales[np.newaxis, :, :])
        # [time, scale]
        f0_diff_ts = np.min(np.abs(f0_diff_tsn), axis=-1)
        # [scale]
        f0_diff_s = np.mean(f0_diff_ts, axis=0)
        scale_idx = np.argmin(f0_diff_s)
        scale = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb',
                 'G', 'Ab', 'A', 'Bb', 'B', 'C'][scale_idx]

        # [time]
        f0_diff_tn = f0_midi[:, np.newaxis] - all_scales[scale_idx][np.newaxis, :]
        note_idx = np.argmin(np.abs(f0_diff_tn), axis=-1)
        midi_diff = np.take_along_axis(
            f0_diff_tn, note_idx[:, np.newaxis], axis=-1)[:, 0]
        print(f'Autotuning... \nInferred key: {scale}  '
              f'\nTuning offset: {int(tuning_factor * 100)} cents')

    # Adjust the midi signal
    return f0_midi - amount * midi_diff


class MusicConversion:
    """Class for converting vocal tracks to instrument sounds using DDSP."""
    
    SUPPORTED_MODELS = {'Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone'}
    
    def __init__(self, song_name: str, model: str, song_path: str):
        """Initialize MusicConversion with song parameters.
        
        Args:
            song_name: Name of the song
            model: Instrument model to convert to (e.g., 'Violin', 'Flute')
            song_path: Path to the vocal audio file
        
        Raises:
            ValueError: If the model is not supported
        """
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}. Supported models: {', '.join(self.SUPPORTED_MODELS)}")
            
        self.song_name = song_name
        self.song_path = song_path
        self.model = model
        
        # Ensure output directory exists
        os.makedirs('output_songs', exist_ok=True)

    def load_song_and_extract_features(self) -> str:
        """Load the song, extract audio features, and convert to the target instrument.
        
        Returns:
            str: Path to the converted audio file
        """
        print(f"Converting vocals to {self.model}...")

        # Load and prepare audio
        audio = self._load_audio()
        
        # Extract audio features
        print("Extracting features...")
        start_time = time.time()
        audio_features = self._compute_audio_features(audio)
        print(f'Audio feature extraction took {time.time() - start_time:.1f} seconds')
        
        # Load model and dataset statistics
        model_dir = self._prepare_model()
        gin_file = os.path.join(model_dir, 'operative_config-0.gin')
        dataset_stats = self._load_dataset_stats(model_dir)
        
        # Configure and load model
        self._configure_model(gin_file, audio)
        model = self._load_model(model_dir)
        
        # Adjust audio features if needed
        audio_features_mod = self._adjust_audio_features(audio_features, dataset_stats)
        
        # Generate audio from the model
        print("Generating instrument audio...")
        start_time = time.time()
        outputs = model(audio_features_mod, training=False)
        audio_gen = model.get_audio_from_outputs(outputs)
        print(f'Prediction took {time.time() - start_time:.1f} seconds')
        
        # Save the generated audio
        output_path = self._save_output(audio_gen)
        
        print(f"Conversion complete. Output saved to {output_path}")
        return output_path
        
    def _load_audio(self) -> np.ndarray:
        """Load and prepare the audio file.
        
        Returns:
            np.ndarray: Processed audio array
        """
        # Load audio file
        audio, sr = librosa.load(self.song_path, sr=DEFAULT_SAMPLE_RATE)
        
        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Reshape for model
        return audio[np.newaxis, :]

    def _compute_audio_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute audio features for the DDSP model.
        
        Args:
            audio: Audio array
            
        Returns:
            Dict: Audio features dictionary
        """
        # Reset CREPE to avoid issues
        ddsp.spectral_ops.reset_crepe()
        
        # Compute features
        audio_features = ddsp.training.metrics.compute_audio_features(audio)
        audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
        
        return audio_features

    def _prepare_model(self) -> str:
        """Prepare the instrument model.
        
        Returns:
            str: Path to the model directory
            
        Raises:
            FileNotFoundError: If the model files are not found
        """
        print(f"Loading {self.model} model...")
        
        # Local model directory
        model_dir = os.path.join('pretrained', self.model.lower())
        
        # Check if model exists locally
        config_path = os.path.join(model_dir, 'operative_config-0.gin')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model {self.model} not found at {config_path}. "
                                    "Please download the models first.")
        
        return model_dir

    def _load_dataset_stats(self, model_dir: str) -> Optional[Dict[str, Any]]:
        """Load dataset statistics for the model.
        
        Args:
            model_dir: Path to the model directory
            
        Returns:
            Optional[Dict]: Dataset statistics or None if not available
        """
        dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
        print(f'Loading dataset statistics from {dataset_stats_file}')
        
        try:
            if tf.io.gfile.exists(dataset_stats_file):
                with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as err:
            print(f'Loading dataset statistics failed: {err}')
            return None

    def _configure_model(self, gin_file: str, audio: np.ndarray) -> None:
        """Configure the DDSP model parameters.
        
        Args:
            gin_file: Path to the gin config file
            audio: Audio array
        """
        with gin.unlock_config():
            gin.parse_config_file(gin_file, skip_unknown=True)
        
        # Get parameters from gin
        time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
        n_samples_train = gin.query_parameter('Harmonic.n_samples')
        hop_size = int(n_samples_train / time_steps_train)
        
        # Calculate new dimensions
        time_steps = int(audio.shape[1] / hop_size)
        n_samples = time_steps * hop_size
        
        # Update gin parameters
        gin_params = [
            f'Harmonic.n_samples = {n_samples}',
            f'FilteredNoise.n_samples = {n_samples}',
            f'F0LoudnessPreprocessor.time_steps = {time_steps}',
            'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors
        ]
        
        with gin.unlock_config():
            gin.parse_config(gin_params)

    def _load_model(self, model_dir: str) -> ddsp.training.models.Autoencoder:
        """Load the DDSP model.
        
        Args:
            model_dir: Path to the model directory
            
        Returns:
            ddsp.training.models.Autoencoder: Loaded model
            
        Raises:
            FileNotFoundError: If no checkpoint is found
        """
        # Find checkpoint
        ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")
            
        ckpt_name = ckpt_files[0].split('.')[0]
        ckpt = os.path.join(model_dir, ckpt_name)
        
        # Load model
        model = ddsp.training.models.Autoencoder()
        model.restore(ckpt)
        
        return model

    def _adjust_audio_features(self, audio_features: Dict[str, np.ndarray], 
                              dataset_stats: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Adjust audio features based on dataset statistics.
        
        Args:
            audio_features: Original audio features
            dataset_stats: Dataset statistics
            
        Returns:
            Dict: Modified audio features
        """
        # Create a copy of audio features
        audio_features_mod = {k: v.copy() for k, v in audio_features.items()}
        
        # Apply adjustments if dataset stats are available
        if dataset_stats is not None:
            # Detect sections that are "on" (actual singing/notes)
            mask_on, note_on_value = detect_notes(
                audio_features['loudness_db'],
                audio_features['f0_confidence'],
                threshold=1
            )
            
            if np.any(mask_on):
                # Shift pitch to match dataset mean
                target_mean_pitch = dataset_stats['mean_pitch']
                pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
                mean_pitch = np.mean(pitch[mask_on])
                p_diff = target_mean_pitch - mean_pitch
                p_diff_octave = p_diff / 12.0
                
                # Round to nearest octave
                round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
                p_diff_octave = round_fn(p_diff_octave)
                
                # Apply pitch shift
                audio_features_mod = self._shift_f0(audio_features_mod, p_diff_octave)
                
                # Apply loudness transform
                _, loudness_norm = self._fit_quantile_transform(
                    audio_features['loudness_db'],
                    mask_on,
                    dataset_stats['quantile_transform']
                )
                
                # Turn down the note_off parts
                mask_off = np.logical_not(mask_on)
                quiet = 20  # dB
                loudness_norm[mask_off] -= quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
                loudness_norm = np.reshape(loudness_norm, audio_features['loudness_db'].shape)
                
                audio_features_mod['loudness_db'] = loudness_norm
                
                # Optional auto-tune (uncomment to enable)
                # audio_features_mod = self._apply_autotune(audio_features_mod, mask_on)
            
        return audio_features_mod

    def _shift_f0(self, audio_features: Dict[str, np.ndarray], pitch_shift: float = 0.0) -> Dict[str, np.ndarray]:
        """Shift the fundamental frequency.
        
        Args:
            audio_features: Audio features
            pitch_shift: Amount to shift in octaves
            
        Returns:
            Dict: Modified audio features
        """
        audio_features['f0_hz'] *= 2.0 ** pitch_shift
        audio_features['f0_hz'] = np.clip(
            audio_features['f0_hz'], 
            0.0, 
            librosa.midi_to_hz(110.0)
        )
        return audio_features

    def _shift_ld(self, audio_features: Dict[str, np.ndarray], ld_shift: float = 0.0) -> Dict[str, np.ndarray]:
        """Shift the loudness.
        
        Args:
            audio_features: Audio features
            ld_shift: Amount to shift in dB
            
        Returns:
            Dict: Modified audio features
        """
        audio_features['loudness_db'] += ld_shift
        return audio_features

    def _fit_quantile_transform(self, loudness_db: np.ndarray, mask: np.ndarray, 
                               inv_quantile: callable) -> Tuple[np.ndarray, np.ndarray]:
        """Apply quantile transform to loudness.
        
        Args:
            loudness_db: Loudness in dB
            mask: Boolean mask for active regions
            inv_quantile: Inverse quantile transform function
            
        Returns:
            Tuple: Original and transformed loudness
        """
        loudness_flat = np.reshape(loudness_db, (-1, 1))
        loudness_masked = loudness_flat[mask.reshape(-1)]
        
        # Sort the masked loudness
        loudness_sorted = np.sort(loudness_masked, axis=0)
        
        # Create mapping
        loudness_mapping = []
        n_loud = loudness_masked.shape[0]
        
        for i in range(n_loud):
            idx = np.searchsorted(loudness_sorted, loudness_masked[i])
            loudness_mapping.append(np.array([loudness_masked[i], inv_quantile(idx / n_loud)]))
        
        loudness_mapping = np.array(loudness_mapping)
        
        # Apply transform to all values
        loudness_norm = np.zeros_like(loudness_flat)
        for i in range(len(loudness_flat)):
            if mask.reshape(-1)[i]:
                idx = np.argmin(np.abs(loudness_flat[i] - loudness_mapping[:, 0]))
                loudness_norm[i] = loudness_mapping[idx, 1]
            else:
                loudness_norm[i] = loudness_flat[i]
        
        return loudness_flat, loudness_norm

    def _apply_autotune(self, audio_features_mod: Dict[str, np.ndarray], mask_on: np.ndarray, 
                       amount: float = 1.0) -> Dict[str, np.ndarray]:
        """Apply auto-tune to the audio.
        
        Args:
            audio_features_mod: Audio features
            mask_on: Boolean mask for active regions
            amount: Amount of auto-tune to apply
            
        Returns:
            Dict: Modified audio features
        """
        f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod['f0_hz']))
        tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
        f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=amount)
        audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)
        return audio_features_mod
        
    def _save_output(self, audio_gen: np.ndarray) -> str:
        """Save the generated audio file.
        
        Args:
            audio_gen: Generated audio array
            
        Returns:
            str: Path to the saved file
        """
        # Create timestamped output directory
        timestamp = str(time.time())
        output_dir = os.path.join('output_songs', timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save file
        output_path = os.path.join(output_dir, f"{self.song_name}.wav")
        librosa.output.write_wav(output_path, np.ravel(audio_gen), sr=DEFAULT_SAMPLE_RATE)
        
        return output_path