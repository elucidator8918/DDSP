"""
Music Conversion Module

This module contains the MusicConversion class for transforming vocal tracks into various instruments.
"""
import os
import time
import numpy as np
import librosa
import tensorflow as tf
import ddsp
import ddsp.training
import gin
import pickle
from scipy.io import wavfile
from ddsp.colab.colab_utils import auto_tune, detect_notes, get_tuning_factor

# Default sample rate for audio processing
DEFAULT_SAMPLE_RATE = 16000

class MusicConversion:
    """
    Class for converting vocal tracks to instrument sounds using DDSP
    """
    def __init__(self, song_name, model, song_path):
        """
        Initialize MusicConversion with song parameters
        
        Args:
            song_name (str): Name of the song
            model (str): Instrument model to convert to (e.g., 'Violin', 'Flute')
            song_path (str): Path to the vocal audio file
        """
        self.song_name = song_name
        self.song_path = song_path
        self.model = model
        os.makedirs('output_songs', exist_ok=True)

    def load_song_and_extract_features(self):
        """
        Load the song, extract audio features, and convert to the target instrument
        
        Returns:
            str: Path to the converted audio file
        """
        print(f"Converting vocals to {self.model}...")

        # Load audio file
        audio, sr = librosa.load(self.song_path, sr=DEFAULT_SAMPLE_RATE)
        
        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Reshape for model
        audio = audio[np.newaxis, :]
        
        # Extract audio features
        print("Extracting features...")
        start_time = time.time()
        audio_features = self._compute_audio_features(audio)
        print(f'Audio feature extraction took {time.time() - start_time:.1f} seconds')
        
        # Load and prepare the model
        model_dir = self._prepare_model()
        gin_file = os.path.join(model_dir, 'operative_config-0.gin')
        
        # Load dataset statistics for auto-adjustment
        dataset_stats = self._load_dataset_stats(model_dir)
        
        # Configure model parameters
        self._configure_model(gin_file, audio)
        
        # Load model checkpoint
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
        upload_time = str(time.time())
        output_dir = os.path.join('output_songs', upload_time)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{self.song_name}.wav")
        librosa.output.write_wav(output_path, np.ravel(audio_gen), sr=DEFAULT_SAMPLE_RATE)
        
        print(f"Conversion complete. Output saved to {output_path}")
        return output_path

    def _compute_audio_features(self, audio):
        """
        Compute audio features for the DDSP model
        
        Args:
            audio (np.ndarray): Audio array
            
        Returns:
            dict: Audio features dictionary
        """
        # Reset CREPE to avoid issues
        ddsp.spectral_ops.reset_crepe()
        
        # Compute features
        audio_features = ddsp.training.metrics.compute_audio_features(audio)
        audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
        
        return audio_features

    def _prepare_model(self):
        """
        Prepare the instrument model
        
        Returns:
            str: Path to the model directory
        """
        if self.model in ('Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone'):
            print(f"Loading {self.model} model...")
            
            # Local model directory
            model_dir = os.path.join('pretrained', self.model.lower())
            os.makedirs(model_dir, exist_ok=True)
            
            # Check if model exists locally, if not download it
            if not os.path.exists(os.path.join(model_dir, 'operative_config-0.gin')):
                raise ValueError(f"Model {self.model} not found. Please download the models first.")
            
            return model_dir
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _load_dataset_stats(self, model_dir):
        """
        Load dataset statistics for the model
        
        Args:
            model_dir (str): Path to the model directory
            
        Returns:
            dict: Dataset statistics
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

    def _configure_model(self, gin_file, audio):
        """
        Configure the DDSP model parameters
        
        Args:
            gin_file (str): Path to the gin config file
            audio (np.ndarray): Audio array
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

    def _load_model(self, model_dir):
        """
        Load the DDSP model
        
        Args:
            model_dir (str): Path to the model directory
            
        Returns:
            ddsp.training.models.Autoencoder: Loaded model
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

    def _adjust_audio_features(self, audio_features, dataset_stats):
        """
        Adjust audio features based on dataset statistics
        
        Args:
            audio_features (dict): Original audio features
            dataset_stats (dict): Dataset statistics
            
        Returns:
            dict: Modified audio features
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
                
                # Optional auto-tune (disabled for now)
                # self._apply_autotune(audio_features_mod, mask_on)
            
        return audio_features_mod

    def _shift_f0(self, audio_features, pitch_shift=0.0):
        """
        Shift the fundamental frequency
        
        Args:
            audio_features (dict): Audio features
            pitch_shift (float): Amount to shift in octaves
            
        Returns:
            dict: Modified audio features
        """
        audio_features['f0_hz'] *= 2.0 ** pitch_shift
        audio_features['f0_hz'] = np.clip(
            audio_features['f0_hz'], 
            0.0, 
            librosa.midi_to_hz(110.0)
        )
        return audio_features

    def _shift_ld(self, audio_features, ld_shift=0.0):
        """
        Shift the loudness
        
        Args:
            audio_features (dict): Audio features
            ld_shift (float): Amount to shift in dB
            
        Returns:
            dict: Modified audio features
        """
        audio_features['loudness_db'] += ld_shift
        return audio_features

    def _fit_quantile_transform(self, loudness_db, mask, inv_quantile):
        """
        Apply quantile transform to loudness
        
        Args:
            loudness_db (np.ndarray): Loudness in dB
            mask (np.ndarray): Boolean mask for active regions
            inv_quantile (callable): Inverse quantile transform function
            
        Returns:
            tuple: Original and transformed loudness
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

    def _apply_autotune(self, audio_features_mod, mask_on, amount=1.0):
        """
        Apply auto-tune to the audio
        
        Args:
            audio_features_mod (dict): Audio features
            mask_on (np.ndarray): Boolean mask for active regions
            amount (float): Amount of auto-tune to apply
            
        Returns:
            dict: Modified audio features
        """
        f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod['f0_hz']))
        tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
        f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=amount)
        audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)
        return audio_features_mod