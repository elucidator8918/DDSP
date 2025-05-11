"""
Vocal Extraction Module

This module contains the VocalExtraction class for separating vocals from music tracks.
"""
import os
import time
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import subprocess

class VocalExtraction:
    """
    Class for extracting vocals from songs using Spleeter
    """
    def __init__(self, song_path, song_name):
        """
        Initialize VocalExtraction with song path and name
        
        Args:
            song_path (str): Path to the audio file
            song_name (str): Name of the song
        """
        self.song_path = song_path
        self.song_name = song_name
        self.destination_path = ''
        self.extracted_destination_path = ''

    def extract_vocal(self):
        """
        Extract vocals from the song using Spleeter
        """
        print("Extracting Vocals...")
        
        # Create output directory for Spleeter
        os.makedirs("output", exist_ok=True)
        
        # Run Spleeter command
        cmd = f"spleeter separate -p spleeter:2stems -o output {self.song_path}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Spleeter error: {stderr.decode()}")
        
        print('Vocals Extracted Successfully')
        
        # Get extracted path and trim it
        base_filename = os.path.basename(self.song_path)
        filename_no_ext = os.path.splitext(base_filename)[0]
        self.extracted_destination_path = f'output/{filename_no_ext}/vocals.wav'
        
        self.trim_extracted_vocal()

    def trim_extracted_vocal(self):
        """
        Trim the extracted vocals to remove silence and noise
        """
        print('Trimming the extracted vocals...')
        
        try:
            # Read the extracted vocal file
            sampling_frequency, signal_data = wavfile.read(self.extracted_destination_path)
            
            # Filter out very quiet parts
            song_data = []
            for i in tqdm(signal_data):
                if abs(i[0]) + abs(i[1]) > 20:  # Noise threshold
                    song_data.append(i)
            
            # Convert list to numpy array
            song_data = np.array(song_data)
            song_data = song_data.astype(np.int16)
            
            # Create timestamp directory
            upload_time = time.time()
            extracted_dir = os.path.join("extracted_vocals", str(upload_time))
            os.makedirs(extracted_dir, exist_ok=True)
            
            # Save the trimmed vocal file
            self.destination_path = os.path.join(extracted_dir, f"{self.song_name}.wav")
            wavfile.write(self.destination_path, sampling_frequency, song_data)
            
            print('Vocal extraction and trimming completed successfully')
            
        except Exception as e:
            print(f"Error during vocal trimming: {str(e)}")
            raise