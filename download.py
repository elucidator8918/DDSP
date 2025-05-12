import os
import tensorflow as tf

def download_ddsp_model(model_name='Violin', save_dir='./pretrained'):
    """
    Downloads a pretrained DDSP timbre transfer model from a public Google Cloud Storage (GCS) bucket.

    Parameters:
    -----------
    model_name : str
        Name of the instrument model to download. Must be one of:
        ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone']
    
    save_dir : str
        Local directory to store the downloaded model files.

    Requirements:
    -------------
    - TensorFlow must be installed (`pip install tensorflow`)
    - Internet access and ability to read from public GCS buckets

    Example:
    --------
    >>> download_ddsp_model('Violin', './my_models/violin')
    """

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # GCS path to the pretrained model
    gcs_path = f'gs://ddsp/models/timbre_transfer_colab/2021-07-08/solo_{model_name.lower()}_ckpt'

    # List all files in the model's GCS directory
    try:
        files = tf.io.gfile.listdir(gcs_path)
    except Exception as e:
        raise RuntimeError(f"Failed to list files in GCS path '{gcs_path}': {e}")

    # Download each file to the local directory
    for fname in files:
        src = os.path.join(gcs_path, fname)
        dst = os.path.join(save_dir, fname)
        try:
            tf.io.gfile.copy(src, dst, overwrite=True)
        except Exception as e:
            raise RuntimeError(f"Failed to copy {src} to {dst}: {e}")

    print(f"✅ Model '{model_name}' downloaded to: {save_dir}")

# Example Usage
download_ddsp_model('Violin')
