"""Compatibility entry point: stage Kaggle assets without overwriting live weights."""
from kaggle_runner import download_outputs

if __name__ == '__main__':
    download_outputs()
