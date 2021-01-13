import os
import shutil
import subprocess
import config as cfg
from torchvision.datasets.utils import download_url


def download_foa_data(data_dir, mode='eval'):
    urls = [
            'https://zenodo.org/record/2599196/files/foa_dev.z01?download=1',
            'https://zenodo.org/record/2599196/files/foa_dev.z02?download=1',
            'https://zenodo.org/record/2599196/files/foa_dev.zip?download=1',
            'https://zenodo.org/record/2599196/files/metadata_dev.zip?download=1',
            'https://zenodo.org/record/3377088/files/foa_eval.zip?download=1',
            'https://zenodo.org/record/3377088/files/metadata_eval.zip?download=1',
    ]
    md5s = [
            'bd5b18a47a3ed96e80069baa6b221a5a',
            '5194ebf43ae095190ed78691ec9889b1',
            '2154ad0d9e1e45bfc933b39591b49206',
            'c2e5c8b0ab430dfd76c497325171245d',
            '4a8ca8bfb69d7c154a56a672e3b635d5',
            'a0ec7640284ade0744dfe299f7ba107b'
    ]
    names = [
        'foa_dev.z01',
        'foa_dev.z02',
        'foa_dev.zip',
        'metadata_dev.zip',
        'foa_eval.zip',
        'metadata_eval.zip'
    ]

    if mode == 'eval':
        urls, md5s, names = urls[-2:], md5s[-2:], names[-2:]

    os.makedirs(data_dir, exist_ok=True)
    for url, md5, name in zip(urls, md5s, names):
        download_url(url, data_dir, md5=md5, filename=name)


def extract_foa_data(data_dir, output_dir, mode='eval'):
    os.makedirs(data_dir, exist_ok=True)
    subprocess.call(["unzip", os.path.join(data_dir,'metadata_eval.zip'), "-d", output_dir])
    subprocess.call(["unzip", os.path.join(data_dir, 'foa_eval.zip'), "-d", output_dir])

    subprocess.call(f"cp -R {output_dir}/proj/asignal/DCASE2019/dataset/foa_eval -d {output_dir}/foa_eval".split(" "))
    shutil.rmtree(f"{output_dir}/proj")

    if mode == 'train':
        subprocess.call(["unzip", os.path.join(data_dir, 'metadata_dev.zip'), "-d", output_dir])
        subprocess.call(f"zip -s 0 {os.path.join(data_dir,'foa_dev.zip')} --out {os.path.join(data_dir,'unsplit_foa_dev.zip')}".split(" "))
        subprocess.call(f"unzip {os.path.join(data_dir, 'unsplit_foa_dev.zip')} -d {output_dir}".split(" "))


def ensure_tau_data(data_dir, mode='eval'):
    zipped_data_dir = os.path.join(data_dir, 'zipped')
    extracted_data_dir = os.path.join(data_dir, 'raw')
    audio_dir = f"{extracted_data_dir}/foa_{mode}"
    meta_data_dir = f"{extracted_data_dir}/metadata_{mode}"

    # Download and extact data
    if not os.path.exists(zipped_data_dir):
        print("Downloading zipped data")
        download_foa_data(zipped_data_dir, mode)
    if not os.path.exists(audio_dir):
        print("Extracting raw data")
        extract_foa_data(zipped_data_dir, extracted_data_dir, mode)
    else:
        print("Using existing raw data")

    return audio_dir, meta_data_dir