import tarfile

dataset_path = '/home/roth_pu/RESIKOAST/hotspot-detection/VT-ADL/mvtech_datasets'
tar_path = '/home/roth_pu/RESIKOAST/hotspot-detection/VT-ADL/mvtec_anomaly_detection.tar.xz'

# Öffnen und extrahieren der .tar.xz-Datei
with tarfile.open(tar_path, 'r:xz') as tar:
    tar.extractall(dataset_path)