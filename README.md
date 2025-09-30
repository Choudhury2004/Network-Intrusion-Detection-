# Network-Intrusion-Detection-
This project implements a Network Intrusion Detection System (NIDS) to monitor network traffic and detect malicious activities such as DoS attacks, probing, unauthorized access, and data breaches. The system leverages machine learning and packet analysis to identify anomalies in real-time and raise alerts.
Features:
1. Real-time network packet capturing
2. Feature extraction from packet headers
3. Machine Learning–based intrusion detection
4. Detection of common attacks (DoS, Probe, R2L, U2R, etc.)
Tech Stack:
1. Programming Language: Python
2. Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
3. Packet Capture: scapy / pyshark

NIDS-Project/
│── data/                 # Dataset (NSL-KDD / CICIDS2017 / pcap files)
│── models/               # Trained ML models
│── notebooks/            # Jupyter notebooks for experiments
│── src/                  # Core source code
│   ├── preprocess.py     # Data preprocessing & feature engineering
│   ├── train.py          # Training ML/DL models
│   ├── detect.py         # Real-time detection script
│── results/              # Evaluation metrics & graphs
│── README.md             # Project documentation
