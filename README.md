CropGuard Edge — AI Crop Disease Detection

Empowering smallholder farmers in Kenya with Edge-AI for early disease diagnosis.

🎯 The Problem
Kenyan farmers lose an average of 20–40% of their yield to pests and diseases. Many tools require high-speed internet, which isn't always accessible in rural areas. CropGuard Edge focuses on lightweight, offline-first inference.

🧠 The Statistical Approach
As a Statistics major, I focused on optimizing model performance while maintaining a small footprint:

Data: Utilized public datasets (Kaggle/PlantVillage) with localized augmentation for Kenyan climate conditions.

Metrics: Achieved an 97% accuracy rate with a focus on minimizing False Negatives (missing a disease is more costly than a false alarm).

Optimization: Focused on model quantization to ensure the scripts can run on low-resource hardware (Edge computing).

🛠 Tech Stack

Languages: Python

Libraries: scikit-learn, Pandas (Data cleaning), NumPy

Deployment: Vercel (Web inference), Git for version control

📂 Project Structure

data/ — Pre-processed images and CSV metadata.

models/ — Serialized .pkl files optimized for low latency.

notebooks/ — Exploratory Data Analysis (EDA) and training logs (where the "Stats" happens).

scripts/ — Production-ready inference logic.

🚀 Getting Started

Clone the repo: git clone https://github.com/samkiva/cropguard-edge

Install dependencies: pip install -r requirements.txt

Run a test inference: python scripts/predict.py --image test_leaf.jpg

📈 Roadmap

[ ] Implement Time Series forecasting for regional disease outbreaks.

[ ] Integration with USSD for non-smartphone users.
