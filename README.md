# AI Invasive Animal Detection

## Overview

This repository contains a modular AI‑powered system for detecting invasive animal species using ESP32‑CAM nodes, a machine‑learning inference service, a backend API, a MongoDB database, and a web dashboard.

## Project Structure

```
AI-Invasive-Animal-Detection/
│
├── README.md                 # This file
├── .gitignore                # Git ignore rules
├── .env.example              # Example environment configuration
├── requirements.txt          # Top‑level requirements (if any)
│
├── ml/                       # Machine‑learning pipeline
│   ├── data/                 # Raw / processed datasets
│   ├── models/               # Trained model files
│   ├── notebooks/            # Exploratory notebooks
│   ├── src/                  # Source code
│   │   ├── data/             # Data handling scripts
│   │   ├── training/         # Training scripts
│   │   └── inference/        # Inference script
│   └── tests/                # Unit tests for ML code
│
├── backend/                  # FastAPI backend service
│   ├── app/                  # Application package
│   │   ├── main.py           # FastAPI entry point
│   │   ├── config.py         # Configuration handling
│   │   ├── api/              # API route modules
│   │   ├── models/           # Pydantic models
│   │   ├── database/         # Database connection helpers
│   │   ├── services/         # Business logic
│   │   └── utils/            # Utility functions
│   └── tests/                # Backend tests
│
├── firmware/                 # ESP32‑CAM firmware (PlatformIO)
│   ├── common/               # Shared code
│   ├── node_01/              # Example node configuration
│   └── node_02/              # Example node configuration
│
├── dashboard/                # Web dashboard (to be added later)
│
├── docs/                     # Documentation
│   ├── architecture.md
│   ├── ml.md
│   ├── backend.md
│   ├── hardware.md
│   └── api.md
│
└── datasets/                 # External datasets (reference only)
```

## Component Status

| Component | Implemented | Simulated | Pending |
|-----------|-------------|-----------|---------|
| Machine‑Learning pipeline | ❌ | ❌ | ✅ |
| Inference API (FastAPI) | ❌ | ❌ | ✅ |
| ESP32‑CAM firmware | ❌ | ❌ | ✅ |
| MongoDB integration | ❌ | ❌ | ✅ |
| Alert system | ❌ | ❌ | ✅ |
| Multi‑node support | ❌ | ❌ | ✅ |
| Edge AI optimization | ❌ | ❌ | ✅ |
| Dashboard | ❌ | ❌ | ✅ |

## Setup Instructions (Phase 0)

1. **Clone the repository**
   ```bash
   git clone <repository‑url>
   cd AI-Invasive-Animal-Detection
   ```
2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
3. **Install top‑level development tools** (if any)
   ```bash
   pip install -r requirements.txt
   ```
4. **Install component‑specific dependencies** when you start each phase:
   - ML: `pip install -r ml/requirements.txt`
   - Backend: `pip install -r backend/requirements.txt`
5. **Configure environment variables**
   - Copy `.env.example` to `.env` and fill in the values.

## Next Steps

Proceed to **Phase 1 – Machine Learning** where the dataset pipeline, training, evaluation, and inference scripts will be built.

---

*All components are currently placeholders pending implementation.*
