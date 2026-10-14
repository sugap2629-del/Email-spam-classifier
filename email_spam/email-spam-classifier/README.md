# Email Spam Classifier

An AI-powered SMS and email spam detection suite built with Python, scikit-learn, and Streamlit. Features a modern dark cyber-defense interface with real-time explainability, heuristic telemetry, and batch file inspection.

## 🚀 Live Demo & Deployment on Render

This project is configured for one-click deployment on **[Render](https://render.com/)**:

1. Create a new **Web Service** on Render.
2. Connect your GitHub repository: `https://github.com/sugap2629-del/Email-spam-classifier.git`.
3. Configure the service settings:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
4. Click **Deploy Web Service**.

---

## 💻 Local Setup

```powershell
# Clone the repository
git clone https://github.com/sugap2629-del/Email-spam-classifier.git
cd Email-spam-classifier

# Set up virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Launch interactive UI
streamlit run app.py
```

## 🛠️ Model Training & CLI

To retrain the model and regenerate evaluation metrics:
```powershell
python -m src.train
```

To run quick command-line predictions:
```powershell
python -m src.predict "URGENT! You have won a $1,000 gift card. Call now to claim."
```

