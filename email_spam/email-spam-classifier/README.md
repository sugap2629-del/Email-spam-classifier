# Email Spam Classifier

A small SMS spam classifier built with scikit-learn. It accepts the common
Kaggle SMS Spam CSV and UCI's tab-separated SMS Spam Collection format.

## Setup
```powershell
cd email-spam-classifier
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and save the Kaggle dataset to `data/scam.csv` with the scraper, or follow
the manual options in [data/README.md](data/README.md):

```powershell
python scraper.py
```

Then train the model and create the evaluation plots:

```powershell
python -m src.train
```

This writes `models/spam_classifier.pkl`, `outputs/confusion_matrix.png`, and
`outputs/top_spam_words.png`. The model and dataset are local generated/downloaded
artifacts and are intentionally excluded from version control.

## Predict

```powershell
python -m src.predict "Congratulations! Claim your free prize now"
```

For a browser UI:

```powershell
streamlit run app.py
```

Open `notebooks/eda.ipynb` in Jupyter for exploratory analysis. Start Jupyter
from the project root so its relative data path resolves correctly.

