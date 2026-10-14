# Dataset

Place the SMS Spam Collection dataset here as `scam.csv`. The training loader
accepts the common Kaggle columns (`v1`, `v2`) and UCI's tab-separated,
headerless format, as well as columns named `label`/`message` or
`category`/`text`.

## Kaggle source

The scraper downloads the [SMS Spam Collection published on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
and saves the Kaggle CSV as `scam.csv`:

```powershell
python scraper.py --force
```

`spam.csv` is also provided as an identical copy for tools or instructions that
expect the conventional Kaggle filename.

The scraper also accepts the UCI dataset archive, which it saves in its
tab-separated format. Both formats are supported by the training loader.

The dataset is not committed to this project. The model and visualizations are
generated locally by `python -m src.train`.
