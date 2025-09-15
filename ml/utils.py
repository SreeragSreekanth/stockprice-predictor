# ml/utils.py

import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent Tkinter errors
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure static folder exists
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

def make_plot(result):
    """
    Generates and saves a plot of actual prices, predicted test prices, and future forecast.
    Returns the filename of the saved PNG image.
    """
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(result["dates"], result["real"], label="Actual Price")
    ax.plot(result["dates"], result["preds_aligned"], label="Predicted (test)")
    ax.plot(result["future_dates"], result["future_preds"], marker='o', linestyle='--', label="Forecast (future)")

    ax.set_title(f"{result['symbol']} - Actual / Predicted / Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()

    # Save figure
    fname = f"{result['symbol'].replace('.', '_')}_{int(datetime.now().timestamp())}.png"
    fpath = os.path.join(STATIC_DIR, fname)
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)  # Close to free memory

    return fname
