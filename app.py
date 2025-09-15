from flask import Flask, request, render_template
from ml.pipeline import run_pipeline
from ml.utils import make_plot
from ml.search_ticker import get_ticker_from_name  # new function

app = Flask(__name__)

DEFAULT_EPOCHS = 15
DEFAULT_LOOKBACK = 60

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get("company", "").strip()
    symbol = get_ticker_from_name(company)
    if not symbol:
        return render_template("result.html", error=f"Company '{company}' not found. Try another.")

    future_days = int(request.form.get("future_days", 7))

    try:
        result = run_pipeline(symbol,
                              lookback=DEFAULT_LOOKBACK,
                              epochs=DEFAULT_EPOCHS,
                              future_days=future_days)
    except Exception as e:
        return render_template("result.html", error="Not enough historical data for this stock.")

    imgfile = make_plot(result)
    preds = [{"date": str(d.date()), "price": float(round(p, 4))}
             for d, p in zip(result["future_dates"], result["future_preds"])]

    return render_template("result.html",
                           symbol=result["symbol"],
                           imgfile=imgfile,
                           predictions=preds,
                           model_file=result["model_file"])


if __name__ == "__main__":
    app.run(debug=True)
