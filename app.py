from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ============================================================
# 1. LOAD RAW DATA  (same as notebook)
# ============================================================
df_raw = pd.read_csv("data/processed/retail_store_inventory_cleaned.csv")
df_raw['date'] = pd.to_datetime(df_raw['date'])

# ============================================================
# 2. LOAD SAVED ARTEFACTS
# ============================================================
xgb_model      = joblib.load("models/xgboost_model.pkl")   if os.path.exists("models/xgboost_model.pkl")   else None
lgb_model      = joblib.load("models/lightgbm_model.pkl")  if os.path.exists("models/lightgbm_model.pkl")  else None
label_encoders = joblib.load("models/label_encoders.pkl")  if os.path.exists("models/label_encoders.pkl")  else {}
saved_features = joblib.load("models/features.pkl")        if os.path.exists("models/features.pkl")        else None

# ============================================================
# 3. REPLICATE NOTEBOOK ENCODING PIPELINE  (run once at start)
#    This produces the exact same encoded DataFrame the notebook used,
#    so the 80/20 split indices are identical.
# ============================================================
def _encode(df_in):
    """Apply label-encoding + one-hot exactly as the notebook did."""
    df_enc = df_in.copy()

    # Drop date before encoding (notebook does this before splitting)
    if 'date' in df_enc.columns:
        df_enc = df_enc.drop(columns=['date'])

    # Label-encode high-cardinality columns with the SAVED encoders
    for col, le in label_encoders.items():
        if col in df_enc.columns:
            known = set(le.classes_)
            df_enc[col] = df_enc[col].apply(lambda v: v if v in known else le.classes_[0])
            df_enc[col] = le.transform(df_enc[col])

    # One-hot encode low-cardinality columns
    onehot_cols = ['category', 'region', 'weather_condition', 'seasonality']
    present     = [c for c in onehot_cols if c in df_enc.columns]
    df_enc      = pd.get_dummies(df_enc, columns=present, drop_first=True, dtype=int)

    return df_enc


# Full encoded frame (no date column)
df_encoded = _encode(df_raw)

# ── Separate features & target ──────────────────────────────
TARGET = 'units_sold'
x_all  = df_encoded.drop(columns=[TARGET], errors='ignore')
y_all  = df_encoded[TARGET]

# Align columns to what the model was trained on
if saved_features:
    train_feature_cols = [f for f in saved_features if f not in (TARGET, 'date')]
    for col in train_feature_cols:
        if col not in x_all.columns:
            x_all[col] = 0
    x_all = x_all[[c for c in train_feature_cols if c in x_all.columns]]

# ── 80 / 20 time-series split  (exact same as notebook) ─────
split_index = int(len(df_encoded) * 0.8)
x_test      = x_all.iloc[split_index:]          # rows the model NEVER saw
y_test      = y_all.iloc[split_index:]
test_indices = x_test.index                     # original row indices

# ── Attach original readable columns back for display ────────
# (store_id / product_id before label-encoding, inventory_level)
df_test_display = df_raw.loc[test_indices, ['store_id', 'product_id',
                                             'inventory_level',
                                             'units_sold']].copy()

# ── Pre-compute predictions for BOTH models once ─────────────
test_preds = {}
for name, model in [('xgboost', xgb_model), ('lightgbm', lgb_model)]:
    if model is not None:
        preds = model.predict(x_test)
        test_preds[name] = preds

# ============================================================
# 4. HELPERS
# ============================================================
def get_status(row):
    if row['inventory_level'] > row['units_sold'] * 1.5:
        return 'Overstock'
    elif row['inventory_level'] < row['units_sold'] * 0.5:
        return 'Stockout'
    return 'Normal'


def build_results(model_name):
    """Return a DataFrame with predictions + risk flags for the test set."""
    preds = test_preds[model_name]

    df_r = df_test_display.copy().reset_index(drop=True)
    df_r['predicted_units_sold'] = np.round(preds, 1)
    df_r['error']                = np.round(np.abs(df_r['units_sold'] - df_r['predicted_units_sold']), 1)
    df_r['stockout_risk']        = df_r['predicted_units_sold'] > df_r['inventory_level']
    df_r['overstock_risk']       = df_r['inventory_level'] > (1.5 * df_r['predicted_units_sold'])

    def _risk(row):
        if row['stockout_risk']:  return 'Stockout'
        if row['overstock_risk']: return 'Overstock'
        return 'Normal'

    df_r['risk'] = df_r.apply(_risk, axis=1)
    return df_r


def compute_metrics(df_r):
    """MAE, RMSE, R² on the full (unfiltered) test results."""
    y_true = df_r['units_sold']
    y_pred = df_r['predicted_units_sold']
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = round(1 - ss_res / ss_tot, 4) if ss_tot != 0 else 0.0
    return round(mae, 2), round(rmse, 2), r2


# ============================================================
# 5. ROUTE — Inventory Status  (Tab 1)
# ============================================================
@app.route("/")
def dashboard():
    import json

    df_local = df_raw.copy()

    selected_store    = request.args.get('store')
    selected_region   = request.args.get('region')
    selected_category = request.args.get('category')
    selected_product  = request.args.get('product')

    # =========================
    # STORE FILTER
    # =========================
    if selected_store:
        df_local = df_local[df_local['store_id'] == selected_store]
    if selected_product:
        df_local = df_local[df_local['product_id'] == selected_product]

    # =========================
    # CATEGORY FILTER
    # =========================
    if selected_category:
        df_local = df_local[df_local['category'] == selected_category]

    # =========================
    # REGION FILTER (one-hot)
    # =========================
    if selected_region:
        region_col = f"region_{selected_region}"
        if region_col in df_local.columns:
            df_local = df_local[df_local[region_col] == 1]

    df_local['status'] = df_local.apply(get_status, axis=1)
    status_counts      = df_local['status'].value_counts().to_dict()

    product_status = (
        df_local.groupby('product_id')['status']
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )

    # ── Chart 1: Status breakdown (donut) ────────────────────
    chart_status = {
        'labels': ['Normal', 'Overstock', 'Stockout'],
        'values': [
            int(status_counts.get('Normal',    0)),
            int(status_counts.get('Overstock', 0)),
            int(status_counts.get('Stockout',  0)),
        ]
    }

    # ── Chart 2: Inventory vs Units Sold (top 15 products) ───
    inv_vs_sold = (
        df_local.groupby('product_id')[['inventory_level', 'units_sold']]
        .mean()
        .round(1)
        .reset_index()
        .sort_values('inventory_level', ascending=False)
        .head(15)
    )
    chart_inv_sold = {
        'labels':    inv_vs_sold['product_id'].tolist(),
        'inventory': inv_vs_sold['inventory_level'].tolist(),
        'sold':      inv_vs_sold['units_sold'].tolist(),
    }

    # ── Chart 3: Stock level over time (daily avg) ───────────
    time_series = (
        df_local.groupby('date')['inventory_level']
        .mean()
        .round(1)
        .reset_index()
        .sort_values('date')
        .tail(60)                          # last 60 days for readability
    )
    chart_time = {
        'labels': time_series['date'].dt.strftime('%Y-%m-%d').tolist(),
        'values': time_series['inventory_level'].tolist(),
    }

    # ── Chart 4: Top overstocked products ────────────────────
    top_overstock = (
        df_local[df_local['status'] == 'Overstock']
        .groupby('product_id')
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name='count')
    )
    chart_overstock = {
        'labels': top_overstock['product_id'].tolist(),
        'values': top_overstock['count'].tolist(),
    }

    # ── Chart 5: Top stockout-risk products ──────────────────
    top_stockout = (
        df_local[df_local['status'] == 'Stockout']
        .groupby('product_id')
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name='count')
    )

    if top_stockout.empty:
        chart_stockout = {
            'labels': ['No stockout data'],
            'values': [0],
        }
    else:
        chart_stockout = {
            'labels': top_stockout['product_id'].tolist(),
            'values': top_stockout['count'].tolist(),
        }

    # ── Chart 6: Category breakdown (avg inventory by category)
    cat_breakdown = (
        df_local.groupby('category')[['inventory_level', 'units_sold']]
        .mean()
        .round(1)
        .reset_index()
        .sort_values('inventory_level', ascending=False)
    )
    chart_category = {
        'labels':    cat_breakdown['category'].tolist(),
        'inventory': cat_breakdown['inventory_level'].tolist(),
        'sold':      cat_breakdown['units_sold'].tolist(),
    }

    return render_template(
        "dashboard.html",
        active_tab="dashboard",
        data=product_status.to_dict(orient="records"),
        counts=status_counts,
        stores=sorted(df_raw['store_id'].unique()),
        products=sorted(df_raw['product_id'].unique()),
        categories=sorted(df_raw['category'].unique()),
        regions=["North", "South", "West"],
        selected_store=selected_store,
        selected_product=selected_product,
        selected_category=selected_category,
        selected_region=selected_region,
        # chart data (JSON-serialised so Jinja passes them cleanly to JS)
        chart_status   = json.dumps(chart_status),
        chart_inv_sold = json.dumps(chart_inv_sold),
        chart_time     = json.dumps(chart_time),
        chart_overstock= json.dumps(chart_overstock),
        chart_stockout = json.dumps(chart_stockout),
        chart_category = json.dumps(chart_category),
    )


# ============================================================
# 6. ROUTE — Model Predictions  (Tab 2)
# ============================================================
@app.route("/predictions")
def predictions():
    selected_model    = request.args.get('model', 'lightgbm')
    selected_store    = request.args.get('store', '')
    selected_product  = request.args.get('product', '')
    selected_category = request.args.get('category', '')
    selected_risk     = request.args.get('risk', '')
    page              = int(request.args.get('page', 1))
    per_page          = 50

    # Guard: models not loaded
    if not test_preds:
        return render_template(
            "dashboard.html",
            active_tab="predictions",
            error="Model files not found. Make sure models/ directory contains the .pkl files.",
            stores=sorted(df_raw['store_id'].unique()),
            products=sorted(df_raw['product_id'].unique()),
            categories=sorted(df_raw['category'].unique()),
            selected_model=selected_model,
            selected_store=selected_store,
            selected_product=selected_product,
            selected_category=selected_category,
            selected_risk=selected_risk,
            pred_rows=[], summary={}, total_pages=0, page=page,
        )

    if selected_model not in test_preds:
        selected_model = list(test_preds.keys())[0]

    # ── Build results on the CORRECT test split ───────────────
    df_results = build_results(selected_model)

    # ── Metrics are ALWAYS computed on the full test set ─────
    #    (not on the filtered view — matches your notebook output)
    mae, rmse, r2 = compute_metrics(df_results)

    summary_base = {
        'total_test': len(df_results),
        'stockout':   int(df_results['stockout_risk'].sum()),
        'overstock':  int(df_results['overstock_risk'].sum()),
        'normal':     int((df_results['risk'] == 'Normal').sum()),
        'mae':        mae,
        'rmse':       rmse,
        'r2':         r2,
    }

    # ── Apply display filters (doesn't affect metrics) ────────
    df_view = df_results.copy()
    if selected_store:
        df_view = df_view[df_view['store_id'] == selected_store]
    if selected_product:
        df_view = df_view[df_view['product_id'] == selected_product]
    if selected_category:
        # category was one-hot encoded; filter from raw data indices
        cat_indices = df_raw[df_raw['category'] == selected_category].index
        cat_indices_in_test = [i for i in cat_indices if i in test_indices]
        # map original indices to reset_index positions
        reset_map = {orig: pos for pos, orig in enumerate(test_indices)}
        positions = [reset_map[i] for i in cat_indices_in_test if i in reset_map]
        df_view = df_view.iloc[positions]
    if selected_risk == 'stockout':
        df_view = df_view[df_view['stockout_risk']]
    elif selected_risk == 'overstock':
        df_view = df_view[df_view['overstock_risk']]

    summary = {**summary_base, 'total': len(df_view)}

    # ── Pagination ────────────────────────────────────────────
    total_pages = max(1, (len(df_view) + per_page - 1) // per_page)
    page        = max(1, min(page, total_pages))
    df_page     = df_view.iloc[(page - 1) * per_page: page * per_page]

    return render_template(
        "dashboard.html",
        active_tab="predictions",
        stores=sorted(df_raw['store_id'].unique()),
        products=sorted(df_raw['product_id'].unique()),
        categories=sorted(df_raw['category'].unique()),
        selected_model=selected_model,
        selected_store=selected_store,
        selected_product=selected_product,
        selected_category=selected_category,
        selected_risk=selected_risk,
        pred_rows=df_page.to_dict(orient='records'),
        summary=summary,
        total_pages=total_pages,
        page=page,
    )


# ============================================================
# 7. RUN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
