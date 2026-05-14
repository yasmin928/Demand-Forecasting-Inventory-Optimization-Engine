from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# =========================
# Load Data
# =========================
df = pd.read_csv("data/processed/retail_store_inventory_cleaned.csv")
df['date'] = pd.to_datetime(df['date'])

# =========================
# Dashboard
# =========================
@app.route("/")
def dashboard():

    df_local = df.copy()

    # =========================
    # Get filters
    # =========================
    selected_store = request.args.get('store')
    selected_region = request.args.get('region')
    selected_category = request.args.get('category')
    selected_product = request.args.get('product')

    # =========================
    # STORE FILTER
    # =========================
    if selected_store:
        df_local = df_local[df_local['store_id'] == selected_store]

    # =========================
    # PRODUCT FILTER
    # =========================
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
        col = f"region_{selected_region}"
        if col in df_local.columns:
            df_local = df_local[df_local[col] == 1]

    # =========================
    # Inventory Status Logic
    # =========================
    def get_status(row):
        if row['inventory_level'] > row['units_sold'] * 1.5:
            return "Overstock"
        elif row['inventory_level'] < row['units_sold'] * 0.5:
            return "Stockout"
        else:
            return "Normal"

    df_local['status'] = df_local.apply(get_status, axis=1)

    # =========================
    # Counts
    # =========================
    status_counts = df_local['status'].value_counts().to_dict()

    # =========================
    # Table (per product)
    # =========================
    product_status = df_local.groupby('product_id')['status'] \
        .agg(lambda x: x.mode()[0]).reset_index()

    return render_template(
        "dashboard.html",
        data=product_status.to_dict(orient="records"),
        counts=status_counts,

        # dropdown values
        stores=sorted(df['store_id'].unique()),
        products=sorted(df['product_id'].unique()),
        categories=sorted(df['category'].unique()),
        regions=["North", "South", "West"],

        # keep selected values
        selected_store=selected_store,
        selected_product=selected_product,
        selected_category=selected_category,
        selected_region=selected_region
    )

# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)