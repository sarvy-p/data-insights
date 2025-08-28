APP_TITLE = "Procurement & Shipment Insights"
EXPECTED_DATE_COLS = ["po_date", "planned_ship_date", "actual_ship_date", "planned_eta", "actual_eta"]
EXPECTED_BASE_COLS = [
    "po_number", "supplier", "origin_country", "destination_country", "lane", "mode",
    "incoterm", "status", "quantity", "unit_price", "freight_cost", "duty_cost", "shipment_id"
]
EXPECTED_COLS = EXPECTED_BASE_COLS + EXPECTED_DATE_COLS

KPI_FORMATS = {
    "total_shipments": "{:,}",
    "on_time_rate": "{:.1f}%",
    "avg_lead": "{:.1f}",
    "late_rate": "{:.1f}%",
    "total_spend": "{:,.0f}",
}
