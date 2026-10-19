import pandas as pd
import config

#load raw adverse event reports
df = pd.read_json(config.RAW_DIR / "events_sample.jsonl", lines=True, convert_dates=False, dtype=False)

print("rows before:", len(df))

#inspect label and check for duplicates 
print(df["event_type"].value_counts(dropna=False))
print("duplicate mdr_report_key:", df["mdr_report_key"].duplicated().sum())
print("duplicate report_number:", df["report_number"].duplicated().sum())

#remove duplicate reports
df = df.drop_duplicates(subset="mdr_report_key")
print("rows after dedupe:", len(df))

#standardize the dates into correct format 
df["date_received"] = pd.to_datetime(
    df["date_received"].astype(str), format="%Y%m%d", errors="coerce"
)
print("unparseable dates:", df["date_received"].isna().sum())
df = df.dropna(subset=["date_received"])

df["year"] = df["date_received"].dt.year
df["month"] = df["date_received"].dt.month
print(df["year"].value_counts().sort_index())

#build binary label: 1=death or injury, 0=malfunction. Drop unlabeled rows (blank, other)
df = df[df["event_type"].isin(["Malfunction", "Injury", "Death"])].copy()
df["label"] = (df["event_type"] != "Malfunction").astype(int)

print("rows after dropping unlabeled:", len(df))
print("injury/death share overall:", round(df["label"].mean(), 4))
print(df.groupby("year")["label"].mean().round(4))

#flatten the nested fields and event-description fields
def get_device_field(devices, field):
    if isinstance(devices, list) and len(devices) > 0:
        return devices[0].get(field, "")
    return ""

df["product_code"] = df["device"].apply(lambda d: get_device_field(d, "device_report_product_code"))
df = df[df["product_code"].isin(config.PRODUCT_CODES)]
df["brand_name"] = df["device"].apply(lambda d: get_device_field(d, "brand_name"))
df["generic_name"] = df["device"].apply(lambda d: get_device_field(d, "generic_name"))
df["device_manufacturer"] = df["device"].apply(lambda d: get_device_field(d, "manufacturer_d_name"))
df["device_operator"] = df["device"].apply(lambda d: get_device_field(d, "device_operator"))
df["device_class"] = df["device"].apply(
    lambda d: d[0].get("openfda", {}).get("device_class", "") if isinstance(d, list) and len(d) > 0 else ""
)

def get_description(mdr_text):
    if not isinstance(mdr_text, list):
        return ""
    parts = [t.get("text", "") for t in mdr_text
             if t.get("text_type_code") == "Description of Event or Problem"]
    return " ".join(parts)

df["event_description"] = df["mdr_text"].apply(get_description)

new_cols = ["product_code", "brand_name", "generic_name", "device_manufacturer",
            "device_operator", "device_class", "event_description"]
print((df[new_cols] == "").mean().round(4))
print(df["product_code"].value_counts().head(10))

#handle missing values 
cat_cols = ["brand_name", "generic_name", "device_manufacturer", "device_operator", "device_class"]
for col in cat_cols:
    df[col] = df[col].replace("", "Unknown")
df["device_operator"] = df["device_operator"].replace({"I": "Unknown", "0": "Unknown"})
    
#flatten list columns that we plan to use as features
df["product_problems"] = df["product_problems"].apply(
    lambda x: "|".join(sorted({p for p in x if isinstance(p, str) and p})) if isinstance(x, list) else ""
)

#keep only the columns we need for modeling (and drop label leakage)
keep_cols = [
    "mdr_report_key", "report_number", "date_received", "year", "month",
    "label", "product_code", "brand_name", "generic_name",
    "device_manufacturer", "device_operator", "device_class",
    "report_source_code", "product_problems", "event_description",
]
clean = df[keep_cols].copy()

#save cleaned dataset 
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
clean.to_csv(config.PROCESSED_DIR / "events_clean.csv", index=False)

print("saved:", clean.shape)
print(clean.groupby("product_code")["generic_name"].agg(lambda s: s.value_counts().index[0]))
print(clean["device_operator"].value_counts().head(8))

#export a readable excel file for manual inspection
excel_path = config.PROCESSED_DIR / "events_clean_preview.xlsx"
summary = (
    clean.groupby(["year", "product_code"])["label"]
    .agg(reports="count", injury_death="sum")
    .reset_index()
)
summary["injury_death_share"] = (summary["injury_death"] / summary["reports"]).round(4)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    clean.sample(min(5000, len(clean)), random_state=42).to_excel(writer, sheet_name="sample", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

    ## Make the sample sheet easier to read
    ws = writer.sheets["sample"]
    ws.freeze_panes = "A2"
    for col in ws.columns:
        letter = col[0].column_letter
        ws.column_dimensions[letter].width = 60 if col[0].value == "event_description" else 22

print("excel saved:", excel_path)