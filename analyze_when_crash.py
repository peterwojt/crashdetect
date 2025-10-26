import pandas as pd
import matplotlib.pyplot as plt

# --- Load timestamp log ---
# Must have columns: filename,timestamp
log_path = "traffic_cam_videos/download_log.csv"  # adjust if needed
df = pd.read_csv(log_path)

# --- Load crash list ---
# First column is filename, rest ignored
crash_path = "traffic_cam_videos/crash_log.csv"  # adjust if needed
crash_df = pd.read_csv(crash_path, header=None)
crash_df = crash_df.rename(columns={0: "filename"})

# --- Normalize filenames ---
df["filename"] = df["filename"].astype(str).str.strip()
crash_df["filename"] = crash_df["filename"].astype(str).str.strip()

# --- Merge crash list with timestamps ---
merged = pd.merge(crash_df[["filename"]], df, on="filename", how="inner")

# --- Convert timestamps to datetime (UTC) ---
merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
merged = merged.dropna(subset=["timestamp"])

# --- Extract hour (UTC) ---
merged["hour"] = merged["timestamp"].dt.hour

# --- Print all crash timestamps ---
print("🕒 Crash timestamps (UTC):")
print(merged[["filename", "timestamp"]].sort_values("timestamp").to_string(index=False))

# --- Count crashes per hour ---
crash_by_hour = merged["hour"].value_counts().sort_index()

# --- Summary ---
if not crash_by_hour.empty:
    peak_hour = crash_by_hour.idxmax()
    print(f"\n🔥 Most crashes occur around {peak_hour:02d}:00 UTC "
          f"with {crash_by_hour.max()} crashes.")
else:
    print("\n⚠️ No matching crash filenames found in the log file.")

# --- Plot ---
if not crash_by_hour.empty:
    plt.figure(figsize=(8, 4))
    crash_by_hour.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Crashes by Hour (UTC)")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Number of Crashes")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
