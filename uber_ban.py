"""
Bangalore Rides — Full Analysis Script (Adapted for your CSV)
Saves all outputs (plots, cleaned CSV, summary CSV, PDF report) into a single folder:
    ./bangalore_analysis_output/
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------
# Config
# -----------------------
INPUT_PATH = r"C:\Users\Hariharan\Desktop\python\uber_project\bangalore_ride_data.csv"
OUT_DIR = r"C:\Users\Hariharan\Desktop\python\uber_project\bangalore_analysis_output"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv(INPUT_PATH)
data = df.copy()

# -----------------------
# Column mapping to expected format
# -----------------------
# Combine Date + Time into START_DATE
data['START_DATE'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')

# END_DATE: copy START_DATE (no duration info)
data['END_DATE'] = data['START_DATE']

# CATEGORY: Vehicle Type
data['CATEGORY'] = data['Vehicle Type']

# START and STOP locations
data['START'] = data['Pickup Location']
data['STOP'] = data['Drop Location']

# MILES: Ride Distance
data['MILES'] = pd.to_numeric(data['Ride Distance'], errors='coerce')

# PURPOSE: Booking Status
data['PURPOSE'] = data['Booking Status'].fillna('NOT')

# -----------------------
# Data cleaning & feature engineering
# -----------------------
data = data.dropna(subset=['START_DATE']).copy()

# Derived features
data['Date'] = data['START_DATE'].dt.date
data['Hour'] = data['START_DATE'].dt.hour
data['Month'] = data['START_DATE'].dt.month_name()
data['Year'] = data['START_DATE'].dt.year
data['DayOfWeek'] = data['START_DATE'].dt.day_name()
data['Duration_mins'] = (data['END_DATE'] - data['START_DATE']).dt.total_seconds() / 60.0

# Day period
data['DayPeriod'] = pd.cut(
    data['Hour'].fillna(-1),
    bins=[-1, 5, 11, 16, 20, 24],
    labels=['LateNight','Morning','Afternoon','Evening','Night'],
    include_lowest=True
)

# Remove negative miles
data = data[(data['MILES'] >= 0) | (data['MILES'].isna())]

# -----------------------
# Basic summaries & stats
# -----------------------
summary = {
    'total_rides': len(data),
    'unique_start_locations': data['START'].nunique(),
    'unique_stop_locations': data['STOP'].nunique(),
    'date_range': (data['START_DATE'].min(), data['START_DATE'].max()),
    'mean_distance_miles': float(data['MILES'].mean(skipna=True)),
    'median_distance_miles': float(data['MILES'].median(skipna=True)),
    'mean_duration_mins': float(data['Duration_mins'].mean(skipna=True)),
    'median_duration_mins': float(data['Duration_mins'].median(skipna=True))
}

purpose_counts = data['PURPOSE'].value_counts()
top_start = data['START'].value_counts().head(15)
top_stop = data['STOP'].value_counts().head(15)
monthly_counts = data.groupby('Month').size()
month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
monthly_counts = monthly_counts.reindex(month_order).fillna(0).astype(int)
yearly_counts = data['Year'].value_counts().sort_index()
hourly_counts = data.groupby('Hour').size().reindex(range(0,24), fill_value=0)
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday_counts = data['DayOfWeek'].value_counts().reindex(weekday_order).fillna(0).astype(int)
distances = data['MILES'].dropna()
durations = data['Duration_mins'].dropna()
daily_counts = data.groupby('Date').size().sort_index()
moving_avg_7d = daily_counts.rolling(window=7, min_periods=1).mean()

# -----------------------
# Save cleaned dataset + summary CSV
# -----------------------
cleaned_csv = os.path.join(OUT_DIR, "bangalore_cleaned_data.csv")
data.to_csv(cleaned_csv, index=False)

summary_df = pd.DataFrame({
    'metric': [
        'total_rides', 'unique_start_locations','unique_stop_locations',
        'date_range_start','date_range_end',
        'mean_distance_miles','median_distance_miles',
        'mean_duration_mins','median_duration_mins'
    ],
    'value': [
        summary['total_rides'],
        summary['unique_start_locations'],
        summary['unique_stop_locations'],
        summary['date_range'][0].strftime("%Y-%m-%d %H:%M:%S"),
        summary['date_range'][1].strftime("%Y-%m-%d %H:%M:%S"),
        summary['mean_distance_miles'],
        summary['median_distance_miles'],
        summary['mean_duration_mins'],
        summary['median_duration_mins']
    ]
})
summary_csv = os.path.join(OUT_DIR, "bangalore_summary_metrics.csv")
summary_df.to_csv(summary_csv, index=False)

# Detailed CSVs
top_start.to_csv(os.path.join(OUT_DIR, "top_start_locations.csv"), header=['count'])
top_stop.to_csv(os.path.join(OUT_DIR, "top_stop_locations.csv"), header=['count'])
monthly_counts.to_csv(os.path.join(OUT_DIR, "monthly_counts.csv"), header=['count'])
yearly_counts.to_csv(os.path.join(OUT_DIR, "yearly_counts.csv"), header=['count'])
hourly_counts.to_csv(os.path.join(OUT_DIR, "hourly_counts.csv"), header=['count'])
weekday_counts.to_csv(os.path.join(OUT_DIR, "weekday_counts.csv"), header=['count'])
daily_counts.to_csv(os.path.join(OUT_DIR, "daily_counts.csv"), header=['count'])

# -----------------------
# Plotting
# -----------------------
sns.set_style("whitegrid")
def savefig(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

# Purpose distribution
fig = plt.figure(figsize=(9,5))
ax = sns.countplot(y='PURPOSE', data=data, order=purpose_counts.index)
ax.set_title("Ride Purpose Distribution (Top categories)")
ax.set_xlabel("Count")
savefig(fig, "purpose_distribution.png")

# Day period counts
fig = plt.figure(figsize=(7,4))
ax = sns.countplot(x='DayPeriod', data=data, order=data['DayPeriod'].cat.categories)
ax.set_title("Rides by Day Period")
ax.set_xlabel("Day Period")
ax.set_ylabel("Count")
savefig(fig, "dayperiod_counts.png")

# Rides by Day of Week
fig = plt.figure(figsize=(9,5))
ax = sns.barplot(x=weekday_counts.index, y=weekday_counts.values)
ax.set_title("Rides by Day of Week")
ax.set_xlabel("Day")
ax.set_ylabel("Count")
plt.xticks(rotation=45)
savefig(fig, "weekday_counts.png")

# Hourly trend
fig = plt.figure(figsize=(9,4))
ax = sns.lineplot(x=hourly_counts.index, y=hourly_counts.values, marker='o')
ax.set_title("Hourly Ride Trend (0-23 hours)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Count")
savefig(fig, "hourly_trend.png")

# Monthly trend
fig = plt.figure(figsize=(10,4))
ax = sns.barplot(x=monthly_counts.index, y=monthly_counts.values)
ax.set_title("Monthly Ride Counts")
ax.set_xlabel("Month")
ax.set_ylabel("Count")
plt.xticks(rotation=45)
savefig(fig, "monthly_trend.png")

# Yearly trend
fig = plt.figure(figsize=(6,4))
ax = sns.barplot(x=yearly_counts.index.astype(str), y=yearly_counts.values)
ax.set_title("Yearly Ride Counts")
ax.set_xlabel("Year")
ax.set_ylabel("Count")
savefig(fig, "yearly_trend.png")

# Top pickup locations
fig = plt.figure(figsize=(8,6))
top_start.plot(kind='barh', color='tab:green', legend=False)
plt.gca().invert_yaxis()
plt.title("Top Pickup Locations (Top 15)")
plt.xlabel("Count")
savefig(fig, "top_pickup_locations.png")

# Top dropoff locations
fig = plt.figure(figsize=(8,6))
top_stop.plot(kind='barh', color='tab:orange', legend=False)
plt.gca().invert_yaxis()
plt.title("Top Dropoff Locations (Top 15)")
plt.xlabel("Count")
savefig(fig, "top_dropoff_locations.png")

# Distance distribution
fig = plt.figure(figsize=(8,4))
ax = sns.histplot(distances, bins=30, kde=True)
ax.set_title("Distribution of Trip Distances (MILES)")
ax.set_xlabel("Miles")
ax.set_ylabel("Frequency")
savefig(fig, "distance_distribution.png")

# Duration distribution
fig = plt.figure(figsize=(8,4))
ax = sns.histplot(durations, bins=30, kde=True)
ax.set_title("Distribution of Trip Duration (minutes)")
ax.set_xlabel("Duration (mins)")
ax.set_ylabel("Frequency")
savefig(fig, "duration_distribution.png")

# Distance vs Duration scatter (sample 2000)
if (not distances.empty) and (not durations.empty):
    fig = plt.figure(figsize=(7,6))
    ax = sns.scatterplot(x='MILES', y='Duration_mins', data=data.sample(min(2000, len(data)), random_state=1))
    ax.set_title("Distance vs Duration (sample)")
    ax.set_xlabel("Miles")
    ax.set_ylabel("Duration (mins)")
    savefig(fig, "distance_vs_duration.png")

# 7-day moving average
fig = plt.figure(figsize=(10,4))
ax = sns.lineplot(x=moving_avg_7d.index, y=moving_avg_7d.values)
ax.set_title("7-Day Moving Average of Daily Ride Counts")
ax.set_xlabel("Date")
ax.set_ylabel("Average Rides")
savefig(fig, "moving_average_7day.png")

# -----------------------
# PDF Report
# -----------------------
pdf_path = os.path.join(OUT_DIR, "bangalore_analysis_report.pdf")
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Bangalore Rides — Analysis Summary", fontsize=16, y=0.95)
    text_lines = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total rides analyzed: {summary['total_rides']}",
        f"Date range: {summary['date_range'][0].strftime('%Y-%m-%d')} to {summary['date_range'][1].strftime('%Y-%m-%d')}",
        f"Unique pickup locations: {summary['unique_start_locations']}",
        f"Unique dropoff locations: {summary['unique_stop_locations']}",
        "",
        "Distance (miles):",
        f"  - Mean: {summary['mean_distance_miles']:.2f}",
        f"  - Median: {summary['median_distance_miles']:.2f}",
        "",
        "Duration (minutes):",
        f"  - Mean: {summary['mean_duration_mins']:.2f}",
        f"  - Median: {summary['median_duration_mins']:.2f}",
        "",
        "Top 5 ride purposes:",
    ]
    for purpose, cnt in purpose_counts.head(5).items():
        text_lines.append(f"  - {purpose}: {int(cnt)} rides")
    text_lines.append("")

    plt.axis('off')
    y = 0.92
    for line in text_lines:
        plt.text(0.05, y, line, fontsize=10, transform=plt.gcf().transFigure)
        y -= 0.03
    pdf.savefig(fig)
    plt.close(fig)

    image_order = [
        "purpose_distribution.png","dayperiod_counts.png","weekday_counts.png",
        "hourly_trend.png","monthly_trend.png","yearly_trend.png",
        "top_pickup_locations.png","top_dropoff_locations.png",
        "distance_distribution.png","duration_distribution.png",
        "distance_vs_duration.png","moving_average_7day.png"
    ]
    for img in image_order:
        img_path = os.path.join(OUT_DIR, img)
        if os.path.exists(img_path):
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.axis('off')
            plt.imshow(plt.imread(img_path))
            pdf.savefig(fig)
            plt.close(fig)

# -----------------------
# Final output
# -----------------------
print("=== Bangalore Rides Analysis Complete ===")
print(f"Cleaned dataset: {cleaned_csv}")
print(f"Summary CSV: {summary_csv}")
print(f"PDF report: {pdf_path}")
print(f"Other outputs (plots + detailed CSVs) saved in: {OUT_DIR}")
