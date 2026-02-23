import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
df = pd.read_csv(r'C:\Users\gunda\Digital Learning Intelligence\digital_learning_analytics_100k.csv')

print(f"Total Learners: {df.shape[0]} | Total Features: {df.shape[1]}")
df.head()
# Checking for data gaps
print("Missing Value Audit:\n", df.isnull().sum())

# Date formatting for activity analysis
df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])
df['last_activity_date'] = pd.to_datetime(df['last_activity_date'])

# Technical summary
df.info()
education_counts = df['education_level'].value_counts()

plt.figure(figsize=(10, 8))
colors = ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5']

plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', 
        startangle=140, colors=colors, pctdistance=0.85, explode=[0.03]*len(education_counts))

# Converting Pie to Donut for a modern look
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Learner Distribution by Education Level', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.show()
# Aggregating average efficiency per platform
platform_efficiency = df.groupby('mooc_platform')['learning_efficiency_score'].mean().sort_values()

plt.figure(figsize=(12, 7))

# Using a professional color palette to distinguish platforms
colors = ['#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1']

# Creating the bar chart
bars = plt.bar(platform_efficiency.index, platform_efficiency.values, color=colors)

# Adding data labels on top of each bar for clarity
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), 
             va='bottom', ha='center', fontsize=10, fontweight='bold')

# CRITICAL FIX: Adjusting Y-axis to zoom in on the differences
min_val = platform_efficiency.min() * 0.95  # Starts at 5% below the lowest value
max_val = platform_efficiency.max() * 1.05  # Ends at 5% above the highest value
plt.ylim(min_val, max_val)

plt.title('Performance Variance: Average Learning Efficiency by Platform', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Mean Efficiency Score', fontsize=12)
plt.xlabel('MOOC Platform', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.show()
# Calculating mean daily minutes per category
app_usage = df.groupby('app_category')['daily_app_minutes'].mean().sort_values(ascending=True)

plt.figure(figsize=(12, 8))

# Using a distinct color palette to emphasize the ranking
colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1']

# Creating a horizontal bar chart
bars = plt.barh(app_usage.index, app_usage.values, color=colors[-len(app_usage):])

# Adding precise data labels to the end of each bar
for bar in bars:
    xval = bar.get_width()
    plt.text(xval + 0.2, bar.get_y() + bar.get_height()/2, round(xval, 2), 
             va='center', ha='left', fontsize=10, fontweight='bold', color='#333')

# ADJUSTING THE SCALE: Zooming in on the x-axis to show variance
x_min = app_usage.min() * 0.98  # Start 2% below the minimum value
x_max = app_usage.max() * 1.05  # End 5% above the maximum value
plt.xlim(x_min, x_max)

plt.title('Daily Engagement Depth by App Category', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Average Daily Minutes', fontsize=12)
plt.ylabel('App Category', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
# Calculating mean engagement per learning path
path_metrics = df.groupby('learning_path_type')['engagement_consistency'].mean().sort_values()

plt.figure(figsize=(10, 6))

# Plotting a simple, clean bar chart
plt.bar(path_metrics.index, path_metrics.values, color='#3498db', width=0.6)

# Adding the exact values on top of each bar
for i, v in enumerate(path_metrics.values):
    plt.text(i, v + (v * 0.01), f'{v:.3f}', ha='center', fontweight='bold')

# Adjusting the Y-axis scale to make the differences visible
plt.ylim(path_metrics.min() * 0.95, path_metrics.max() * 1.05)

plt.title('Engagement Consistency by Learning Path', fontsize=14, fontweight='bold')
plt.ylabel('Consistency Score')
plt.xlabel('Learning Path Type')
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.show()
numeric_cols = ['digital_literacy_score', 'daily_app_minutes', 'app_completion_rate', 
                'in_app_quiz_score', 'mastery_score', 'learning_efficiency_score']
corr = df[numeric_cols].corr()

plt.figure(figsize=(12, 8))
plt.imshow(corr, cmap='RdBu_r', interpolation='nearest')
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Learning Behavioral Correlation Matrix', fontsize=16, fontweight='bold')
plt.show()