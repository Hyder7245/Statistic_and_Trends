
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""# Importing the Dataset"""

df = pd.read_csv('/content/drive/MyDrive/Datasets/avocado.csv')

# Define the plotting function
def plot_avg_price_trend_multiple_regions(data, regions):
    """
    Plots a line chart showing the trend of average avocado prices over time for multiple regions.

    Parameters:
    data (DataFrame): Filtered DataFrame containing data for multiple regions.
    regions (list): List of region names to plot.

    Returns:
    None
    """
    plt.figure(figsize=(12, 8))

    # Loop through each region and plot
    for region in regions:
        # Filter data for each specific region
        region_specific_data = data[data['region'] == region]
        avg_price_over_time = region_specific_data.groupby('Date')['AveragePrice'].mean()

        # Plot the line for the region
        plt.plot(avg_price_over_time, label=region)

    # Plot styling
    plt.title('Average Avocado Price Trend Over Time by Region')
    plt.xlabel('Date')
    plt.ylabel('Average Price ($)')
    plt.legend(title='Region')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_avg_price_by_region(data):
    """
    Plots a bar chart showing the average price of avocados by region, with values on top of each bar
    and different colors for each bar.

    Parameters:
    data (DataFrame): The avocado dataset.

    Returns:
    None
    """

    # Generate a color map with a unique color for each bar
    colors = plt.cm.viridis(range(len(data)))

    # Plotting
    plt.figure(figsize=(12, 8))
    bars = plt.bar(data.index, data.values, color=colors)
    plt.title('Top 5 Regions by Average Avocado Price')
    plt.xlabel('Region')
    plt.ylabel('Average Price ($)')
    plt.xticks(rotation=90)

    # Display the values on top of each bar
    for bar, value in zip(bars, data.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

# Define the function to plot the price distribution of the top 5 regions
def plot_price_distribution_top5_regions(data):
    """
    Plots a box plot showing the distribution of avocado prices for the top 5 regions with highest average prices.

    Parameters:
    data (DataFrame): The filtered avocado dataset for the top 5 regions.

    Returns:
    None
    """
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='region', y='AveragePrice', data=data, hue='region', palette='viridis', legend=False)
    plt.title('Price Distribution of Avocados in Top 5 Regions')
    plt.xlabel('Region')
    plt.ylabel('Average Price ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# List of regions to plot
regions = ['Albany', 'Chicago', 'LosAngeles']  # Replace with the desired list of regions

# Filter data for the specified regions
region_data = df[df['region'].isin(regions)]

# Convert the Date column in region_data with .loc to avoid the warning
region_data.loc[:, 'Date'] = pd.to_datetime(region_data['Date'])

# Call the function with the filtered data and region list
plot_avg_price_trend_multiple_regions(region_data, regions)

# Call the function with the avocado data
# Calculate the average price by region and select the top 5 regions
avg_price_by_region = df.groupby('region')['AveragePrice'].mean().nlargest(5)
plot_avg_price_by_region(avg_price_by_region)

# Calculate the top 5 regions based on average price
top5_regions = df.groupby('region')['AveragePrice'].mean().nlargest(5).index.tolist()

# Filter the dataset to include only these top 5 regions
top5_data = df[df['region'].isin(top5_regions)]
# Call the function with the filtered data for the top 5 regions
plot_price_distribution_top5_regions(top5_data)