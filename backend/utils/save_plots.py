import matplotlib.pyplot as plt
import os

# Function to save the plot with standardized naming
def save_plot(fig, model, pollutant, output_dir="backend/weather_data_files/cleaned_plots"):
    """
    Saves a plot to a standardized location and file name.

    Parameters:
    - fig: The plot figure to save (matplotlib figure)
    - model: The model used to generate the forecast (e.g., 'prophet', 'sarima', 'xgboost')
    - pollutant: The weather variable being predicted (e.g., 'Max_Temperature', 'Min_Temperature')
    - output_dir: The directory to save the plot in (default is 'cleaned_plots')
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a standardized filename for the plot
    plot_filename = f"{model}_{pollutant}_forecast.png"
    plot_path = os.path.join(output_dir, plot_filename)

    # Save the plot
    fig.savefig(plot_path)
    
    # Close the plot to release memory
    plt.close(fig)

    # Print confirmation message
    print(f"Saved plot to {plot_path}")

# Example usage:
# Assuming 'fig' is the figure object from any of the models (Prophet, SARIMA, XGBoost)
# save_plot(fig, "prophet", "Max_Temperature")
