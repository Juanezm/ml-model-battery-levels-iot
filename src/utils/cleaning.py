import datetime
import imgkit
import os
import pandas as pd
import seaborn as sns
import numpy as np
import requests

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt


def df_to_image(df, filename):
    """
    Convert a pandas DataFrame into a styled PNG image.

    Args:
      df (pandas.DataFrame): The DataFrame to convert.
      filename (str): The base name of the files to create. The HTML file will be named
          '{filename}.html' and the image file will be named '{filename}.png'.
    """
    # Style the DataFrame
    styled = df.head(10).style.set_table_styles(
        [dict(selector="tr:nth-of-type(odd) td", props=[("background", "#eee")]),
         dict(selector="tr:nth-of-type(even) td", props=[("background", "white")])])

    # Save the styled DataFrame to an HTML file
    html_file = f'{filename}.html'
    styled.to_html(html_file)

    # Convert the HTML file to a PNG image
    imgkit.from_file(html_file, f'{filename}.png')

    # Delete the HTML file
    os.remove(html_file)


def plot_variable_counts(dataframe, filename):
    """
    Plots a chart showing the number of rows for each type of variable in the dataframe.

    Args:
      dataframe (pd.DataFrame): The input dataframe containing the data.
      filename (str): The base name of the files to create. The HTML file will be named
      '{filename}.html' and the image file will be named '{filename}.png'.
    Returns:
      None

    """

    # Count the number of rows for each type of variable
    variable_counts = dataframe['variable'].value_counts()

    # Plot the variable counts
    ax = variable_counts.plot(kind='bar')

    # Set the plot title and labels
    plt.title('Number of Rows per Variable')
    plt.xlabel('Variable')
    plt.ylabel('Count')

    # Add text annotations for the count values
    for i, count in enumerate(variable_counts):
        ax.text(i, count + 1, str(count), ha='center', va='bottom')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


def convert_unix_to_datetime(df, date_column):
    """
    Converts Unix timestamps to datetime format in a specified DataFrame column.

    This function iterates over a DataFrame column and checks if each entry is a Unix
    timestamp (a string of length 10). If it is, the function converts the timestamp
    to a datetime object.

    Args:
        df (pd.DataFrame): The DataFrame containing the date column.
        date_column (str): The name of the column with the dates to convert.

    Returns:
        pd.DataFrame: The DataFrame with the converted date column.
    """
    for i in range(len(df)):
        if isinstance(df[date_column][i], str) and len(df[date_column][i]) == 10:
            df.loc[i, date_column] = datetime.datetime.fromtimestamp(int(df[date_column][i]))
    return df


def outrange_percentage_by_variable(dataframe, null_threshold):
    """
    Calculates and prints the percentage of false entries and null values by variable,
    and returns a string with the variables where the percentage of null values exceeds a given threshold.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing the data.
        null_threshold (float): The threshold for the percentage of null values.

    Returns:
        str: A string with the percentages of non-null and null values by variable and the variables
             with a high percentage of null values.
        list: The list of variables with a high percentage of null values.
    """
    # Check if 'range' column contains tuples
    if not isinstance(dataframe['range'].iloc[0], tuple):
        # Convert the range column to interval tuples
        dataframe['range'] = dataframe['range'].apply(convert_range_string_to_interval)

    # Check if values are within the specified range
    dataframe.loc[:, 'within_range'] = dataframe.apply(
        lambda row: row['range'][0] <= row['value'] <= row['range'][1] if row['range'] is not None else False, axis=1)

    # Filter the dataframe by rows where within_range is False
    filtered_data = dataframe[dataframe['within_range'] == False]

    # Group the filtered data by variable and count the number of rows
    count_by_variable = filtered_data.groupby('variable')['within_range'].count()

    # Calculate the percentage of false entries by variable
    total_by_variable = dataframe.groupby('variable')['within_range'].count()
    percentage_by_variable = (count_by_variable / total_by_variable).fillna(0) * 100

    # Calculate the percentage of null values by variable
    null_count_by_variable = dataframe['value'].isnull().groupby(dataframe['variable']).sum()
    null_percentage_by_variable = (null_count_by_variable / total_by_variable) * 100

    # Create a string with the percentage of non-null and null values by variable
    percentages_string = ""
    for var in percentage_by_variable.index:
        percentages_string += f"{var}: {percentage_by_variable[var]:.2f}% (null: {null_percentage_by_variable[var]:.2f}%)\n"

    # Get a list of variables with a high percentage of null values
    high_null_vars = null_percentage_by_variable[null_percentage_by_variable > null_threshold].index.tolist()

    # Append the variables with a high percentage of null values to the string
    percentages_string += "\nVariables with more than 5% null values: " + ', '.join(high_null_vars)

    return percentages_string, high_null_vars


def print_boxplots(dataframe: pd.DataFrame, filename):
    """
    Creates a grid of box plots for each unique variable in the dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing the data.
        filename (str): The base name of the files to create. The HTML file will be named
      '{filename}.html' and the image file will be named '{filename}.png'.
    Returns:
        None: Displays the box plots.

    """

    # Get the unique values in the variable column
    unique_variables = dataframe['variable'].unique()

    # Calculate the number of rows and columns for the subplots grid
    num_rows = (len(unique_variables) + 3) // 4  # Round up to the nearest multiple of 4
    num_cols = min(len(unique_variables), 4)

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 10))

    # Loop through each unique variable and create a box plot
    for i, variable in enumerate(unique_variables):
        # Filter the data by variable
        variable_data = dataframe[dataframe['variable'] == variable]

        # Calculate the subplot coordinates
        row = i // num_cols
        col = i % num_cols

        # Create the box plot in the current subplot
        sns.boxplot(x=variable_data['variable'], y=variable_data['value'], ax=axes[row, col])
        axes[row, col].set_title(variable)

    # Hide any unused subplots
    for i in range(len(unique_variables), num_rows * num_cols):
        axes.flat[i].set_visible(False)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


def convert_range_string_to_interval(range_string):
    """
    Converts a range in string format to a tuple interval.

    This function accepts a range string in the format "[a, b]" and returns a tuple (a, b).
    If the input is NaN or not in the expected format, the function returns None.

    Args:
        range_string (str): The range in string format.

    Returns:
        tuple: The range as a tuple (a, b), where a and b are either int or float.
        If the input is NaN or not in the expected format, returns None.
    """
    if pd.isna(range_string) or range_string == 'NaN' or range_string == 'nan':
        return None
    # Remove the square brackets
    range_string = range_string.strip('[]')
    # Split the string by comma
    range_values = range_string.split(',')
    # Convert the values to float or int
    range_values = [float(x) if '.' in x else int(x) for x in range_values]
    # Create a tuple for the interval
    range_interval = tuple(range_values)
    return range_interval


def trim_sensor_data(df):
    """
    Trim the sensor data by removing values outside the specified range for each variable.

    Args:
        df (pd.DataFrame): The input dataframe containing the data.

    Returns:
        pd.DataFrame: The dataframe with trimmed sensor data.
    """

    # Get unique sensor variables
    variables = df['variable'].unique()

    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Iterate over each sensor variable
    for var in variables:
        # Get the range for this sensor variable
        sensor_range = df[df['variable'] == var]['range'].iloc[0]

        # Check if the range is not None before proceeding
        if sensor_range is not None:
            # Trim the data by removing values outside the range
            sensor_data = df_copy[df_copy['variable'] == var]
            trimmed_data = sensor_data[
                (sensor_data['value'] >= sensor_range[0]) & (sensor_data['value'] <= sensor_range[1])]

            # Update the 'value' column in the copied dataframe
            df_copy.loc[df_copy['variable'] == var, 'value'] = trimmed_data['value']

    # Update the 'within_range' column
    df_copy.loc[:, 'within_range'] = df_copy.apply(
        lambda row: row['range'][0] <= row['value'] <= row['range'][1] if row['range'] is not None else False, axis=1)

    return df_copy


def impute_missing_values(dataframe, columns_of_interest, high_null_vars, max_iter=10, random_state=0):
    """
    Imputes missing values in specified columns of a dataframe using IterativeImputer.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing the data.
        columns_of_interest (list of str): The names of the columns in which to impute missing values.
        high_null_vars (list of str): The names of the columns with a high percentage of null values.
        max_iter (int, optional): The maximum number of imputation rounds to perform before returning the imputations.
            Defaults to 10.
        random_state (int, optional): The seed of the pseudo random number generator to use when shuffling the data.
            Defaults to 0.

    Returns:
        pd.DataFrame: The dataframe with missing values imputed.
    """

    # Select columns of interest
    df_subset = dataframe[columns_of_interest]

    # Initialize the imputer
    imp = IterativeImputer(max_iter=max_iter, random_state=random_state)

    # Fit the imputer and transform the dataframe subset
    df_imputed_subset = imp.fit_transform(df_subset)

    # The output is a numpy array, convert it back to a dataframe
    df_imputed_subset = pd.DataFrame(df_imputed_subset, columns=df_subset.columns, index=df_subset.index)

    # Merge the imputed subset back into the original dataframe
    for variable in high_null_vars:
        dataframe[variable] = df_imputed_subset[variable]

    return dataframe

def plot_sensors(df, output_filename):
    """
    Generates subplots for each month from a dataframe and saves them as a single PNG file using matplotlib.

    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        output_filename (str): The name of the output PNG file.
    """
    # Group the dataframe by month
    df_monthly = df.groupby(pd.Grouper(freq='M'))

    num_months = len(df_monthly)

    # Create the figure and subplots
    fig, axes = plt.subplots(nrows=num_months, ncols=1, figsize=(30, 6 * num_months))

    # Define a color palette for the lines
    color_palette = sns.color_palette("Set2")

    # Flatten the axes array
    axes = np.array(axes).flatten()

    for i, (_, data) in enumerate(df_monthly):
        # Select the subplot
        ax = axes[i]

        # Add a line for each column with a different color
        for j, column in enumerate(data.columns):
            # Scale down the "Barometric pressure" column by 100 times
            if column == 'Barometric pressure':
                ax.plot(data.index, data[column] / 100, label=column, color=color_palette[j])
            else:
                ax.plot(data.index, data[column], label=column, color=color_palette[j])

        ax.set_xlabel('')
        ax.set_ylabel('Sensor Readings')
        ax.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(f'{output_filename}.png')
    plt.close()


def filter_max_group_by_time_threshold(df, time_threshold='60 min'):
    """
    Filtra un DataFrame por el grupo con la mayor cantidad de filas,
    basado en una diferencia de tiempo entre registros y un umbral.

    Args:
    df (pandas.DataFrame): El DataFrame original.
    time_threshold (str, optional): El umbral de tiempo para considerar que la diferencia es significativa.
        Debe ser un formato de tiempo válido admitido por pandas.Timedelta. Por defecto, es '60 min'.

    Returns:
    pandas.DataFrame: El DataFrame filtrado con el grupo de mayor cantidad de filas.
    """

    df.index = pd.to_datetime(df.index)  # Asegurarse de que el índice sea una fecha
    df = df.sort_index()  # Ordenar el índice por si acaso

    # Calcular la diferencia entre cada registro
    df['time_diff'] = df.index.to_series().diff()

    # Establecer el umbral
    threshold = pd.Timedelta(time_threshold)

    # Crear una nueva columna 'group' que cambia de valor cada vez que la diferencia de tiempo supera el umbral
    df['group'] = (df['time_diff'] > threshold).cumsum()

    # Identificar el grupo con la mayor cantidad de filas
    max_group = df['group'].value_counts().idxmax()

    # Filtrar el DataFrame para mantener solo las filas del grupo con la mayor cantidad de filas
    filtered_df = df[df['group'] == max_group]

    # Eliminar las columnas auxiliares 'time_diff' y 'group' si ya no se necesitan
    filtered_df = filtered_df.drop(columns=['time_diff', 'group'])

    return filtered_df

def get_day_length(latitude, longitude, date):
    """
    Get the length of daylight for a given latitude, longitude, and date.

    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
        date (str): The date for which to retrieve the daylight length in 'YYYY-MM-DD' format.

    Returns:
        str: The length of daylight in hours and minutes, formatted as 'HH:MM'.

    Example:
        get_day_length(37.7749, -122.4194, '2023-05-29')
        '14:26'
    """
    url = f"https://api.sunrise-sunset.org/json?lat={latitude}&lng={longitude}&date={date}"
    response = requests.get(url)
    sunrise_sunset = response.json()
    return sunrise_sunset.get('results', {}).get('day_length', '')
