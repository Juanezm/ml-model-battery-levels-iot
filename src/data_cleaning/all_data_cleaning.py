from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate
import pandas as pd
import time
import numpy as np

from src.utils.cleaning import plot_variable_counts, convert_unix_to_datetime, outrange_percentage_by_variable, \
    print_boxplots, trim_sensor_data, impute_missing_values, plot_sensors, filter_max_group_by_time_threshold, \
    get_day_length
from src.utils.report import add_text_to_story, add_section_title_to_story, add_image_to_story, add_table_to_story


sensor_ids = [
    '200034001951343334363036',
    '270043001951343334363036',
    '380033001951343334363036',
    '46004e000251353337353037',
    '46005a000351353337353037',
    '4e0022000251353337353037',
    '4e0031000251353337353037',
]

for sensor_id in sensor_ids:
    # Create PDF document with A4 size and margins
    doc = SimpleDocTemplate(f"./data_cleaning/{sensor_id}_data_cleaning_report.pdf", pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)

    # Get default stylesheet for formatting the PDF
    styles = getSampleStyleSheet()

    # Initialize an empty list for the story
    Story = []

    # REPORT
    #
    add_section_title_to_story("Sección 1: Carga de datos", Story)
    ##############################################################################

    # Load sensor data from csv file
    sensor1_raw_data = pd.read_csv(f'./data/{sensor_id}_raw_data.csv',
                                   usecols=["date", "value", "variable", "units", "range"])

    # REPORT
    #
    add_text_to_story("Se cargaron los datos desde el archivo CSV para el sensor " + sensor_id, styles["BodyText"],
                      Story)
    add_table_to_story(sensor1_raw_data, 'sensor1_raw_data_1', Story)
    ##############################################################################

    # Convert Unix timestamps to datetime format
    sensor1_raw_data = convert_unix_to_datetime(sensor1_raw_data, 'date')

    # Convert remaining strings to datetime format
    sensor1_raw_data['date'] = pd.to_datetime(sensor1_raw_data['date'], errors='coerce')

    # Set the date column as the index and sort the dataframe by date
    sensor1_raw_data = sensor1_raw_data.set_index('date').sort_index()

    # REPORT
    #
    add_text_to_story(
        "Se conviertien las marcas de tiempo Unix a formato de fecha y hora y se establece como índice. Además se ordena el dataframe por fecha.",
        styles["BodyText"], Story)
    add_table_to_story(sensor1_raw_data, 'sensor1_raw_data_timestamps', Story)
    ##############################################################################

    # Remap 'Soil moisture' values to percentage
    soilmoist_data = sensor1_raw_data[sensor1_raw_data['variable'] == 'Soil moisture']
    sensor1_raw_data.loc[sensor1_raw_data['variable'] == 'Soil moisture', 'value'] = (soilmoist_data[
                                                                                          'value'] / 4096) * 100

    # REPORT
    #
    add_text_to_story("Se convierten los valores de 'Soil moisture' a porcentaje.", styles["BodyText"], Story)
    add_table_to_story(sensor1_raw_data[sensor1_raw_data['variable'] == 'Soil moisture'],
                       'sensor1_raw_data_soil_moisture', Story)
    ##############################################################################

    # Calculate percentage of out-of-range and null entries
    percentages_string, high_null_vars = outrange_percentage_by_variable(sensor1_raw_data, 5)

    # REPORT
    #
    add_text_to_story(
        "Se realiza un análisis de cada variable y se calcula el porcentaje de valores que están fuera de rango y cuantos de ellos corresponden a valores nulos.",
        styles["BodyText"], Story)
    plot_variable_counts(sensor1_raw_data, 'plot_variable_counts')
    add_image_to_story('plot_variable_counts', Story)
    add_text_to_story(percentages_string, styles["BodyText"], Story)
    time.sleep(1)
    ##############################################################################

    # Trim sensor data to remove outliers and unneeded observations
    sensor1_raw_data = trim_sensor_data(sensor1_raw_data)

    # Calculate the percentage of out of range entries by variable after trimming
    percentages_string, high_null_vars = outrange_percentage_by_variable(sensor1_raw_data, 5)

    # REPORT
    #
    # Add text to the story and plot boxplots of the sensor data
    add_text_to_story(
        "Se ponen a nulo los datos que están fuera de rango y se vuelve a calcular que porcentaje de valores corresponden a valores nulo.",
        styles["BodyText"], Story)
    print_boxplots(sensor1_raw_data, 'boxplots')
    add_image_to_story('boxplots', Story)
    add_text_to_story(percentages_string, styles["BodyText"], Story)
    ##############################################################################

    # Round down to the nearest minute for timestamps
    sensor1_raw_data.index = sensor1_raw_data.index.floor('T')

    # Simplify the dataframe by removing unnecessary columns
    df_simplified = sensor1_raw_data.drop(['units', 'range', 'within_range'], axis=1)

    # Pivot the dataframe to take the mean of measurements within the same minute for each variable
    df_pivot = df_simplified.pivot_table(index=df_simplified.index, columns='variable', values='value', aggfunc='mean')

    # Flatten the columns and drop the 'Wind direction' column
    df_pivot.columns = df_pivot.columns.get_level_values(0)
    sensor1_clean_df = df_pivot.drop(['Wind direction'], axis=1)

    # Drop rows where all values are NA
    sensor1_clean_df.dropna(how='all', inplace=True)

    # REPORT
    #
    add_text_to_story(
        "Se simplifica el dataframe eliminando columnas innecesarias ('units', 'range', 'within_range') y se pivota uniendo todas las mediciones realizadas en el mismo minuto por todos los sensores. Se elimina también la variable 'Wind direction' dado que tiende a contener muchos valores vacios y el número de mediciones suele ser menor al resto de variables.",
        styles["BodyText"], Story)
    add_table_to_story(sensor1_clean_df, 'sensor1_clean_df_1', Story)
    ##############################################################################

    # Impute missing values in the cleaned dataframe
    sensor1_clean_df = impute_missing_values(
        sensor1_clean_df,
        ['Temperature', 'Soil moisture', 'Rain meter', 'Barometric pressure', 'Humidity'],
        high_null_vars
    )

    # REPORT
    #
    add_text_to_story(
        "Se imputan los valores faltantes usando el método IterativeImpute. El método IterativeImputer es una técnica de imputación de valores que utiliza un regresor bayesiano como estimador y utilizando el resto de columnas para esa fila.",
        styles["BodyText"], Story)
    add_table_to_story(sensor1_clean_df, 'sensor1_clean_df_2', Story)
    ##############################################################################

    # Perform time-based interpolation on the dataframe
    sensor1_clean_df = sensor1_clean_df.interpolate(method='time')

    # Round all values in the DataFrame to 2 decimal places
    sensor1_clean_df = sensor1_clean_df.round(2)

    # REPORT
    #
    add_text_to_story(
        "Se realizó una interpolación basada en el tiempo en el dataframe y redondearon todos los valores a 2 decimales.",
        styles["BodyText"], Story)
    add_table_to_story(sensor1_clean_df, 'sensor1_clean_df_impute_missing_values', Story)
    ##############################################################################

    # REPORT
    #
    add_text_to_story("Lecturas de los sensores después de toda la limpieza y procesamiento de datos.",
                      styles["BodyText"], Story)
    plot_sensors(sensor1_clean_df, 'plot_sensor_readings')
    add_image_to_story('plot_sensor_readings', Story)
    ##############################################################################

    sensor1_clean_df = filter_max_group_by_time_threshold(sensor1_clean_df, time_threshold='300 min')

    # REPORT
    #
    add_text_to_story("Lecturas de los sensores después de eliminar los periodos invalidos.",
                      styles["BodyText"], Story)
    plot_sensors(sensor1_clean_df, 'plot_sensor_readings_trimmed')
    add_image_to_story('plot_sensor_readings_trimmed', Story)
    ##############################################################################

    # Perform time-based interpolation on the dataframe again
    sensor1_clean_df = sensor1_clean_df.interpolate(method='time')
    # Delete rows containing NaN located at the beginning and the end of the DataFrame
    sensor1_clean_df = sensor1_clean_df.dropna()

    # REPORT
    #
    add_text_to_story(
        "Se realizó una interpolación basada en el tiempo en el dataframe de nuevo.",
        styles["BodyText"], Story)
    add_table_to_story(sensor1_clean_df, 'sensor1_clean_df_final', Story)
    ##############################################################################

    # Read latitude and longitude from the provided DataFrame
    lat_lon = pd.read_csv(f'./data/{sensor_id}_raw_data.csv', usecols=["geo_lat", "geo_lon"], nrows=1)

    latitude = lat_lon['geo_lat'].values[0]
    longitude = lat_lon['geo_lon'].values[0]

    # Get unique dates in the DataFrame
    unique_dates = np.unique(sensor1_clean_df.index.date).tolist()
    unique_dates_str = [date.strftime('%Y-%m-%d') for date in unique_dates]

    # Create a dictionary to store day lengths for each date
    day_length_dict = {}

    # Iterate over unique dates and retrieve day length (only make API call once per unique date)
    for date in unique_dates:
        day_length = get_day_length(latitude, longitude, str(date))
        day_length_dict[str(date)] = day_length

    # Convert day lengths to minutes
    day_length_minutes = {date: int(day_length.split(':')[0]) * 60 + int(day_length.split(':')[1]) for date, day_length
                          in day_length_dict.items()}

    # Convert day lengths to datetime
    day_length_date = {pd.to_datetime(k).date(): v for k, v in
                       day_length_minutes.items()}  # Ensure dictionary keys are date objects

    # Map day lengths to a new column in the DataFrame based on the date
    sensor1_clean_df['day_length'] = sensor1_clean_df.index.to_series().apply(
        lambda x: pd.Series(day_length_date).get(x.date(), None))

    # REPORT
    #
    add_text_to_story(
        "Se añade una nueva columna para cada entrada con el numero de minutos total de luz",
        styles["BodyText"], Story)
    add_table_to_story(sensor1_clean_df, 'sensor1_clean_df_final_day_length', Story)
    ##############################################################################

    # Build and save the final PDF document
    doc.build(Story)

    # Save each sensor_id's data to a separate CSV
    sensor1_clean_df.to_csv(f'./data_cleaning/{sensor_id}_cleaned_data.csv')
