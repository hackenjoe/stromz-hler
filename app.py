# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:01:05 2021

@author: Pascal
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from PIL import Image, ImageOps


#### init
st.set_page_config(
    page_title="Elektrozähler",
    page_icon=":zap:",
    layout="wide",
    )
st.title("Stromverbrauch einer Kantine :zap:")

#### sidebar
st.sidebar.title("Einstellungen:")
start_date = st.sidebar.date_input(label="Startdatum", 
                                   min_value=date(2016, 7, 1), 
                                   value=date(2016, 7, 1), 
                                   max_value=date(2021, 4, 30),
                                   help="Schränkt das Datumsinterval ein")

end_date = st.sidebar.date_input(label="Enddatum", 
                                 min_value=date(2016, 7, 1), 
                                 value=date(2021, 4, 30), 
                                 max_value=date(2021, 4, 30),
                                 help="Schränkt das Datumsinterval ein")

pred_length = st.sidebar.slider(label="Vorhersagelänge", 
                                      min_value=1, 
                                      max_value=365, 
                                      value=30, 
                                      help="Hier kann die Vorhersagelänge für Vorhersagen zum voraussichtlichen Stromverbrauch in der Kantine festgelegt werden. Angabe in Tagen.")


#### main window
## data preparation
st.markdown(">**Szenario:** Eine große Kantine hat Platz für viele hungrige Personen. Die Küchengeräte und die elektrischen Speisewärmer sind notwendig, um das Essen schnell und warm auszuteilen. Nach einiger Zeit kommt eine Stromabrechnung der Musterwerke GmbH und der Kantinenbesitzer  wundert sich über die Höhe der Kosten. Daraufhin schließt er einen Stromzähler in seiner Mensa an. Nach etwa 5 Jahren lässt er sich die Daten von einem Datenexperte auswerten.")
            
image = Image.open('canteen.jpg')
image = image.resize((300, 450))
image = ImageOps.expand(image, border=1)
st.image(image, caption='Kantine der HBC')
            
"1. Zunächst ist eine Betrachtung der Rohdaten des Stromzählers notwendig. Hier Bild von Mensa und Bild von Stromzähler einfügen."
df = pd.read_csv("D169-440009-MP-510-Istwert-Value_clean_15m_kW.csv", sep=";", skiprows=1).drop(columns="Orginalwert")
"Auszug aus dem Datensatz (unverändert):"
df[:25]

"2. Die Abtastrate beträgt offensichtlich $15$ Minuten. Bei knapp $5$ Jahren sind das $169310$ Datensätze. Aus Darstellungsgründen ist daher eine Aggregierung auf Tagesebene sinnvoll. Außerdem sollten die Daten bereinigt werden (d.h. Duplikate, fehlende Werte...)."
df["Zeit"] = pd.to_datetime(df["Zeit"], format="%d.%m.%Y %H:%M")

print(f"Höchster Wert {df.sort_values(by='Wert', ascending=False)[:1]}")
print(f"Niedrigster Wert {df.sort_values(by='Wert')[:1]}")

st.text(f"Doppelte Zeitstempel gefunden: {df.shape[0] - df.drop_duplicates(subset='Zeit').shape[0]}!")

df.drop_duplicates(subset="Zeit", inplace=True)


## plots
"3. Visuelle Aufbereitung der Daten ist ein zentraler Bestandteil bei der Auswertung des Stromzählers. Eine geeignete Darstellungsoption für sequentielle Daten stellt das *Liniendiagramm* dar."
df_daily = df.resample('D', on='Zeit').sum()
df_daily = df_daily.loc[start_date:end_date]
df_daily.plot(style=['--'])
plt.ylabel('kW');
plt.title("Elektro Leistung Mensa 2016.01.07 - 2021.12.04")
# plt.legend(['Gemessene Werte', 'Daten-Aggregation', 'Datenselektion'], loc='upper left');

#df[_daily"Wochentag"] = df_daily.apply(lambda x: x["Zeit"].strftime("%A"), axis=1)
df_daily
st.subheader(f"Elektro Leistung Mensa {start_date.year}/{start_date.month}/{start_date.day} - {end_date.year}/{end_date.month}/{end_date.day}")
st.line_chart(df_daily.rename(columns={'Wert':'kW'}), height=520, width=1200)

rolling = df_daily.rolling(365, center=True)
data = df_daily.copy(True)
data['Gleitender Mittelwert 365 Tage'] = rolling.mean()
rolling = df_daily.rolling(30, center=True)
data['Gleitender Mittelwert 30 Tage'] = rolling.mean()
ax = data.plot(style=['-', ':'])
ax.lines[0].set_alpha(0.3)

st.subheader("Weitere Darstellung")
line_fig = px.line(data, height=520, width=1200)
line_fig.update_yaxes(title_text='Kumulierter Kilowattverbrauch')
line_fig.update_layout(xaxis_rangeslider_visible=True)
st.plotly_chart(line_fig, use_container_width=True)


## Forecast
"4. Zeitreihen können mithilfer *datengetriebener* Methoden analysiert werden. Mit dem Forecast Algorithmus [Prophet](https://github.com/facebook/prophet) von Facebook können univariate Zeitreihen *vorhergesagt* werden."

f"Beispielweise könnte der Stromverbrauch der ersten $30$ Tage aus der Vergangenheit vorhergesagt werden. Durch die Historie (Streuwolke) lässt sich dann schnell prüfen, wie gut das Modell vorhersagt."
# prepare expected column names
df_time = df_daily.copy(True)
df_time.reset_index(inplace=True)
df_time.columns = ['ds', 'y']
# define model
model = Prophet(daily_seasonality=True)
# fit the model
model.fit(df_time)

# define period for which we want a prediction (in-sample forecast)
# prevents months with less than 30 days
time_diff = end_date - start_date
if (time_diff.days < 30):
    in_sample_limit = time_diff.days
else:
    in_sample_limit = 30
if (time_diff.days > 3):
    future = model.make_future_dataframe(periods=30).set_index("ds").loc[pd.to_datetime(start_date):].iloc[:in_sample_limit].reset_index()
    forecast_in_sample = model.predict(future)
    print(forecast_in_sample[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # plot forecast
    model.plot(forecast_in_sample)
    #st.pyplot(model.plot(forecast_in_sample))

    fig = plot_plotly(model, forecast_in_sample)
    fig.update_yaxes(title_text="Kumulierter Kilowattverbrauch")
    fig.update_xaxes(title_text="Zeit")
    future["date_string"] = future['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
    fig['layout']['xaxis'].update(range=[future["date_string"].iloc[0], future["date_string"].iloc[-1]])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.text("Der gewählte Zeitraum ist zu kurz für eine Vorhersage. Bitte wähle ein späteres Enddatum.")

#fig = px.scatter(df_daily)
#fig.show()

f"Natürlich ist auch ein Blick in die Zukunft möglich, um zu sehen, wie hoch in etwa der erwartete Stromverbrauch in den nächsten ${pred_length}$ Tagen sein wird."
# out-sample forecast
future = model.make_future_dataframe(periods=pred_length).iloc[-pred_length:]
forecast_out_sample = model.predict(future)
#ds: datastamp, yhat: prediction of measurement, yhat_lower & yhat_upper: uncertaintiy_intervals
print(forecast_out_sample[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
model.plot(forecast_out_sample)
# clamp up negative values to 0
forecast_out_sample["yhat"] = forecast_out_sample["yhat"].apply(lambda x: x if x>=0 else 0)
# again plotly plot for streamlit
fig = plot_plotly(model, forecast_out_sample)
fig.update_yaxes(title_text="Kumulierter Kilowattverbrauch")
fig.update_xaxes(title_text="Zeit")
future["date_string"] = future['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
fig['layout']['xaxis'].update(range=[future["date_string"].iloc[0], future["date_string"].iloc[-1]])
st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Komponenten anzeigen"):
    "5. Zeitreihen können auch in ihre Komponenten zerlegt werden:"
    fig = model.plot_components(forecast_out_sample)
    fig.set_size_inches(12.5, 10.5)
    fig.suptitle(f"Komponenten für p={pred_length}")
    fig