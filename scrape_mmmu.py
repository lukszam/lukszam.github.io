import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
# from lxml import html
# from flask import Flask, jsonify

# app = Flask(__name__)

# @app.route('/scrape')
# def scrape():

# URL of the webpage containing the table
url = 'https://mmmu-benchmark.github.io/'

# Set up Selenium WebDriver with headless Chrome options
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--remote-debugging-port=9222')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920x1080')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--allow-running-insecure-content')

driver = webdriver.Chrome(options=options)
driver.get(url)

# Wait for the table to be populated (adjust the wait time and condition as needed)
try:
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, "//table[@id='mmmu-table']/tbody/tr"))
    )
except Exception as e:
    print("Table rows not found or page took too long to load")
    driver.quit()
    exit()

# Get the page source and parse it with BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Close the WebDriver
driver.quit()

# Find the table you want to scrape
table = soup.find('table', {'id': 'mmmu-table'})
if table is None:
    print("Table not found")
    exit()

# Extract table headers
headers = []
for header in table.find_all('th'):
    if header.text.strip() in headers:
        header_text = header.text.strip()  + "_2"
    else:
        header_text = header.text.strip()
    headers.append(header_text)

# Extract table rows
rows = []
tbody = table.find('tbody')
if tbody is None:
    print("Table body not found")
    exit()

for idx, row in enumerate(tbody.find_all('tr')):
    cells = row.find_all('td')
    row_data = [cell.text.strip() for cell in cells]
    
    row_data.append(row.get("class")[0])
    # if "Human" not in row_data[0]:
    rows.append(row_data)

# Convert the data to a pandas DataFrame
headers[17] = "Overall_3"
headers[10] = "Accuracy"
headers.append("Type")
df = pd.DataFrame(rows, columns=headers[4:])

data = df.to_json(orient='records')

data_json =  json.loads(data)

with open("data_mmmu.json", 'w') as file:
    json.dump(data_json, file, indent=4)

df = pd.read_json('data_mmmu.json')

df['Accuracy'] = df['Accuracy'].str.rstrip('*')
df = df[df['Accuracy'] != '-']
df['Accuracy'] = pd.to_numeric(df['Accuracy'])

# Convert date to numerical format for regression
df['date_num'] = (df['Date'] - df['Date'].min()).dt.days

# Fit linear regression model
X = df['date_num'].values.reshape(-1, 1)
y = df['Accuracy'].values
model = LinearRegression()
model.fit(X, y)
df['Trend'] = model.predict(X)

# Assuming you have a DataFrame named df with columns 'Date' and 'Overall_2'
# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Create the scatter plot
fig = px.scatter(df, x='Date', y='Accuracy', hover_name='Name', color='Type', title='Performance on MMMU benchmark', hover_data={'Size': True})

# Add the trend line
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Trend'],
    mode='lines',
    name='Trend Line',
    line=dict(color="rgba(0, 0, 255, 0.3)", width=10)
))

# # Control y-axis range
fig.update_yaxes(range=[0.0, 100.0], autorange=False)

# Add horizontal line at y = 0.9 (threshold)
fig.add_shape(
    type="rect",
    x0=df['Date'].min(),  # Start of the line on the x-axis
    y0=76.1,               # y value where the line starts
    x1=df['Date'].max(),  # End of the line on the x-axis
    y1=88.6,              # y value where the line ends (same as y0 for horizontal line)
    fillcolor="rgba(255, 0, 0, 0.2)",
    line=dict(
        width=0
    ),
)

# Add dummy trace for the rectangle to appear in the legend
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=10, color='rgba(255, 0, 0, 0.2)', symbol='square'),
    showlegend=True,
    name='Human Experts'
))
# Save the plot as an HTML file
fig.write_html('scatter_plot_mmmu.html')

    # return jsonify(data)


# if __name__ == '__main__':
#     app.run(debug=True)