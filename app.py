import numpy as np
import pandas as pd
import panel as pn
from panel.interact import interactive
import hvplot.pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


pn.config.throttled = True


global X_train, y_train, X_test, y_test


def add_noise(y, noise_level=0.1):
    return y + np.random.normal(0, np.sqrt(noise_level), len(y))

def generate_data(size=1000, type='Simple', random_state=42):
    np.random.seed(random_state)
    X = np.random.rand(size)
    if type == 'Simple':
        y = 2 * X
    elif type == 'Complex':
        y = 2 * X - 1.5*X**2
    else:  # Very Complex
        y = 1*X**6 - 1*X - (1-X)**7
    
    return X.reshape(-1, 1), 100*y


# Function to generate and store data
def generate_and_store_data(event):
    global X_train, X_test, y_train, y_test
    X, y = generate_data(size=size_select.value, type=dataset_select.value)
    y = add_noise(y, noise_level=noise_select.value)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Sorting the data (if necessary)
    train_indices = np.argsort(X_train, axis=0).flatten()
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    test_indices = np.argsort(X_test, axis=0).flatten()
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]



def fit_and_evaluate_model(model_complexity, X_train, y_train, X_test, y_test):
    model = make_pipeline(PolynomialFeatures(degree=model_complexity), LinearRegression())
    
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    
    
    return y_train_pred, y_test_pred, mse_train, mse_test

def plot_data(model_complexity, dataset, size, noise):

    global X_train, y_train, X_test, y_test
    y_train_pred, y_test_pred, mse_train, mse_test, = fit_and_evaluate_model(
        model_complexity, X_train, y_train, X_test, y_test)

    # Convert NumPy arrays to Pandas DataFrames for training and testing data
    df_train = pd.DataFrame({
        'X': X_train.flatten(),
        'y': y_train,
        'y_pred': y_train_pred
    })
    
    df_test = pd.DataFrame({
        'X': X_test.flatten(),
        'y': y_test,
        'y_pred': y_test_pred
    })
    
    # Create hvPlot visualizations
    plot_train = df_train.hvplot.scatter(x='X', y='y', color='navy', tools=[]) * \
                 df_train.hvplot.line(x='X', y='y_pred', color='navy', tools=[])
    plot_train.opts(title=f'Training Data. MSE {mse_train:.1f}', xlabel='X', ylabel='y', width=600, height=400)
    
    plot_test = df_test.hvplot.scatter(x='X', y='y', color='gold', tools=[]) * \
                df_test.hvplot.line(x='X', y='y_pred', color='navy', tools=[])
    plot_test.opts(title=f'Testing Data. MSE {mse_test:.1f}', xlabel='X', ylabel='y', width=600, height=400)


    return plot_train + plot_test

model_select = pn.widgets.IntSlider(name='Model complexity', value=1, start=1, end=20, step=1)
dataset_select = pn.widgets.Select(name='Data complexity', value='Complex', options=['Simple', 'Complex', 'Very Complex'])
size_select = pn.widgets.IntSlider(name='Sample size', value=20, start=20, end=100, step=1)
noise_select = pn.widgets.FloatSlider(name='Noise level', value=5, start=0, end=100, step=1)

# Init data
generate_and_store_data("")


# Attach the generate_and_store_data function to widget changes
dataset_select.param.watch(generate_and_store_data, ['value'])
size_select.param.watch(generate_and_store_data, ['value'])
noise_select.param.watch(generate_and_store_data, ['value'])

interact_panel = interactive(plot_data, 
                             model_complexity=model_select, 
                             dataset=dataset_select,
                             size=size_select, 
                             noise=noise_select)

app_layout = pn.Column(
    pn.Row(model_select, dataset_select, size_select, noise_select), 
    interact_panel
)
app_layout.servable()
