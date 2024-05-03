importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.4/pyc/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/1.2.3/dist/wheels/bokeh-3.2.2-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.2.3/dist/wheels/panel-1.2.3-py3-none-any.whl', 'pyodide-http==0.2.1', 'hvplot', 'numpy', 'pandas', 'scikit-learn']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

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

def generate_data(size=1000, type='Simple', X=None):
    if X is None:
        X = np.random.rand(size)
    if type == 'Simple':
        y = X - 1
    elif type == 'Complex':
        y = 2 * X - 1.5*X**2 - 1
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

    # Retrieve the LinearRegression object from the pipeline
    linear_model = model.named_steps['linearregression']
    
    formula = f"{linear_model.intercept_:2.1f}"
    for i, c in enumerate(linear_model.coef_[1:]):
        if c < 0:
            sign = ""
        else:
            sign = "+ "
        if i == 0:
            formula += f" {sign}{c:2.1f} x"
        else:
            formula += f" {sign}{c:2.1f} x^{i+1}"

    pred = model.predict(np.linspace(0, 1, 100).reshape(100,1))
    
    return y_train_pred, y_test_pred, mse_train, mse_test, formula, pred

def plot_data(model_complexity, dataset, size, noise, show_test, new_data):

    global X_train, y_train, X_test, y_test
    y_train_pred, y_test_pred, mse_train, mse_test, formula, pred = fit_and_evaluate_model(
        model_complexity, X_train, y_train, X_test, y_test)

    
    # Convert NumPy arrays to Pandas DataFrames for training and testing data
    df_train = pd.DataFrame({
        'X': X_train.flatten(),
        'y': y_train,
        
        
    })

    df_all = pd.DataFrame({
        'X': np.linspace(0, 1, 100),
        'y_pred': pred,
        'y_real': generate_data(size=size_select.value, type=dataset_select.value, X=np.linspace(0, 1, 100))[1]
    })
    
    df_test = pd.DataFrame({
        'X': X_test.flatten(),
        'y': y_test,
    })
    
    # Create hvPlot visualizations
    plot_train = df_train.hvplot.scatter(x='X', y='y', color='navy', tools=[], ylim=(-200,100)) * \
                 df_all.hvplot.line(x='X', y='y_pred', color='navy', tools=[]) * \
                 df_all.hvplot.line(x='X', y='y_real', color='cornflowerblue', tools=[])
    
    plot_train.opts(title=f'Training Data. MSE {mse_train:.1f}. \\ny-hat = {formula}', xlabel='X', ylabel='y', width=600, height=400)
    
    plot_test = df_test.hvplot.scatter(x='X', y='y', color='maroon', tools=[], ylim=(-200,100)) * \
                df_all.hvplot.line(x='X', y='y_pred', color='navy', tools=[]) * \
                 df_all.hvplot.line(x='X', y='y_real', color='cornflowerblue', tools=[]).redim.range(X=(0, 1), Y=(-200, 100))
    
    plot_test.opts(title=f'Testing Data. MSE {mse_test:.1f}', xlabel='X', ylabel='y', width=600, height=400)

    return plot_train + plot_test if show_test else plot_train

model_select = pn.widgets.IntSlider(name='Model complexity', value=1, start=1, end=20, step=1)
dataset_select = pn.widgets.Select(name='Data complexity', value='Complex', options=['Simple', 'Complex', 'Very Complex'])
size_select = pn.widgets.IntSlider(name='Sample size', value=10, start=20, end=100, step=1)
noise_select = pn.widgets.FloatSlider(name='Noise level', value=5, start=0, end=100, step=1)
show_test_plot = pn.widgets.Checkbox(name='Show Test Plot', value=True)  # Checkbox for toggling test plot visibility
show_new_data = pn.widgets.Button(name='Create new data')  # Checkbox for toggling test plot visibility

# Init data
generate_and_store_data("")


# Attach the generate_and_store_data function to widget changes
dataset_select.param.watch(generate_and_store_data, ['value'])
size_select.param.watch(generate_and_store_data, ['value'])
noise_select.param.watch(generate_and_store_data, ['value'])
show_new_data.param.watch(generate_and_store_data, ['value'])

interact_panel = interactive(plot_data, 
                             model_complexity=model_select, 
                             dataset=dataset_select,
                             size=size_select, 
                             noise=noise_select,
                             show_test=show_test_plot,
                            new_data=show_new_data)

app_layout = pn.Column(
    pn.Row(model_select, dataset_select, size_select, noise_select), 
    show_test_plot,
    show_new_data,
    interact_panel
)
app_layout.servable()


await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.globals.set('patch', msg.patch)
    self.pyodide.runPythonAsync(`
    state.curdoc.apply_json_patch(patch.to_py(), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.globals.set('location', msg.location)
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads(location)
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()
