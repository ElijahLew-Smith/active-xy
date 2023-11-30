import flocking_app
import numpy as np
import importlib
importlib.reload(flocking_app)
from flocking_app import ActiveXY
import json



import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.graph_objs as go


import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly



sim_params = {
    'X': 100,
    'Y': 100,
    'model_type':         'XY',
    'active_component':   ['dV', 'muV'],
    'damping_noise_temp': [1, None, 0.5],#[0, 0.01, 'inf'],#
    'activity':             1,
    'dt':                  0.02,
    'boundary_conditions':'periodic',
    'background':         'random',
    'perturbation_type':  'none',
    'background_angle':   0,
    'perturbation_angle': np.pi,
    'perturbation_size': 20,
}
# log_damping_noise_temp = [None, None, None]
# for i in range(0,3): 
#     damping_noise_temp = sim_params['damping_noise_temp'][i]
#     if damping_noise_temp is not None: 
#         log_damping_noise_temp[i] = np.log10(damping_noise_temp)

simulation = ActiveXY(**sim_params)
# simulation.time_evolution()




simulation.initialize_single_time_evolution()

default_params = {
    'interval': 50, 
    'page_size': 1600
}

slider_mapping = {
    'interval_value': 'interval_value',
    'dt': 'dt',
    'size': 'size',
    'background_angle': 'background_angle', 
    'perturbation_angle': 'perturbation_angle', 
    'perturbation_size': 'perturbation_size', 
    'damping': 'damping',
    'noise': 'noise',
    'temp': 'temp',
    # 'arrow-number-slider': 'arrow_number', 
}


slider_style = { 'alignItems': 'center',  'width': (3/8)*default_params['page_size'], 'justifyContent': 'center', 'margin': 'auto'}
dropdown_style = {'padding': '10px', 'width': '200px', 'margin': 'auto'}

#This is from [0.01, 100] on a linear scale 
fdt_log_range = [-2, 2]
dt_log_range = [-3, 0]




# Define the figure layout once outside of the callback
figure_layout = go.Layout(
    width=default_params['page_size']/2,
    height=default_params['page_size']/2,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)


log_scale_sliders = {'dt', 
                      'damping', 'noise','temp'
                     }


#Type is linear or log10 
def create_slider(label, slider_id, scale_type, min_value, max_value, step, default_value, marks):


    if scale_type == 'log10':
        displayed_value = 10**default_value
    elif scale_type == 'linear':
        displayed_value = default_value
    else: 
        raise ValueError('Invalid Scale Type')

    return html.Div([
        html.Div([html.Label(label)], style={'flex': '1'}),
        html.Div([
            dcc.Input(
                id={'type': 'input-for-slider', 'index': slider_id},
                type='number',
                value=displayed_value,
                style={'width': '60px'}  # Fixed width of 20px
            )
        ], style={'flex': 'none'}),  # Remove flex to avoid stretching
        html.Div([
            dcc.Slider(
                id={'type': 'dynamic-slider', 'index': slider_id},
                min=min_value,
                max=max_value,
                step=step,
                value=default_value,
                marks=marks
            )
        ], style={'flex': '3'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'marginBottom': '10px'})

app = dash.Dash(__name__)


app.layout = html.Div([
    
    # Component on Left 
    html.Div([
        # Graph
        html.Div(
            dcc.Graph(id='simulation-graph'),
            style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}
        ),

        # Reset and Play/Pause Buttons
        html.Div([
            html.Button('Reset', id='reset-button', n_clicks=0),
            html.Button('Play/Pause', id='run-button', n_clicks=0),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexDirection': 'row'}),
        
    ], style={'padding': 10, 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexDirection': 'column', 'flex': 1}),



    # Controls on Right 
    html.Div([

    # Update interval and timestep 
    html.Div([
        html.Div(id='output-div') ,
        create_slider('Update Interval (in milliseconds):', 'interval_value',  'linear', 10, 300, 10,  default_params['interval'],  {i: str(i) for i in range(10, 301, 50)}),
        create_slider('Timestep:', 'dt',  'log10', dt_log_range[0], dt_log_range[1], 0.001, np.log10(sim_params['dt']), {i: f"{10**float(i):.1}" for i in range(dt_log_range[0], dt_log_range[1] + 1)} ),
        create_slider('Linear Size:', 'size',  'linear', 0, 400, 1, sim_params['X'],{i: f"{i}" for i in range(0, 400, 20)} ),

    ]),

    # Dropdown for Background and Perturbation Type
    html.Div([
        html.Label('Boundary Conditions:'),
        dcc.Dropdown(
            id='boundary_conditions',
            options=[
                {'label': 'Periodic', 'value': 'periodic'},
                {'label': 'Open', 'value': 'open'},
            ],
            value=sim_params['boundary_conditions'],  
            multi=False  
        ),
        html.Label('Background:'),
        dcc.Dropdown(
            id='initial-background',
            options=[
                {'label': 'Random', 'value': 'random'},
                {'label': 'Uniform', 'value': 'uniform'},
            ],
            value=sim_params['background'],  
            multi=False  
        ),
    ], style=dropdown_style),

    html.Div([
        create_slider('Background Angle:', 'background_angle',  'linear', 0, np.pi*2, 0.01, sim_params['background_angle'],
                      {
                        0: '0',
                        np.pi/2: 'π/2',
                        np.pi: 'π',
                        3*np.pi/2: '3π/2',
                        2*np.pi: '2π'
                    } ),
    ]), 
    html.Div([
        html.Label('Perturbation Type:'),
        dcc.Dropdown(
            id='perturbation-type',
            options=[
                {'label': 'None', 'value': 'none'},
                {'label': 'Blob', 'value': 'blob'},
                {'label': 'Vorticies', 'value': 'opposite_vortex'},
                {'label': 'Vortex', 'value': 'vortex'},
                {'label': 'Square', 'value': 'perturbation'},
            ],
            value=sim_params['perturbation_type'], 
            multi=False 
        ),
    ], style=dropdown_style),

    # Sliders for Perturbation Angle, Size, and Arrow Number
    html.Div([
        create_slider('Perturbation Angle:', 'perturbation_angle',  'linear', 0, np.pi*2, 0.01, sim_params['perturbation_angle'],
                {
                0: '0',
                np.pi/2: 'π/2',
                np.pi: 'π',
                3*np.pi/2: '3π/2',
                2*np.pi: '2π'
            } ),
        create_slider('Perturbation Size:', 'perturbation_size',  'linear', 1, simulation.X, 1, sim_params['perturbation_size'],
            {
                int(0): '0',
                int(simulation.X/4): '1/4',
                int(simulation.X/2): '1/2',
                int(3*simulation.X/4): '3/4',
                int(simulation.X): '1'
            } ),
        # create_slider('Arrow Number:', 'arrow-number-slider', 1, simulation.X, 100, sim_params['arrow_number'],
        #         {
        #         int(0): '0',
        #         int(simulation.X/4): '1/4',
        #         int(simulation.X/2): '1/2',
        #         int(3*simulation.X/4): '3/4',
        #         int(simulation.X): '1'
        #     }),
    ]),


    # Interval Component
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # Default interval (1 second)
        n_intervals=0
    ),


    # Parameter Controls for Damping, Noise, Temperature
    # Damping Controls
    html.Div([
        html.Label('Whats Fixed in FDT:', style={'marginRight': '10px'}),
        dcc.Dropdown(
                id='fixed-fdt-parameter',
                options=[
                    {'label': 'Damping', 'value': 0},
                    {'label': 'Noise',   'value': 1},
                    {'label': 'Temperature',    'value': 2},
                ],
                value=1,  #Default is noise set to constant 
                multi=False  
            ),
        ], style=dropdown_style),
    html.Div([
        create_slider('Damping:',   'damping',  'log10', fdt_log_range[0], fdt_log_range[1], 0.01, 0, {i: f"{10**float(i):.1}" for i in range(fdt_log_range[0], fdt_log_range[1] + 1)} ),
        create_slider('Noise:',     'noise',    'log10', fdt_log_range[0], fdt_log_range[1], 0.01, 0, {i: f"{10**float(i):.1}" for i in range(fdt_log_range[0], fdt_log_range[1] + 1)} ),
        create_slider('Temperature:','temp',    'log10', fdt_log_range[0], fdt_log_range[1], 0.01, np.log10(.5), {i: f"{10**float(i):.1}" for i in range(fdt_log_range[0], fdt_log_range[1] + 1)} ),
    ]),


    # Model Types
    html.Div([
        html.Label('Model Type:'),
        dcc.Dropdown(
            id='model-type',
            options=[
                {'label': 'Equilibrium XY',             'value': 'XY'},
                {'label': 'Kuramoto',                   'value': 'kuramoto'},
                {'label': '1st Discretiation (Andy)',   'value': '1'},
                {'label': '2nd Discretiation (Marvin)', 'value': '2'},
                {'label': '3rd Discretiation (Eli)',    'value': '3'},
            ],
            value=sim_params['model_type'],  
            multi=False  
        ),
        dcc.Checklist(
            id='active-component',
            options=[
                {'label': 'dV', 'value': 'dV'},
                {'label': 'mu*V', 'value': 'muV'}
            ],
            value=['dV', 'muV']
        ),
    ], style=dropdown_style),
        # create_slider('Activity:', 'activity',  'linear', 0.1, 10, 0.01,  1, {}),


],  style={'padding': 10, 'flex': 1, 'justifyContent': 'center'}  ), 

], style={'display': 'flex', 'flexDirection': 'row'} )






# Updates the interval 
@app.callback(
    Output('interval-component', 'interval'),
    [Input('run-button', 'n_clicks'),
     Input({'type': 'dynamic-slider', 'index': 'interval_value'}, 'value')],
    [State('interval-component', 'interval')]
)
def set_interval(run_clicks, slider_interval, current_interval):
    ctx = dash.callback_context

    # Check if the run button was clicked
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'run-button':
            # Toggle the simulation state
            if current_interval == slider_interval:
                return 24 * 60 * 60 * 100000  # High value to pause updates
            else:
                return slider_interval  # Regular interval to resume updates
        elif trigger_id == 'interval_value':
            return slider_interval

    # If the callback wasn't triggered by the run button, set it to the default. This happens during initialzation
    return default_params['interval']   






# Keeps sliders and inputs the same for the base slider 
@app.callback(
    [Output({'type':'input-for-slider',  'index': ALL}, 'value'),
     Output({'type':'dynamic-slider',    'index': ALL}, 'value')],
    [Input('fixed-fdt-parameter',   'value'),
     Input({'type': 'input-for-slider',  'index': ALL}, 'value'),
     Input({'type': 'dynamic-slider',    'index': ALL}, 'value')],
    [State({'type': 'dynamic-slider',    'index': ALL}, 'id')], 
    prevent_initial_call=True)
def sync_input_and_slider(fixed_fdt_parameter, input_values, slider_values, slider_ids):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id']
    input_updates = [dash.no_update] * len(slider_ids)
    slider_updates = [dash.no_update] * len(slider_ids)


    # Check if trigger_id is not empty and is valid JSON
    triggered_index = None
    if trigger_id:
        try:
            triggered_dict = json.loads(trigger_id.split('.')[0])
            triggered_index = triggered_dict.get('index')
        except json.JSONDecodeError:
            print("Error decoding JSON from trigger_id:", trigger_id)
    for i, slider_id in enumerate(slider_ids):
        if slider_id['index'] == triggered_index:
            if 'input-for-slider' in trigger_id:
                # Logic for updating from input
                if slider_id['index'] in log_scale_sliders:
                    try:
                        slider_updates[i] = np.log10(float(input_values[i]))
                    except ValueError:
                        pass
                else:
                    slider_updates[i] = input_values[i]

            elif 'dynamic-slider' in trigger_id:
                # Logic for updating from slider
                if slider_id['index'] in log_scale_sliders:
                    try:
                        input_updates[i] = 10**float(slider_values[i])
                    except ValueError:
                        pass
                else:
                    input_updates[i] = slider_values[i]


    if trigger_id == 'fixed-fdt-parameter.value' or triggered_index in ['damping', 'noise', 'temp']:
        input_value_dict = {slider_id['index']: value for slider_id, value in zip(slider_ids, input_values)}
        slider_value_dict = {slider_id['index']: value for slider_id, value in zip(slider_ids, slider_values)}

        if 'input-for-slider' in trigger_id:
            damping  = input_value_dict.get('damping')
            noise    = input_value_dict.get('noise')
            temp     = input_value_dict.get('temp', None)
        else: 
            damping  = 10**slider_value_dict.get('damping')
            noise    = 10**slider_value_dict.get('noise')
            temp     = 10**slider_value_dict.get('temp', None)
        print(temp)
        damping_noise_temp = [damping, noise, temp]
        damping_noise_temp[fixed_fdt_parameter] = None 
        simulation.damping_noise_temp = damping_noise_temp
        simulation.initialize_FDT()

        for j, sid in enumerate(slider_ids):
            if sid['index'] in ['damping', 'noise', 'temp']:
                param_value = getattr(simulation, sid['index'])
                input_updates[j] = param_value
                slider_updates[j] = np.log10(param_value)
    # Create a dictionary of slider values
    # slider_values = {slider_mapping[id['index']]: value for id, value in zip(slider_values, slider_values)} 
    
    return input_updates, slider_updates




@app.callback(
    Output('simulation-graph',      'figure'),
    [Input('interval-component',    'n_intervals'),
     Input('reset-button',          'n_clicks'),
     Input({'type': 'input-for-slider', 'index': ALL}, 'value'),
     Input({'type': 'input-for-slider', 'index': ALL}, 'id'),
     Input('boundary_conditions',   'value'), 
     Input('initial-background',    'value'), 
     Input('perturbation-type',     'value'), 
     Input('fixed-fdt-parameter',   'value'),
     Input('model-type',            'value'),
     Input('active-component',      'value')],
    # [State('arrow-number',          'value')], 
    
)

def update_graph(n, reset_clicks, 
                 values, ids, 
                 boundary_conditions, initial_background, perturbation_type, 
                 fixed_fdt_parameter, 
                 model_type, active_component,
                #  arrow_number
                 ):

    # Create a dictionary of slider values
    # print(values)
    input_values = {slider_mapping[id['index']]: value for id, value in zip(ids, values)}
    # print(input_values)


    # Now you can use the named variables directly
    interval_value       = input_values['interval_value']
    dt               = input_values['dt']
    size                 = input_values['size']

    background_angle     = input_values['background_angle']

    perturbation_angle   = input_values['perturbation_angle']
    perturbation_size    = input_values['perturbation_size']
    
    damping = input_values['damping']
    noise = input_values['noise']
    temp = input_values['temp']

 


    ctx = callback_context

    # We need to check if the input is from the unpacking or put in regularly 
    trigger_id_string = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id_string.startswith('{') and trigger_id_string.endswith('}'):
        # This looks like a JSON string for a pattern-matching ID
        try:
            trigger_id = json.loads(trigger_id_string)
            triggered_index = trigger_id.get('index')
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {trigger_id_string}")
            triggered_index = None
    else:
        # This is a standard ID, not a pattern-matching ID
        triggered_index = trigger_id_string
                
    # Reinitialize simulation if its required 
    if triggered_index in ['size', 'reset-button', 'initial-background',  'background_angle', 'perturbation-type', 'perturbation_size', 'perturbation_angle']: 
        simulation.X = size
        simulation.Y = size 
        simulation.background = initial_background
        simulation.background_angle = background_angle
        simulation.perturbation_type = perturbation_type
        simulation.perturbation_angle = perturbation_angle
        simulation.perturbation_size = perturbation_size
        simulation.initial_condition()  
    else: 
        simulation.dt = dt
        simulation.boundary_conditions = boundary_conditions
        # Reset FDT if required 
        damping_noise_temp = [damping, noise, temp]
        damping_noise_temp[fixed_fdt_parameter] = None 
        simulation.damping_noise_temp = damping_noise_temp
        simulation.initialize_FDT()
        simulation.model_type = model_type
        simulation.active_component = active_component
        simulation.initialize_active_force()

        simulation.lattice = simulation.runge_kutta_step() % (np.pi * 2)





    fig = simulation.poltly_configuration(arrow_number =None, configuration=simulation.lattice.T)

    fig.update_layout(figure_layout)


    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
