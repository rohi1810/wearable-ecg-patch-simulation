import numpy as np
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from functools import lru_cache
import psutil

# Initialize the app
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)

app.layout = dbc.Container([
    html.H1("HRV Simulation Dashboard", className='mb-4 text-center'),

    dbc.Row([
        dbc.Col([
            html.Label("Select Condition:", className='font-weight-bold'),
            dcc.RadioItems(
                id='condition-selector',
                options=[
                    {'label': ' Rest', 'value': 'rest'},
                    {'label': ' Stress', 'value': 'stress'},
                    {'label': ' Exercise', 'value': 'exercise'}
                ],
                value='rest',
                labelStyle={'display': 'block', 'margin': '5px'}
            )
        ], width=4),

        dbc.Col([
            html.Label("Select Duration:", className='font-weight-bold'),
            dcc.Dropdown(
                id='duration-selector',
                options=[
                    {'label': '1 minute', 'value': 1},
                    {'label': '5 minutes', 'value': 5},
                    {'label': '10 minutes', 'value': 10},
                    {'label': '30 minutes', 'value': 30},
                    {'label': '1 hour', 'value': 60},
                    {'label': '4 hours', 'value': 240},
                    {'label': '8 hours', 'value': 480},
                    {'label': '12 hours', 'value': 720},
                    {'label': '24 hours', 'value': 1440}
                ],
                value=60,
                clearable=False
            )
        ], width=4)
    ], className='mb-4'),

    dcc.Graph(id='hrv-plot'),
    dcc.Graph(id='hr-plot'),
    dcc.Graph(id='ecg-plot'),
    html.Div(id='memory-usage', className='mt-2 text-muted small')
], fluid=True)


@lru_cache(maxsize=3)
def generate_hrv_data(duration_min, condition):
    duration_sec = duration_min * 60

    if duration_min <= 10:
        fs = 1.0
    elif duration_min <= 60:
        fs = 0.5
    else:
        fs = 0.1

    time = np.linspace(0, duration_sec, int(duration_sec * fs), endpoint=False)

    if condition == 'rest': # low hr, high variability
        base_hr = 60 + 5 * np.sin(2 * np.pi * time / (0 * 20)) # oscillates slowly (every 20 minutes)
        variability = 0.1 #	 time between each heartbeat changes
    elif condition == 'stress': # mid hr, low variability
        base_hr = 75 + 3 * np.sin(2 * np.pi * time / (60 * 10))
        variability = 0.05
    else:  # exercise , high hr, medium variability
        base_hr = 100 + 10 * np.sin(2 * np.pi * time / (60 * 5))
        variability = 0.08

    rr_intervals = (60 / base_hr) * (1 + variability * np.random.randn(len(time)))
    heart_rate = 60 / rr_intervals

    return time, rr_intervals, heart_rate


def calculate_hrv_metrics(rr_intervals):
    rr_ms = rr_intervals * 1000  # convert to ms
    diff_rr = np.diff(rr_ms)

    sdnn = np.std(rr_ms) # standard deviation high sdnn - healthier, lower sdnn - stress
    rmssd = np.sqrt(np.mean(diff_rr**2)) # root mean square of successive differences
    nn50 = np.sum(np.abs(diff_rr) > 50) # percentage of successive RR intervals
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0

    return {
        "SDNN (ms)": sdnn,
        "RMSSD (ms)": rmssd,
        "pNN50 (%)": pnn50
    }


def generate_ecg_waveform(duration_min, condition):
    # ECG sampling frequency (typical for wearable ECG)
    fs_ecg = 250  # 250 Hz

    duration_sec = duration_min * 60
    t = np.linspace(0, duration_sec, int(fs_ecg * duration_sec), endpoint=False)

    # Heart rate baseline per condition
    if condition == 'rest':
        hr_base = 60
    elif condition == 'stress':
        hr_base = 75
    else:  # exercise
        hr_base = 100

    # Heart period in seconds (beat-to-beat interval)
    # Add some variability using sin + noise
    instantaneous_hr = hr_base + 5 * np.sin(2 * np.pi * t / 20)  # slow sinusoidal variation
    instantaneous_hr += np.random.normal(0, 1, size=t.shape)  # small noise
    instantaneous_hr = np.clip(instantaneous_hr, 40, 180)

    instantaneous_rr = 60 / instantaneous_hr  # seconds per beat

    # Generate heartbeat template (one beat), duration ~1s (250 samples)
    beat_length = int(fs_ecg * 1)  # samples per beat

    # Simple synthetic ECG beat template (P, QRS, T waves)
    beat = np.zeros(beat_length)
    # P wave ~ small positive bump at 0.2s
    beat[int(0.2 * fs_ecg):int(0.25 * fs_ecg)] = 0.1 * np.sin(np.linspace(0, np.pi, int(0.05 * fs_ecg)))
    # Q wave ~ small negative deflection before R
    beat[int(0.4 * fs_ecg):int(0.42 * fs_ecg)] = -0.15
    # R wave ~ big positive peak at 0.43s
    beat[int(0.42 * fs_ecg):int(0.45 * fs_ecg)] = 1.0
    # S wave ~ negative dip at 0.46s
    beat[int(0.45 * fs_ecg):int(0.48 * fs_ecg)] = -0.25
    # T wave ~ medium positive bump at 0.6s to 0.8s
    beat[int(0.6 * fs_ecg):int(0.8 * fs_ecg)] = 0.3 * np.sin(np.linspace(0, np.pi, int(0.2 * fs_ecg)))

    # Construct ECG by repeating the beat at intervals defined by instantaneous RR intervals
    ecg_signal = np.zeros_like(t)
    current_idx = 0
    beat_idx = 0

    while current_idx < len(t):
        # Calculate beat duration in samples from instantaneous RR at current index
        if beat_idx < len(instantaneous_rr):
            rr_sec = instantaneous_rr[beat_idx]
        else:
            rr_sec = 60 / hr_base  # fallback

        beat_samples = int(rr_sec * fs_ecg)

        # Add beat template scaled to beat_samples length
        if current_idx + beat_samples > len(t):
            # last beat truncated
            ecg_signal[current_idx:] = beat[:len(t) - current_idx]
            break
        else:
            # Stretch or compress beat template to fit beat_samples
            beat_resampled = np.interp(
                np.linspace(0, beat_length, beat_samples, endpoint=False),
                np.arange(beat_length),
                beat
            )
            ecg_signal[current_idx:current_idx + beat_samples] = beat_resampled
            current_idx += beat_samples
            beat_idx += 1

    # Add some baseline noise
    noise = 0.05 * np.random.randn(len(t))
    ecg_signal += noise

    return t, ecg_signal


@callback(
    [Output('hrv-plot', 'figure'),
     Output('hr-plot', 'figure'),
     Output('ecg-plot', 'figure'),
     Output('memory-usage', 'children')],
    [Input('condition-selector', 'value'),
     Input('duration-selector', 'value')]
)
def update_plots(condition, duration_min):
    time, rr_intervals, heart_rate = generate_hrv_data(duration_min, condition)

    metrics = calculate_hrv_metrics(rr_intervals)
    metrics_text = " | ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])

    # HRV plot
    hrv_fig = go.Figure()
    hrv_fig.add_trace(go.Scatter(
        x=time / 60,
        y=rr_intervals * 1000,
        mode='lines',
        name='RR Intervals',
        line=dict(width=1)
    ))
    hrv_fig.update_layout(
        title=f'HRV - {condition.capitalize()} Condition',
        xaxis_title='Time (minutes)',
        yaxis_title='RR Interval (ms)',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Heart Rate plot
    hr_fig = go.Figure()
    hr_fig.add_trace(go.Scatter(
        x=time / 60,
        y=heart_rate,
        mode='lines',
        name='Heart Rate',
        line=dict(color='red', width=1)
    ))
    hr_fig.update_layout(
        title=f'Heart Rate - {condition.capitalize()} Condition',
        xaxis_title='Time (minutes)',
        yaxis_title='Heart Rate (bpm)',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # ECG waveform plot
    t_ecg, ecg_signal = generate_ecg_waveform(duration_min, condition)
    ecg_fig = go.Figure()
    ecg_fig.add_trace(go.Scatter(
        x=t_ecg,
        y=ecg_signal,
        mode='lines',
        name='ECG Signal',
        line=dict(color='green', width=1)
    ))
    ecg_fig.update_layout(
        title=f'ECG Waveform - {condition.capitalize()} Condition',
        xaxis_title='Time (seconds)',
        yaxis_title='Voltage (mV)',
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )

    mem_usage = psutil.Process().memory_info().rss / (1024 * 1024)
    mem_info = f"Memory usage: {mem_usage:.1f} MB | Points: {len(time):,} | {metrics_text}"

    return hrv_fig, hr_fig, ecg_fig, mem_info


if __name__ == '__main__':
    app.run(debug=True)
