import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from operation_analysis import _plot_logs


def test_plot_logs_radiation_level(tmp_path, monkeypatch):
    rad_df = pd.DataFrame({
        'TimeStamp': [100, 101, 102],
        'RadiationLevel': [0.1, 0.2, 0.3],
        'Dose': [1.0, 2.0, 3.0],
    })
    power_df = pd.DataFrame()

    subplot_calls = []
    orig_subplot = plt.subplot
    def capture_subplot(*args, **kwargs):
        subplot_calls.append(args)
        return orig_subplot(*args, **kwargs)
    monkeypatch.setattr(plt, 'subplot', capture_subplot)

    plot_calls = []
    orig_plot = plt.plot
    def capture_plot(x, y, *args, **kwargs):
        plot_calls.append(list(x))
        return orig_plot(x, y, *args, **kwargs)
    monkeypatch.setattr(plt, 'plot', capture_plot)

    out = tmp_path / 'plot.png'
    _plot_logs(rad_df, power_df, str(out))

    assert out.exists()
    # Expect two subplots: RadiationLevel and Dose
    assert len(subplot_calls) == 2
    # Time axis should start at zero
    assert plot_calls[0][0] == 0
