import matplotlib
import matplotlib.pyplot as plt
import io
import numpy as np

def create_boxplot(data):
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=128)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    return fig.canvas.tostring_rgb()

def prettify_name(name):
    if "test" in name:
        name = name.replace('test_', '')
    name = name.replace('_', ' ')
    return name[0].upper() + name[1:]

def filter_out(names):
    return filter(lambda name: "train" not in name and "time" not in name, names)

def build_cv_graphics(scores):
    #matplotlib.use('Agg')

    result = {}
    metric_names = list(scores.keys())
    filtered_metrics = filter_out(metric_names)

    for metric_name in filtered_metrics:
        img = create_boxplot(scores[metric_name])
        pretty_name = prettify_name(metric_name)
        result[pretty_name] = img

    return result