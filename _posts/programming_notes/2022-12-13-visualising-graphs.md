---
title: Visualising Networks & Graphs in Python
blog_type: programming_notes
excerpt: Some libraries to plot graphs in Python.
layout: post
last_modified_at: 2022-12-13
---

### Networkx

{: .code title="Networkx code to visualise graph" .x}
``` python
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(graph:  Any, figsize=(6,6), **kwargs):
    G = nx.Graph(graph)
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=figsize)
    default_kwargs = {
        "node_size": 1300, "node_color": "#636EFA",
        "font_size": 25, "font_color": "lightgray",
    }
    default_kwargs.update(kwargs)
    nx.draw(G,pos, with_labels=True, **default_kwargs)

cyclic_graph = {
    'a': {'c', 'd', 'e', 'b'}, 'b': {'e', 'a'}, 'c': {'a'},
    'd': {'a'}, 'e': {'b', 'a'}, 'f': {},
}
draw_graph(cyclic_graph)
```

{% include image.html id="/assets/Images/posts/programming_notes/graph-viz-nx.png" width="50%" %}

### Plotly

Use the library ([github]((https://gist.github.com/mogproject/50668d3ca60188c50e6ef3f5f3ace101))) explained in [this blogpost](https://www.cs.utah.edu/~yos/2021/02/02/plotly-python.html).

{: .code title="Plotly code to visualise graph" .x}
``` python
import random
import plotly.express as px
import visualize as gvz
# Optional if you want to upload to chart-studio
import chart_studio.plotly as py

def _node_color(self, *args, **kwargs):
    return random.choice(px.colors.qualitative.Plotly[:3])

cyclic_graph = {
    'a': {'c', 'd', 'e', 'b'}, 'b': {'e', 'a'}, 'c': {'a'},
    'd': {'a'}, 'e': {'b', 'a'}, 'f': {},
}
fig = gvz.GraphVisualization(
    G=nx.Graph(non_cyclic_graph),
    # Describes where to place in 2-D. There are other layouts. The other one I
    # recommend is nx.spring_layout(G)
    pos=nx.kamada_kawai_layout(G),
    node_text_position='top center',
    node_color=_node_color,
    node_size=25,
    node_text_font_size={4: 30},
    edge_color={(0, 1): '#ff0000'},
).create_figure()
fig.show()

# Only if you want to upload to chart-studio
py.plot(fig, filename='Network graph plotting in plotly', auto_open=True)
```

<iframe width="100%" height="400" frameborder="0" scrolling="no" src="//plotly.com/~visperz/4.embed"></iframe>
