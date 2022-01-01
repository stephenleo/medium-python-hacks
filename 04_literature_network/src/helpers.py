import streamlit as st
from pyvis.network import Network
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import networkx as nx
import textwrap


def reset_default_topic_sliders(min_topic_size, n_gram_range):
    st.session_state['min_topic_size'] = min_topic_size
    st.session_state['n_gram_range'] = n_gram_range


def reset_default_threshold_slider(threshold):
    st.session_state['threshold'] = threshold


@st.cache()
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)

    return data


@st.cache()
def embedding_gen(data):
    return SentenceTransformer('allenai-specter').encode(data['Text'])


@st.cache()
def load_bertopic_model(min_topic_size, n_gram_range):
    return BERTopic(
        vectorizer_model=CountVectorizer(
            stop_words='english', ngram_range=n_gram_range
        ),
        min_topic_size=min_topic_size,
        verbose=True
    )


@st.cache()
def topic_modeling(data, min_topic_size, n_gram_range):
    """Topic modeling using BERTopic
    """
    topic_model = load_bertopic_model(min_topic_size, n_gram_range)

    # Train the topic model
    topic_data = data.copy()
    topic_data["Topic"], topic_data["Probs"] = topic_model.fit_transform(
        data['Text'], embeddings=embedding_gen(data))

    # Merge topic results
    topic_df = topic_model.get_topic_info()
    topic_df.columns = ['Topic', 'Topic_Count', 'Topic_Name']
    topic_df = topic_df.sort_values(by='Topic_Count', ascending=False)
    topic_data = topic_data.merge(topic_df, on='Topic', how='left')

    # Topics
    # Optimization: Only take top 10 largest topics
    topics = topic_df.head(10).set_index('Topic').to_dict(orient='index')

    return topic_data, topic_model, topics


@st.cache()
def cosine_sim(data):
    cosine_sim_matrix = cosine_similarity(embedding_gen(data))

    # Take only upper triangular matrix
    cosine_sim_matrix = np.triu(cosine_sim_matrix, k=1)

    return cosine_sim_matrix


@st.cache()
def calc_max_connections(num_papers, ratio):
    n = ratio*num_papers

    return n*(n-1)/2


@st.cache()
def calc_optimal_threshold(cosine_sim_matrix, max_connections):
    """Calculates the optimal threshold for the cosine similarity matrix.
    Allows a max of max_connections
    """
    thresh_sweep = np.arange(0.05, 1.05, 0.05)[::-1]
    for idx, threshold in enumerate(thresh_sweep):
        neighbors = np.argwhere(cosine_sim_matrix >= threshold).tolist()
        if len(neighbors) > max_connections:
            break

    return round(thresh_sweep[idx-1], 2).item(), round(thresh_sweep[idx], 2).item()


@st.cache()
def calc_neighbors(cosine_sim_matrix, threshold):
    neighbors = np.argwhere(cosine_sim_matrix >= threshold).tolist()

    return neighbors, len(neighbors)


def nx_hash_func(nx_net):
    """Hash function for NetworkX graphs.
    """
    return (list(nx_net.nodes()), list(nx_net.edges()))


def pyvis_hash_func(pyvis_net):
    """Hash function for pyvis graphs.
    """
    return (pyvis_net.nodes, pyvis_net.edges)


@st.cache(hash_funcs={nx.Graph: nx_hash_func, Network: pyvis_hash_func})
def network_plot(topic_data, topics, neighbors):
    """Creates a network plot of connected papers. Colored by Topic Model topics.
    """
    nx_net = nx.Graph()
    pyvis_net = Network(height='750px', width='100%', bgcolor='#222222')

    # Add Nodes
    nodes = [
        (
            row.Index,
            {
                'group': row.Topic,
                'label': row.Index,
                'title': row.Text,
                'size': 20, 'font': {'size': 20, 'color': 'white'}
            }
        )
        for row in topic_data.itertuples()
    ]
    nx_net.add_nodes_from(nodes)
    assert(nx_net.number_of_nodes() == len(topic_data))

    # Add Edges
    nx_net.add_edges_from(neighbors)
    assert(nx_net.number_of_edges() == len(neighbors))

    # Optimization: Remove Isolated nodes
    nx_net.remove_nodes_from(list(nx.isolates(nx_net)))

    # Add Legend Nodes
    step = 150
    x = -2000
    y = -500
    legend_nodes = [
        (
            len(topic_data)+idx,
            {
                'group': key, 'label': ', '.join(value['Topic_Name'].split('_')[1:]),
                'size': 30, 'physics': False, 'x': x, 'y': f'{y + idx*step}px',
                # , 'fixed': True,
                'shape': 'box', 'widthConstraint': 1000, 'font': {'size': 40, 'color': 'black'}
            }
        )
        for idx, (key, value) in enumerate(topics.items())
    ]
    nx_net.add_nodes_from(legend_nodes)

    # Plot the Pyvis graph
    pyvis_net.from_nx(nx_net)

    return nx_net, pyvis_net


def text_processing(text):
    text = text.split('[SEP]')
    text = '<br><br>'.join(text)
    text = '<br>'.join(textwrap.wrap(text, width=50))[:500]
    text = text + '...'
    return text


@st.cache()
def network_centrality(topic_data, centrality, centrality_option):
    """Calculates the centrality of the network
    """
    # Sort Top 10 Central nodes
    central_nodes = sorted(
        centrality.items(), key=lambda item: item[1], reverse=True)
    central_nodes = pd.DataFrame(central_nodes, columns=[
                                 'node', centrality_option]).set_index('node')

    joined_data = topic_data.join(central_nodes)

    top_central_nodes = joined_data.sort_values(
        centrality_option, ascending=False).head(10)

    # Prepare for plot
    top_central_nodes = top_central_nodes.reset_index()
    top_central_nodes['index'] = top_central_nodes['index'].astype(str)
    top_central_nodes['Topic_Name'] = top_central_nodes['Topic_Name'].apply(
        lambda x: ', '.join(x.split('_')[1:]))
    top_central_nodes['Text'] = top_central_nodes['Text'].apply(
        text_processing)

    # Plot the Top 10 Central nodes
    fig = px.bar(top_central_nodes, x=centrality_option, y='index',
                 color='Topic_Name', hover_data=['Text'], orientation='h')
    fig.update_layout(yaxis={'categoryorder': 'total ascending', 'visible': False, 'showticklabels': False},
                      font={'size': 15}, height=800)
    return fig
