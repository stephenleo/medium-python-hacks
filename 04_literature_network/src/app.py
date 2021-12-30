import streamlit as st
st.set_page_config(layout="wide")
from streamlit.components.v1 import html

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

import networkx as nx
from pyvis.network import Network

# Initialize models
@st.cache
def initialize_models():
    embedding_model = SentenceTransformer('allenai-specter')

    return embedding_model

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)

    data = data[['Title', 'Abstract']]
    data = data.dropna()
    data = data.reset_index(drop=True)

    return data

def topic_modeling(data, embedding_model, min_topic_size, n_gram_range):
    topic_model = BERTopic(
        embedding_model=embedding_model, 
        n_gram_range=n_gram_range, 
        vectorizer_model=CountVectorizer(stop_words='english'),
        min_topic_size=min_topic_size
    )

    # For 'allenai-specter'
    data['Title + Abstract'] = data['Title'] + '[SEP]' + data['Abstract']

    # Train the topic model
    data["Topic"], data["Probs"] = topic_model.fit_transform(data['Title + Abstract'])

    # Merge topic results
    topic_df = topic_model.get_topic_info()[['Topic', 'Name']]
    data = data.merge(topic_df, on='Topic', how='left')

    # Topics
    topics = topic_df.set_index('Topic').to_dict(orient='index')

    return data, topic_model, topics

def embeddings(data, embedding_model):
    data['embedding'] = embedding_model.encode(data['Title + Abstract']).tolist()

    return data

def cosine_sim(data):
    cosine_sim_matrix = cosine_similarity(data['embedding'].values.tolist())

    # Take only upper triangular matrix
    cosine_sim_matrix = np.triu(cosine_sim_matrix, k=1)

    return cosine_sim_matrix

def network_plot(data, cosine_sim_matrix, topics, threshold = 0.85):
    nx_net = nx.Graph()
    pyvis_net = Network(height='750px', width='100%', notebook=True, bgcolor='#222222')

    neighbors = np.argwhere(cosine_sim_matrix >= threshold).tolist()

    # Add Nodes
    nodes = [
        (
            row.Index, 
            {
                'group': row.Topic, 
                'label': row.Index,
                'title': row.Title,
                'size': 20, 'font': {'size': 20, 'color': 'white'}
            }
        )
        for row in data.itertuples()
    ]
    nx_net.add_nodes_from(nodes)
    assert(nx_net.number_of_nodes() == len(data))

    # Add Legend Nodes
    step = 150
    x = -2000
    y = -500
    legend_nodes = [
        (
            len(data)+idx, 
            {
                'group': key, 'label': ', '.join(value['Name'].split('_')[1:]),
                'size': 30, 'physics': False, 'x': x, 'y': f'{y + idx*step}px',
                'shape': 'box', 'widthConstraint': 1000, 'font': {'size': 40, 'color': 'black'}#, 'fixed': True, 
            }
        )
        for idx, (key, value) in enumerate(topics.items())
    ]
    nx_net.add_nodes_from(legend_nodes)

    # Add Edges
    nx_net.add_edges_from(neighbors)
    assert(nx_net.number_of_edges() == len(neighbors))

    # Plot the Pyvis graph
    pyvis_net.from_nx(nx_net)
    # pyvis_net.show('nx.html')
    return nx_net, pyvis_net

st.title('Semantic Similarity of Scientific Papers')
uploaded_file = st.file_uploader("Choose a CSV file")

with st.spinner('Initializing models...'):
    embedding_model = initialize_models()

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.header('Loaded Data')
    st.write(data.head())
else:
    data = None

if data is not None:
    st.header('Topic Modeling')
    with st.spinner('Topic Modeling'):
        data, topic_model, topics = topic_modeling(data, embedding_model, min_topic_size=3, n_gram_range=(1,3))

        mapping = {
            'Topic Keywords': topic_model.visualize_barchart(),
            'Topic Similarities': topic_model.visualize_heatmap(),
            'Topic Hierarchies': topic_model.visualize_hierarchy(),
            'Intertopic Distance': topic_model.visualize_topics()
        }

        topic_model_vis_option = st.selectbox('Select Topic Modeling Visualization', mapping.keys())

        fig = mapping[topic_model_vis_option]
        fig.update_layout(title='')
        st.plotly_chart(fig, use_container_width=True)

if data is not None:
    st.header('Semantic Similarity Network')

    with st.spinner('Embedding generation'):
        data = embeddings(data, embedding_model)

    with st.spinner('Cosine Similarity Calculation'):
        cosine_sim_matrix = cosine_sim(data)

    with st.spinner('Network Generation'):
        nx_net, pyvis_net = network_plot(data, cosine_sim_matrix, topics)

        # Save and read graph as HTML file (on Streamlit Sharing)
        try:
            path = '/tmp'
            pyvis_net.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
        except:
            path = '/html_files'
            pyvis_net.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        html(HtmlFile.read(), height=1000)

if data is not None:
    st.header('Most Important Papers')

    centrality_mapping = {
        'Closeness Centrality': nx.closeness_centrality,
        'Degree Centrality': nx.degree_centrality,
        'Eigenvector Centrality': nx.eigenvector_centrality,
        'Betweenness Centrality': nx.betweenness_centrality,
    }

    centrality_option = st.selectbox('Select Centrality Measure', centrality_mapping.keys())

    with st.spinner('Network Centrality Calculation'):
        # Calculate centrality
        centrality = centrality_mapping[centrality_option](nx_net)

        # Sort Top 5 Central nodes
        central_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        central_nodes = pd.DataFrame(central_nodes, columns=['node', centrality_option]).set_index('node')

        data = data.join(central_nodes)
        top_central_nodes = data.sort_values(centrality_option, ascending=False).head()

        # Plot the Top 5 Central nodes
        fig = px.bar(top_central_nodes, x=centrality_option, y='Title')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500, width=1000)
        st.plotly_chart(fig, use_container_width=True)