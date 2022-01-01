import networkx as nx
from streamlit.components.v1 import html
import streamlit as st
import helpers
import logging

# Setup Basic Configuration
st.set_page_config(layout='wide',
                   page_title='STriP: Semantic Similarity of Scientific Papers!',
                   page_icon='💡'
                   )


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('main')


def load_data():
    """Loads the data from the uploaded file.
    """

    st.header('📂 Load Data')
    uploaded_file = st.file_uploader("Choose a CSV file",
                                     help='Upload a CSV file with the following columns: Title, Abstract')

    if uploaded_file is not None:
        df = helpers.load_data(uploaded_file)
    else:
        df = helpers.load_data('data.csv')
    data = df.copy()

    # Column Selection. By default, any column called 'title' and 'abstract' are selected
    st.subheader('Select columns to analyze')
    selected_cols = st.multiselect(label='Select one or more columns. All the selected columns are concatenated before analyzing', options=data.columns,
                                   default=[col for col in data.columns if col.lower() in ['title', 'abstract']])

    if not selected_cols:
        st.error('No columns selected! Please select some text columns to analyze')

    data = data[selected_cols]

    # Minor cleanup
    data = data.dropna()
    data = data.reset_index(drop=True)

    # Load max 200 rows only
    st.write(f'Number of rows: {len(data)}')
    if len(data) > 200:
        data = data.iloc[:200]
        st.write(f'Only first 200 rows will be analyzed')

    # Prints
    st.write('First 5 rows of loaded data:')
    st.write(data[selected_cols].head())

    # Combine all selected columns
    if (data is not None) and selected_cols:
        data['Text'] = data[data.columns[0]]
        for column in data.columns[1:]:
            data['Text'] = data['Text'] + '[SEP]' + data[column].astype(str)

    return data, selected_cols


def topic_modeling(data):
    """Runs the topic modeling step.
    """

    st.header('🔥 Topic Modeling')
    cols = st.columns(3)
    with cols[0]:
        min_topic_size = st.slider('Minimum topic size', key='min_topic_size', min_value=2,
                                   max_value=min(round(len(data)*0.25), 100), step=1, value=min(round(len(data)/25), 10),
                                   help='The minimum size of the topic. Increasing this value will lead to a lower number of clusters/topics.')
    with cols[1]:
        n_gram_range = st.slider('N-gram range', key='n_gram_range', min_value=1,
                                 max_value=3, step=1, value=(1, 2),
                                 help='N-gram range for the topic model')
    with cols[2]:
        st.text('')
        st.text('')
        st.button('Reset Defaults', on_click=helpers.reset_default_topic_sliders, key='reset_topic_sliders',
                  kwargs={'min_topic_size': min(round(len(data)/25), 10), 'n_gram_range': (1, 2)})

    with st.spinner('Topic Modeling'):
        with helpers.st_stdout("success"), helpers.st_stderr("code"):
            topic_data, topic_model, topics = helpers.topic_modeling(
                data, min_topic_size=min_topic_size, n_gram_range=n_gram_range)

        mapping = {
            'Topic Keywords': topic_model.visualize_barchart,
            'Topic Similarities': topic_model.visualize_heatmap,
            'Topic Hierarchies': topic_model.visualize_hierarchy,
            'Intertopic Distance': topic_model.visualize_topics
        }

        cols = st.columns(3)
        with cols[0]:
            topic_model_vis_option = st.selectbox(
                'Select Topic Modeling Visualization', mapping.keys())
        try:
            fig = mapping[topic_model_vis_option](top_n_topics=10)
            fig.update_layout(title='')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning(
                'No visualization available. Try a lower Minimum topic size!')

    return topic_data, topics


def strip_network(data, topic_data, topics):
    """Generated the STriP network.
    """

    st.header('🚀 STriP Network')

    with st.spinner('Cosine Similarity Calculation'):
        cosine_sim_matrix = helpers.cosine_sim(data)

    value, min_value = helpers.calc_optimal_threshold(
        cosine_sim_matrix,
        # 25% is a good value for the number of papers
        max_connections=min(
            helpers.calc_max_connections(len(data), 0.25), 5_000
        )
    )

    cols = st.columns(3)
    with cols[0]:
        threshold = st.slider('Cosine Similarity Threshold', key='threshold', min_value=min_value,
                              max_value=1.0, step=0.01, value=value,
                              help='The minimum cosine similarity between papers to draw a connection. Increasing this value will lead to a lesser connections.')

        neighbors, num_connections = helpers.calc_neighbors(
            cosine_sim_matrix, threshold)
        st.write(f'Number of connections: {num_connections}')

    with cols[1]:
        st.text('')
        st.text('')
        st.button('Reset Defaults', on_click=helpers.reset_default_threshold_slider, key='reset_threshold',
                  kwargs={'threshold': value})

    with st.spinner('Network Generation'):
        nx_net, pyvis_net = helpers.network_plot(
            topic_data, topics, neighbors)

        # Save and read graph as HTML file (on Streamlit Sharing)
        try:
            path = '/tmp'
            pyvis_net.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html',
                            'r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
        except:
            path = '/html_files'
            pyvis_net.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html',
                            'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        html(HtmlFile.read(), height=800)

        return nx_net


def network_centrality(nx_net, topic_data):
    """Finds most important papers using network centrality measures.
    """

    st.header('🏅 Most Important Papers')

    centrality_mapping = {
        'Closeness Centrality': nx.closeness_centrality,
        'Degree Centrality': nx.degree_centrality,
        'Eigenvector Centrality': nx.eigenvector_centrality,
        'Betweenness Centrality': nx.betweenness_centrality,
    }

    cols = st.columns(3)
    with cols[0]:
        centrality_option = st.selectbox(
            'Select Centrality Measure', centrality_mapping.keys())

    # Calculate centrality
    centrality = centrality_mapping[centrality_option](nx_net)

    cols = st.columns([1, 10, 1])
    with cols[1]:
        with st.spinner('Network Centrality Calculation'):
            fig = helpers.network_centrality(
                topic_data, centrality, centrality_option)
            st.plotly_chart(fig, use_container_width=True)


def about_me():
    st.markdown(
        """
        💡🔥🚀 STriP v1.0 🚀🔥💡

        👨‍🔬 Author: Marie Stephen Leo

        👔 Linkedin: [Marie Stephen Leo](https://www.linkedin.com/in/marie-stephen-leo/)

        📝 Medium: [@stephen-leo](https://stephen-leo.medium.com/)

        💻 Github: [stephenleo](https://github.com/stephenleo)
        """
    )


def main():
    st.title('STriP (S3P): Semantic Similarity of Scientific Papers!')

    logger.info('========== Step1: Loading data ==========')
    data, selected_cols = load_data()

    if (data is not None) and selected_cols:
        logger.info('========== Step2: Topic modeling ==========')
        topic_data, topics = topic_modeling(data)

        logger.info('========== Step3: STriP Network ==========')
        nx_net = strip_network(data, topic_data, topics)

        logger.info('========== Step4: Network Centrality ==========')
        network_centrality(nx_net, topic_data)

    about_me()


if __name__ == '__main__':
    main()


# Optimizations
# Pareto: Embedding Generation, Topic Modeling, Network Generation, Plotting
# 1. Generate embeddings exactly once - BERTopic generated embedding internally if not specified. Replace this with an external generated embeddings so it can be reused for Semantic Clustering
# 2. Generate topic model exactly once - Streamlit cache. Dynamic heuristically tuned hyperparameters
# 3. Generate network exactly once - Streamlit cache with custom hash funcs
# 4. Tune the threshold dynamically until the number of connections is within a certain range
# 5. Remove isolated nodes
# 6. Prune the legend to only top 10 topics
