import networkx as nx
from streamlit.components.v1 import html
import streamlit as st
import helpers
st.set_page_config(layout='wide',
                   page_title='STriP: Semantic Similarity of Scientific Papers!',
                   page_icon='💡'
                   )


def main():
    st.title('STriP (S3P): Semantic Similarity of Scientific Papers!')

    st.header('📂 Load Data')
    uploaded_file = st.file_uploader("Choose a CSV file",
                                     help='Upload a CSV file with the following columns: Title, Abstract')

    ##########
    # Load data
    ##########
    if uploaded_file is not None:
        df = helpers.load_data(uploaded_file)
    else:
        df = helpers.load_data('data.csv')

    data = df.copy()
    selected_cols = st.multiselect('Select columns to analyse', options=data.columns,
                                   default=[col for col in data.columns if col.lower() in ['title', 'abstract']])
    data = data[selected_cols]
    data = data.dropna()
    data = data.reset_index(drop=True)
    st.write(f'Number of papers: {len(data)}')
    st.write('First 5 rows of loaded data:')
    st.write(data[selected_cols].head())

    if data is not None:
        ##########
        # Topic modeling
        ##########
        st.header('🔥 Topic Modeling')

        cols = st.columns(3)
        with cols[0]:
            min_topic_size = st.slider('Minimum topic size', key='min_topic_size', min_value=2,
                                       max_value=round(len(data)/3), step=1, value=round(len(data)/25),
                                       help='The minimum size of the topic. Increasing this value will lead to a lower number of clusters/topics.')
        with cols[1]:
            n_gram_range = st.slider('N-gram range', key='n_gram_range', min_value=1,
                                     max_value=4, step=1, value=(1, 2),
                                     help='N-gram range for the topic model')
        with cols[2]:
            st.text('')
            st.text('')
            st.button('Reset Defaults', on_click=helpers.reset_default_topic_sliders, key='reset_topic_sliders',
                      kwargs={'min_topic_size': round(len(data)/25), 'n_gram_range': (1, 2)})

        with st.spinner('Topic Modeling'):
            data, topic_model, topics = helpers.topic_modeling(
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

        ##########
        # STriP Network
        ##########
        st.header('🚀 STriP Network')

        with st.spinner('Embedding generation'):
            data = helpers.embeddings(data)

        with st.spinner('Cosine Similarity Calculation'):
            cosine_sim_matrix = helpers.cosine_sim(data)

        min_value, value = helpers.calc_optimal_threshold(
            cosine_sim_matrix,
            # 25% is a good value for the number of papers
            max_connections=helpers.calc_max_connections(len(data), 0.25)
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
                data, topics, neighbors)

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

        ##########
        # Centrality
        ##########
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
                    data, centrality, centrality_option)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        💡🔥🚀 STriP v1.0 🚀🔥💡

        👨‍🔬 Author: Marie Stephen Leo

        👔 Linkedin: [Marie Stephen Leo](https://www.linkedin.com/in/marie-stephen-leo/)

        📝 Medium: [@stephen-leo](https://stephen-leo.medium.com/)

        💻 Github: [stephenleo](https://github.com/stephenleo)
        """
    )


if __name__ == '__main__':
    main()
