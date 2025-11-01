import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import io
from PIL import Image
import cv2

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cluster-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    .movie-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        background: white;
    }
    .movie-card:hover {
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
        transform: translateY(-5px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df_pca = pd.read_csv('pca_features_clustered.csv')
        df_umap = pd.read_csv('umap_features_clustered.csv')

        if 'cluster' in df_pca.columns:
            df = df_pca
            method = 'PCA'
        elif 'cluster' in df_umap.columns:
            df = df_umap
            method = 'UMAP'
        else:
            st.error("No se encontró la columna 'cluster' en los archivos")
            return None, None, None
    except FileNotFoundError:
        st.error("No se encontraron los archivos clusterizados. Ejecuta el script de K-means primero.")
        return None, None, None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['cluster', 'movieId', 'id', 'index', 'year']]
    return df, feature_cols, method

@st.cache_data
def load_movie_metadata():
    try:
        metadata = pd.read_csv('movies_train.csv')
        return metadata
    except:
        return None

def extract_features_from_image(image):
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (100, 150))
    features = []
    for channel in range(3):
        hist = cv2.calcHist([img_resized], [channel], None, [32], [0, 256])
        features.extend(hist.flatten())
    features.extend([
        img_resized.mean(),
        img_resized.std(),
        img_resized.min(),
        img_resized.max()
    ])
    return np.array(features)

def find_similar_movies(df, feature_cols, query_features, top_k=10):
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    query_scaled = scaler.transform(query_features.reshape(1, -1))
    similarities = cosine_similarity(query_scaled, X_scaled)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results

def get_cluster_representatives(df, feature_cols, n_per_cluster=5):
    representatives = []
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        centroid = cluster_data[feature_cols].mean().values
        distances = np.linalg.norm(cluster_data[feature_cols].values - centroid, axis=1)
        closest_indices = np.argsort(distances)[:n_per_cluster]
        cluster_reps = cluster_data.iloc[closest_indices].copy()
        cluster_reps['distance_to_centroid'] = distances[closest_indices]
        representatives.append(cluster_reps)
    return pd.concat(representatives, ignore_index=True)

def plot_clusters_interactive(df, feature_cols, color_by='cluster'):
    plot_data = df.copy()
    plot_data['x'] = df[feature_cols[0]]
    plot_data['y'] = df[feature_cols[1]]

    hover_cols = ['x', 'y', 'cluster']
    if 'title' in df.columns:
        hover_cols.append('title')
    if 'genres' in df.columns:
        hover_cols.append('genres')
    if 'year' in df.columns:
        hover_cols.append('year')

    fig = px.scatter(
        plot_data,
        x='x',
        y='y',
        color=color_by,
        hover_data=hover_cols,
        title='Distribución de Películas en Espacio de Características',
        color_continuous_scale='viridis' if color_by != 'cluster' else None,
        height=600,
        labels={'x': 'Componente 1', 'y': 'Componente 2'}
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')))
    fig.update_layout(
        plot_bgcolor='rgba(240, 242, 246, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

def plot_cluster_distribution(df):
    cluster_counts = df['cluster'].value_counts().sort_index()
    colors = px.colors.qualitative.Set3[:len(cluster_counts)]
    fig = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in cluster_counts.index],
            y=cluster_counts.values,
            marker_color=colors,
            text=cluster_counts.values,
            textposition='auto',
        )
    ])
    fig.update_layout(
        title='Distribución de Películas por Cluster',
        xaxis_title='Cluster',
        yaxis_title='Número de Películas',
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(240, 242, 246, 0.5)',
        paper_bgcolor='white'
    )
    return fig

def display_movie_card(movie_data, show_similarity=False):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://via.placeholder.com/150x225/667eea/white?text=Movie+Poster",
                 use_container_width=True)
    with col2:
        title = movie_data.get('title', f"Movie ID: {movie_data.get('movieId', 'Unknown')}")
        st.markdown(f"**{title}**")
        cluster_color = px.colors.qualitative.Set3[int(movie_data['cluster']) % 10]
        st.markdown(
            f'<span class="cluster-badge" style="background-color: {cluster_color};">'
            f'Cluster {int(movie_data["cluster"])}</span>',
            unsafe_allow_html=True
        )
        info_cols = st.columns(3)
        if 'genres' in movie_data:
            info_cols[0].write(f" {movie_data['genres']}")
        if 'year' in movie_data:
            info_cols[1].write(f" {movie_data['year']}")
        if show_similarity and 'similarity' in movie_data:
            info_cols[2].write(f" Similitud: {movie_data['similarity']:.2%}")

def main():
    st.markdown('<h1 class="main-header"> Sistema de Recomendación de Películas</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    df, feature_cols, method = load_data()
    metadata = load_movie_metadata()
    if df is None:
        st.stop()

    # Combinar con metadatos si están disponibles
    if metadata is not None:
        if 'movieId' in df.columns and 'movieId' in metadata.columns:
            df = df.merge(metadata[['movieId', 'title', 'genres']], on='movieId', how='left')

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/white?text=Movie+Recommender",
                 use_container_width=True)
        st.markdown("---")
        st.markdown("### Estadísticas del Dataset")
        st.metric("Total de Películas", len(df))
        st.metric("Número de Clusters", df['cluster'].nunique())
        st.metric("Método de Reducción", method)
        st.metric("Dimensiones de Features", len(feature_cols))
        st.markdown("---")
        st.markdown("### Información")
        st.info("""
        Este sistema agrupa películas basándose en características visuales 
        extraídas de sus pósters usando técnicas de clustering.
        """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Búsqueda por Similitud",
        "Clusters y Representantes",
        "Visualización 2D",
        "Filtros Avanzados"
    ])

    with tab1:
        st.markdown('<div class="sub-header">Buscar Películas Similares</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("####Opción 1: Seleccionar de la Base de Datos")
            filter_cluster = st.checkbox("Filtrar por cluster específico")
            if filter_cluster:
                selected_cluster = st.selectbox(
                    "Selecciona un cluster",
                    sorted(df['cluster'].unique())
                )
                search_df = df[df['cluster'] == selected_cluster]
            else:
                search_df = df

            if 'title' in df.columns:
                movie_options = search_df['title'].dropna().tolist()
                if movie_options:
                    selected_movie = st.selectbox("Selecciona una película", movie_options)
                    if st.button("Buscar Similares", key="search_db"):
                        movie_data = df[df['title'] == selected_movie].iloc[0]
                        query_features = movie_data[feature_cols].values
                        with st.spinner("Buscando películas similares..."):
                            similar_movies = find_similar_movies(
                                df, feature_cols, query_features, top_k=10
                            )
                        st.success(f"Se encontraron {len(similar_movies)} películas similares")
                        st.markdown("### Películas Similares")
                        for idx, movie in similar_movies.iterrows():
                            with st.container():
                                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                                display_movie_card(movie, show_similarity=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No hay películas con títulos disponibles")
            else:
                movie_ids = search_df.index.tolist()
                selected_id = st.selectbox("Selecciona ID de película", movie_ids)
                if st.button("Buscar Similares", key="search_db_id"):
                    query_features = df.iloc[selected_id][feature_cols].values
                    with st.spinner("Buscando películas similares..."):
                        similar_movies = find_similar_movies(
                            df, feature_cols, query_features, top_k=10
                        )
                    st.success(f" Se encontraron {len(similar_movies)} películas similares")
                    for idx, movie in similar_movies.iterrows():
                        display_movie_card(movie, show_similarity=True)

        with col2:
            st.markdown("####  Opción 2: Subir una Imagen")
            st.info("⚠️ Funcionalidad en desarrollo: requiere el mismo pipeline de features que el entrenamiento.")
            uploaded_file = st.file_uploader(
                "Sube un póster de película",
                type=['jpg', 'jpeg', 'png'],
                help="Sube una imagen de un póster para encontrar películas visualmente similares"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen subida", use_container_width=True)
                if st.button("Buscar por Imagen"):
                    st.warning("""
                    Esta funcionalidad requiere:
                    1) El modelo de extracción de features entrenado
                    2) Procesar la imagen con el mismo pipeline usado en el entrenamiento
                    """)

    with tab2:
        st.markdown('<div class="sub-header">Explorar Clusters de Películas</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_dist = plot_cluster_distribution(df)
            # KEY ÚNICO
            st.plotly_chart(fig_dist, use_container_width=True, key="plot_cluster_dist")

        with col2:
            st.markdown("### Estadísticas por Cluster")
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_size = len(df[df['cluster'] == cluster_id])
                percentage = (cluster_size / len(df)) * 100
                st.metric(f"Cluster {cluster_id}", f"{cluster_size} películas", f"{percentage:.1f}%")

        st.markdown("---")
        st.markdown("### Películas Representativas de Cada Cluster")
        n_representatives = st.slider(
            "Número de representantes por cluster", min_value=3, max_value=10, value=5
        )
        with st.spinner("Calculando películas representativas..."):
            representatives = get_cluster_representatives(df, feature_cols, n_representatives)

        for cluster_id in sorted(representatives['cluster'].unique()):
            with st.expander(f"🎬 Cluster {cluster_id}", expanded=True):
                cluster_reps = representatives[representatives['cluster'] == cluster_id]
                cols = st.columns(min(3, len(cluster_reps)))
                for idx, (_, movie) in enumerate(cluster_reps.iterrows()):
                    with cols[idx % 3]:
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                        display_movie_card(movie, show_similarity=False)
                        st.caption(f"Distancia al centroide: {movie['distance_to_centroid']:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="sub-header">Visualización del Espacio de Características</div>',
                    unsafe_allow_html=True)
        st.info("""
        Esta visualización muestra la distribución de películas en un espacio 2D.
        Películas cercanas tienen características visuales similares.
        """)

        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown("### Opciones de Visualización")
            color_option = st.radio("Colorear por:", ["Cluster", "Año (si disponible)", "Género (si disponible)"])
            show_all = st.checkbox("Mostrar todas las películas", value=True)
            if not show_all:
                sample_size = st.slider(
                    "Número de películas a mostrar",
                    min_value=100,
                    max_value=min(5000, len(df)),
                    value=min(1000, len(df))
                )
                plot_df = df.sample(n=sample_size, random_state=42)
            else:
                plot_df = df

        with col1:
            if color_option == "Cluster":
                color_by = 'cluster'
            elif color_option == "Año (si disponible)" and 'year' in df.columns:
                color_by = 'year'
            elif color_option == "Género (si disponible)" and 'genres' in df.columns:
                plot_df = plot_df.copy()
                plot_df['main_genre'] = plot_df['genres'].str.split('|').str[0]
                color_by = 'main_genre'
            else:
                color_by = 'cluster'

            fig_scatter = plot_clusters_interactive(plot_df, feature_cols, color_by)
            # KEY ÚNICO
            st.plotly_chart(fig_scatter, use_container_width=True, key="plot_scatter_main")

        st.markdown("### Análisis Estadístico")
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Películas Visualizadas", len(plot_df))
        with stat_cols[1]:
            st.metric("Clusters Únicos", plot_df['cluster'].nunique())
        with stat_cols[2]:
            avg_cluster_size = len(plot_df) / plot_df['cluster'].nunique()
            st.metric("Tamaño Promedio de Cluster", f"{avg_cluster_size:.1f}")
        with stat_cols[3]:
            if 'genres' in plot_df.columns:
                unique_genres = plot_df['genres'].str.split('|').explode().nunique()
                st.metric("Géneros Únicos", unique_genres)

    with tab4:
        st.markdown('<div class="sub-header">Búsqueda y Filtros Avanzados</div>',
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Filtrar por Cluster")
            selected_clusters = st.multiselect(
                "Selecciona clusters",
                options=sorted(df['cluster'].nunique() and df['cluster'].unique()),
                default=sorted(df['cluster'].unique())
            )
        with col2:
            if 'year' in df.columns:
                st.markdown("#### Filtrar por Año")
                min_year = int(df['year'].min()) if pd.notna(df['year'].min()) else 1900
                max_year = int(df['year'].max()) if pd.notna(df['year'].max()) else 2024
                year_range = st.slider("Rango de años", min_value=min_year, max_value=max_year, value=(min_year, max_year))
            else:
                year_range = None
        with col3:
            if 'genres' in df.columns:
                st.markdown("#### Filtrar por Género")
                all_genres = df['genres'].str.split('|').explode().unique()
                all_genres = [g for g in all_genres if pd.notna(g)]
                selected_genres = st.multiselect("Selecciona géneros", options=sorted(all_genres))
            else:
                selected_genres = []

        filtered_df = df[df['cluster'].isin(selected_clusters)]
        if year_range and 'year' in df.columns:
            filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
        if selected_genres and 'genres' in df.columns:
            mask = filtered_df['genres'].apply(lambda x: any(genre in str(x) for genre in selected_genres))
            filtered_df = filtered_df[mask]

        st.markdown("---")
        st.markdown(f"### Resultados: {len(filtered_df)} películas encontradas")

        if len(filtered_df) > 0:
            view_option = st.radio("Vista:", ["Lista", "Tabla", "Gráfico"], horizontal=True)
            if view_option == "Lista":
                n_cols = 3
                rows = (len(filtered_df) + n_cols - 1) // n_cols
                for row in range(min(rows, 10)):  # limitar a 10 filas
                    cols = st.columns(n_cols)
                    for col_idx in range(n_cols):
                        idx = row * n_cols + col_idx
                        if idx < len(filtered_df):
                            with cols[col_idx]:
                                movie = filtered_df.iloc[idx]
                                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                                display_movie_card(movie)
                                st.markdown('</div>', unsafe_allow_html=True)

                if len(filtered_df) > 30:
                    st.info(f"Mostrando las primeras 30 de {len(filtered_df)} películas")

            elif view_option == "Tabla":
                display_cols = ['cluster']
                if 'title' in filtered_df.columns:
                    display_cols.append('title')
                if 'genres' in filtered_df.columns:
                    display_cols.append('genres')
                if 'year' in filtered_df.columns:
                    display_cols.append('year')

                st.dataframe(filtered_df[display_cols].head(100), use_container_width=True, height=400)

                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Descargar resultados (CSV)",
                    data=csv,
                    file_name="peliculas_filtradas.csv",
                    mime="text/csv"
                )
            else:  # Gráfico
                fig = plot_clusters_interactive(filtered_df, feature_cols, 'cluster')
                # KEY ÚNICO
                st.plotly_chart(fig, use_container_width=True, key="plot_filtered_scatter")
        else:
            st.warning("No se encontraron películas con los filtros seleccionados")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p> Sistema de Recomendación de Películas basado en Clustering Visual</p>
        <p>Desarrollado con usando Streamlit, Scikit-learn y Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
