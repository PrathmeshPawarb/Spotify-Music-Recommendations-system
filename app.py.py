import pandas as pd
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as html_component

df = pd.read_csv('/content/spotify_tracks')

# Standardize numerical features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Standardize numerical features
df_numeric = pd.DataFrame(sc.fit_transform(df_numeric), columns=df_numeric.columns)
df_numeric = df_numeric.round(2)

# Convert the DataFrame to a sparse matrix
from scipy.sparse import csr_matrix
sparse_matrix = csr_matrix(df_numeric.values)


from annoy import AnnoyIndex

# Build Annoy Index
annoy_index = AnnoyIndex(sparse_matrix.shape[1], 'angular')
for i in range(sparse_matrix.shape[0]):
    vector = sparse_matrix.getrow(i).toarray().flatten()
    annoy_index.add_item(i, vector)
# decideing trees
annoy_index.build(75) 

def recommend_tracks_annoy(track_name, annoy_index, df, num_recommendations=5):
    track_index = df[df['track_name'] == track_name].index[0]
    num_recommendations = min(num_recommendations, annoy_index.get_n_items())  # Limit recommendations to index size
    related_tracks_indices = annoy_index.get_nns_by_item(track_index, num_recommendations)
    recommendations = df.iloc[related_tracks_indices][['track_name', 'artists','track_genre','popularity']]
    return recommendations



# Streamlit app
def main():
   

   # Header
    st.title("Spotify Track Recommender")

   # Sidebar
    st.sidebar.header("User Input")

    # Track selection dropdown
    selected_track = st.sidebar.selectbox("Select a Track", df['track_name'])

    # Number of recommendations
    num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    if st.sidebar.button("Get Recommendations"):
        recommendations_annoy = recommend_tracks_annoy(selected_track, annoy_index, df, num_recommendations)
        
        # Display recommendations
        st.subheader(f"Recommendations for track '{selected_track}':")
        st.table(recommendations_annoy)

if __name__ == "__main__":
    main()
