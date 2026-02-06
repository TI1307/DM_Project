import streamlit as st
import pandas as pd

from app.data_loader import DataLoader
from app.recommendation_engine import RecommendationEngine
from app.clustering import ClusteringAnalysis
from app.visualizations import Visualizations

st.set_page_config(layout="wide", page_title="Library Analytics & Recommendations")

# -----------------------------
# Load and Prepare Everything
# -----------------------------
@st.cache_resource
def initialize_system():
    loader = DataLoader()
    recommender = RecommendationEngine(loader)
    clustering = ClusteringAnalysis(loader)
    clustering.prepare_data()
    clustering.create_profiles()
    
    # Run clustering model once so labels are available for recommendations
    with st.spinner("Preparing your personalized experience..."):
        clustering.prepare_features()
        clustering.cluster_kmeans(n_clusters=4)
        
    viz = Visualizations(loader)
    return loader, recommender, clustering, viz


loader, recommender, clustering, viz = initialize_system()

st.title("Library Data Analysis & Recommendation System")

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Library Analysis", "Recommend by Book", "Recommend by User"],
)

# ============================================================
# PAGE 1 — LIBRARY ANALYSIS WITH ALL VISUALIZATIONS
# ============================================================
if page == "Library Analysis":
    st.header("Library Overview & Analytics")

    # Get statistics
    stats = loader.get_statistics()
    total_books = stats.get("total_books", 0) or 0
    borrowed_books = stats.get("borrowed_books", 0) or 0
    unborrowed_books = stats.get("unborrowed_books", 0) or 0
    total_users = stats.get("unique_users", 0) or 0
    total_borrowings = stats.get("total_borrowings", 0) or 0

    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Books", f"{total_books:,}")
    col2.metric("Books in Use", f"{borrowed_books:,}")
    col3.metric("Available", f"{unborrowed_books:,}")
    col4.metric("Users", f"{total_users:,}")
    col5.metric("Total Checkouts", f"{total_borrowings:,}")

    st.divider()

    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_borrowings_per_user = stats.get("avg_borrowings_per_user", 0)
        st.metric("Avg Checkouts per User", f"{avg_borrowings_per_user:.1f}")
    with col2:
        avg_borrowings_per_book = stats.get("avg_borrowings_per_book", 0)
        st.metric("Avg Checkouts per Book", f"{avg_borrowings_per_book:.1f}")
    with col3:
        utilization = (borrowed_books / total_books * 100) if total_books > 0 else 0
        st.metric("Collection Usage Rate", f"{utilization:.1f}%")

    st.divider()

    # User Segmentation metrics
    if stats.get('light_readers') or stats.get('moderate_readers') or stats.get('heavy_readers'):
        st.subheader("User Activity Levels")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Light Users (1-5 books)", stats.get('light_readers', 0))
        with col2:
            st.metric("Regular Users (6-15 books)", stats.get('moderate_readers', 0))
        with col3:
            st.metric("Heavy Users (15+ books)", stats.get('heavy_readers', 0))
        st.divider()

    st.subheader("Most Popular Books")
    popular = loader.get_popular_books(15)
    if popular is not None and not popular.empty:
        display_cols = ['Titre', 'Count', 'Auteur']
        popular_clean = popular[display_cols].fillna("Unknown").copy()
        popular_clean.columns = ['Title', 'Checkouts', 'Author']
        st.dataframe(popular_clean, use_container_width=True, hide_index=True)

    st.divider()

    # Top Books Visualization
    viz.show_top_books_chart(15)

    st.divider()

    # User Segmentation Visualization
    st.subheader("User Activity Distribution")
    viz.show_user_segmentation_by_borrowing_frequency()

    st.divider()

    # Books per User Distribution
    st.subheader("Checkout Patterns")
    viz.show_books_per_user_distribution()

    st.divider()

    # Category Distribution
    st.subheader("Collection by Category")
    cat = loader.get_category_distribution()
    if cat is not None and len(cat) > 0:
        cat_df = cat.reset_index()
        cat_df.columns = ['Category', 'Number of Books']
        cat_df = cat_df.fillna("Unknown")
        st.dataframe(cat_df.head(15), use_container_width=True, hide_index=True)
    
    viz.show_category_distribution()

    st.divider()

    # Top Borrowed Categories
    viz.show_top_borrowed_categories(15)

    st.divider()

    # Top Authors
    st.subheader("Top Authors in Collection")
    authors = loader.get_author_statistics(12)
    if authors is not None and not authors.empty:
        authors_clean = authors.fillna("Unknown").copy()
        authors_clean.columns = ['Author', 'Books in Library']
        st.dataframe(authors_clean, use_container_width=True, hide_index=True)
    
    viz.show_top_authors(12)

    st.divider()

    # User Category Distribution
    st.subheader("Checkouts by User Type")
    viz.show_user_category_distribution()

# ============================================================
# PAGE 2 — BOOK RECOMMENDATIONS
# ============================================================
elif page == "Recommend by Book":
    st.header("Book Recommendations")

    books_df = loader.get_all_books()
    if books_df is not None and not books_df.empty:
        title_col = "Titre_clean" if "Titre_clean" in books_df.columns else "Titre"
        books = sorted(books_df[title_col].dropna().unique().tolist())
    else:
        books = []

    book_title = st.selectbox("Select a book", [""] + books, index=0)

    num_recs = st.text_input("How many recommendations? (1-20)", value="3")
    
    try:
        num_recs = int(num_recs)
        if num_recs < 1 or num_recs > 20:
            st.warning("Please enter a number between 1 and 20")
            num_recs = 10
    except:
        st.warning("Please enter a valid number")
        num_recs = 10

    if st.button("Find Similar Books", type="primary"):
        if not book_title:
            st.warning("Please select a book first")
        else:
            with st.spinner("Finding similar books..."):
                st.success(f"Selected: **{book_title}**")
                # Use clustering for book-to-book recommendations by default
                recs = recommender.get_book_recommendations_by_clustering(book_title, clustering, num_recommendations=int(num_recs))
                
                if recs is None or recs.empty:
                    st.warning("No similar books found in the current collection.")
                else:
                    st.subheader("You might also like:")
                    
                    # Clean display - only show Title and Author
                    display_data = []
                    for idx, row in recs.iterrows():
                        book_info = {'Title': row.get('Titre', 'Unknown')}
                        
                        # Use Auteur with proper fallback
                        author = row.get('Auteur')
                        if pd.isna(author) or str(author).strip() == '' or str(author).lower() == 'nan':
                            author = 'Unknown'
                        book_info['Author'] = author
                        
                        display_data.append(book_info)
                    
                    display_df = pd.DataFrame(display_data)
                    # Force Author to be visible and Title to be clean
                    display_df.columns = ['Recommended Title', 'Author']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================
# PAGE 3 — USER RECOMMENDATIONS
# ============================================================
elif page == "Recommend by User":
    st.header("Personalized Recommendations")
    st.info("Enter user details to get personalized book suggestions")

    col1, col2 = st.columns(2)
    with col1:
        first_name = st.text_input("First Name (Prénom)")
    with col2:
        last_name = st.text_input("Last Name (Nom)")

    num_recs_user = st.text_input("How many recommendations? (1-20)", value="3", key="user_num_recs")
    
    try:
        num_recs_user = int(num_recs_user)
        if num_recs_user < 1 or num_recs_user > 20:
            st.warning("Please enter a number between 1 and 20")
            num_recs_user = 10
    except:
        st.warning("Please enter a valid number")
        num_recs_user = 10

    if st.button("Get Recommendations", type="primary"):
        if not first_name.strip() or not last_name.strip():
            st.warning("Please enter both first name and last name")
        else:
            # Find user ID by name
            users_match = loader.borrowings_df[
                (loader.borrowings_df['Prénom'].str.lower().str.strip() == first_name.lower().strip()) &
                (loader.borrowings_df['Nom'].str.lower().str.strip() == last_name.lower().strip())
            ]
            
            if users_match.empty:
                st.error(f"No user found with name: {first_name} {last_name}")
                st.info("Please check the spelling and try again")
            else:
                user_id = users_match.iloc[0]['N° lecteur']
                
                with st.spinner("Finding recommendations..."):
                    st.success(f"Found user: **{first_name} {last_name}**")
                    
                    # Uses behavioral clustering by default
                    recs = recommender.get_recommendations_by_clustering(user_id, clustering, num_recommendations=int(num_recs_user))
                    
                    if recs is None or recs.empty:
                        st.warning("Unable to generate recommendations for this user.")
                    else:
                        st.subheader("Recommended for you:")
                        
                        # Clean display - only show Title and Author
                        display_data = []
                        for idx, row in recs.iterrows():
                            book_info = {'Title': row.get('Titre', 'Unknown')}
                            
                            # Use Auteur with proper fallback
                            author = row.get('Auteur')
                            if pd.isna(author) or str(author).strip() == '' or str(author).lower() == 'nan':
                                author = 'Unknown'
                            book_info['Author'] = author
                            
                            display_data.append(book_info)
                        
                        display_df = pd.DataFrame(display_data)
                        # Force Author to be visible
                        display_df.columns = ['Recommended Title', 'Author']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)


