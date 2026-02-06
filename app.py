import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from recommendation_engine import RecommendationEngine
from visualizations import Visualizations

# Page configuration
st.set_page_config(
    page_title="Library Discovery System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Brown Design CSS
st.markdown("""
<style>
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    /* Load FontAwesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    :root {
        --primary: #4a2c1a;
        --accent: #8d6e63;
        --secondary: #d7ccc8;
        --background: #fdfaf6;
        --text-dark: #3e2723;
        --text-light: #5d4037;
        --white: #ffffff;
    }

    /* Main background with texture */
    .main {
        background: linear-gradient(135deg, #fdfaf6 0%, #f5f0eb 100%);
    }
    
    .stApp {
        background-color: var(--background);
    }

    /* Sidebar - Rich Brown */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3e2723 0%, #4a2c1a 100%);
        box-shadow: 4px 0 10px rgba(0,0,0,0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #f5f0eb !important;
    }
    
    /* Headers - Elegant Serif */
    .page-header {
        font-family: 'Playfair Display', serif;
        color: var(--primary);
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-align: center;
    }

    .page-subtitle {
        font-family: 'Source Sans Pro', sans-serif;
        color: var(--accent);
        font-size: 1.2rem;
        margin-bottom: 3rem;
        text-align: center;
    }

    /* Cards */
    .stat-card {
        background: var(--white);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(107, 68, 35, 0.08);
        border: 1px solid var(--secondary);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(107, 68, 35, 0.15);
    }
    
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.2rem;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    .book-card {
        background: var(--white);
        padding: 1.75rem;
        border-radius: 15px;
        margin-bottom: 1.25rem;
        border-left: 6px solid var(--accent);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    .book-card:hover {
        transform: translateX(8px);
        box-shadow: 0 6px 20px rgba(74, 44, 26, 0.12);
    }

    .book-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.4rem;
    }

    .book-author {
        color: var(--accent);
        font-style: italic;
        margin-bottom: 0.75rem;
        font-size: 1.05rem;
    }

    .book-category {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        background-color: #f7f2ed;
        color: #4e342e; /* Dark brown for high contrast */
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid var(--secondary);
    }

    .match-badge {
        background-color: var(--primary);
        color: var(--white);
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        display: inline-block;
        box-shadow: 0 2px 6px rgba(74, 44, 26, 0.2);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 0;
        padding-top: 10px;
        padding-bottom: 10px;
        color: var(--accent);
        font-weight: 600;
        border-bottom: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--primary) !important;
        border-bottom: 3px solid var(--primary) !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--primary) 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6rem 2.5rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 15px rgba(62, 39, 35, 0.3) !important;
    }

    /* Metric Override */
    [data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-family: 'Source Sans Pro', sans-serif !important;
    }

    /* Icon Color */
    .fa-solid, .fa-regular {
        color: var(--accent);
        margin-right: 8px;
    }
    
    .sidebar-icon {
        color: #f5f0eb !important;
        margin-right: 12px;
        width: 20px;
    }

    /* Fix White Font on White issues */
    .stAlert p {
        color: #3e2723 !important;
    }
    
    /* Style for match percentage */
    .match-score {
        font-size: 2.8rem;
        color: var(--primary);
        font-weight: 800;
        margin-bottom: -5px;
    }
    
    .match-label {
        font-size: 0.85rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
</style>
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loader' not in st.session_state:
    with st.spinner("Loading library collection..."):
        try:
            st.session_state.data_loader = DataLoader()
            st.session_state.recommendation_engine = RecommendationEngine(
                st.session_state.data_loader,
                min_support=0.005,
                min_confidence=0.3
            )
            st.session_state.visualizations = Visualizations(st.session_state.data_loader)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

# Sidebar Navigation
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("# Library Discovery", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem; margin-bottom: 2rem; opacity: 0.9;'>Your personalized reading companion</p>", unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
        <div style='text-align: center; padding: 1rem; background-color: white; border-radius: 12px; border: 1px solid #d7ccc8; margin-bottom: 1rem;'>
            <div style='font-size: 0.8rem; color: #8d6e63; text-transform: uppercase; letter-spacing: 1px;'>Library Status</div>
            <div style='font-size: 1.2rem; color: #6b4423; font-weight: 700;'>
                <i class="fa-solid fa-check-circle" style="margin-right: 5px;"></i> Active & Optimized
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Find Similar Books", "Personal Recommendations", "Library Insights", "Reader Profiles", "Reader Clusters"],
        format_func=lambda x: {
            "Home": "Home",
            "Find Similar Books": "Find Similar Books",
            "Personal Recommendations": "Personal Recommendations",
            "Library Insights": "Library Insights",
            "Reader Profiles": "Reader Profiles",
            "Reader Clusters": "Reader Clusters"
        }.get(x, x)
    )

    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <div style='padding: 1rem; background-color: rgba(107, 68, 35, 0.05); border-radius: 8px;'>
            <div style='font-size: 0.75rem; color: #6b4423; font-weight: 600; margin-bottom: 0.5rem;'>SYSTEM INFO</div>
            <div style='font-size: 0.7rem; color: #8d6e63;'>
                <i class="fa-solid fa-database" style="margin-right: 5px;"></i> {len(st.session_state.data_loader.books_df)} Books<br>
                <i class="fa-solid fa-users" style="margin-right: 5px;"></i> {st.session_state.data_loader.borrowings_df['N° lecteur'].nunique()} Readers
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick stats in sidebar
    stats = st.session_state.data_loader.get_statistics()
    st.markdown("### Quick Overview")
    st.markdown(f"<p style='font-size: 1.1rem; margin: 0.5rem 0;'><strong>{stats['total_books']:,}</strong> Books</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1.1rem; margin: 0.5rem 0;'><strong>{stats['unique_users']:,}</strong> Readers</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1.1rem; margin: 0.5rem 0;'><strong>{stats['total_borrowings']:,}</strong> Circulations</p>", unsafe_allow_html=True)

# Main Content
if page == "Home":
    st.markdown('<div class="page-header">Welcome to Your Library</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Discover your next favorite book through personalized recommendations</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key metrics
    stats = st.session_state.data_loader.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['total_books']:,}</div>
                <div class="stat-label"><i class="fa-solid fa-book"></i> Collection</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['unique_users']:,}</div>
                <div class="stat-label"><i class="fa-solid fa-users"></i> Readers</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['total_borrowings']:,}</div>
                <div class="stat-label"><i class="fa-solid fa-repeat"></i> Borrows</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['avg_borrowings_per_user']:.1f}</div>
                <div class="stat-label"><i class="fa-solid fa-chart-line"></i> Avg Read</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Collection overview
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("## Collection Status")
        st.markdown("<p style='color: #8d6e63; margin-bottom: 1.5rem;'>Overview of library circulation</p>", unsafe_allow_html=True)
        st.session_state.visualizations.show_library_collection_overview()
    
    with col2:
        st.markdown("## Currently Popular")
        st.markdown("<p style='color: #8d6e63; margin-bottom: 1.5rem;'>Books readers are enjoying</p>", unsafe_allow_html=True)
        
        popular_books = st.session_state.data_loader.get_popular_books(6)
        
        for idx, (_, book) in enumerate(popular_books.iterrows(), 1):
            st.markdown('<div class="book-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="book-title">{idx}. {book["Titre"]}</div>', unsafe_allow_html=True)
            if pd.notna(book.get('Auteur')):
                st.markdown(f'<div class="book-author">by {book["Auteur"]}</div>', unsafe_allow_html=True)
            if pd.notna(book.get('topic_fr')):
                st.markdown(f'<div class="book-category">{book["topic_fr"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="margin-top: 0.75rem; font-size: 0.9rem; color: #8d6e63;">Borrowed {book["Count"]} times</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Find Similar Books":
    st.markdown('<div class="page-header">Find Similar Books</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Looking for something like a book you enjoyed? We will help you find similar titles</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Book selection
    title_col = 'Titre_clean' if 'Titre_clean' in st.session_state.data_loader.borrowings_df.columns else 'Titre'
    all_books = sorted(st.session_state.data_loader.borrowings_df[title_col].unique())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_book = st.selectbox(
            "Select a book you enjoyed",
            all_books,
            index=0
        )
    
    with col2:
        num_recommendations = st.selectbox(
            "Number of suggestions",
            [5, 10, 15, 20],
            index=1
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Find Similar Books", type="primary", use_container_width=True):
        with st.spinner("Searching our collection..."):
            recommendations = st.session_state.recommendation_engine.get_recommendations(
                selected_book,
                num_recommendations=num_recommendations
            )
            
            if recommendations is not None and len(recommendations) > 0:
                st.markdown(f"## Books Similar to *{selected_book}*")
                st.markdown(f"<p style='font-size: 1.1rem; color: #8d6e63; margin-bottom: 2rem;'>We found <strong>{len(recommendations)}</strong> books you might enjoy</p>", unsafe_allow_html=True)
                
                # Display recommendations
                for idx, row in recommendations.iterrows():
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f'<span class="match-badge">#{idx + 1}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="book-title">{row["Titre"]}</div>', unsafe_allow_html=True)
                        
                        if 'Auteur' in row and pd.notna(row['Auteur']):
                            st.markdown(f'<div class="book-author">by {row["Auteur"]}</div>', unsafe_allow_html=True)
                        elif 'Auteur_merged1' in row and pd.notna(row['Auteur_merged1']):
                            st.markdown(f'<div class="book-author">by {row["Auteur_merged1"]}</div>', unsafe_allow_html=True)
                        
                        if 'topic_fr' in row and pd.notna(row['topic_fr']):
                            st.markdown(f'<div class="book-category">{row["topic_fr"]}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        # Match strength indicator
                        confidence_pct = int(row['confidence'] * 100)
                        st.markdown(f"""
                        <div style='text-align: center; padding: 0.5rem;'>
                            <div class="match-score">{confidence_pct}%</div>
                            <div class="match-label">Match</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("We couldn't find similar books at this time. Try selecting a different title or explore our popular books on the home page.")

elif page == "Personal Recommendations":
    st.markdown('<div class="page-header">Personal Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Based on your reading history, here are books curated just for you</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # User selection
    all_users = sorted(st.session_state.data_loader.borrowings_df['N° lecteur'].unique())
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_user = st.selectbox(
            "Reader ID",
            all_users,
            index=0
        )
    
    with col2:
        rec_method = st.selectbox(
            "Recommendation Strategy",
            ["Association Rules", "Clustering (Category)", "Clustering (Advanced)"],
            index=0
        )
        if rec_method == "Association Rules":
            method_key = 'association'
        elif rec_method == "Clustering (Category)":
            method_key = 'clustering_category'
        else: # Clustering (Advanced)
            method_key = 'clustering_advanced'
    
    with col3:
        num_recommendations = st.selectbox(
            "Quantity",
            [5, 10, 15, 20],
            index=1,
            key="user_rec_num"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show user's reading profile
    user_books = st.session_state.data_loader.get_user_borrowed_books(selected_user)
    user_category = st.session_state.data_loader.get_user_category(selected_user)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(user_books)}</div>
                <div class="stat-label"><i class="fa-solid fa-book-open icon-container"></i> Books Read</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="font-size: 1.5rem;">{user_category}</div>
                <div class="stat-label"><i class="fa-solid fa-graduation-cap icon-container"></i> Reader Type</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("View Reading History"):
        if user_books:
            for i, book in enumerate(user_books[:30], 1):
                st.markdown(f"{i}. {book}")
        else:
            st.write("No borrowing history available.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing patterns and curating recommendations..."):
            recommendations = st.session_state.recommendation_engine.get_recommendations_for_user(
                selected_user,
                num_recommendations=num_recommendations,
                method=method_key
            )
            
            if recommendations is not None and len(recommendations) > 0:
                st.markdown(f"## Recommended Just For You")
                st.markdown(f"<p style='font-size: 1.1rem; color: #8d6e63; margin-bottom: 2rem;'>Based on your reading history, we've selected <strong>{len(recommendations)}</strong> books</p>", unsafe_allow_html=True)
                
                # Display recommendations
                for idx, row in recommendations.iterrows():
                    st.markdown('<div class="book-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f'<span class="match-badge">#{idx + 1}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="book-title">{row["Titre"]}</div>', unsafe_allow_html=True)
                        
                        if 'Auteur_merged1' in row and pd.notna(row['Auteur_merged1']):
                            st.markdown(f'<div class="book-author">by {row["Auteur_merged1"]}</div>', unsafe_allow_html=True)
                        
                        if 'topic_fr' in row and pd.notna(row['topic_fr']):
                            st.markdown(f'<div class="book-category">{row["topic_fr"]}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        confidence_pct = int(row['confidence'] * 100)
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1rem;'>
                            <div style='font-size: 2.5rem; color: #6b4423; font-weight: 700;'>{confidence_pct}%</div>
                            <div style='font-size: 0.9rem; color: #8d6e63; text-transform: uppercase; letter-spacing: 1px;'>Match</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("We're still learning about your preferences. Keep reading, and we'll have great recommendations for you soon!")

elif page == "Library Insights":
    st.markdown('<div class="page-header">Library Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Explore trends and discover what is happening in our library</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Books", "Readers", "Authors"])
    
    with tab1:
        st.markdown("### Most Popular Books")
        st.markdown("<p style='color: #8d6e63; margin-bottom: 1.5rem;'>Titles capturing readers' attention</p>", unsafe_allow_html=True)
        st.session_state.visualizations.show_top_books_chart(15)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("### Collection by Category")
        st.markdown("<p style='color: #8d6e63; margin-bottom: 1.5rem;'>Diversity of our collection</p>", unsafe_allow_html=True)
        st.session_state.visualizations.show_category_distribution()
    
    with tab2:
        st.markdown("### Reading Patterns")
        st.markdown("<p style='color: #8d6e63; margin-bottom: 1.5rem;'>How our community reads</p>", unsafe_allow_html=True)
        st.session_state.visualizations.show_books_per_user_distribution()
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("### Reader Categories")
        st.markdown("<p style='color: #8d6e63; margin-bottom: 1.5rem;'>Who visits our library</p>", unsafe_allow_html=True)
        st.session_state.visualizations.show_user_category_distribution()
    
    with tab3:
        st.markdown("### Popular Authors")
        st.markdown("<p style='color: #8d6e63; margin-bottom: 1.5rem;'>Most represented in our collection</p>", unsafe_allow_html=True)
        st.session_state.visualizations.show_top_authors(15)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        author_stats = st.session_state.data_loader.get_author_statistics(20)
        if not author_stats.empty:
            st.markdown("### Author Collection")
            st.dataframe(author_stats, use_container_width=True, height=450)

elif page == "Reader Profiles":
    st.markdown('<div class="page-header">Reader Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understanding our reading community</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # User segmentation
    st.markdown("### Reader Types")
    st.markdown("<p style='color: #8d6e63; margin-bottom: 1.5rem;'>Our readers categorized by reading habits</p>", unsafe_allow_html=True)
    
    st.session_state.visualizations.show_user_segmentation_pie()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Reading patterns
    st.markdown("### Reading Patterns by Community")
    st.markdown("<p style='color: #8d6e63; margin-bottom: 1.5rem;'>How different groups engage with books</p>", unsafe_allow_html=True)
    
    st.session_state.visualizations.show_borrowing_by_student_year()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Detailed statistics
    if st.session_state.data_loader.user_stats is not None:
        st.markdown("### Reader Insights")
        
        col1, col2, col3 = st.columns(3)
        
        categories = ['Light Reader', 'Moderate Reader', 'Heavy Reader']
        descriptions = [
            'Occasional readers exploring the collection',
            'Regular readers with steady habits',
            'Avid readers with deep engagement'
        ]
        ranges = ['1-5 books', '6-15 books', '16+ books']
        
        for col, category, desc, range_text in zip([col1, col2, col3], categories, descriptions, ranges):
            with col:
                category_data = st.session_state.data_loader.user_stats[
                    st.session_state.data_loader.user_stats['category'] == category
                ]
                
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; margin-bottom: 0.5rem;'>{category}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: #8d6e63; font-size: 0.9rem; margin-bottom: 1rem;'>{desc}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: #a1887f; font-size: 0.85rem; margin-bottom: 1rem;'>{range_text}</p>", unsafe_allow_html=True)
                
                if not category_data.empty:
                    st.metric("Readers", f"{len(category_data):,}", label_visibility="collapsed")
                    st.markdown(f"<p style='text-align: center; font-size: 0.9rem; margin-top: 0.5rem;'>Avg: {category_data['total_borrowings'].mean():.0f} books</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

elif page == "Reader Clusters":
    st.markdown('<div class="page-header">Reader Clusters</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Advanced reader segmentation and behavior analysis</div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Check if clustering data is available
    clustering_data = st.session_state.data_loader.get_clustering_data()
    
    if clustering_data is not None:
        st.info("This feature is currently in development. Advanced clustering analysis will be available soon.")
        
        # Show some basic info about clustering data
        st.markdown("### Available Data")
        st.markdown(f"<p style='color: #8d6e63; margin-bottom: 1rem;'>We have {len(clustering_data)} records ready for analysis</p>", unsafe_allow_html=True)
        
    else:
        st.info("Reader clustering analysis is currently unavailable. This feature will help identify distinct reader groups based on their preferences and behaviors.")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 2px 12px rgba(74, 44, 26, 0.08);'>
        <h3 style='margin-top: 0;'>Coming Soon</h3>
        <p style='color: #8d6e63; line-height: 1.8;'>
        Advanced clustering will reveal hidden patterns in reader behavior, helping us better understand and serve our community. 
        This analysis will identify distinct reader groups based on their reading preferences, borrowing patterns, and engagement levels.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; font-family: "Source Sans Pro", sans-serif;'>
    <p style='margin: 0; color: #8d6e63; font-size: 1.1rem; font-weight: 600;'>Library Discovery System</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #a1887f;'>Helping readers find their next favorite book</p>
</div>
""", unsafe_allow_html=True)