"""Melbourne Property Intelligence — Streamlit Dashboard.

Interactive dashboard for exploring Melbourne property market data
and querying the AI-powered knowledge base.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Melbourne Property Intelligence",
    page_icon="🏠",
    layout="wide",
)

st.title("Melbourne Property Intelligence")
st.markdown("LLM-powered insights into Melbourne's property market")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Chat with Market Data", "Suburb Explorer", "Data Status"],
)

if page == "Chat with Market Data":
    st.header("Ask the Market Expert")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask about Melbourne property market..."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from src.query.rag import rag_query

                result = rag_query(query)
                st.markdown(result.answer)

                if result.sources:
                    with st.expander("Sources"):
                        for source in result.sources:
                            st.markdown(
                                f"**{source['source']}** (relevance: {source['score']:.2f})\n\n"
                                f"{source['text']}"
                            )

            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer}
            )

elif page == "Suburb Explorer":
    st.header("Suburb Statistics")

    from src.ingestion.storage import get_engine, init_db, query_suburb_stats

    init_db()
    engine = get_engine()

    # Get list of suburbs
    import pandas as pd
    from sqlalchemy import text

    with engine.connect() as conn:
        suburbs_df = pd.read_sql(
            text("SELECT DISTINCT suburb FROM auction_results ORDER BY suburb"),
            conn,
        )

    if suburbs_df.empty:
        st.info("No auction data yet. Run the data ingestion pipeline first.")
    else:
        suburb_list = suburbs_df["suburb"].tolist()
        selected_suburb = st.selectbox("Select a suburb", suburb_list)

        if selected_suburb:
            stats = query_suburb_stats(selected_suburb)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Sales", stats.get("total_sales", 0))
            with col2:
                st.metric(
                    "Median Price",
                    f"${stats['median_price']:,.0f}" if stats.get("median_price") else "N/A",
                )
            with col3:
                st.metric(
                    "Avg Distance to CBD",
                    f"{stats['avg_distance_km']:.1f} km" if stats.get("avg_distance_km") else "N/A",
                )

elif page == "Data Status":
    st.header("Data Status")

    from src.index.vectorstore import get_collection_stats

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("News Articles")
        from src.ingestion.storage import get_engine
        from sqlalchemy import text

        engine = get_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM news_articles"))
                count = result.scalar()
                st.metric("Articles", count or 0)
        except Exception:
            st.metric("Articles", 0)

    with col2:
        st.subheader("Vector Store")
        stats = get_collection_stats()
        st.metric("Documents", stats["count"])
        st.caption(f"Status: {stats['status']}")

    st.divider()
    st.subheader("Quick Actions")

    if st.button("Ingest News"):
        with st.spinner("Collecting news..."):
            from datetime import date
            from src.ingestion.news_scraper import collect_property_news
            from src.ingestion.storage import store_news_articles

            articles = collect_property_news()
            article_dicts = [
                {
                    "title": a.title,
                    "url": a.url,
                    "source": a.source,
                    "published_date": a.published_date,
                    "content": a.content,
                    "summary": a.summary,
                    "date_scraped": date.today(),
                }
                for a in articles
            ]
            store_news_articles(article_dicts)
            st.success(f"Collected {len(article_dicts)} articles!")
