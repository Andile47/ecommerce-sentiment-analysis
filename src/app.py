import streamlit as st

from predict import predict_sentiment


st.set_page_config(
    page_title="Review Sentiment",
    layout="centered",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f6f8f4 0%, #eef4f0 100%);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 50rem;
        padding-top: 2.5rem;
    }
    h1, h2 {
        color: #183c32;
    }
    [data-testid="stForm"] {
        border-top: 1px solid #c7d7ce;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Review sentiment")
st.write("Classify a product review as positive, negative, or neutral.")

with st.form("review_form"):
    review = st.text_area(
        "Product review",
        placeholder="The product broke after one day and the quality is terrible.",
        height=150,
    )
    submitted = st.form_submit_button("Analyze review", type="primary")

if submitted:
    if not review.strip():
        st.warning("Enter a review before analyzing it.")
    else:
        try:
            sentiment = predict_sentiment(review)
            if sentiment == "positive":
                st.success("Predicted sentiment: Positive")
            elif sentiment == "negative":
                st.error("Predicted sentiment: Negative")
            else:
                st.info("Predicted sentiment: Neutral")
            st.caption("This is a model estimate, not a guaranteed judgment.")
        except FileNotFoundError as error:
            st.error(str(error))