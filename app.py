import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt
import os
import google.generativeai as genai

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from deep_translator import GoogleTranslator
from functools import lru_cache
import re
import numpy as np
from PIL import Image
import base64
import anthropic

# -------------------------
# Configurable Category Profiles
# -------------------------
# Each profile defines: aspects, aspect_display, and optional intent/emotion rules
CATEGORY_PROFILES = {
    'Electronics': {
        'aspects': [
            'delivery', 'service', 'price', 'quality', 'packaging', 'usability',
            'performance', 'warranty', 'return_policy', 'durability', 'availability',
            'support', 'features', 'design', 'charging'
        ],
        'aspect_display': {
            'delivery': 'Delivery',
            'service': 'Customer Service',
            'price': 'Price',
            'quality': 'Product Quality',
            'packaging': 'Packaging',
            'usability': 'Ease of Use',
            'performance': 'Performance',
            'warranty': 'Warranty',
            'return_policy': 'Return/Refund Policy',
            'durability': 'Durability',
            'availability': 'Availability',
            'support': 'Technical Support',
            'features': 'Features',
            'design': 'Design & Look',
            'charging': 'Charging/Battery'
        },
        'aspect_keywords': {
            'delivery': ['delivery', 'shipping', 'courier', 'dispatched', 'arrived', 'arrival'],
            'service': ['service', 'customer care', 'helpdesk', 'representative', 'agent', 'staff'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'worth', 'overpriced', 'budget'],
            'quality': ['quality', 'build quality', 'sturdy', 'flimsy', 'premium', 'cheap feel'],
            'packaging': ['packaging', 'package', 'box', 'wrapped', 'packing', 'packed'],
            'usability': ['usability', 'easy to use', 'user friendly', 'intuitive', 'simple', 'complicated', 'difficult to use'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'smooth', 'efficient', 'powerful', 'sluggish'],
            'warranty': ['warranty', 'guarantee', 'repair', 'service center'],
            'return_policy': ['return', 'refund', 'exchange', 'replacement', 'return policy'],
            'durability': ['durability', 'durable', 'lasting', 'broke', 'broken', 'fragile', 'long lasting', 'wear and tear'],
            'availability': ['availability', 'stock', 'out of stock', 'available', 'in stock'],
            'support': ['support', 'technical support', 'helpline', 'assistance', 'tech support'],
            'features': ['feature', 'features', 'function', 'capability', 'specs', 'specification', 'functionality'],
            'design': ['design', 'look', 'appearance', 'aesthetic', 'style', 'sleek', 'slim', 'bulky'],
            'charging': ['charging', 'battery', 'charge', 'drain', 'drains', 'power', 'battery life', 'fast charging', 'mah'],
        }
    },
    'Fashion': {
        'aspects': [
            'delivery', 'service', 'price', 'quality', 'packaging', 'availability',
            'return_policy', 'design', 'comfort', 'fit', 'material', 'color',
            'stitching', 'size', 'washing', 'durability'
        ],
        'aspect_display': {
            'delivery': 'Delivery',
            'service': 'Customer Service',
            'price': 'Price',
            'quality': 'Fabric Quality',
            'packaging': 'Packaging',
            'availability': 'Availability/Stock',
            'return_policy': 'Return/Exchange Policy',
            'design': 'Design/Style',
            'comfort': 'Comfort',
            'fit': 'Fit',
            'material': 'Material',
            'color': 'Color',
            'stitching': 'Stitching',
            'size': 'Size Accuracy',
            'washing': 'Washing/Care',
            'durability': 'Durability'
        },
        'aspect_keywords': {
            'delivery': ['delivery', 'shipping', 'courier', 'dispatched', 'arrived'],
            'service': ['service', 'customer care', 'helpdesk', 'agent', 'staff'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'worth', 'overpriced'],
            'quality': ['quality', 'fabric quality', 'well made', 'poorly made'],
            'packaging': ['packaging', 'package', 'box', 'wrapped', 'packing'],
            'availability': ['availability', 'stock', 'out of stock', 'available'],
            'return_policy': ['return', 'refund', 'exchange', 'replacement'],
            'design': ['design', 'style', 'pattern', 'look', 'appearance', 'aesthetic', 'print'],
            'comfort': ['comfort', 'comfortable', 'cozy', 'soft', 'rough', 'itchy', 'scratchy'],
            'fit': ['fit', 'fitting', 'tight', 'loose', 'baggy', 'slim fit', 'oversized'],
            'material': ['material', 'fabric', 'cloth', 'texture', 'feel', 'cotton', 'polyester', 'silk', 'wool'],
            'color': ['color', 'colour', 'shade', 'fade', 'faded', 'bright', 'dark', 'vibrant'],
            'stitching': ['stitching', 'stitch', 'seam', 'thread', 'sewing', 'stitched'],
            'size': ['size', 'sizing', 'dimension', 'measurement', 'runs small', 'runs large', 'true to size'],
            'washing': ['washing', 'wash', 'laundry', 'shrink', 'shrank', 'care', 'fading after wash'],
            'durability': ['durability', 'durable', 'lasting', 'wear', 'tear', 'long lasting', 'pilling'],
        }
    }
}

# Generic, lightweight rules for intents and emotions (can be overridden per profile)
INTENT_RULES_DEFAULT = {
    'praise': [r"\blove\b", r"\bamazing\b", r"\bgreat\b", r"\bexcellent\b", r"\bhappy\b", r"\bsatisfied\b"],
    'complaint': [r"\bnot working\b", r"\bworst\b", r"\bbroken\b", r"\blate\b", r"\bdelay(ed)?\b", r"\brefund\b", r"\breturn\b", r"\bdisappoint(ed)?\b"],
    'inquiry': [r"\bhow\b", r"\bwhen\b", r"\bwhere\b", r"\bwhat\b", r"\bdoes it\b", r"\bcan i\b", r"\bis it\b"],
    'feature_request': [r"\bplease add\b", r"\bwish it had\b", r"\bfeature request\b"],
    'purchase_intent': [r"\bwill buy\b", r"\border(ing)?\b", r"\badd(ed)? to cart\b", r"\bthinking to buy\b"],
    'return_refund': [r"\brefund\b", r"\breturn\b", r"\breplacement\b"],
    'support_needed': [r"\bhelp\b", r"\bsupport\b", r"\bassist\b", r"\bcontact\b"],
    'comparison': [r"\bbetter than\b", r"\bworse than\b", r"\bvs\b", r"\bcompared to\b"]
}

EMOTION_RULES_DEFAULT = {
    'joy': [r"\blove\b", r"\bhappy\b", r"\bdelight(ed)?\b", r"\bpleased\b"],
    'anger': [r"\bangry\b", r"\bfurious\b", r"\bwaste\b", r"\bterrible\b"],
    'sadness': [r"\bsad\b", r"\bdisappoint(ed)?\b", r"\bupset\b"],
    'surprise': [r"\bsurpris(ed|ing)\b", r"\bdidn'?t expect\b"]
}

# Optional: profile-specific overrides can be added under CATEGORY_PROFILES[profile]['intent_rules'/'emotion_rules']

# -------------------------
# Rule Helpers
# -------------------------
def _compile_rules(rules_dict):
    compiled = {}
    for tag, patterns in rules_dict.items():
        compiled[tag] = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    return compiled

def _aspect_pattern(aspect, profile):
    """Return a regex pattern matching any keyword for this aspect."""
    keywords = profile.get('aspect_keywords', {}).get(aspect, [aspect])
    return '|'.join(re.escape(kw) for kw in keywords)

def _aspect_in_text(text, aspect, profile):
    keywords = profile.get('aspect_keywords', {}).get(aspect, [aspect])
    t = str(text).lower()
    return any(kw in t for kw in keywords)

def detect_tags_from_text(text, compiled_rules):
    text = str(text)
    detected = []
    for tag, patterns in compiled_rules.items():
        if any(p.search(text) for p in patterns):
            detected.append(tag)
    return detected

# Initialize analyzers
analyzer = SentimentIntensityAnalyzer()
aspects = CATEGORY_PROFILES['Electronics']['aspects']
aspect_display = CATEGORY_PROFILES['Electronics']['aspect_display']


# -------------------------
# Helper Functions
# -------------------------
@lru_cache(maxsize=10000)
def cached_translate_to_english(text):
    try:
        lang = detect(text)
        if lang == 'en':
            return text
        else:
            return GoogleTranslator(source=lang, target='en').translate(text)
    except Exception:
        return text

def get_vader_sentiment_score(text):
    return analyzer.polarity_scores(str(text))['compound']

def analyze_vader_sentiment(score):
    if score >= 0.5:
        return 'Positive'
    elif score <= -0.2:
        return 'Negative'
    else:
        return 'Neutral'

def final_sentiment(row):
    rating = row['Rating']
    text_sent = row['text_sentiment']
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return text_sent
    elif rating <= 2:
        return text_sent
    else:
        return 'Neutral'

def simple_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation and numbers
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip()

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------------
# Gemini Helpers
# -------------------------
def _get_anthropic_api_key():
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY")

def _get_gemini_api_key():
    try:
        # Prefer Streamlit secrets if available
        key = st.secrets.get("GEMINI_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.environ.get("GEMINI_API_KEY")

def generate_final_report_with_gemini(summary_df, reviews_list, category_name: str):
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Set it in Streamlit secrets or environment.")
    genai.configure(api_key=api_key)
    # Limit reviews to a reasonable number to keep prompt size manageable
    reviews_list = [str(r) for r in reviews_list if isinstance(r, str) or r is not None]
    if len(reviews_list) > 200:
        reviews_list = reviews_list[:200]
    reviews_blob = "\n".join(f"- {r}" for r in reviews_list)

    aspect_table_csv = summary_df.to_csv(index=False)

    prompt = (
        "You are a senior product insights analyst creating an executive-ready report for a company client.\n"
        f"Category context: {category_name}\n\n"
        "Use the aspect-based sentiment table and representative customer reviews below to synthesize a crisp, actionable report.\n"
        "Focus on business relevance. Avoid repeating raw data. Be concise.\n\n"
        "Deliverables:\n"
        "1) Strengths (bulleted)\n"
        "2) Weaknesses (bulleted)\n"
        "3) Opportunities (market/product opportunities; bulleted)\n"
        "4) Risks (bulleted)\n"
        "5) Recommended Actions (prioritized, with rationale)\n"
        "6) Top 5 Short Customer Quotes (verbatim, diverse angles)\n"
        "7) Closing Summary (3-4 lines)\n\n"
        "Aspect Sentiment Table (CSV):\n"
        f"{aspect_table_csv}\n\n"
        "Representative Customer Reviews:\n"
        f"{reviews_blob}\n\n"
        "Constraints: Keep it structured with clear headings, no code fences, no markdown tables."
    )

    # Determine an available model at runtime (prefer latest 1.5 variants)
    preferred_models = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",  # legacy text-only fallback
    ]
    available_models = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                available_models.append(m.name)
    except Exception:
        # If listing fails, we'll rely on preferred set
        available_models = []

    candidate_models = [m for m in preferred_models if m in available_models] or available_models or preferred_models
    last_error = None
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return (getattr(response, "text", "") or "").strip()
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("🧠 Product Review Sentiment Analyzer")

# -------------------------
# Category/Profile Selection
# -------------------------
st.sidebar.markdown("### Category Profile")
selected_profile_name = st.sidebar.selectbox(
    "Select a category profile",
    list(CATEGORY_PROFILES.keys()),
    index=0
)

# Apply selected profile
profile = CATEGORY_PROFILES[selected_profile_name]
aspects = profile['aspects']
aspect_display = profile['aspect_display']

# Optional: quick custom aspect override per client (comma-separated)
custom_aspects_csv = st.sidebar.text_input("Custom aspects (comma-separated)", value="")
if custom_aspects_csv.strip():
    custom_aspect_list = [a.strip().lower() for a in custom_aspects_csv.split(',') if a.strip()]
    if custom_aspect_list:
        aspects = custom_aspect_list
        # Try to preserve display names from profile when possible
        aspect_display = {a: profile['aspect_display'].get(a, a.title()) for a in aspects}

# -------------------------
# 1. Single Text Analysis
# -------------------------
with st.expander('🔍 Single Text Analysis'):
    text = st.text_input('Enter a review:')
    if text:
        lang = detect(text)
        if lang != 'en':
            english_text = cached_translate_to_english(text)
            st.write('📝 English Translation:', english_text)
        else:
            english_text = text
        score = get_vader_sentiment_score(english_text)
        sentiment = analyze_vader_sentiment(score)
        st.write('📊 Sentiment:', sentiment)
        st.write('📊 VADER Score:', round(score, 2))
        # Compile rules based on selected profile
        intent_rules = _compile_rules(profile.get('intent_rules', INTENT_RULES_DEFAULT))
        emotion_rules = _compile_rules(profile.get('emotion_rules', EMOTION_RULES_DEFAULT))

        aspect_mentions = [aspect_display.get(aspect, aspect.title()) for aspect in aspects if _aspect_in_text(english_text, aspect, profile)]
        intents = detect_tags_from_text(english_text, intent_rules)
        emotions = detect_tags_from_text(english_text, emotion_rules)
        st.write('🔎 Aspect Mentions:', aspect_mentions or ['none'])
        st.write('🎯 Intents:', intents or ['none'])
        st.write('💬 Emotions:', emotions or ['none'])

# -------------------------
# 2. File Upload
# -------------------------
st.subheader("📂 Upload a Review CSV File")
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

# -------------------------
# 3. Main Analysis
# -------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    if 'Review_Summary' not in df.columns or 'Rating' not in df.columns:
        st.error("❌ CSV must contain both 'Review_Summary' and 'Rating' columns.")
    else:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df = df.dropna(subset=['Rating'])

        # User option for translation
        st.sidebar.markdown("### Translation Options")
        translate_reviews = st.sidebar.checkbox("Translate non-English reviews to English", value=True)

        # Translation with progress bar and caching
        if translate_reviews:
            st.info("Translating reviews to English (if needed)...")
            english_reviews = []
            progress_bar = st.progress(0)
            review_list = df['Review_Summary'].astype(str).tolist()
            for i, review in enumerate(review_list):
                english_reviews.append(cached_translate_to_english(review))
                if len(review_list) > 1:
                    progress_bar.progress((i+1)/len(review_list))
            df['Review_Summary_English'] = english_reviews
            progress_bar.empty()
        else:
            df['Review_Summary_English'] = df['Review_Summary'].astype(str)

        # VADER sentiment analysis
        df['score'] = df['Review_Summary_English'].apply(get_vader_sentiment_score)
        df['text_sentiment'] = df['score'].apply(analyze_vader_sentiment)
        df['final_sentiment'] = df.apply(final_sentiment, axis=1)
        df['mismatch'] = df['text_sentiment'] != df['final_sentiment']

        # Aspect extraction
        for aspect in aspects:
            col_name = aspect + '_mention'
            pattern = _aspect_pattern(aspect, profile)
            df[col_name] = df['Review_Summary_English'].str.lower().str.contains(pattern, regex=True, na=False)

        # Intent & Emotion tagging
        intent_rules = _compile_rules(profile.get('intent_rules', INTENT_RULES_DEFAULT))
        emotion_rules = _compile_rules(profile.get('emotion_rules', EMOTION_RULES_DEFAULT))
        df['intents'] = df['Review_Summary_English'].apply(lambda t: detect_tags_from_text(t, intent_rules))
        df['emotions'] = df['Review_Summary_English'].apply(lambda t: detect_tags_from_text(t, emotion_rules))

        st.sidebar.title("📊 Filters")
        min_rating, max_rating = int(df['Rating'].min()), int(df['Rating'].max())
        selected_range = st.sidebar.slider("Select Rating Range", min_rating, max_rating, (min_rating, max_rating))
        df = df[(df['Rating'] >= selected_range[0]) & (df['Rating'] <= selected_range[1])]

        st.success(f"✅ Total reviews analyzed: {len(df)}")
        st.write(df.head())

              # -------------------------
        # 🔹 Sentiment Distribution
        # -------------------------
        st.subheader("📊 Sentiment Distribution")
        col1, col2 = st.columns(2)

        with col1:
                sentiment_counts = df['final_sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                color_scale = alt.Scale(domain=['Positive', 'Neutral', 'Negative'],
                                        range=['#21ba45', '#a0a0a0', '#db2828'])  # green, gray, red

                bar_chart = alt.Chart(sentiment_counts).mark_bar().encode(
                    x=alt.X('Sentiment', sort=['Positive', 'Neutral', 'Negative']),
                    y='Count',
                    color=alt.Color('Sentiment', scale=color_scale),
                    tooltip=['Sentiment', 'Count']
                ).properties(
                    width='container',
                    height=350
                )
                st.altair_chart(bar_chart, use_container_width=True)


        with col2:
                sentiment_counts = df['final_sentiment'].value_counts()
                # Ensure the order matches: Positive, Neutral, Negative
                order = ['Positive', 'Neutral', 'Negative']
                sentiment_counts = sentiment_counts.reindex(order).fillna(0)
                colors = ['green', 'grey', 'red']  # green for positive, grey for neutral, red for negative
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts, labels=sentiment_counts.index,
                    autopct="%1.1f%%", startangle=90, colors=colors)
                ax.axis('equal')
                st.pyplot(fig)
      
        # -------------------------
        # Word Cloud
        # -------------------------
        st.subheader("☁️ Word Cloud")
        wc_sentiment_filter = st.selectbox(
            "Generate word cloud for:", ["All Reviews", "Positive", "Negative", "Neutral"], key="wc_filter"
        )
        if wc_sentiment_filter == "All Reviews":
            wc_text_series = df['Review_Summary_English']
        else:
            wc_text_series = df[df['final_sentiment'] == wc_sentiment_filter]['Review_Summary_English']

        wc_text = " ".join(wc_text_series.dropna().astype(str).tolist())
        wc_text_clean = simple_clean(wc_text)

        if wc_text_clean.strip():
            wc = WordCloud(width=900, height=350, background_color='white', colormap='viridis',
                           max_words=150, collocations=False).generate(wc_text_clean)
            fig_wc, ax_wc = plt.subplots(figsize=(12, 4))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.info("Not enough text to generate a word cloud for this filter.")

        # -------------------------
        # Intent & Emotion Summaries
        # -------------------------
        st.subheader("🎯 Intent & 💬 Emotion Summaries")
        col3, col4 = st.columns(2)
        with col3:
            # explode intents
            intents_exploded = df.explode('intents')
            intents_exploded['intents'] = intents_exploded['intents'].fillna('none')
            intent_counts = intents_exploded['intents'].value_counts().reset_index()
            intent_counts.columns = ['Intent', 'Count']
            chart_i = alt.Chart(intent_counts).mark_bar().encode(
                x=alt.X('Intent', sort='-y'),
                y='Count',
                tooltip=['Intent', 'Count']
            ).properties(width='container', height=300)
            st.altair_chart(chart_i, use_container_width=True)
        with col4:
            emotions_exploded = df.explode('emotions')
            emotions_exploded['emotions'] = emotions_exploded['emotions'].fillna('none')
            emotion_counts = emotions_exploded['emotions'].value_counts().reset_index()
            emotion_counts.columns = ['Emotion', 'Count']
            chart_e = alt.Chart(emotion_counts).mark_bar().encode(
                x=alt.X('Emotion', sort='-y'),
                y='Count',
                tooltip=['Emotion', 'Count']
            ).properties(width='container', height=300)
            st.altair_chart(chart_e, use_container_width=True)

        # -------------------------
        # Aspect Breakdown (Counts)
        # -------------------------
        aspect_breakdown = []
        for aspect in aspects:
            aspect_col = aspect + '_mention'
            pos = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Positive'))
            neg = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Negative'))
            neu = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Neutral'))
            total = pos + neg + neu
            aspect_breakdown.append({
                'Aspect': aspect_display.get(aspect, aspect.title()),
                'Positive Count': pos,
                'Negative Count': neg,
                'Neutral Count': neu,
                'Total Mentions': total
            })
        aspect_breakdown_df = pd.DataFrame(aspect_breakdown)
        st.subheader("📊 Aspect Breakdown (Counts)")
        st.dataframe(aspect_breakdown_df)

        # -------------------------
        # Aspect-Based Sentiment Summary Table (Percentages)
        # -------------------------
        aspect_sentiment_summary = []
        for aspect in aspects:
            aspect_col = aspect + '_mention'
            pos = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Positive'))
            neg = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Negative'))
            neu = np.sum((df[aspect_col]) & (df['text_sentiment'] == 'Neutral'))
            total = pos + neg + neu
            pos_pct = round(100 * pos / total, 1) if total else 0
            neg_pct = round(100 * neg / total, 1) if total else 0
            neu_pct = round(100 * neu / total, 1) if total else 0
            aspect_sentiment_summary.append({
                "Aspect": aspect_display.get(aspect, aspect.title()),
                "Positive (%)": pos_pct,
                "Negative (%)": neg_pct,
                "Neutral (%)": neu_pct
            })
        st.subheader("📋 Aspect-Based Sentiment Summary (%)")
        st.table(pd.DataFrame(aspect_sentiment_summary))

                # Find the aspect with the lowest positive and highest negative sentiment
        summary_df = pd.DataFrame(aspect_sentiment_summary)
        # Convert percentages to numbers for sorting
        summary_df['Positive (%)'] = pd.to_numeric(summary_df['Positive (%)'])
        summary_df['Negative (%)'] = pd.to_numeric(summary_df['Negative (%)'])

        # Identify the aspect needing most improvement
        weakest_aspect = summary_df.sort_values('Negative (%)', ascending=False).iloc[0]
        strongest_aspect = summary_df.sort_values('Positive (%)', ascending=False).iloc[0]

        # Display AI section with only final report generation
        st.markdown("## 🤖 AI Analysis & Recommendations")
        if st.button("Final report generation"):
            with st.spinner("Generating final report with Gemini..."):
                try:
                    reviews_for_llm = df['Review_Summary_English'].dropna().astype(str).tolist()
                    final_report = generate_final_report_with_gemini(summary_df, reviews_for_llm, selected_profile_name)
                    st.session_state['final_report_text'] = final_report
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")
        if 'final_report_text' in st.session_state and st.session_state['final_report_text']:
            st.markdown(st.session_state['final_report_text'])
            st.download_button(
                "Download Final Report",
                st.session_state['final_report_text'],
                file_name="final_report.txt",
                mime="text/plain"
            )

       

      
        # -------------------------
        # Download Results
        # -------------------------
        st.subheader("⬇ Download Sentiment CSV")
        csv = convert_df(df)
        st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

        # -------------------------
        # Show Full Data
        # -------------------------
        with st.expander("🗂 Show All Reviews"):
            st.dataframe(df)

# -------------------------
# Platform Comparison
# -------------------------
st.markdown("---")
st.subheader("🆚 Platform Comparison")
st.caption("Compare sentiment across any two platforms. Supports CSV upload, pasted text, or review screenshots.")

# --- Helpers ---
def _process_platform_df(df_p, prof):
    df_p = df_p.copy()
    if 'Review_Summary' not in df_p.columns or 'Rating' not in df_p.columns:
        return None, "Data must have 'Review_Summary' and 'Rating' columns."
    df_p['Rating'] = pd.to_numeric(df_p['Rating'], errors='coerce')
    df_p = df_p.dropna(subset=['Rating'])
    df_p['Review_Summary_English'] = df_p['Review_Summary'].astype(str)
    df_p['score'] = df_p['Review_Summary_English'].apply(get_vader_sentiment_score)
    df_p['text_sentiment'] = df_p['score'].apply(analyze_vader_sentiment)
    df_p['final_sentiment'] = df_p.apply(final_sentiment, axis=1)
    for asp in prof['aspects']:
        pattern = _aspect_pattern(asp, prof)
        df_p[asp + '_mention'] = df_p['Review_Summary_English'].str.lower().str.contains(pattern, regex=True, na=False)
    return df_p, None

def _text_to_df(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None, "No reviews found in pasted text."
    return pd.DataFrame({'Review_Summary': lines, 'Rating': 3}), None

def _images_to_df(image_files):
    api_key = _get_anthropic_api_key()
    if not api_key:
        return None, "ANTHROPIC_API_KEY not found — required for image extraction."
    client = anthropic.Anthropic(api_key=api_key, base_url="https://api.anthropic.com")
    media_map = {'jpeg': 'image/jpeg', 'jpg': 'image/jpeg', 'png': 'image/png',
                 'webp': 'image/webp', 'gif': 'image/gif'}
    all_reviews = []
    for img_file in image_files:
        img_bytes = img_file.read()
        img_file.seek(0)
        img = Image.open(img_file)
        fmt = (img.format or 'PNG').lower()
        media_type = media_map.get(fmt, 'image/png')
        img_b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                    {"type": "text", "text": (
                        "Extract all customer review text visible in this image. "
                        "Return each review on a separate new line. "
                        "Only return the review text itself — no star ratings, dates, usernames, or labels."
                    )}
                ]}]
            )
            lines = [l.strip() for l in response.content[0].text.strip().splitlines() if l.strip()]
            all_reviews.extend(lines)
        except Exception as e:
            err_str = str(e)
            if "credit balance is too low" in err_str or "402" in err_str:
                return None, "Image extraction requires Anthropic API credits. Please use CSV or Text input instead."
            return None, f"Image extraction failed: {e}"
    if not all_reviews:
        return None, "No review text could be extracted from the uploaded image(s)."
    return pd.DataFrame({'Review_Summary': all_reviews, 'Rating': 3}), None

def _resolve_platform_input(csv_file, text, imgs):
    if csv_file:
        df = pd.read_csv(csv_file)
        return df.drop(columns=['Unnamed: 0'], errors='ignore'), None
    elif text and text.strip():
        return _text_to_df(text)
    elif imgs:
        return _images_to_df(imgs)
    return None, None

# --- Input widgets (name + 3 input tabs per platform) ---
col_p1, col_p2 = st.columns(2)

with col_p1:
    platform1_name = st.text_input("Platform 1 name", value="Platform 1", key="p1name")
    t1_csv, t1_text, t1_img = st.tabs(["📄 CSV", "📝 Text", "🖼️ Image"])
    with t1_csv:
        p1_csv = st.file_uploader("Upload CSV", type=["csv"], key="p1_csv")
    with t1_text:
        p1_text = st.text_area(
            "Paste reviews (one per line)",
            height=160,
            placeholder="Great product, fast delivery!\nBattery life could be better.\nExcellent value for money.",
            key="p1_text"
        )
    with t1_img:
        p1_imgs = st.file_uploader(
            "Upload review screenshot(s)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="p1_img"
        )

with col_p2:
    platform2_name = st.text_input("Platform 2 name", value="Platform 2", key="p2name")
    t2_csv, t2_text, t2_img = st.tabs(["📄 CSV", "📝 Text", "🖼️ Image"])
    with t2_csv:
        p2_csv = st.file_uploader("Upload CSV", type=["csv"], key="p2_csv")
    with t2_text:
        p2_text = st.text_area(
            "Paste reviews (one per line)",
            height=160,
            placeholder="Good but overpriced.\nAmazing quality, will buy again!\nTook too long to arrive.",
            key="p2_text"
        )
    with t2_img:
        p2_imgs = st.file_uploader(
            "Upload review screenshot(s)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="p2_img"
        )

# --- Resolve inputs ---
df_p1_raw, err1 = _resolve_platform_input(p1_csv, p1_text, p1_imgs)
df_p2_raw, err2 = _resolve_platform_input(p2_csv, p2_text, p2_imgs)

if err1:
    st.error(f"{platform1_name}: {err1}")
if err2:
    st.error(f"{platform2_name}: {err2}")

if df_p1_raw is not None and df_p2_raw is not None:
    with st.spinner("Analyzing platforms..."):
        df_p1, err1 = _process_platform_df(df_p1_raw, profile)
        df_p2, err2 = _process_platform_df(df_p2_raw, profile)

    if err1:
        st.error(f"{platform1_name}: {err1}")
    elif err2:
        st.error(f"{platform2_name}: {err2}")
    else:
        st.markdown(f"### {platform1_name} vs {platform2_name} — Sentiment Distribution")
        col_s1, col_s2 = st.columns(2)

        def _sentiment_bar(df_in, name):
            counts = df_in['final_sentiment'].value_counts().reset_index()
            counts.columns = ['Sentiment', 'Count']
            color_scale = alt.Scale(domain=['Positive', 'Neutral', 'Negative'],
                                    range=['#21ba45', '#a0a0a0', '#db2828'])
            return alt.Chart(counts).mark_bar().encode(
                x=alt.X('Sentiment', sort=['Positive', 'Neutral', 'Negative']),
                y='Count',
                color=alt.Color('Sentiment', scale=color_scale),
                tooltip=['Sentiment', 'Count']
            ).properties(title=name, width='container', height=300)

        with col_s1:
            st.altair_chart(_sentiment_bar(df_p1, platform1_name), use_container_width=True)
        with col_s2:
            st.altair_chart(_sentiment_bar(df_p2, platform2_name), use_container_width=True)

        st.markdown("### Average Rating")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(f"{platform1_name} Avg Rating", f"{df_p1['Rating'].mean():.2f} / 5")
        with col_r2:
            st.metric(f"{platform2_name} Avg Rating", f"{df_p2['Rating'].mean():.2f} / 5")

        st.markdown("### Aspect Positive Sentiment Comparison (%)")

        def _aspect_pos_pct(df_in, prof):
            rows = []
            for asp in prof['aspects']:
                col = asp + '_mention'
                pos = np.sum((df_in[col]) & (df_in['text_sentiment'] == 'Positive'))
                total = np.sum(df_in[col])
                rows.append({'Aspect': prof['aspect_display'].get(asp, asp.title()),
                              'Positive (%)': round(100 * pos / total, 1) if total else 0})
            return pd.DataFrame(rows)

        asp_p1 = _aspect_pos_pct(df_p1, profile).rename(columns={'Positive (%)': platform1_name})
        asp_p2 = _aspect_pos_pct(df_p2, profile).rename(columns={'Positive (%)': platform2_name})
        asp_merged = asp_p1.merge(asp_p2, on='Aspect')
        asp_melted = asp_merged.melt(id_vars='Aspect', var_name='Platform', value_name='Positive (%)')

        comparison_chart = alt.Chart(asp_melted).mark_bar().encode(
            x=alt.X('Aspect', sort='-y'),
            y=alt.Y('Positive (%)'),
            color='Platform',
            xOffset='Platform',
            tooltip=['Aspect', 'Platform', 'Positive (%)']
        ).properties(width='container', height=380)
        st.altair_chart(comparison_chart, use_container_width=True)

        st.dataframe(asp_merged.set_index('Aspect'))
else:
    st.info("Provide input for both platforms above to enable comparison.")
