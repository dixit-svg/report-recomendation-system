import logging
import os
import re

import streamlit as st

from recommend_v2 import get_report_recommendations_from_url, BASE_REPORT_URL

logging.disable(logging.CRITICAL)

st.set_page_config(
    page_title="Ghost Research — Recommended Reports",
    page_icon="📊",
    layout="centered",
)

st.title("Find Reports That Matter to You")
st.markdown(
    "Paste your LinkedIn profile link below and we'll suggest "
    "research reports tailored to your background and interests."
)

linkedin_url = st.text_input(
    "Your LinkedIn profile link",
    placeholder="https://www.linkedin.com/in/your-name/",
)

with st.expander("Having trouble? Update your LinkedIn session token here"):
    st.markdown(
        "If you see a message saying your session has expired, paste a fresh "
        "**li_at** token below. You can find it in your browser:\n\n"
        "1. Open [linkedin.com](https://www.linkedin.com) and make sure you're logged in.\n"
        "2. Open **Developer Tools** (press F12).\n"
        "3. Go to **Application → Cookies → linkedin.com**.\n"
        "4. Copy the value next to **li_at** and paste it here."
    )
    li_at_input = st.text_input(
        "li_at token",
        type="password",
        placeholder="Paste your token here",
        label_visibility="collapsed",
    )

if st.button("Show My Recommendations", type="primary", disabled=not linkedin_url):
    pattern = r"^https?://(www\.)?linkedin\.com/in/[\w-]+/?$"
    if not re.match(pattern, linkedin_url.strip()):
        st.error(
            "That doesn't look like a LinkedIn profile link. "
            "It should look something like **https://www.linkedin.com/in/your-name/**"
        )
    else:
        if li_at_input:
            os.environ["LI_AT"] = li_at_input.strip()

        with st.spinner("Analysing your profile — this usually takes about a minute…"):
            try:
                urls = get_report_recommendations_from_url(linkedin_url.strip())
                error = None
            except (PermissionError, RuntimeError) as e:
                urls = None
                error = "token_expired" if "li_at" in str(e).lower() or "expired" in str(e).lower() else "generic"
            except Exception:
                urls = None
                error = "generic"

        if error == "token_expired":
            st.error(
                "Your LinkedIn session has expired. "
                "Expand the **\"Having trouble?\"** section above, "
                "paste a fresh token, and try again."
            )
        elif error == "generic":
            st.error(
                "We weren't able to fetch recommendations right now. "
                "Please double-check the link and try again in a moment."
            )
        elif len(urls) == 0:
            st.info("We couldn't find any matching reports for this profile. Try a different profile link.")
        else:
            st.success(f"Here are **{len(urls)}** reports picked for you!")
            for i, url in enumerate(urls, 1):
                slug = url.removeprefix(f"{BASE_REPORT_URL}/")
                title = slug.replace("-", " ").title()
                st.markdown(f"**{i}.** [{title}]({url})")
