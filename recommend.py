import json
import os
import logging
import time
from typing import Dict, List, Any

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from sentence_transformers import CrossEncoder
from urllib.parse import urlparse

from linkedin_scraper import LinkedInScraper

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("recommendation_pipeline")

SOFT_THRESHOLD = 0.30
RRF_K = 60
RESULTS_PER_QUERY = 20
TOP_N_FOR_RERANK = 30
TOP_N_FINAL = 10
ALLOWED_CATEGORY_SLUGS = {"for-you", "sector", "theme", "region"}
OPENSEARCH_INDEX = "ghost_research_report_overviews"
BASE_REPORT_URL = "https://www.ghostresearch.com/reports"
DATA_FILE = os.path.join(os.path.dirname(__file__), "test_data.json")
CATEGORIES_FILE = os.path.join(os.path.dirname(__file__), "catagories.json")


# ---------------------------------------------------------------------------
# Lazy-loaded singletons — heavy resources initialised once on first call
# ---------------------------------------------------------------------------
_embedding_model = None
_vector_db = None
_os_client = None
_cross_encoder = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("HUGGINGFACE_EMBEDDINGS_MODEL")
        logger.info("Loading HuggingFace embedding model: %s", model_name)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name, show_progress=False
        )
        logger.info("Embedding model loaded.")
    return _embedding_model


def _get_aws_auth():
    credentials = boto3.Session(
        aws_access_key_id=os.getenv("AWS_OPENSEARCH_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_OPENSEARCH_SECRET_ACCESS_KEY"),
    ).get_credentials()
    return AWS4Auth(
        os.getenv("AWS_OPENSEARCH_ACCESS_KEY_ID"),
        os.getenv("AWS_OPENSEARCH_SECRET_ACCESS_KEY"),
        os.getenv("AWS_OPENSEARCH_REGION_NAME"),
        "es",
        session_token=credentials.token,
    )


def _get_vector_db() -> OpenSearchVectorSearch:
    global _vector_db
    if _vector_db is None:
        logger.info("Initialising OpenSearch vector store…")
        _vector_db = OpenSearchVectorSearch(
            embedding_function=_get_embedding_model(),
            opensearch_url=os.getenv("OPENSEARCH_URL"),
            http_auth=_get_aws_auth(),
            timeout=300,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            index_name=OPENSEARCH_INDEX,
            engine="faiss",
        )
        logger.info("OpenSearch vector store ready.")
    return _vector_db


def _get_os_client() -> OpenSearch:
    global _os_client
    if _os_client is None:
        parsed = urlparse(os.getenv("OPENSEARCH_URL"))
        logger.info("Initialising raw OpenSearch client…")
        _os_client = OpenSearch(
            hosts=[{"host": parsed.hostname, "port": parsed.port or 443}],
            http_auth=_get_aws_auth(),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
        )
        logger.info("OpenSearch client ready.")
    return _os_client


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder reranker…")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


# ---------------------------------------------------------------------------
# Step 1 — LLM category recommendation
# ---------------------------------------------------------------------------
def recommend_categories(
    profile: dict,
    categories: list[dict] | None = None,
) -> list[dict]:
    """Ask GPT to pick relevant subcategories for *profile*."""
    logger.info("Step 1: Recommending categories via LLM…")
    t0 = time.perf_counter()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if categories is None:
        with open(CATEGORIES_FILE, "r") as f:
            categories = json.load(f)

    system_prompt = (
        "You are an expert at matching professional profiles to relevant report categories.\n"
        "You will receive a person's LinkedIn profile data and a list of category groups.\n"
        "From ALL the groups, pick only the subcategories that are relevant to this person.\n"
        "Return ONLY valid JSON."
    )

    user_prompt = f"""Analyze this LinkedIn profile and recommend the most relevant subcategories.

## Profile
{json.dumps(profile, indent=2)}

## Available Category Groups
{json.dumps(categories, indent=2)}

Return a JSON object with this exact structure:
{{
    "recommended": [
        {{"category": "<subcategory name>", "relevance_score": <float 0-1>, "reasoning": "<one sentence>"}},
        {{"category": "<subcategory name>", "relevance_score": <float 0-1>, "reasoning": "<one sentence>"}}
    ]
}}

Rules:
- Only pick subcategories from the provided lists (e.g. "Management", "Energy", "Global", "Deep Tech").
- Only include subcategories that are genuinely relevant to this person's profile.
- Return them ranked by relevance_score (highest first)."""

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    recommended = json.loads(response.choices[0].message.content)["recommended"]
    elapsed = time.perf_counter() - t0
    logger.info(
        "LLM returned %d categories in %.2fs", len(recommended), elapsed
    )
    for cat in recommended:
        logger.info(
            "  %-30s score=%.2f | %s",
            cat["category"],
            cat["relevance_score"],
            cat["reasoning"],
        )
    return recommended


# ---------------------------------------------------------------------------
# Step 2 — Soft category filtering + DB candidate expansion
# ---------------------------------------------------------------------------
def resolve_candidates(
    all_recommended: list[dict],
) -> tuple[dict, list[int], dict, dict]:
    """
    Apply soft threshold, resolve subcategory slugs against DB, expand
    the candidate report pool.

    Returns:
        category_weights  — {slug: float}
        report_ids        — sorted list of candidate report IDs
        report_slug_by_id — {report_id: slug}
        report_to_slugs   — {report_id: set_of_matching_category_slugs}
    """
    logger.info("Step 2: Resolving candidates from DB (threshold >= %.2f)…", SOFT_THRESHOLD)
    t0 = time.perf_counter()

    kept = [c for c in all_recommended if c["relevance_score"] >= SOFT_THRESHOLD]
    category_weights = {c["category"]: c["relevance_score"] for c in kept}
    category_slugs = list(category_weights.keys())

    logger.info(
        "Kept %d/%d categories after soft threshold",
        len(kept),
        len(all_recommended),
    )
    for slug, w in category_weights.items():
        logger.info("  %-30s weight=%.2f", slug, w)

    with open(DATA_FILE, "r") as f:
        db = json.load(f)

    db_categories = db["categories"]
    db_subcategories = db["subcategories"]
    db_reports = db["reports"]
    db_report_subcategory = db["report_subcategory"]

    subcat_by_slug = {sc["slug"]: sc for sc in db_subcategories}
    report_by_id = {r["id"]: r for r in db_reports}
    allowed_cat_ids = {
        cat["id"] for cat in db_categories if cat["slug"] in ALLOWED_CATEGORY_SLUGS
    }

    valid_slugs = [
        s for s in category_slugs
        if s in subcat_by_slug and subcat_by_slug[s]["cat_id"] in allowed_cat_ids
    ]
    slug_to_subcat_id = {s: subcat_by_slug[s]["id"] for s in valid_slugs}
    subcat_ids = set(slug_to_subcat_id.values())

    report_id_set = sorted(
        {
            link["report_id"]
            for link in db_report_subcategory
            if link["subcat_id"] in subcat_ids
        }
    )

    reports = [
        {"id": rid, "slug": report_by_id[rid]["slug"]}
        for rid in report_id_set
        if rid in report_by_id
    ]
    report_ids = [r["id"] for r in reports]
    report_slug_by_id = {r["id"]: r["slug"] for r in reports}

    report_to_slugs: Dict[int, set] = {}
    for link in db_report_subcategory:
        if link["subcat_id"] in subcat_ids:
            rid = link["report_id"]
            if rid not in report_to_slugs:
                report_to_slugs[rid] = set()
            for slug, sid in slug_to_subcat_id.items():
                if link["subcat_id"] == sid:
                    report_to_slugs[rid].add(slug)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Valid slugs: %s | Candidate pool: %d reports | DB total: %d reports (%.2fs)",
        valid_slugs,
        len(reports),
        len(db_reports),
        elapsed,
    )
    return category_weights, report_ids, report_slug_by_id, report_to_slugs


# ---------------------------------------------------------------------------
# Step 3 — Multi-query generation (one per category)
# ---------------------------------------------------------------------------
def generate_category_queries(
    profile: dict,
    kept_categories: list[dict],
) -> dict[str, str]:
    """Generate a focused search query per category via LLM."""
    logger.info("Step 3: Generating %d focused queries…", len(kept_categories))
    t0 = time.perf_counter()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    queries: dict[str, str] = {}

    for cat in kept_categories:
        prompt = (
            f'Based on this LinkedIn profile, write a 1-2 sentence search query '
            f'to find research reports relevant to the "{cat["category"]}" topic area.\n\n'
            f'Focus on what specifically about {cat["category"]} matters to this person '
            f'given their role, industry, and experience. Be concrete and specific.\n\n'
            f'Profile:\n{json.dumps(profile, indent=2)}\n\n'
            f'Category context: {cat["reasoning"]}\n\n'
            f'Write ONLY the 1-2 sentence search query, nothing else.'
        )
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        query = response.choices[0].message.content.strip()
        queries[cat["category"]] = query
        logger.info(
            "  [%s] (weight=%.2f): %s",
            cat["category"],
            cat["relevance_score"],
            query[:120],
        )

    elapsed = time.perf_counter() - t0
    logger.info("Generated %d queries in %.2fs", len(queries), elapsed)
    return queries


# ---------------------------------------------------------------------------
# Step 4 — Weighted RRF (vector + BM25 per query)
# ---------------------------------------------------------------------------
def weighted_rrf(
    category_queries: dict[str, str],
    category_weights: dict[str, float],
    report_ids: list[int],
) -> tuple[list[tuple[int, float]], dict[int, str]]:
    """
    Run vector + BM25 per category query, merge via weighted RRF.

    Returns:
        rrf_ranked     — [(report_id, rrf_score), …] descending
        content_lookup — {report_id: page_content}
    """
    logger.info("Step 4: Running weighted RRF retrieval…")
    t0 = time.perf_counter()

    vector_db = _get_vector_db()
    os_client = _get_os_client()

    all_rankings: list[tuple[float, dict[int, int]]] = []
    content_lookup: dict[int, str] = {}

    for slug, query_text in category_queries.items():
        weight = category_weights[slug]

        vector_docs = vector_db.similarity_search_with_score(
            query=query_text,
            k=RESULTS_PER_QUERY,
            filter={"terms": {"metadata.report_id": report_ids}},
        )
        vector_ranking: dict[int, int] = {}
        for rank, (doc, _score) in enumerate(vector_docs):
            rid = doc.metadata["report_id"]
            vector_ranking[rid] = rank
            content_lookup[rid] = doc.page_content

        bm25_body = {
            "size": RESULTS_PER_QUERY,
            "query": {
                "bool": {
                    "must": [{"match": {"text": query_text}}],
                    "filter": [{"terms": {"metadata.report_id": report_ids}}],
                }
            },
        }
        bm25_response = os_client.search(index=OPENSEARCH_INDEX, body=bm25_body)
        bm25_ranking: dict[int, int] = {}
        for rank, hit in enumerate(bm25_response["hits"]["hits"]):
            rid = hit["_source"]["metadata"]["report_id"]
            bm25_ranking[rid] = rank
            if rid not in content_lookup:
                content_lookup[rid] = hit["_source"].get("text", "")

        all_rankings.append((weight, vector_ranking))
        all_rankings.append((weight, bm25_ranking))

        logger.info(
            "  [%s] Vector: %d | BM25: %d",
            slug,
            len(vector_ranking),
            len(bm25_ranking),
        )

    all_candidate_ids: set[int] = set()
    for _, ranking in all_rankings:
        all_candidate_ids |= set(ranking.keys())

    rrf_scores: dict[int, float] = {}
    for rid in all_candidate_ids:
        score = 0.0
        for weight, ranking in all_rankings:
            if rid in ranking:
                score += weight / (RRF_K + ranking[rid])
        rrf_scores[rid] = score

    rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    elapsed = time.perf_counter() - t0
    logger.info(
        "RRF complete: %d unique candidates in %.2fs", len(rrf_ranked), elapsed
    )
    for i, (rid, score) in enumerate(rrf_ranked[:5], 1):
        logger.info("  RRF #%d: report %d  score=%.6f", i, rid, score)

    return rrf_ranked, content_lookup


# ---------------------------------------------------------------------------
# Step 5 — Build profile summary for the cross-encoder
# ---------------------------------------------------------------------------
def build_profile_summary(profile: dict) -> str:
    parts: list[str] = []
    if profile.get("scraped_job_title"):
        parts.append(
            f"{profile['scraped_job_title']} at {profile.get('recent_company_name', '')}"
        )
    if profile.get("recent_company_industry"):
        parts.append(f"Industry: {profile['recent_company_industry']}")
    if profile.get("recent_company_details", {}).get("specialties"):
        parts.append(
            f"Specialties: {profile['recent_company_details']['specialties']}"
        )

    exp_titles: list[str] = []
    for exp in profile.get("scraped_experience", [])[:6]:
        title = exp.get("title", "")
        company = exp.get("company_name", "")
        if title and not title.startswith("-") and company:
            exp_titles.append(f"{title} at {company}")
    if exp_titles:
        parts.append(f"Experience: {'; '.join(exp_titles)}")

    summary = ". ".join(parts)
    logger.info("Profile summary (%d chars): %s", len(summary), summary[:150])
    return summary


# ---------------------------------------------------------------------------
# Step 6 — Cross-encoder reranking
# ---------------------------------------------------------------------------
def cross_encoder_rerank(
    profile_summary: str,
    rrf_ranked: list[tuple[int, float]],
    content_lookup: dict[int, str],
    report_slug_by_id: dict[int, str],
    report_to_slugs: dict[int, set],
) -> list[dict]:
    """Rerank top RRF candidates with the cross-encoder."""
    logger.info(
        "Step 5: Cross-encoder reranking top %d candidates…", TOP_N_FOR_RERANK
    )
    t0 = time.perf_counter()

    candidates = rrf_ranked[:TOP_N_FOR_RERANK]
    reranker = _get_cross_encoder()

    pairs = []
    candidate_rids = []
    for rid, _rrf_score in candidates:
        pairs.append((profile_summary, content_lookup.get(rid, "")))
        candidate_rids.append(rid)

    rerank_scores = reranker.predict(pairs)

    results: list[dict] = []
    for rid, (_, rrf_score), rerank_score in zip(
        candidate_rids, candidates, rerank_scores
    ):
        results.append(
            {
                "report_id": rid,
                "report_slug": report_slug_by_id.get(rid, "unknown"),
                "content": content_lookup.get(rid, ""),
                "rrf_score": rrf_score,
                "rerank_score": float(rerank_score),
                "matching_categories": sorted(report_to_slugs.get(rid, set())),
            }
        )

    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    elapsed = time.perf_counter() - t0
    logger.info("Cross-encoder reranking done in %.2fs", elapsed)
    for i, r in enumerate(results[:TOP_N_FINAL], 1):
        logger.info(
            "  #%d  report=%d  slug=%-60s  rerank=%.4f  rrf=%.6f  cats=%s",
            i,
            r["report_id"],
            r["report_slug"],
            r["rerank_score"],
            r["rrf_score"],
            r["matching_categories"],
        )

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_report_recommendations(profile: dict, top_n: int = TOP_N_FINAL) -> list[str]:
    """
    Full pipeline: profile dict → list of recommended report URLs.

    Args:
        profile: A single LinkedIn profile dict (as scraped).
        top_n:   Number of report URLs to return.

    Returns:
        List of report URLs, e.g.
        ["https://www.ghostresearch.com/reports/some-report-slug", …]
    """
    pipeline_start = time.perf_counter()

    logger.info("=" * 90)
    logger.info("RECOMMENDATION PIPELINE START")
    logger.info(
        "Profile: %s at %s | Industry: %s",
        profile.get("scraped_job_title", "N/A"),
        profile.get("recent_company_name", "N/A"),
        profile.get("recent_company_industry", "N/A"),
    )
    logger.info("=" * 90)

    # 1 — LLM category recommendation
    all_recommended = recommend_categories(profile)

    # 2 — Soft filtering + DB candidate expansion
    category_weights, report_ids, report_slug_by_id, report_to_slugs = (
        resolve_candidates(all_recommended)
    )
    kept_categories = [
        c for c in all_recommended if c["relevance_score"] >= SOFT_THRESHOLD
    ]

    # 3 — Multi-query generation
    category_queries = generate_category_queries(profile, kept_categories)

    # 4 — Weighted RRF retrieval
    rrf_ranked, content_lookup = weighted_rrf(
        category_queries, category_weights, report_ids
    )

    # 5 — Cross-encoder reranking
    profile_summary = build_profile_summary(profile)
    final_results = cross_encoder_rerank(
        profile_summary,
        rrf_ranked,
        content_lookup,
        report_slug_by_id,
        report_to_slugs,
    )

    # 6 — Build final URL list
    urls = [
        f"{BASE_REPORT_URL}/{r['report_slug']}"
        for r in final_results[:top_n]
    ]

    total_elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 90)
    logger.info("PIPELINE COMPLETE in %.2fs — returning %d URLs", total_elapsed, len(urls))
    for i, url in enumerate(urls, 1):
        logger.info("  %2d. %s", i, url)
    logger.info("=" * 90)

    return urls


def scrape_profile(linkedin_url: str) -> dict:
    """
    Scrape a LinkedIn profile by URL using the Voyager API.

    Requires the LI_AT environment variable to be set with a valid li_at cookie.
    """
    li_at = os.getenv("LI_AT", "").strip()
    if not li_at:
        raise RuntimeError("LI_AT environment variable is not set or empty")
    scraper = LinkedInScraper(li_at_token=li_at)
    return scraper.scrape_profile(linkedin_url)


def get_report_recommendations_from_url(
    linkedin_url: str, top_n: int = TOP_N_FINAL
) -> list[str]:
    """
    End-to-end pipeline: LinkedIn URL → list of recommended report URLs.

    Scrapes the profile, then runs the full recommendation pipeline.
    """
    logger.info("Scraping LinkedIn profile: %s", linkedin_url)
    profile = scrape_profile(linkedin_url)
    return get_report_recommendations(profile, top_n=top_n)


# # ---------------------------------------------------------------------------
# # CLI entry point
# # ---------------------------------------------------------------------------
# def main():
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Recommend Ghost Research reports for a LinkedIn profile."
#     )
#     parser.add_argument(
#         "linkedin_url",
#         type=str,
#         help="LinkedIn profile URL, e.g. https://www.linkedin.com/in/someone/",
#     )
#     parser.add_argument(
#         "--top-n",
#         type=int,
#         default=TOP_N_FINAL,
#         help="Number of report URLs to return (default: 10).",
#     )
#     args = parser.parse_args()

#     urls = get_report_recommendations_from_url(args.linkedin_url, top_n=args.top_n)

#     print("\n" + "=" * 70)
#     print("RECOMMENDED REPORT URLs")
#     print("=" * 70)
#     for url in urls:
#         print(f"  {url}")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()
