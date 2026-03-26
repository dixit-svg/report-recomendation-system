import json
import os
import logging
import time
from typing import Dict, List, Any, Tuple

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

# Set up production level logs: add both a file handler and a console handler.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("RECOMMEND_V2_LOG_PATH", "recommend_v2.log")

root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)

# Remove old handlers before adding new
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
root_logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
root_logger.addHandler(file_handler)

logger = logging.getLogger("recommendation_v2")

SOFT_THRESHOLD = 0.30
RRF_K = 60
RESULTS_PER_QUERY = 20
TOP_N_FINAL = 5
ALLOWED_CATEGORY_SLUGS = {"for-you", "sector", "theme", "region"}
OPENSEARCH_INDEX = "ghost_research_report_overviews"
BASE_REPORT_URL = "https://www.ghostresearch.com/reports"
DATA_FILE = os.path.join(os.path.dirname(__file__), "test_data.json")
CATEGORIES_FILE = os.path.join(os.path.dirname(__file__), "catagories.json")

HIGH_RELEVANCE_THRESHOLD = 0.75
MED_RELEVANCE_THRESHOLD = 0.50
TOP_K_HIGH = 4
TOP_K_MED = 3
TOP_K_LOW = 2


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
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
    try:
        credentials = boto3.Session(
            aws_access_key_id=os.getenv("AWS_OPENSEARCH_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_OPENSEARCH_SECRET_ACCESS_KEY"),
        ).get_credentials()
        logger.debug("AWS credentials loaded successfully in _get_aws_auth")
    except Exception as e:
        logger.error("Failed to load AWS credentials: %s", str(e), exc_info=True)
        raise
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
        try:
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
        except Exception as e:
            logger.exception("Failed to initialize OpenSearch vector store: %s", str(e))
            raise
    return _vector_db


def _get_os_client() -> OpenSearch:
    global _os_client
    if _os_client is None:
        try:
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
        except Exception as e:
            logger.exception("Failed to initialize OpenSearch client: %s", str(e))
            raise
    return _os_client


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        try:
            logger.info("Loading cross-encoder reranker…")
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
            logger.info("Cross-encoder loaded.")
        except Exception as e:
            logger.exception("Failed to load cross-encoder: %s", str(e))
            raise
    return _cross_encoder


def _top_k_for_relevance(relevance_score: float) -> int:
    if relevance_score >= HIGH_RELEVANCE_THRESHOLD:
        return TOP_K_HIGH
    if relevance_score >= MED_RELEVANCE_THRESHOLD:
        return TOP_K_MED
    return TOP_K_LOW


# ---------------------------------------------------------------------------
# Step 1 — LLM subcategory evaluation
# ---------------------------------------------------------------------------
def recommend_categories(
    profile: dict,
    categories: list[dict] | None = None,
) -> list[dict]:
    """Ask the LLM to pick relevant subcategories for *profile*."""
    logger.info("Step 1: Recommending subcategories via LLM…")
    t0 = time.perf_counter()

    try:
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
        logger.debug("OpenAI response: %s", response)
        recommended = json.loads(response.choices[0].message.content)["recommended"]
        elapsed = time.perf_counter() - t0
        logger.info("LLM returned %d subcategories in %.2fs", len(recommended), elapsed)
        for cat in recommended:
            logger.info(
                "  %-30s score=%.2f | %s",
                cat["category"],
                cat["relevance_score"],
                cat["reasoning"],
            )
        return recommended
    except Exception as e:
        logger.error("Error in recommend_categories: %s", str(e), exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 2 — Soft filtering + DB candidate pool expansion
# ---------------------------------------------------------------------------
def resolve_candidates(
    all_recommended: list[dict],
) -> Tuple[Dict[str, float], List[int], Dict[int, str], Dict[int, set]]:
    """
    Apply soft threshold, resolve subcategory slugs against the DB,
    and expand the candidate report pool.

    Returns:
        category_weights  — {slug: relevance_score}
        report_ids        — sorted candidate report IDs
        report_slug_by_id — {report_id: url_slug}
        report_to_slugs   — {report_id: {matching_subcategory_slugs}}
    """
    logger.info("Step 2: Resolving candidates from DB (threshold >= %.2f)…", SOFT_THRESHOLD)
    t0 = time.perf_counter()
    try:
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
            s
            for s in category_slugs
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
            "Valid slugs: %s | Candidate pool: %d reports | DB total: %d (%.2fs)",
            valid_slugs,
            len(reports),
            len(db_reports),
            elapsed,
        )
        return category_weights, report_ids, report_slug_by_id, report_to_slugs
    except Exception as e:
        logger.error("Error in resolve_candidates: %s", str(e), exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 3 — Generate one focused search query per subcategory
# ---------------------------------------------------------------------------
def generate_category_queries(
    profile: dict,
    kept_categories: list[dict],
) -> Dict[str, str]:
    """Return {subcategory_slug: search_query} via one LLM call per category."""
    logger.info("Step 3: Generating %d focused queries…", len(kept_categories))
    t0 = time.perf_counter()

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        queries: Dict[str, str] = {}

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
                "  [%s] (score=%.2f): %s",
                cat["category"],
                cat["relevance_score"],
                query[:120],
            )

        elapsed = time.perf_counter() - t0
        logger.info("Generated %d queries in %.2fs", len(queries), elapsed)
        return queries
    except Exception as e:
        logger.error("Error in generate_category_queries: %s", str(e), exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 4 — Per-subcategory retrieval + RRF + variable top-K selection
# ---------------------------------------------------------------------------
def per_subcategory_rrf(
    category_queries: Dict[str, str],
    category_weights: Dict[str, float],
    report_ids: List[int],
) -> Tuple[List[dict], Dict[int, str]]:
    """
    For each subcategory:
      1. Run vector search (k=RESULTS_PER_QUERY) and BM25 search independently.
      2. Merge the two result lists with standard RRF (no weighting — same query).
      3. Select top-K results where K depends on the subcategory's relevance score.

    Deduplicate across subcategories using MAX weighted score
    (weighted_score = rrf_score × relevance_score).  When a report appears
    in multiple subcategories' top-K, the entry with the highest weighted
    score wins attribution; all matching subcategories are recorded.

    Returns:
        selected_reports — list of dicts with report_id, weighted_score, source info, etc.
        content_lookup   — {report_id: page_content}
    """
    logger.info("Step 4: Per-subcategory RRF retrieval…")
    t0 = time.perf_counter()

    try:
        vector_db = _get_vector_db()
        os_client = _get_os_client()
        content_lookup: Dict[int, str] = {}

        per_subcat_results: Dict[str, List[Tuple[int, float]]] = {}

        for slug, query_text in category_queries.items():
            relevance = category_weights[slug]

            # --- Vector search ---
            try:
                vector_docs = vector_db.similarity_search_with_score(
                    query=query_text,
                    k=RESULTS_PER_QUERY,
                    filter={"terms": {"metadata.report_id": report_ids}},
                )
            except Exception as e:
                logger.error("Vector search failed for %s: %s", slug, str(e), exc_info=True)
                vector_docs = []

            vector_ranking: Dict[int, int] = {}
            for rank, (doc, _score) in enumerate(vector_docs):
                rid = doc.metadata["report_id"]
                vector_ranking[rid] = rank
                content_lookup[rid] = doc.page_content

            # --- BM25 search ---
            bm25_body = {
                "size": RESULTS_PER_QUERY,
                "query": {
                    "bool": {
                        "must": [{"match": {"text": query_text}}],
                        "filter": [{"terms": {"metadata.report_id": report_ids}}],
                    }
                },
            }
            try:
                bm25_response = os_client.search(index=OPENSEARCH_INDEX, body=bm25_body)
                bm25_hits = bm25_response["hits"]["hits"]
            except Exception as e:
                logger.error("BM25 search failed for %s: %s", slug, str(e), exc_info=True)
                bm25_hits = []
            bm25_ranking: Dict[int, int] = {}
            for rank, hit in enumerate(bm25_hits):
                rid = hit["_source"]["metadata"]["report_id"]
                bm25_ranking[rid] = rank
                if rid not in content_lookup:
                    content_lookup[rid] = hit["_source"].get("text", "")

            logger.info(
                "  [%s] (relevance=%.2f) Vector: %d | BM25: %d",
                slug,
                relevance,
                len(vector_ranking),
                len(bm25_ranking),
            )

            # --- RRF within this subcategory ---
            all_rids = set(vector_ranking.keys()) | set(bm25_ranking.keys())
            rrf_scores: Dict[int, float] = {}
            for rid in all_rids:
                score = 0.0
                if rid in vector_ranking:
                    score += 1.0 / (RRF_K + vector_ranking[rid])
                if rid in bm25_ranking:
                    score += 1.0 / (RRF_K + bm25_ranking[rid])
                rrf_scores[rid] = score

            ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            per_subcat_results[slug] = ranked

            top_k = _top_k_for_relevance(relevance)
            logger.info(
                "    RRF produced %d unique docs → selecting top %d",
                len(ranked),
                top_k,
            )
            for i, (rid, score) in enumerate(ranked[:top_k], 1):
                logger.debug("    #%d  report=%d  rrf=%.6f", i, rid, score)

        # --- Variable top-K selection per subcategory ---
        # Collect top-K from each subcategory (duplicates allowed at this stage).
        all_candidates: List[dict] = []

        for slug, ranked in per_subcat_results.items():
            relevance = category_weights[slug]
            top_k = _top_k_for_relevance(relevance)

            for rid, rrf_score in ranked[:top_k]:
                weighted_score = rrf_score * relevance
                all_candidates.append(
                    {
                        "report_id": rid,
                        "rrf_score": rrf_score,
                        "weighted_score": weighted_score,
                        "source_subcategory": slug,
                        "source_relevance": relevance,
                    }
                )

        # --- Deduplicate using MAX weighted score ---
        # For each report_id keep the entry with the highest weighted_score
        # and track every subcategory that selected it.
        best_entry: Dict[int, dict] = {}
        found_in: Dict[int, List[str]] = {}

        for entry in all_candidates:
            rid = entry["report_id"]
            found_in.setdefault(rid, []).append(entry["source_subcategory"])

            prev = best_entry.get(rid)
            if prev is None or entry["weighted_score"] > prev["weighted_score"]:
                best_entry[rid] = entry

        selected: List[dict] = []
        for rid, entry in best_entry.items():
            entry["found_in_subcategories"] = sorted(set(found_in[rid]))
            selected.append(entry)

        selected.sort(key=lambda x: x["weighted_score"], reverse=True)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Per-subcategory RRF complete: %d unique reports selected "
            "(from %d total slots) in %.2fs",
            len(selected),
            len(all_candidates),
            elapsed,
        )
        for r in selected:
            logger.info(
                "  report=%d  weighted=%.6f  rrf=%.6f  from=%s (%.2f)  also_in=%s",
                r["report_id"],
                r["weighted_score"],
                r["rrf_score"],
                r["source_subcategory"],
                r["source_relevance"],
                r["found_in_subcategories"],
            )

        return selected, content_lookup
    except Exception as e:
        logger.error("Error in per_subcategory_rrf: %s", str(e), exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 5 — Build enhanced profile summary for cross-encoder
# ---------------------------------------------------------------------------
def build_enhanced_profile_summary(
    profile: dict, top_categories: list[dict]
) -> str:
    """
    Build a rich profile summary that includes job context, about section,
    skills, experience, AND the user's identified interest areas.
    """
    parts: List[str] = []

    try:
        if profile.get("scraped_job_title"):
            parts.append(
                f"{profile['scraped_job_title']} at "
                f"{profile.get('recent_company_name', '')}"
            )

        if profile.get("scraped_headline"):
            parts.append(profile["scraped_headline"])

        if profile.get("recent_company_industry"):
            parts.append(f"Industry: {profile['recent_company_industry']}")

        specialties = profile.get("recent_company_details", {}).get("specialties")
        if specialties:
            parts.append(f"Company specialties: {specialties}")

        about = profile.get("scraped_about", "")
        if about:
            parts.append(f"About: {about[:300]}")

        skills = profile.get("scraped_skills", [])[:10]
        if skills:
            parts.append(f"Key skills: {', '.join(skills)}")

        exp_titles: List[str] = []
        for exp in profile.get("scraped_experience", [])[:4]:
            title = exp.get("title", "")
            company = exp.get("company_name", "")
            if title and not title.startswith("-") and company:
                exp_titles.append(f"{title} at {company}")
        if exp_titles:
            parts.append(f"Experience: {'; '.join(exp_titles)}")

        interest_parts: List[str] = []
        for cat in top_categories[:5]:
            interest_parts.append(f"{cat['category']} ({cat['reasoning']})")
        if interest_parts:
            parts.append(f"Key interest areas: {'; '.join(interest_parts)}")

        summary = ". ".join(parts)
        logger.info(
            "Enhanced profile summary (%d chars): %s…", len(summary), summary[:200]
        )
        return summary
    except Exception as e:
        logger.error("Error in build_enhanced_profile_summary: %s", str(e), exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 6 — Cross-encoder reranking
# ---------------------------------------------------------------------------
def cross_encoder_rerank(
    profile_summary: str,
    selected_reports: List[dict],
    content_lookup: Dict[int, str],
    report_slug_by_id: Dict[int, str],
    report_to_slugs: Dict[int, set],
) -> List[dict]:
    """Score each (profile_summary, report_content) pair with the cross-encoder."""
    logger.info(
        "Step 6: Cross-encoder reranking %d candidates…", len(selected_reports)
    )
    t0 = time.perf_counter()

    try:
        reranker = _get_cross_encoder()

        pairs = []
        rids = []
        for entry in selected_reports:
            rid = entry["report_id"]
            pairs.append((profile_summary, content_lookup.get(rid, "")))
            rids.append(rid)

        rerank_scores = reranker.predict(pairs)

        results: List[dict] = []
        for entry, rerank_score in zip(selected_reports, rerank_scores):
            rid = entry["report_id"]
            results.append(
                {
                    "report_id": rid,
                    "report_slug": report_slug_by_id.get(rid, "unknown"),
                    "content": content_lookup.get(rid, ""),
                    "rrf_score": entry["rrf_score"],
                    "weighted_score": entry["weighted_score"],
                    "rerank_score": float(rerank_score),
                    "source_subcategory": entry["source_subcategory"],
                    "source_relevance": entry["source_relevance"],
                    "found_in_subcategories": entry["found_in_subcategories"],
                    "matching_categories": sorted(report_to_slugs.get(rid, set())),
                }
            )

        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        elapsed = time.perf_counter() - t0
        logger.info("Cross-encoder reranking done in %.2fs", elapsed)
        for i, r in enumerate(results[:TOP_N_FINAL], 1):
            logger.info(
                "  #%d  report=%d  slug=%-55s  rerank=%.4f  "
                "weighted=%.6f  from=%s  found_in=%s",
                i,
                r["report_id"],
                r["report_slug"],
                r["rerank_score"],
                r["weighted_score"],
                r["source_subcategory"],
                r["found_in_subcategories"],
            )

        return results
    except Exception as e:
        logger.error("Error in cross_encoder_rerank: %s", str(e), exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_report_recommendations(
    profile: dict, top_n: int = TOP_N_FINAL
) -> list[str]:
    """
    Full pipeline: profile dict → list of recommended report URLs.

    Pipeline stages:
        1. LLM subcategory evaluation
        2. Soft filtering + DB candidate pool expansion
        3. One search query per subcategory (LLM)
        4. Per-subcategory vector+BM25 → per-subcategory RRF → variable top-K
        5. Enhanced profile summary construction
        6. Cross-encoder reranking → final top-N
    """
    pipeline_start = time.perf_counter()

    logger.info("=" * 90)
    logger.info("RECOMMENDATION PIPELINE V2 START")
    logger.info(
        "Profile: %s at %s | Industry: %s",
        profile.get("scraped_job_title", "N/A"),
        profile.get("recent_company_name", "N/A"),
        profile.get("recent_company_industry", "N/A"),
    )
    logger.info("=" * 90)

    try:
        # 1 — LLM subcategory evaluation
        all_recommended = recommend_categories(profile)

        # 2 — Soft filtering + DB candidate expansion
        category_weights, report_ids, report_slug_by_id, report_to_slugs = (
            resolve_candidates(all_recommended)
        )
        kept_categories = [
            c for c in all_recommended if c["relevance_score"] >= SOFT_THRESHOLD
        ]

        # 3 — One search query per subcategory
        category_queries = generate_category_queries(profile, kept_categories)

        # 4 — Per-subcategory RRF + variable top-K selection
        selected_reports, content_lookup = per_subcategory_rrf(
            category_queries, category_weights, report_ids
        )

        # 5 — Enhanced profile summary
        profile_summary = build_enhanced_profile_summary(profile, kept_categories)

        # 6 — Cross-encoder reranking
        final_results = cross_encoder_rerank(
            profile_summary,
            selected_reports,
            content_lookup,
            report_slug_by_id,
            report_to_slugs,
        )

        # Build final URL list
        urls = [
            f"{BASE_REPORT_URL}/{r['report_slug']}" for r in final_results[:top_n]
        ]

        total_elapsed = time.perf_counter() - pipeline_start
        logger.info("=" * 90)
        logger.info(
            "PIPELINE V2 COMPLETE in %.2fs — returning %d URLs", total_elapsed, len(urls)
        )
        for i, url in enumerate(urls, 1):
            logger.info("  %2d. %s", i, url)
        logger.info("=" * 90)

        return urls
    except Exception as e:
        logger.error("Error in get_report_recommendations: %s", str(e), exc_info=True)
        raise


def scrape_profile(linkedin_url: str) -> dict:
    """Scrape a LinkedIn profile by URL using the Voyager API."""
    li_at = os.getenv("LI_AT", "").strip()
    if not li_at:
        logger.critical("LI_AT environment variable is not set or empty")
        raise RuntimeError("LI_AT environment variable is not set or empty")
    try:
        scraper = LinkedInScraper(li_at_token=li_at)
        logger.info("Attempting to scrape profile from LinkedIn URL: %s", linkedin_url)
        profile = scraper.scrape_profile(linkedin_url)
        logger.info("Successfully scraped LinkedIn profile: %s", linkedin_url)
        return profile
    except Exception as e:
        logger.error("Failed to scrape profile from LinkedIn URL %s: %s", linkedin_url, str(e), exc_info=True)
        raise


def get_report_recommendations_from_url(
    linkedin_url: str, top_n: int = TOP_N_FINAL
) -> list[str]:
    """End-to-end: LinkedIn URL → list of recommended report URLs."""
    logger.info("Scraping LinkedIn profile: %s", linkedin_url)
    try:
        profile = scrape_profile(linkedin_url)
        with open("profile.json", "w") as f:
            json.dump(profile, f, indent=2)
        logger.info("Profile scraped and saved to profile.json")
        return get_report_recommendations(profile, top_n=top_n)
    except Exception as e:
        logger.error("Error in get_report_recommendations_from_url: %s", str(e), exc_info=True)
        raise
