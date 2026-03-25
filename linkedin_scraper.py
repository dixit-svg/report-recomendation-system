"""
Standalone LinkedIn profile scraper using LinkedIn's Voyager API.
No browser or src.* dependencies required — just needs a li_at cookie token.

Usage:
    python linkedin_scraper.py "https://www.linkedin.com/in/some-person/"
    python linkedin_scraper.py url1 url2 url3
    python linkedin_scraper.py                  # prompts for URL interactively
"""

import json
import logging
import os
import random
import re
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote as urlquote, urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

INTER_REQUEST_DELAY = (1.0, 2.5)
INTER_PROFILE_DELAY = (2.0, 4.0)

FULL_PROFILE_DECORATION = (
    "com.linkedin.voyager.dash.deco.identity.profile."
    "FullProfileWithEntities-108"
)

MONTH_ABBR = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def normalize_profile_url(url: str) -> Optional[str]:
    """Normalize a LinkedIn profile URL to https://www.linkedin.com/in/<id>."""
    if not url:
        return None
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower().replace("www.", "")
    if "linkedin.com" not in host:
        return None
    path = parsed.path.rstrip("/")
    match = re.search(r"/in/([^/]+)", path)
    if not match:
        return None
    return f"https://www.linkedin.com/in/{match.group(1)}"


def extract_public_id(profile_url: str) -> Optional[str]:
    url = normalize_profile_url(profile_url) or profile_url
    path = urlparse(url).path.rstrip("/")
    parts = path.split("/")
    return parts[-1] if parts else None


def format_api_date(date_dict: Optional[Dict]) -> str:
    if not date_dict:
        return ""
    m = date_dict.get("month", 0)
    y = date_dict.get("year", "")
    month_str = MONTH_ABBR[m] if 0 < m < 13 else ""
    return f"{month_str} {y}".strip()


def bucket_api_entities(
    included: List[Dict], target_public_id: Optional[str] = None
) -> Dict[str, Any]:
    """Partition the normalized `included` array by entity type."""
    profile = None
    positions: List[Dict] = []
    skills: List[Dict] = []
    companies: Dict[str, Dict] = {}
    industries: Dict[str, str] = {}
    skills_paging: Dict[str, int] = {}
    skill_urns: List[str] = []
    candidate_profiles: List[Dict] = []

    for item in included:
        t = item.get("$type", "")
        if t.endswith(".Profile"):
            candidate_profiles.append(item)
        elif t.endswith(".Position"):
            positions.append(item)
        elif t.endswith(".Skill"):
            skills.append(item)
        elif "Company" in t or "School" in t:
            urn = item.get("entityUrn")
            if urn:
                companies[urn] = item
        elif t.endswith(".Industry"):
            urn = item.get("entityUrn")
            if urn:
                industries[urn] = item.get("name", "")

        paging = item.get("paging")
        elements = item.get("*elements") or []
        if paging and elements and "fsd_skill" in str(elements[:1]):
            skills_paging = {
                "total": paging.get("total", 0),
                "count": paging.get("count", 0),
                "start": paging.get("start", 0),
            }
            skill_urns = elements

    if candidate_profiles:
        if target_public_id:
            for cp in candidate_profiles:
                if cp.get("publicIdentifier") == target_public_id:
                    profile = cp
                    break
        if not profile:
            for cp in candidate_profiles:
                recipes = cp.get("$recipeTypes") or []
                if any("FullProfileWithEntities" in r for r in recipes):
                    profile = cp
                    break
        if not profile:
            profile = candidate_profiles[0]

    return {
        "profile": profile,
        "positions": positions,
        "skills": skills,
        "companies": companies,
        "industries": industries,
        "skills_paging": skills_paging,
        "skill_urns": skill_urns,
    }


class LinkedInScraper:
    """Lightweight LinkedIn scraper using the Voyager API (HTTP only, no browser)."""

    def __init__(self, li_at_token: str):
        self.li_at_token = li_at_token
        self.session: Optional[requests.Session] = None
        self._init_session()

    def _init_session(self) -> None:
        csrf = f"ajax:{secrets.token_hex(12)}"
        s = requests.Session()
        s.cookies.set("li_at", self.li_at_token, domain=".linkedin.com")
        s.cookies.set("JSESSIONID", f'"{csrf}"', domain=".linkedin.com")
        s.headers.update({
            "csrf-token": csrf,
            "user-agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
            ),
            "accept": "application/vnd.linkedin.normalized+json+2.1",
            "accept-language": "en-AU,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
            "x-li-lang": "en_US",
            "x-restli-protocol-version": "2.0.0",
        })
        self.session = s
        log.info("Voyager API session initialized")

    def _delay(self, lo: float = 0.5, hi: float = 1.5) -> None:
        time.sleep(random.uniform(lo, hi))

    def _api_request(self, path: str, params: Optional[Dict] = None) -> Dict:
        self._delay(*INTER_REQUEST_DELAY)
        url = f"https://www.linkedin.com/voyager/api{path}"
        try:
            resp = self.session.get(url, params=params, timeout=20)
        except requests.TooManyRedirects:
            raise PermissionError(
                "LinkedIn keeps redirecting (TooManyRedirects) — your li_at token is "
                "expired. Get a fresh one from your browser: DevTools → Application → "
                "Cookies → linkedin.com → li_at"
            )
        if resp.status_code == 999:
            raise RuntimeError("LinkedIn returned 999 (anti-bot / rate limit)")
        if resp.status_code in (401, 403):
            raise PermissionError(
                f"LinkedIn API returned {resp.status_code} — your li_at token may be expired"
            )
        if resp.status_code in (301, 302, 303, 307, 308):
            raise PermissionError(
                f"LinkedIn redirected ({resp.status_code}) — your li_at token is likely expired"
            )
        resp.raise_for_status()
        return resp.json()

    def _get_full_profile(self, public_id: str) -> Dict:
        return self._api_request(
            "/identity/dash/profiles",
            params={
                "q": "memberIdentity",
                "memberIdentity": public_id,
                "decorationId": FULL_PROFILE_DECORATION,
            },
        )

    def _get_company(self, universal_name: str) -> Dict:
        return self._api_request(
            "/organization/companies",
            params={
                "decorationId": (
                    "com.linkedin.voyager.deco.organization.web."
                    "WebFullCompanyMain-12"
                ),
                "q": "universalName",
                "universalName": universal_name,
            },
        )

    def scrape_profile(self, profile_url: str) -> Dict[str, Any]:
        """Scrape a single LinkedIn profile. Returns a dict with all profile data."""
        public_id = extract_public_id(profile_url)
        if not public_id:
            raise ValueError(f"Cannot extract public_id from {profile_url}")

        log.info(f"Scraping profile: {public_id}")
        data = self._get_full_profile(public_id)
        entities = bucket_api_entities(data.get("included", []), target_public_id=public_id)
        profile = entities["profile"]
        if not profile:
            raise RuntimeError("No profile entity in API response")

        def _first_locale_value(ml_dict: Optional[Dict]) -> str:
            if not ml_dict:
                return ""
            return ml_dict.get("en_US") or next(iter(ml_dict.values()), "")

        first = _first_locale_value(profile.get("multiLocaleFirstName")) or profile.get("firstName") or ""
        last = _first_locale_value(profile.get("multiLocaleLastName")) or profile.get("lastName") or ""

        all_skills = [s["name"] for s in entities["skills"] if s.get("name")]

        result: Dict[str, Any] = {
            "profile_url": profile_url,
            "status": "success",
            "scraped_name": f"{first} {last}".strip() or None,
            "scraped_headline": profile.get("headline"),
            "scraped_about": profile.get("summary") or "",
            "scraped_skills": all_skills,
            "recent_company_name": None,
            "recent_company_website": None,
            "recent_company_industry": None,
            "recent_company_size": None,
            "scraped_job_title": None,
            "scraped_experience": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Experience (sort most-recent first)
        positions = entities["positions"]

        def _pos_end_key(p: Dict) -> Tuple[int, int]:
            dr = p.get("dateRange") or {}
            end = dr.get("end")
            if not end:
                return (9999, 12)
            return (end.get("year", 0), end.get("month", 0))

        positions.sort(key=_pos_end_key, reverse=True)

        experience_list: List[Dict[str, Any]] = []
        company_universal_name: Optional[str] = None

        for pos in positions:
            dr = pos.get("dateRange") or {}
            start_str = format_api_date(dr.get("start"))
            end_d = dr.get("end")
            end_str = format_api_date(end_d) if end_d else "Present"
            dates = f"{start_str} - {end_str}" if start_str else ""

            cn = pos.get("companyName") or (
                (pos.get("multiLocaleCompanyName") or {}).get("en_US")
            )
            entry: Dict[str, Any] = {
                "title": pos.get("title"),
                "company_name": cn,
                "employment_type": "",
                "date_range": dates,
                "location": pos.get("locationName") or "",
                "description": pos.get("description") or "",
            }
            experience_list.append(entry)

            if not company_universal_name and pos.get("companyUrn"):
                c_ent = entities["companies"].get(pos["companyUrn"])
                if c_ent:
                    company_universal_name = c_ent.get("universalName")

        result["scraped_experience"] = experience_list
        if positions:
            cn = positions[0].get("companyName") or (
                (positions[0].get("multiLocaleCompanyName") or {}).get("en_US")
            )
            result["recent_company_name"] = cn
            result["scraped_job_title"] = positions[0].get("title")

        # Company details (website / industry / size)
        if company_universal_name:
            try:
                company_resp = self._get_company(company_universal_name)
                comp_entities = company_resp.get("included", [])
                for c in comp_entities:
                    ct = c.get("$type", "")
                    if "Company" not in ct:
                        continue
                    if c.get("universalName") != company_universal_name:
                        continue

                    website = (
                        c.get("companyPageUrl")
                        or ((c.get("callToAction") or {}).get("url"))
                    )
                    if website and not website.startswith("http"):
                        website = "https://" + website
                    result["recent_company_website"] = website

                    ind_urns = c.get("*companyIndustries") or c.get("companyIndustries") or []
                    for iurn in ind_urns:
                        iurn_str = iurn if isinstance(iurn, str) else ""
                        for ci in comp_entities:
                            if ci.get("entityUrn") == iurn_str and ci.get("name"):
                                result["recent_company_industry"] = ci["name"]
                                break
                        if result["recent_company_industry"]:
                            break

                    staff_range = c.get("staffCountRange") or {}
                    if staff_range.get("start") or staff_range.get("end"):
                        result["recent_company_size"] = (
                            f"{staff_range.get('start', '')}-"
                            f"{staff_range.get('end', '')} employees"
                        )
                    elif c.get("staffCount"):
                        result["recent_company_size"] = f"{c['staffCount']} employees"

                    if c.get("name") and not result["recent_company_name"]:
                        result["recent_company_name"] = c["name"]

                    founded_raw = (c.get("foundedOn") or {}).get("year")
                    specialties_raw = c.get("specialities") or c.get("specialties")
                    result["recent_company_details"] = {
                        "name": c.get("name"),
                        "website": website,
                        "industry": result.get("recent_company_industry"),
                        "size": result.get("recent_company_size"),
                        "headquarters": (c.get("headquarter") or {}).get("city"),
                        "founded": str(founded_raw) if founded_raw else None,
                        "type": (c.get("companyType") or {}).get("localizedName"),
                        "specialties": (
                            ", ".join(specialties_raw)
                            if isinstance(specialties_raw, list)
                            else (specialties_raw or "")
                        ),
                    }
                    break
            except Exception as exc:
                log.warning(f"Company fetch failed: {exc}")

        log.info(
            f"Done — name='{result.get('scraped_name')}' "
            f"company='{result.get('recent_company_name')}' "
            f"skills={len(result.get('scraped_skills', []))} "
            f"experience={len(experience_list)}"
        )
        return result

    def scrape_profiles(self, profile_urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple LinkedIn profiles. Returns a list of result dicts."""
        results = []
        for idx, url in enumerate(profile_urls, 1):
            log.info(f"[{idx}/{len(profile_urls)}] {url}")
            try:
                results.append(self.scrape_profile(url))
            except Exception as exc:
                log.error(f"Failed to scrape {url}: {exc}")
                results.append({
                    "profile_url": url,
                    "status": "failed",
                    "error": str(exc),
                    "scraped_name": None,
                    "scraped_headline": None,
                    "scraped_experience": [],
                    "timestamp": datetime.utcnow().isoformat(),
                })
            if idx < len(profile_urls):
                self._delay(*INTER_PROFILE_DELAY)
        return results


# def main():
#     li_at = os.getenv("LI_AT", "").strip()
#     if not li_at:
#         print("Error: LI_AT not found in .env file or environment variables.")
#         sys.exit(1)

#     urls = sys.argv[1:]
#     if not urls:
#         raw = input("Enter LinkedIn profile URL(s) (comma or space separated):\n> ").strip()
#         urls = [u.strip() for u in re.split(r"[,\s]+", raw) if u.strip()]

#     if not urls:
#         print("No URLs provided.")
#         sys.exit(1)

#     scraper = LinkedInScraper(li_at_token=li_at)
#     results = scraper.scrape_profiles(urls)

#     output = {
#         "generated_at": datetime.utcnow().isoformat(),
#         "profile_urls": urls,
#         "results": results,
#     }

#     filename = f"linkedin_scrape_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     with open(filename, "w") as f:
#         json.dump(output, f, indent=2)

#     print(f"\nResults saved to {filename}")

#     for r in results:
#         status = r.get("status", "unknown")
#         name = r.get("scraped_name") or "N/A"
#         company = r.get("recent_company_name") or "N/A"
#         title = r.get("scraped_job_title") or "N/A"
#         print(f"  [{status}] {name} — {title} at {company}")


# if __name__ == "__main__":
#     main()
