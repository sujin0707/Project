import os
import time
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

S2_API_KEY = os.getenv("S2_API_KEY", "")
S2_BASE = "https://api.semanticscholar.org/graph/v1"


def s2_headers():
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    return headers


def _get_with_retry(url, params, timeout=20, max_retries=2):
    """Simple GET wrapper with basic retry on 429, and raises for other errors."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=s2_headers(), timeout=timeout)
            if r.status_code == 429:
                # rate limited - wait a bit and retry
                wait = 2 * (attempt + 1)
                time.sleep(wait)
                last_exc = requests.exceptions.HTTPError("429 Too Many Requests")
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(1)
    raise last_exc


@st.cache_data(ttl=3600, show_spinner=False)
def search_paper(query: str, limit: int = 5):
    fields = "paperId,title,abstract,year,authors,venue,citationCount,referenceCount,url"
    url = f"{S2_BASE}/paper/search"
    params = {"query": query, "limit": limit, "fields": fields}
    r = _get_with_retry(url, params, timeout=20)
    return r.json().get("data", [])


@st.cache_data(ttl=3600, show_spinner=False)
def get_paper_core(paper_id: str):
    """Basic paper info only (no refs/citations) - fast and small."""
    fields = "paperId,title,abstract,year,authors,venue,citationCount,referenceCount,url"
    url = f"{S2_BASE}/paper/{paper_id}"
    params = {"fields": fields}
    r = _get_with_retry(url, params, timeout=20)
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def get_paper_references(paper_id: str, limit: int = 50):
    """Fetch references via dedicated paginated endpoint (avoids huge payloads)."""
    fields = "title,abstract,year,authors,venue,citationCount,url"
    url = f"{S2_BASE}/paper/{paper_id}/references"
    params = {"fields": fields, "limit": limit}
    r = _get_with_retry(url, params, timeout=20)
    data = r.json().get("data", [])
    return [d.get("citedPaper") for d in data if d.get("citedPaper")]


@st.cache_data(ttl=3600, show_spinner=False)
def get_paper_citations(paper_id: str, limit: int = 50):
    """Fetch citations via dedicated paginated endpoint (avoids huge payloads)."""
    fields = "title,abstract,year,authors,venue,citationCount,url"
    url = f"{S2_BASE}/paper/{paper_id}/citations"
    params = {"fields": fields, "limit": limit}
    r = _get_with_retry(url, params, timeout=20)
    data = r.json().get("data", [])
    return [d.get("citingPaper") for d in data if d.get("citingPaper")]


def authors_to_str(authors, max_authors=3):
    if not authors:
        return ""
    names = [a.get("name", "") for a in authors if a.get("name")]
    if len(names) > max_authors:
        return ", ".join(names[:max_authors]) + " et al."
    return ", ".join(names)


def paper_list_to_df(papers, category=None):
    rows = []
    for p in papers:
        if not p:
            continue
        rows.append({
            "year": p.get("year"),
            "title": p.get("title"),
            "authors": authors_to_str(p.get("authors", [])),
            "venue": p.get("venue"),
            "citationCount": p.get("citationCount", 0),
            "url": p.get("url"),
            "abstract": p.get("abstract") or "",
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["citationCount"] = pd.to_numeric(df["citationCount"], errors="coerce").fillna(0)
    if category is not None:
        df["category"] = category
    return df


def make_timeline(df):
    """Build a year x category timeline. df must have year, citationCount, title,
    and optionally 'category' columns."""
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["year"]).copy()
    if df.empty:
        return pd.DataFrame()

    # sort by citationCount so 'first' picks the most-cited paper per group
    df_sorted = df.sort_values("citationCount", ascending=False)

    group_cols = ["year"]
    if "category" in df.columns:
        group_cols = ["year", "category"]

    timeline = (
        df_sorted.groupby(group_cols)
        .agg(paper_count=("title", "count"), top_paper=("title", "first"))
        .reset_index()
        .sort_values("year")
    )
    return timeline


def simple_research_flow(seed, refs_df, cites_df):
    seed_title = seed.get("title", "Unknown paper")
    seed_year = seed.get("year", "Unknown year")

    old_refs = refs_df.dropna(subset=["year"]).sort_values(
        ["year", "citationCount"], ascending=[True, False]
    ).head(5)

    recent_cites = cites_df.dropna(subset=["year"]).sort_values(
        ["year", "citationCount"], ascending=[False, False]
    ).head(5)

    lines = []
    lines.append("### 연구 흐름 초안")
    lines.append("")
    lines.append(f"**Seed paper:** {seed_title} ({seed_year})")
    lines.append("")
    lines.append("#### 1. 선행 연구 흐름")
    if old_refs.empty:
        lines.append("- 선행 논문 정보를 충분히 찾지 못했습니다.")
    else:
        for _, row in old_refs.iterrows():
            lines.append(
                f"- {int(row['year'])}: {row['title']} "
                f"({row['venue'] or 'venue unknown'}, citations: {int(row['citationCount'])})"
            )

    lines.append("")
    lines.append("#### 2. 이 논문 이후의 최신 흐름")
    if recent_cites.empty:
        lines.append("- 아직 이 논문을 인용한 최신 논문 정보를 충분히 찾지 못했습니다.")
    else:
        for _, row in recent_cites.iterrows():
            lines.append(
                f"- {int(row['year'])}: {row['title']} "
                f"({row['venue'] or 'venue unknown'}, citations: {int(row['citationCount'])})"
            )

    lines.append("")
    lines.append("#### 3. 해석")
    lines.append(
        "- 위 목록을 기준으로, seed paper 이전에는 어떤 표현 방식/모델링 방식이 쓰였는지 보고, "
        "seed paper 이후에는 어떤 한계를 개선하려는 논문들이 등장했는지 확인하면 됩니다."
    )
    lines.append(
        "- 다음 단계에서는 각 abstract를 LLM에 넣어 `문제 - 방법 - 기여 - 한계 - seed paper와의 관계`로 요약하면 됩니다."
    )

    return "\n".join(lines)


st.set_page_config(page_title="Paper Flow", layout="wide")

st.title("Paper Flow: 논문 인용 흐름 분석기")

query = st.text_input("논문 제목, 키워드, DOI, arXiv ID를 입력하세요")

if st.button("논문 검색") and query.strip():
    with st.spinner("논문 검색 중..."):
        try:
            results = search_paper(query)
        except requests.exceptions.HTTPError:
            st.error("API 요청 제한(429)에 걸렸습니다. 잠시 후 다시 시도해주세요.")
            results = []
        except requests.exceptions.RequestException as e:
            st.error(f"네트워크 오류가 발생했습니다: {e}")
            results = []

    if not results:
        st.session_state.pop("search_results", None)
        st.error("검색 결과가 없습니다.")
    else:
        st.session_state["search_results"] = results
        # clear any previously loaded paper detail when a new search happens
        st.session_state.pop("paper_detail", None)
        st.session_state.pop("refs_raw", None)
        st.session_state.pop("cites_raw", None)

if "search_results" in st.session_state:
    results = st.session_state["search_results"]

    options = [
        f"{p.get('title')} ({p.get('year')}) - citations: {p.get('citationCount', 0)}"
        for p in results
    ]

    selected_idx = st.selectbox("분석할 논문 선택", range(len(options)), format_func=lambda i: options[i])
    selected = results[selected_idx]

    ref_limit = st.slider("가져올 선행 논문(References) 최대 개수", 10, 100, 50, step=10)
    cite_limit = st.slider("가져올 인용 논문(Citations) 최대 개수", 10, 100, 50, step=10)

    if st.button("이 논문 분석"):
        try:
            with st.spinner("논문 기본 정보 가져오는 중..."):
                paper = get_paper_core(selected["paperId"])

            with st.spinner("선행 논문(References) 가져오는 중..."):
                refs_raw = get_paper_references(selected["paperId"], limit=ref_limit)

            with st.spinner("인용 논문(Citations) 가져오는 중..."):
                cites_raw = get_paper_citations(selected["paperId"], limit=cite_limit)

            st.session_state["paper_detail"] = paper
            st.session_state["refs_raw"] = refs_raw
            st.session_state["cites_raw"] = cites_raw
        except requests.exceptions.HTTPError:
            st.error("API 요청 제한(429)에 걸렸습니다. 잠시 후 다시 시도해주세요.")
        except requests.exceptions.RequestException as e:
            st.error(f"네트워크 오류가 발생했습니다: {e}")

if "paper_detail" in st.session_state:
    paper = st.session_state["paper_detail"]
    refs_raw = st.session_state.get("refs_raw", [])
    cites_raw = st.session_state.get("cites_raw", [])

    st.header("Seed Paper")
    st.subheader(paper.get("title"))
    st.write(f"**Year:** {paper.get('year')}")
    st.write(f"**Venue:** {paper.get('venue')}")
    st.write(f"**Authors:** {authors_to_str(paper.get('authors', []), max_authors=10)}")
    st.write(f"**Citations:** {paper.get('citationCount')}")
    st.write(f"**References:** {paper.get('referenceCount')}")
    st.write(paper.get("abstract") or "No abstract available.")

    refs_df = paper_list_to_df(refs_raw, category="선행연구")
    cites_df = paper_list_to_df(cites_raw, category="후속연구")

    tab1, tab2, tab3, tab4 = st.tabs([
        "선행 논문 References",
        "최신 인용 논문 Citations",
        "Timeline",
        "Research Flow 초안",
    ])

    with tab1:
        st.subheader("이 논문이 인용한 선행 논문")
        if refs_df.empty:
            st.warning("References를 찾지 못했습니다.")
        else:
            show = refs_df.sort_values(
                ["citationCount", "year"], ascending=[False, False]
            ).head(20)
            st.dataframe(show[["year", "title", "authors", "venue", "citationCount", "url"]])

    with tab2:
        st.subheader("이 논문을 인용한 최신 논문")
        if cites_df.empty:
            st.warning("Citations를 찾지 못했습니다.")
        else:
            show = cites_df.sort_values(
                ["year", "citationCount"], ascending=[False, False]
            ).head(20)
            st.dataframe(show[["year", "title", "authors", "venue", "citationCount", "url"]])

    with tab3:
        st.subheader("연도별 흐름 (선행연구 vs 후속연구)")
        combined = pd.concat([refs_df, cites_df], ignore_index=True)
        timeline = make_timeline(combined)
        if timeline.empty:
            st.warning("Timeline을 만들 수 없습니다.")
        else:
            st.dataframe(timeline)
            chart_data = timeline.pivot_table(
                index="year", columns="category", values="paper_count", fill_value=0
            )
            st.bar_chart(chart_data)

    with tab4:
        st.markdown(simple_research_flow(paper, refs_df, cites_df))