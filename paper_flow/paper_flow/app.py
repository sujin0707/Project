import os
import time
import json
import requests
import streamlit as st
import pandas as pd

try:
    import anthropic
except ImportError:
    anthropic = None

S2_API_KEY = os.getenv("S2_API_KEY", "s2k-3MWrBBdjX8r0koM2UWR0truwwFRlUNIq9kMLq7IW")
S2_BASE = "https://api.semanticscholar.org/graph/v1"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# 모델 선택지: Sonnet은 품질/속도 균형, Haiku는 저렴하고 빠름
LLM_MODEL_OPTIONS = {
    "Claude Sonnet 5 (권장, 품질과 속도 균형)": "claude-sonnet-5",
    "Claude Haiku 4.5 (저렴하고 빠름)": "claude-haiku-4-5-20251001",
}


def s2_headers():
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    return headers


_last_request_time = [0.0]
MIN_REQUEST_INTERVAL = 1.05  # 초당 1회 제한 대비 여유를 둔 최소 간격(초)


def _throttle():
    """Semantic Scholar의 '초당 1요청' 제한을 지키기 위해 마지막 요청 이후
    최소 MIN_REQUEST_INTERVAL초가 지나지 않았으면 그만큼 대기한다."""
    now = time.monotonic()
    elapsed = now - _last_request_time[0]
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time[0] = time.monotonic()


def _get_with_retry(url, params, timeout=20, max_retries=2):
    """Simple GET wrapper with basic retry on 429, and raises for other errors.
    초당 1요청 제한을 지키기 위해 매 요청 전 throttle을 건다."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            _throttle()
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


def get_anthropic_client():
    if anthropic is None:
        raise RuntimeError(
            "anthropic 패키지가 설치되어 있지 않습니다. 터미널에서 `pip install anthropic`을 실행해주세요."
        )
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY가 설정되어 있지 않습니다. .env 파일에 `ANTHROPIC_API_KEY=your_key`를 추가해주세요."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _extract_text(message):
    return "".join(block.text for block in message.content if block.type == "text").strip()


def _parse_json_response(text):
    """LLM이 코드블록으로 감싸서 응답하는 경우까지 안전하게 JSON 파싱."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def summarize_paper_with_llm(paper_title, paper_abstract, paper_year, seed_title, seed_abstract, model):
    """단일 논문 abstract를 문제/방법/기여/한계/seed와의 관계로 구조화 요약."""
    if not paper_abstract:
        return None

    client = get_anthropic_client()

    system_prompt = (
        "당신은 연구 동향을 분석하는 리서치 어시스턴트입니다. "
        "주어진 논문의 초록을 읽고 핵심 내용을 구조화하여 요약합니다. "
        "반드시 순수 JSON 객체만 응답하고, 다른 설명이나 마크다운 코드블록은 포함하지 마세요."
    )

    user_prompt = f"""[Seed Paper]
제목: {seed_title}
초록: {seed_abstract or "정보 없음"}

[비교 대상 논문]
제목: {paper_title}
연도: {paper_year}
초록: {paper_abstract}

아래 JSON 스키마 형식으로만 응답하세요 (모든 값은 한국어 1문장 내외):
{{
  "problem": "이 논문이 다루는 문제",
  "method": "이 논문이 사용한 방법/접근",
  "contribution": "이 논문의 주요 기여",
  "limitation": "이 논문의 한계 또는 미해결 과제",
  "relation_to_seed": "이 논문이 seed paper와 어떻게 연결되는지"
}}"""

    message = client.messages.create(
        model=model,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    parsed = _parse_json_response(_extract_text(message))
    return parsed


def summarize_papers_batch(papers_df, seed_title, seed_abstract, model, max_papers, progress_label):
    """상위 N개 논문에 대해 순차적으로 LLM 요약을 호출 (진행률 표시 + 속도 조절)."""
    subset = papers_df.dropna(subset=["abstract"]).head(max_papers)
    results = []

    if subset.empty:
        return results

    progress = st.progress(0.0, text=progress_label)
    total = len(subset)

    for i, (_, row) in enumerate(subset.iterrows()):
        try:
            summary = summarize_paper_with_llm(
                row["title"], row["abstract"], row["year"], seed_title, seed_abstract, model
            )
        except RuntimeError as e:
            progress.empty()
            raise e
        except Exception as e:
            st.warning(f"'{row['title']}' 요약 중 오류가 발생했습니다: {e}")
            summary = None

        results.append({"year": row["year"], "title": row["title"], "url": row["url"], "summary": summary})
        progress.progress((i + 1) / total, text=progress_label)
        time.sleep(0.3)  # 너무 빠르게 연속 호출하지 않도록 최소한의 페이싱

    progress.empty()
    return results


@st.cache_data(ttl=86400, show_spinner=False)
def synthesize_research_flow_with_llm(seed_title, seed_abstract, ref_summaries_json, cite_summaries_json, model):
    """개별 요약들을 모아 전체 연구 흐름 리포트를 LLM으로 종합."""
    client = get_anthropic_client()

    system_prompt = (
        "당신은 연구 동향을 종합하는 리서치 어시스턴트입니다. "
        "주어진 선행 연구 요약과 후속 연구 요약을 바탕으로, 이 연구 주제가 시간에 따라 "
        "어떻게 발전해왔는지 한국어로 서술하는 리포트를 마크다운으로 작성합니다."
    )

    user_prompt = f"""[Seed Paper]
제목: {seed_title}
초록: {seed_abstract or "정보 없음"}

[선행 연구 요약 (JSON)]
{ref_summaries_json}

[후속 연구 요약 (JSON)]
{cite_summaries_json}

다음 구조로 마크다운 리포트를 작성하세요:
1. **배경**: seed paper 이전에는 이 주제가 어떻게 다뤄졌는지
2. **Seed Paper의 위치**: seed paper가 해결하려 한 문제와 기여
3. **이후의 발전**: 후속 연구들이 개선한 부분과 새로운 방향
4. **종합 및 전망**: 이 연구 주제가 앞으로 어디로 향할 것으로 보이는지 간단한 전망

각 항목은 2~4문장 정도로 간결하게 작성하세요."""

    message = client.messages.create(
        model=model,
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return _extract_text(message)


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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "선행 논문 References",
        "최신 인용 논문 Citations",
        "Timeline",
        "Research Flow 초안",
        "AI 요약 (LLM)",
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

    with tab5:
        st.subheader("LLM으로 논문별 요약 + 전체 흐름 종합")
        st.caption(
            "각 논문의 abstract를 Claude에 넣어 `문제 - 방법 - 기여 - 한계 - seed paper와의 관계`로 "
            "구조화 요약한 뒤, 이를 모아 전체 연구 흐름 리포트를 생성합니다."
        )

        if anthropic is None:
            st.warning("`anthropic` 패키지가 설치되어 있지 않습니다. 터미널에서 `pip install anthropic`을 실행해주세요.")
        elif not ANTHROPIC_API_KEY:
            st.warning(
                "ANTHROPIC_API_KEY가 설정되어 있지 않습니다. `.env` 파일에 "
                "`ANTHROPIC_API_KEY=your_key`를 추가한 뒤 앱을 재시작해주세요."
            )
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                model_label = st.selectbox("사용할 모델", list(LLM_MODEL_OPTIONS.keys()))
                model = LLM_MODEL_OPTIONS[model_label]
            with col_b:
                n_summarize = st.slider(
                    "요약할 논문 개수 (선행/후속 각각, citation 순 상위)", 3, 10, 5
                )

            st.caption("⚠️ 논문 개수 × 2 (선행+후속) + 종합 리포트 1회만큼 API 호출이 발생하며 비용이 청구됩니다.")

            if st.button("AI 요약 생성", type="primary"):
                seed_title = paper.get("title", "")
                seed_abstract = paper.get("abstract", "")

                top_refs = refs_df.dropna(subset=["abstract"]).sort_values(
                    "citationCount", ascending=False
                )
                top_cites = cites_df.dropna(subset=["abstract"]).sort_values(
                    "citationCount", ascending=False
                )

                try:
                    ref_summaries = summarize_papers_batch(
                        top_refs, seed_title, seed_abstract, model, n_summarize,
                        "선행 논문 요약 생성 중..."
                    )
                    cite_summaries = summarize_papers_batch(
                        top_cites, seed_title, seed_abstract, model, n_summarize,
                        "후속 논문 요약 생성 중..."
                    )

                    st.session_state["llm_ref_summaries"] = ref_summaries
                    st.session_state["llm_cite_summaries"] = cite_summaries

                    with st.spinner("전체 연구 흐름 종합 중..."):
                        report = synthesize_research_flow_with_llm(
                            seed_title,
                            seed_abstract,
                            json.dumps(ref_summaries, ensure_ascii=False),
                            json.dumps(cite_summaries, ensure_ascii=False),
                            model,
                        )
                    st.session_state["llm_report"] = report

                except RuntimeError as e:
                    st.error(str(e))

            if "llm_report" in st.session_state:
                st.markdown("---")
                st.markdown("### 📄 종합 연구 흐름 리포트")
                st.markdown(st.session_state["llm_report"])

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 선행 논문 개별 요약")
                    for item in st.session_state.get("llm_ref_summaries", []):
                        s = item.get("summary")
                        with st.expander(f"{item.get('year')}: {item.get('title')}"):
                            if not s:
                                st.write("요약 실패 또는 초록 없음")
                            else:
                                st.write(f"**문제:** {s.get('problem', '')}")
                                st.write(f"**방법:** {s.get('method', '')}")
                                st.write(f"**기여:** {s.get('contribution', '')}")
                                st.write(f"**한계:** {s.get('limitation', '')}")
                                st.write(f"**Seed와의 관계:** {s.get('relation_to_seed', '')}")

                with col2:
                    st.markdown("#### 후속 논문 개별 요약")
                    for item in st.session_state.get("llm_cite_summaries", []):
                        s = item.get("summary")
                        with st.expander(f"{item.get('year')}: {item.get('title')}"):
                            if not s:
                                st.write("요약 실패 또는 초록 없음")
                            else:
                                st.write(f"**문제:** {s.get('problem', '')}")
                                st.write(f"**방법:** {s.get('method', '')}")
                                st.write(f"**기여:** {s.get('contribution', '')}")
                                st.write(f"**한계:** {s.get('limitation', '')}")
                                st.write(f"**Seed와의 관계:** {s.get('relation_to_seed', '')}")