"""Crawl Kiswahili websites and build an offline training corpus for Ollama.

The corpus is stored in SQLite so the bot stays fully offline after the
initial crawl.  Each entry pairs a Swahili question/topic with a natural
answer, which the engine injects as few-shot examples.

Usage (standalone):
    python -m bot.swahili_corpus --db data/conversations.db --limit 200

Or from the CLI:
    python -m bot.main crawl-swahili
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Sources ──────────────────────────────────────────────────────────────
# Each source has a URL, an extractor that pulls (question, answer) pairs,
# and a max-page budget so we don't hammer sites.

SWAHILI_SOURCES: list[dict[str, Any]] = [
    {
        "name": "wikipedia_sw_faq",
        "url": "https://sw.wikipedia.org/wiki/Mwanzo",
        "type": "wikipedia",
        "max_pages": 30,
    },
    {
        "name": "jamii_forums",
        "url": "https://www.jamboforums.com",
        "type": "jambo",
        "max_pages": 15,
    },
    {
        "name": "tanzania_news_sw",
        "url": "https://www.mwananchi.co.tz",
        "type": "news",
        "max_pages": 10,
    },
    {
        "name": "swahili_dict",
        "url": "https://www.swahilidictionary.org",
        "type": "dictionary",
        "max_pages": 20,
    },
    {
        "name": "taifa_leo",
        "url": "https://www.taifaleo.nation.africa",
        "type": "news",
        "max_pages": 10,
    },
]

# ── Built-in Swahili Q&A pairs (always available offline) ────────────────
# Curated by hand so the bot has a solid baseline even without crawling.

BUILTIN_SWAHILI_QA: list[tuple[str, str]] = [
    # Greetings & social
    ("Habari yako", "Salama, habari yako? Niko poa."),
    ("Mambo vipi", "Mambo vipi! Umeshindaje?"),
    ("Vipi", "Poa, wewe uko aje?"),
    ("Umeshindaje", "Niko poa, asante! Wewe je?"),
    ("Umeamkaje", "Nimeamka salama. Wewe je?"),
    ("Unaendeleaje", " Nzuri, habari yako?"),
    ("Hujambo", "Poa, salama!"),
    ("Shikamoo", "Marahaba! Habari yako?"),
    ("Salama", "Poa kabisa, wewe uko aje?"),
    ("Habari za asubuhi", "Salama, habari za asubuhi! Umeamkaje?"),
    ("Habari za mchana", "Poa, habari za mchana! Kazi ikoje?"),
    ("Habari za jioni", "Salama, habari za jioni! Uko poa?"),
    # Questions
    ("Jina lako ni nini", "Jina langu ni Steve. Wewe unaitwa nani?"),
    ("Unafanya kazi gani", "Niko hapa nikisaidia watu. Wewe unafanya nini?"),
    ("Uko wapi", "Niko hapa tu. Wewe uko wapi?"),
    ("Una umri gani", "Sijana tu! Wewe una umri gani?"),
    ("Upo single", "Sijui hilo bado. Wewe upo single?"),
    ("Una mpenzi", "Bado sijajua. Wewe una mpenzi?"),
    # Help / requests
    ("Naomba unifungulie mlango", "Sawa kijana, nakuja kukufungulia."),
    ("Nisaidie tafadhali", "Sawa, nieleze unahitaji nini hasa?"),
    ("Nipe pesa", "Sawa, lakini hivi sasa sina. Pole sana."),
    ("Niletee chakula", "Sawa, nitakuletea. Subiri kidogo."),
    ("Nitumie hela", "Pole, sasa hivi sina pesa za kutosha."),
    ("Nifungulie", "Sawa, nakuja sasa hivi."),
    # Factual / educational
    ("Biology ni nini", "Biology ni somo la kuangalia vitu vilivyo hai — mimea, wanyama, jinsi maisha yanavyofanya kazi."),
    ("Chemistry ni nini", "Chemistry ni somo la viungo na madawa — jinsi vitu vinavyochanganyika na kubadilika."),
    ("Physics ni nini", "Physics ni somo la nguvu, mwendo, na nishati — jinsi ulimwengu unavyofanya kazi."),
    ("History ni nini", "Historia ni somo la mambo ya zamani — watu, mataifa, na matukio yaliyopita."),
    ("Geography ni nini", "Jiografia ni somo la nchi, mazingira, na tabianchi duniani."),
    ("Maths ni nini", "Hisabati ni somo la nambari, aljebra, na hesabu — msingi wa sayansi na biashara."),
    ("Kiswahili ni nini", "Kiswahili ni lugha inayozungumzwa na watu wengi Afrika Mashariki — Tanzania, Kenya, na sehemu nyingine."),
    ("English ni nini", "Kiingereza ni lugha inayozungumzwa na watu wengi duniani — inatumika katika biashara na elimu."),
    ("Ndoa ni nini", "Ndoa ni wakati watu wawili wanaahidi kukaa pamoja kwa muda mrefu — mara nyingi na sherehe na usajili wa kisheria."),
    ("Malaria ni nini", "Malaria ni ugonjwa unaosababishwa na mbu — husababisha homa, baridi, na uchovu."),
    ("HIV ni nini", "Vиру vya HIV ni virusi vinavyoharibu mfumo wa kinga wa mwili — kinazuilika na kinga na matibabu."),
    ("Corona ni nini", "COVID-19 ni ugonjwa unaosababishwa na virusi vya corona — unaweza kuzuilika kwa kuvaa barakoa na kujikinga."),
    # Emotional support
    ("Nimechoka leo", "Pole, siku ndefu? Pumzika kidogo."),
    ("Nimekata tamaa", "Pole sana. Kila mtu anapitia wakati mgumu. Niko hapa."),
    ("Nina wasiwasi", "Pole. Wasiwasi ni wa kawaida. Unataka kuongea kuhusu hilo?"),
    ("Nina furaha leo", "Hongera! Nini kimetokea? Nieleze zaidi!"),
    ("Nimesikitika", "Pole sana. Kuna nini? Unataka kuongea?"),
    ("Nina stress", "Pole. Stress ni ngumu. Nini kinasababisha?"),
    ("Nafikiri sana", "Pole. Mawazo mengi ni vigumu. Unataka kuzungumza?"),
    ("Nina uchovu", "Pole. Pumzika kidogo, mtu anahitaji pumziko."),
    ("Nimependwa", "Hongera! That's beautiful. Mwenye furaha!"),
    ("Nimevunjika moyo", "Pole sana. Moyo unavunjika ni ngumu. Niko hapa kukusaidia."),
    # Weather & daily life
    ("Hali ya hewa leo", "Hali ya hewa leo ni nzuri — jua nzuri na joto la wastani."),
    ("Joto leo ni", "Joto leo ni la wastani. Vaa nguo nyepesi."),
    ("Mvua inakaribia", "Mvua inaweza kuja, beba mwavuli."),
    ("Baridi leo", "Baridi leo? Vaa nguo nzuri na joto."),
    # Food & daily
    ("Umeshindaje kula", "Nimekula vizuri. Wewe umekula?"),
    ("Chakula kizuri", "Chakula kizuri ni kile kinachotoa nguvu mwilini — mchemsho, mboga, na matunda."),
    ("Niko njaa", "Pole! Kula kitu kidogo upate nguvu."),
    ("Maji tafadhali", "Maji ni muhimu kwa afya — kila siku kunua maji ya kutosha."),
    # Culture & society
    ("Utamaduni wa Tanzania", "Utamaduni wa Tanzania ni tajiri — lugha, ngoma, mavazi, na chakula vyote ni vya kipekee."),
    ("Utamaduni wa Kenya", "Utamaduni wa Kenya ni mbalimbali — kila kabila lina desturi zake."),
    ("Sherehe za Krismasi", "Krismasi ni wakati wa furaha — watu hukusanyika, kula chakula, na kutoa zawadi."),
    ("Sikukuu ya Eid", "Eid ni sikukuu ya Waislamu — wakati wa furaha, kula chakula, na kuwasalimia wengine."),
    # Technology
    ("Simu ni nini", "Simu ni kifaa cha mawasiliano — kinatumika kupiga simu, kutuma ujumbe, na kuunganika na mtandao."),
    ("Internet ni nini", "Internet ni mtandao wa kimataifa — unaunganisha watu na taifa lote duniani."),
    ("WhatsApp ni nini", "WhatsApp ni programu ya mawasiliano — inatumika kutuma ujumbe, picha, na sauti bure."),
]

# ── SQLite schema for corpus ─────────────────────────────────────────────

CORPUS_SCHEMA = """
CREATE TABLE IF NOT EXISTS swahili_corpus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    quality_score REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corpus_category
    ON swahili_corpus(category, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_corpus_source
    ON swahili_corpus(source, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_unique
    ON swahili_corpus(question, answer);
"""


def init_corpus_db(db_path: Path) -> None:
    """Create the swahili_corpus table if it doesn't exist."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(CORPUS_SCHEMA)
    conn.commit()
    conn.close()
    logger.info("Swahili corpus table initialized at %s", db_path)


def seed_builtin_swahili(db_path: Path) -> int:
    """Insert hand-curated Swahili Q&A pairs. Returns number inserted."""
    init_corpus_db(db_path)
    conn = sqlite3.connect(str(db_path))
    now = time.time()
    inserted = 0
    for question, answer in BUILTIN_SWAHILI_QA:
        try:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO swahili_corpus
                    (source, question, answer, category, quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("builtin", question, answer, _classify_qa(question, answer), 1.0, now),
            )
            if cur.rowcount > 0:
                inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    logger.info("Seeded %d built-in Swahili Q&A pairs", inserted)
    return inserted


def _classify_qa(question: str, answer: str) -> str:
    """Auto-classify a Q&A pair into a category."""
    combined = f"{question} {answer}".lower()
    categories = {
        "greeting": ("habari", "mambo", "vipi", "hujambo", "umeshindaje", "umeamkaje", "shikamoo", "salama"),
        "help": ("naomba", "nisaidie", "saidia", "nifungulie", "nipe", "niletee", "nitumie"),
        "factual": ("ni nini", "ni somo", "maana", "definition", "meaning"),
        "emotion": ("nimechoka", "nimesikitika", "nina wasiwasi", "nimefurahi", "stress", "uchovu", "moyo"),
        "food": ("chakula", "kula", "njaa", "maji", "kunywa"),
        "weather": ("hali ya hewa", "joto", "mvua", "baridi", "jua"),
        "culture": ("utamaduni", "sherehe", "krismasi", "eid", "desturi"),
        "technology": ("simu", "internet", "whatsapp", "computer", "teknolojia"),
    }
    for cat, markers in categories.items():
        if any(m in combined for m in markers):
            return cat
    return "general"


# ── Web Crawler ──────────────────────────────────────────────────────────

def crawl_swahili_sources(
    db_path: Path,
    *,
    sources: list[dict[str, Any]] | None = None,
    max_total: int = 200,
) -> int:
    """Crawl Swahili websites and store Q&A pairs in the corpus DB.

    Returns the number of new entries inserted.
    """
    init_corpus_db(db_path)
    sources = sources or SWAHILI_SOURCES
    total_inserted = 0

    for source in sources:
        if total_inserted >= max_total:
            break
        remaining = max_total - total_inserted
        try:
            inserted = _crawl_source(db_path, source, max_entries=remaining)
            total_inserted += inserted
            logger.info(
                "Crawled %s: %d entries (total: %d)",
                source["name"],
                inserted,
                total_inserted,
            )
        except Exception:
            logger.exception("Failed to crawl source: %s", source["name"])
        time.sleep(1.0)  # Be polite

    return total_inserted


def _crawl_source(
    db_path: Path,
    source: dict[str, Any],
    *,
    max_entries: int = 100,
) -> int:
    """Crawl a single source and insert entries into the DB."""
    name = source["name"]
    url = source["url"]
    source_type = source["type"]
    max_pages = min(source.get("max_pages", 10), max_entries)

    logger.info("Starting crawl of %s (%s)", name, url)

    if source_type == "wikipedia":
        return _crawl_wikipedia_sw(db_path, name, url, max_pages=max_entries)
    elif source_type == "dictionary":
        return _crawl_dictionary(db_path, name, url, max_pages=max_entries)
    elif source_type == "news":
        return _crawl_news(db_path, name, url, max_pages=max_entries)
    else:
        logger.warning("Unknown source type: %s for %s", source_type, name)
        return 0


def _crawl_wikipedia_sw(
    db_path: Path,
    source_name: str,
    base_url: str,
    *,
    max_entries: int = 100,
) -> int:
    """Crawl Swahili Wikipedia for articles and extract Q&A pairs."""
    conn = sqlite3.connect(str(db_path))
    now = time.time()
    inserted = 0

    # Get list of Swahili Wikipedia articles
    api_url = "https://sw.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Kategoria:Mada_za_Wikipedia",
        "cmlimit": str(min(max_entries, 50)),
        "format": "json",
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            # First, get some article titles from the main category
            resp = client.get(api_url, params=params)
            if resp.status_code != 200:
                # Fallback: try to get random articles
                params = {
                    "action": "query",
                    "list": "random",
                    "rnnamespace": "0",
                    "rnlimit": str(min(max_entries, 50)),
                    "format": "json",
                }
                resp = client.get(api_url, params=params)

            if resp.status_code != 200:
                logger.warning("Wikipedia API returned %d", resp.status_code)
                conn.close()
                return 0

            data = resp.json()
            pages = data.get("query", {}).get("categorymembers", []) or data.get("query", {}).get("random", [])

            for page in pages:
                if inserted >= max_entries:
                    break
                title = page.get("title", "")
                if not title:
                    continue

                # Fetch article content
                article = _fetch_wikipedia_article(client, title)
                if article:
                    qa_pairs = _extract_wikipedia_qa(article["title"], article["extract"])
                    for question, answer in qa_pairs:
                        try:
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO swahili_corpus
                                    (source, question, answer, category, quality_score, created_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (source_name, question, answer, _classify_qa(question, answer), 0.9, now),
                            )
                            inserted += 1
                        except sqlite3.IntegrityError:
                            pass

    except Exception:
        logger.exception("Wikipedia crawl failed for %s", source_name)

    conn.commit()
    conn.close()
    return inserted


def _fetch_wikipedia_article(client: httpx.Client, title: str) -> dict | None:
    """Fetch a single Wikipedia article's extract."""
    try:
        resp = client.get(
            "https://sw.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "exintro": "true",
                "explaintext": "true",
                "format": "json",
            },
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "extract" in page:
                return {"title": page.get("title", title), "extract": page["extract"]}
    except Exception:
        pass
    return None


def _extract_wikipedia_qa(title: str, extract: str) -> list[tuple[str, str]]:
    """Extract Q&A pairs from a Wikipedia article extract."""
    pairs: list[tuple[str, str]] = []
    if not extract or len(extract) < 30:
        return pairs

    # Clean the extract
    clean = re.sub(r"\[.*?\]", "", extract)  # Remove references
    clean = re.sub(r"\s+", " ", clean).strip()

    # Generate a question from the title
    question_sw = f"{title} ni nini"
    answer = clean[:300] if len(clean) > 300 else clean
    if answer and not answer.endswith("."):
        answer += "."
    pairs.append((question_sw, answer))

    # Split into sentences and create additional Q&A pairs
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    if len(sentences) >= 3:
        # Create a "tell me about" style question
        summary = " ".join(sentences[:3])
        if len(summary) > 300:
            summary = summary[:297] + "..."
        pairs.append((f"Nieleze kuhusu {title.lower()}", summary))

    return pairs


def _crawl_dictionary(
    db_path: Path,
    source_name: str,
    base_url: str,
    *,
    max_entries: int = 100,
) -> int:
    """Crawl a Swahili dictionary for word definitions."""
    conn = sqlite3.connect(str(db_path))
    now = time.time()
    inserted = 0

    # Common Swahili words to look up
    common_words = [
        "amani", "furaha", "upendo", "busara", "imani", "heshima",
        "tumaini", "nguvu", "hekima", "ujuzi", "utu", "haki",
        "amani", "utulivu", "mwangaza", "maarifa", "ushindi",
        "ushujaa", "uwazi", "ukarimu", "ukweli", "urafiki",
        "uzuri", "uthabiti", "uaminifu", "ubunifu", "ulimwengu",
        "maisha", "kazi", "elimu", "afya", "salamu",
        "miji", "vijiji", "mashamba", "mikoko", "milima",
        "mito", "bahr", "ziwa", "joto", "baridi",
    ]

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            for word in common_words:
                if inserted >= max_entries:
                    break
                try:
                    # Try to get word definition
                    resp = client.get(
                        f"{base_url}/define/{word}",
                        headers={"User-Agent": "SwahiliBot/1.0 (educational crawler)"},
                    )
                    if resp.status_code == 200:
                        text = resp.text
                        # Extract definition using simple regex
                        definition = _extract_definition(text, word)
                        if definition:
                            question = f"Maana ya {word} ni nini?"
                            answer = f"'{word}' maana yake ni: {definition}"
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO swahili_corpus
                                    (source, question, answer, category, quality_score, created_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (source_name, question, answer, "factual", 0.8, now),
                            )
                            inserted += 1
                except Exception:
                    pass
                time.sleep(0.3)
    except Exception:
        logger.exception("Dictionary crawl failed")

    conn.commit()
    conn.close()
    return inserted


def _extract_definition(html: str, word: str) -> str | None:
    """Extract a definition from dictionary HTML content."""
    # Simple extraction - look for definition patterns
    patterns = [
        rf'class="def"[^>]*>([^<]+)<',
        rf'class="meaning"[^>]*>([^<]+)<',
        rf'<p[^>]*>([^<]*(?:{word})[^<]*)</p>',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _crawl_news(
    db_path: Path,
    source_name: str,
    base_url: str,
    *,
    max_entries: int = 100,
) -> int:
    """Crawl a Swahili news site for headlines and summaries."""
    conn = sqlite3.connect(str(db_path))
    now = time.time()
    inserted = 0

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(
                base_url,
                headers={"User-Agent": "SwahiliBot/1.0 (educational crawler)"},
            )
            if resp.status_code != 200:
                conn.close()
                return 0

            # Extract article titles and summaries from HTML
            articles = _extract_news_articles(resp.text)

            for title, summary in articles[:max_entries]:
                if not title or not summary:
                    continue
                # Generate Q&A from news
                question = f"Kuhusu {title.lower()[:50]}, unajua nini?"
                answer = summary[:300] if len(summary) > 300 else summary
                conn.execute(
                    """
                    INSERT OR IGNORE INTO swahili_corpus
                        (source, question, answer, category, quality_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (source_name, question, answer, "news", 0.7, now),
                )
                inserted += 1
    except Exception:
        logger.exception("News crawl failed")

    conn.commit()
    conn.close()
    return inserted


def _extract_news_articles(html: str) -> list[tuple[str, str]]:
    """Extract article titles and summaries from news HTML."""
    articles: list[tuple[str, str]] = []
    # Look for article headings and paragraphs
    title_pattern = re.compile(r"<h[1-4][^>]*>(.*?)</h[1-4]>", re.DOTALL)
    for match in title_pattern.finditer(html):
        title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if title and len(title) > 10:
            articles.append((title, ""))

    # Try to get summaries/paragraphs after each title
    p_pattern = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL)
    paragraphs = [re.sub(r"<[^>]+>", "", m.group(1)).strip() for m in p_pattern.finditer(html)]
    for i, (title, _) in enumerate(articles):
        if i < len(paragraphs) and paragraphs[i]:
            articles[i] = (title, paragraphs[i])

    return articles[:50]


# ── Corpus retrieval ─────────────────────────────────────────────────────

def search_corpus(
    db_path: Path,
    query: str,
    *,
    category: str = "",
    limit: int = 5,
) -> list[tuple[str, str, str]]:
    """Search the Swahili corpus for relevant Q&A pairs.

    Returns list of (question, answer, category).
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Tokenize the query
    tokens = set(re.findall(r"[a-z']+", query.lower()))
    if not tokens:
        conn.close()
        return []

    # Build search condition
    conditions = []
    params: list[Any] = []

    if category:
        conditions.append("category = ?")
        params.append(category)

    # FTS-like search using LIKE
    token_conditions = []
    for token in tokens:
        if len(token) >= 3:
            token_conditions.append("(question LIKE ? OR answer LIKE ?)")
            params.extend([f"%{token}%", f"%{token}%"])

    if token_conditions:
        conditions.append(f"({' OR '.join(token_conditions)})")

    where = " AND ".join(conditions) if conditions else "1=1"
    query_sql = f"""
        SELECT question, answer, category
        FROM swahili_corpus
        WHERE {where}
        ORDER BY quality_score DESC, created_at DESC
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(query_sql, params).fetchall()
    conn.close()
    return [(r["question"], r["answer"], r["category"]) for r in rows]


def corpus_stats(db_path: Path) -> dict[str, Any]:
    """Get statistics about the Swahili corpus."""
    if not db_path.exists():
        return {"total": 0, "by_category": {}, "by_source": {}}

    conn = sqlite3.connect(str(db_path))
    try:
        total = conn.execute("SELECT COUNT(*) FROM swahili_corpus").fetchone()[0]
        cats = conn.execute(
            "SELECT category, COUNT(*) as c FROM swahili_corpus GROUP BY category"
        ).fetchall()
        sources = conn.execute(
            "SELECT source, COUNT(*) as c FROM swahili_corpus GROUP BY source"
        ).fetchall()
        return {
            "total": total,
            "by_category": {r[0]: r[1] for r in cats},
            "by_source": {r[0]: r[1] for r in sources},
        }
    except sqlite3.OperationalError:
        return {"total": 0, "by_category": {}, "by_source": {}}
    finally:
        conn.close()


# ── Standalone entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Kiswahili training corpus")
    parser.add_argument("--db", default="data/conversations.db", help="Database path")
    parser.add_argument("--limit", type=int, default=200, help="Max entries to crawl")
    parser.add_argument("--seed-only", action="store_true", help="Only seed built-in pairs")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    db = Path(args.db)
    if args.stats:
        stats = corpus_stats(db)
        print(f"Total entries: {stats['total']}")
        print(f"By category: {stats['by_category']}")
        print(f"By source: {stats['by_source']}")
    elif args.seed_only:
        inserted = seed_builtin_swahili(db)
        print(f"Seeded {inserted} built-in Swahili Q&A pairs")
    else:
        # First seed built-ins
        seed_builtin_swahili(db)
        # Then crawl
        inserted = crawl_swahili_sources(db, max_total=args.limit)
        print(f"Crawled {inserted} additional entries")
        stats = corpus_stats(db)
        print(f"Total corpus: {stats['total']} entries")
