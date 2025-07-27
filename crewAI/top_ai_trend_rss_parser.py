from pydantic import BaseModel, HttpUrl, Field
from typing import List
import feedparser
import openai
import datetime
import subprocess
import asyncio
from crewai import Agent, Task, Crew

# ----- Config -----
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI key

RSS_FEEDS = [
    "https://openai.com/blog/rss.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://techcrunch.com/tag/ai/feed/",
    "https://spectrum.ieee.org/rss/topic/artificial-intelligence",
    "https://www.producthunt.com/feed",
    "https://towardsdatascience.com/feed",
    "https://huggingface.co/blog/rss.xml",
    "https://www.reddit.com/r/MachineLearning/.rss",
    "https://www.reddit.com/r/ArtificialInteligence/.rss",
    "https://ai.googleblog.com/feeds/posts/default",
    "https://www.microsoft.com/en-us/research/feed/",
    "https://www.deepmind.com/blog/rss.xml",
    "https://blog.ml.cmu.edu/rss/",
    "https://www.ibm.com/blogs/research/tag/artificial-intelligence/feed/",
    "https://feeds.feedburner.com/TheBaML",
    "https://blog.research.google/feeds/posts/default",
    "https://openai.substack.com/feed",
    "https://ai.facebook.com/blog/rss/",
    "https://www.getrevue.co/profile/importai",
    "https://www.nvidia.com/en-us/research/ai/rss/"
]

# ----- Data Model -----
class FeedEntry(BaseModel):
    title: str
    link: HttpUrl
    published: str = Field(default="")
    summary: str
    source: str

# ----- Agents and Tasks -----
class FeedFetcher:
    def run(self) -> List[FeedEntry]:
        entries = []
        for url in RSS_FEEDS:
            feed = feedparser.parse(url)
            for item in feed.entries:
                try:
                    entry = FeedEntry(
                        title=item.title,
                        link=item.link,
                        published=item.get("published", ""),
                        summary=item.get("summary", "")[:500],
                        source=feed.feed.get("title", "Unknown")
                    )
                    entries.append(entry)
                except Exception:
                    continue
        return entries

class OpenAICurator:
    def run(self, entries: List[FeedEntry]) -> List[FeedEntry]:
        content = "\n".join(f"{i+1}. {e.title} - {e.summary}" for i, e in enumerate(entries[:100]))
        messages = [
            {"role": "system", "content": "You're an expert AI news curator."},
            {"role": "user", "content": f"Select the 50 most important or innovative AI updates from below:\n{content}"}
        ]
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        output = response.choices[0].message.content
        top_titles = [line.split(" - ")[0].strip(".1234567890 ") for line in output.split("\n") if line.strip()]
        return [e for e in entries if any(e.title.startswith(title) for title in top_titles)]

class OllamaCurator:
    def run(self, entries: List[FeedEntry]) -> List[FeedEntry]:
        prompt = "Curate the 10 most insightful AI feed entries from the following list:\n"
        for i, e in enumerate(entries):
            prompt += f"{i+1}. {e.title} - {e.summary}\n"
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output = result.stdout.decode()
        top_titles = [line.strip().split(" - ")[0] for line in output.split("\n") if line.strip()]
        return [e for e in entries if any(e.title.startswith(title) for title in top_titles)]

class SummaryReporter:
    def run(self, entries: List[FeedEntry]) -> str:
        report = "# 🧠 Top AI Trends Summary\n\n"
        report += f"🗓️ Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
        for i, e in enumerate(entries, 1):
            report += f"### {i}. {e.title}\n"
            report += f"**Source**: {e.source}  \n"
            report += f"**Published**: {e.published}  \n"
            report += f"**Summary**: {e.summary.strip()}  \n"
            report += f"[🔗 Read more]({e.link})\n\n"
        with open("ai_trends_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        return "✅ Report saved to `ai_trends_report.md`"

# ----- Async CrewAI Pipeline -----
async def main():
    fetcher = Agent(name="Fetcher", role="RSS fetcher", goal="Fetch AI news entries", backstory="Gathers news feeds")
    curator = Agent(name="OpenAI Curator", role="Curate top 50", goal="Filter best updates using GPT-4", backstory="Uses OpenAI to choose top stories")
    reducer = Agent(name="Ollama Curator", role="Final 10", goal="Refine to 10 with Ollama", backstory="Local LLM insight")
    reporter = Agent(name="Reporter", role="Generate Report", goal="Summarize in Markdown", backstory="Writes clean summaries")

    task1 = Task(agent=fetcher, async_execution=False, expected_output="RSS entries")
    task2 = Task(agent=curator, async_execution=False, expected_output="Top 50 entries")
    task3 = Task(agent=reducer, async_execution=False, expected_output="Top 10 entries")
    task4 = Task(agent=reporter, async_execution=False, expected_output="Markdown summary")

    crew = Crew(agents=[fetcher, curator, reducer, reporter], tasks=[task1, task2, task3, task4])

    # Step-by-step with return sharing
    raw_entries = FeedFetcher().run()
    top_50 = OpenAICurator().run(raw_entries)
    top_10 = OllamaCurator().run(top_50)
    report = SummaryReporter().run(top_10)
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
