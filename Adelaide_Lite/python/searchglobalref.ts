import { parseArgs } from "jsr:@std/cli/parse-args";
import { addExtra } from "npm:playwright-extra";
import * as playwright from "npm:playwright";
import stealth from "npm:puppeteer-extra-plugin-stealth";

// Install stealth plugin
const chromium = addExtra(playwright.chromium);
chromium.use(stealth());

const OLLAMA_BASE_URL = Deno.env.get("OLLAMA_PROXY_URL") || "http://localhost:11435";
const OLLAMA_EMBED_ENDPOINT = `${OLLAMA_BASE_URL}/api/embed`;
const OLLAMA_MODEL = "qwen3-embedding:0.6b";

async function getEmbedding(text: string): Promise<number[] | null> {
  if (!text) return null;
  try {
    const res = await fetch(OLLAMA_EMBED_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: OLLAMA_MODEL, input: text }),
    });
    if (!res.ok) return null;
    const data = await res.json();
    if (data.embeddings && data.embeddings.length > 0) return data.embeddings[0];
    if (data.embedding) return data.embedding;
  } catch {
    return null;
  }
  return null;
}

function generateApa7Reference(title: string, url: string): string {
  const date = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  const cleanTitle = title.trim().replace(/\.$/, "");
  return `${cleanTitle}. (Fetched: ${date}). ${url}`;
}

async function searchDuckDuckGo(page: any, query: string, numResults: number) {
  const results = [];
  try {
    await page.goto(`https://duckduckgo.com/?q=${encodeURIComponent(query)}&ia=web`, { waitUntil: "domcontentloaded", timeout: 15000 });
    await page.waitForSelector("article[data-testid='result']", { timeout: 10000 });
    const blocks = await page.$$("article[data-testid='result']");
    for (const block of blocks) {
      if (results.length >= numResults) break;
      try {
        const titleEl = await block.$("a[data-testid='result-title-a']");
        const url = await titleEl.getAttribute("href");
        let snippet = "";
        try { snippet = await (await block.$("div[data-testid='result-snippet']")).innerText(); } catch {}
        if (url && titleEl) {
          results.push({ title: await titleEl.innerText(), url, snippet });
        }
      } catch {}
    }
  } catch (e) {
    console.error(`DDG Error: ${e.message}`);
  }
  return results;
}

async function searchGoogle(page: any, query: string, numResults: number) {
  const results = [];
  try {
    await page.goto(`https://www.google.com/search?q=${encodeURIComponent(query)}`, { waitUntil: "domcontentloaded", timeout: 15000 });
    await page.waitForSelector("div.g", { timeout: 10000 });
    const blocks = await page.$$("div.g");
    for (const block of blocks) {
      if (results.length >= numResults) break;
      try {
        const h3 = await block.$("h3");
        const a = await block.$("a");
        const url = await a.getAttribute("href");
        const text = await block.innerText();
        if (url && h3 && !url.includes("google.com")) {
          results.push({ title: await h3.innerText(), url, snippet: text.substring(0, 200) });
        }
      } catch {}
    }
  } catch (e) {
    console.error(`Google Error: ${e.message}`);
  }
  return results;
}

async function searchScholar(page: any, query: string, numResults: number) {
  const results = [];
  try {
    await page.goto(`https://scholar.google.com/scholar?q=${encodeURIComponent(query)}`, { waitUntil: "domcontentloaded", timeout: 15000 });
    await page.waitForSelector("div.gs_r.gs_or.gs_scl", { timeout: 10000 });
    const blocks = await page.$$("div.gs_r.gs_or.gs_scl");
    for (const block of blocks) {
      if (results.length >= numResults) break;
      try {
        const a = await block.$("h3.gs_rt a");
        const url = await a.getAttribute("href");
        const text = await block.innerText();
        if (url && a) {
          results.push({ title: await a.innerText(), url, snippet: text.substring(0, 200) });
        }
      } catch {}
    }
  } catch (e) {
    console.error(`Scholar Error: ${e.message}`);
  }
  return results;
}

async function capturePageData(page: any, result: any) {
  try {
    await page.goto(result.url, { waitUntil: "domcontentloaded", timeout: 20000 });
    
    // Snippet
    try {
      const ps = await page.$$eval("p", els => els.map(e => e.innerText).filter(t => t.length > 20).join(" "));
      result.snippet = ps.substring(0, 400) + (ps.length > 400 ? "..." : "");
    } catch {}

    // Screenshot
    try {
      const screenshot = await page.screenshot({ type: "jpeg", quality: 27, scale: "css" });
      const base64 = screenshot.toString('base64');
      result.screenshot_base64 = `data:image/jpeg;base64,${base64}`;
    } catch {}

    // Save PDF
    try {
      const title = result.title.replace(/[^a-zA-Z0-9.\-_ ]/g, "").trim().substring(0, 100);
      const home = Deno.env.get("HOME") || "/Users/albertstarfield";
      const pdfPath = `${home}/Downloads/${title}.pdf`;
      await page.pdf({ path: pdfPath, printBackground: true });
    } catch {}

  } catch (e) {
    console.error(`Error capturing ${result.url}: ${e.message}`);
  }
}

async function main() {
  const args = parseArgs(Deno.args, {
    boolean: ["jsonIO"],
    string: ["ollamaExternal", "ollamaHost", "engines", "num", "timeout", "pages"],
    default: { num: "5", timeout: "60", engines: "all", pages: "1" }
  });

  const query = args._[0] as string;
  if (!query) {
    console.error("Usage: deno run -A searchglobalref.ts <query>");
    Deno.exit(1);
  }

  const numResults = parseInt(args.num);
  let engines = args.engines.split(",");
  if (engines.includes("all") || args.engines === "all") engines = ["ddg", "google", "scholar"];

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  let allFlat = [];

  for (const eng of engines) {
    let res = [];
    if (eng === "ddg") res = await searchDuckDuckGo(page, query, numResults);
    else if (eng === "google") res = await searchGoogle(page, query, numResults);
    else if (eng === "scholar") res = await searchScholar(page, query, numResults);

    res.forEach(r => {
      r.source_engine = eng;
      r.apa7_reference = generateApa7Reference(r.title, r.url);
      allFlat.push(r);
    });
  }

  // Visit pages
  for (const r of allFlat) {
    await capturePageData(page, r);
  }

  await browser.close();

  // Semantic ranking
  const qEmb = await getEmbedding(query);
  let finalResults = allFlat;

  if (qEmb && allFlat.length > 0) {
    const ranked = [];
    for (const r of allFlat) {
      const textToEmbed = `${r.title} ${r.snippet || ""}`;
      const rEmb = await getEmbedding(textToEmbed);
      let score = 0;
      if (rEmb) {
        let dot = 0, norm1 = 0, norm2 = 0;
        for (let i = 0; i < qEmb.length; i++) {
          dot += qEmb[i] * rEmb[i];
          norm1 += qEmb[i] * qEmb[i];
          norm2 += rEmb[i] * rEmb[i];
        }
        score = dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
      }
      ranked.push({ score, r });
    }
    ranked.sort((a, b) => b.score - a.score);
    finalResults = ranked.slice(0, 7).map((x, i) => {
      x.r.semantic_rank = i + 1;
      x.r.semantic_score = x.score;
      return x.r;
    });
  } else {
    finalResults = allFlat.slice(0, 7);
  }

  if (args.jsonIO) {
    console.log(JSON.stringify({ phase: 2, status: "complete", results: finalResults }));
  } else {
    console.log("# Global Search Results");
    console.log(`*Query: ${query}*\n`);
    for (let i = 0; i < finalResults.length; i++) {
      const r = finalResults[i];
      console.log(`## ${i + 1}. ${r.title}`);
      console.log(`- **URL:** ${r.url}`);
      console.log(`- **Engine:** ${r.source_engine || "unknown"}`);
      if (r.semantic_rank) console.log(`- **Semantic Rank:** ${r.semantic_rank}`);
      console.log(`- **Reference:** ${r.apa7_reference}\n`);
      console.log(`### Snippet\n${r.snippet || "No snippet available."}\n`);
      if (r.screenshot_base64) console.log(`### Visual Evidence (Page Snapshot)\n![Page Snapshot](${r.screenshot_base64})\n`);
      console.log(`---\n`);
    }
  }

  // Fire off memorythoughts.py
  for (const r of finalResults) {
    const mem = `Source: ${r.url}\nReference: ${r.apa7_reference}\nSnippet: ${r.snippet || ""}`;
    const cmd = new Deno.Command("python3", {
      args: ["python/memorythoughts.py", "--string", mem]
    });
    cmd.spawn();
  }
}

main().catch(console.error);
