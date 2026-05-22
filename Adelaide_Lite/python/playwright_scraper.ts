import { parseArgs } from "jsr:@std/cli/parse-args";
import { addExtra } from "npm:playwright-extra";
import * as playwright from "npm:playwright";
import stealth from "npm:puppeteer-extra-plugin-stealth";

// Install stealth plugin
const chromium = addExtra(playwright.chromium);
chromium.use(stealth());



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
    string: ["engines", "num", "timeout"],
    default: { num: "5", timeout: "60", engines: "all" }
  });

  const query = args._[0] as string;
  if (!query) {
    console.error("Usage: deno run -A python/playwright_scraper.ts <query> [--engines=ddg,google,scholar] [--num=5]");
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
      allFlat.push(r);
    });
  }

  // Visit pages to get deeper snippets
  for (const r of allFlat) {
    await capturePageData(page, r);
  }

  await browser.close();

  // Print plain JSON to standard output so Python can parse it
  console.log(JSON.stringify(allFlat));
}

main().catch((e) => {
  console.error("Scraper Error:", e);
  Deno.exit(1);
});
