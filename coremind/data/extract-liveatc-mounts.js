// Paste this in your browser console on any LiveATC airport search page:
//   https://www.liveatc.net/search/?icao=KIAD
//   https://www.liveatc.net/search/?icao=KDCA
//   etc.
//
// It outputs a JSON array — copy it and append to data/browser_mounts.json.
// Run once per airport page. The scraper reads browser_mounts.json automatically.

(function extractFeeds() {
  const feeds = [];

  document.querySelectorAll('a[href*="hlisten.php"]').forEach(a => {
    const url = new URL(a.href, location.origin);
    const mount = url.searchParams.get('mount');
    const icaoParam = url.searchParams.get('icao') || '';
    if (!mount) return;

    // Grab the surrounding table row for label context
    const row = a.closest('tr');
    const label = row
      ? Array.from(row.querySelectorAll('td'))
          .map(td => td.innerText.trim())
          .filter(Boolean)
          .join(' | ')
      : a.innerText.trim();

    feeds.push({
      mount,
      icao: icaoParam.toUpperCase(),
      label,
    });
  });

  if (feeds.length === 0) {
    console.warn('No feeds found. Are you on a liveatc.net/search/?icao=XXXX page?');
    return;
  }

  const json = JSON.stringify(feeds, null, 2);
  console.log(json);

  if (navigator.clipboard) {
    navigator.clipboard.writeText(json)
      .then(() => console.log(`Copied ${feeds.length} feeds to clipboard.`))
      .catch(() => console.log('Clipboard copy failed — copy the output above manually.'));
  }

  return feeds;
})();
