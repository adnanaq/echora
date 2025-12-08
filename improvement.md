How can we improve efficiency?

The code is already quite efficient by running API calls in parallel. However, there are a few key areas where performance could
be further improved without compromising integrity:

a. Implement Throttled Parallelism for Jikan

- Problem: The biggest bottleneck is in \_fetch_jikan_complete. To avoid Jikan's strict rate limit (e.g., 3 requests/sec), the code
  currently fetches episode and character details sequentially. For a long series, this negates much of the benefit of
  parallelism.
- Improvement: Instead of full sequential execution, we can implement a throttled parallel fetcher. By using a tool like
  asyncio.Semaphore(3) combined with a 1-second delay, we can run up to three Jikan requests concurrently, fully saturating their
  rate limit without exceeding it. This would change the Jikan fetching from 1 + 1 + 1... seconds to max(1, 1, 1) seconds for
  every batch of three requests, providing a significant speedup on series with many episodes or characters.

b. Refactor the AniList Helper to be Natively Asynchronous

- Problem: The \_fetch_anilist method uses loop.run_in_executor. This runs the code in a separate thread, which is a heavy-handed
  way to handle potentially blocking code. It adds overhead from thread creation and context switching.
- Improvement: The ideal solution is to investigate the AniListEnrichmentHelper and refactor it to be purely and natively
  asynchronous. This would allow it to run directly on the main event loop without the need for a separate thread, reducing
  resource consumption and complexity.

c. Use a Faster JSON Library

- Problem: For very large API responses, the standard json library can become a minor CPU bottleneck during deserialization.
- Improvement: We could integrate a high-performance JSON library like orjson. It is significantly faster than the standard
  library and can provide a small but noticeable speed boost, especially when processing hundreds of megabytes of JSON data across
  all API calls.
