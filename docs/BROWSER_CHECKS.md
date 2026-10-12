# Native Chrome controlled-page checklist (not yet executed)

Node tests use the actual popup client plus a real loopback API, and mock tab/DOM behavior. They do not execute Chrome permissions or UI.

1. Train both models and export an offline record using README commands. Load unpacked extension in Chrome Developer mode and note ID.
2. Run `python app.py --extension-id YOUR_ID`. In another terminal run `python -m http.server 8000 --bind 127.0.0.1 --directory tests/fixtures`.
3. Visit `http://127.0.0.1:8000/controlled.html?check=1`. Click **Evaluate active page**. Expect Active URL, numeric score, live threshold .712132751457169 (for the recorded dataset/seed), captured URL and URL-only limitation. Either suspicious or below threshold is permissible; no site-safety guarantee.
4. Change the URL to `/controlled.html?check=123&extra=1`, click again and confirm the actual captured URL/request changed. The score may or may not change: equal predictions are possible.
5. Load artifacts/record.json separately. Expect explicitly Offline record and the offline threshold; changing the active page must not alter this record's score.
6. Stop the API and repeat both operations: expect Unable to evaluate. Restart to confirm recovery. Try chrome://extensions and a malformed record: expect an error.
7. Inspect popup DevTools: requests only to 127.0.0.1:5000, from extension context, no content-script network traffic. `/predict-url` request has actual URL and live-url-v1; response has 39 measurements. Confirm any preflight and exact configured Origin allowlist.
8. Record Chrome version/OS/policies and any Local Network Access allow/deny prompt behavior. Denied local access must stay Unable to evaluate; do not disable protections to claim compatibility.
9. Confirm all external strings are rendered as text. Record actual outcomes in reports/VALIDATION.md before asserting native compatibility.

No live phishing URL is needed. Local synthetic page contents are not training/evaluation data; live benchmark metrics came from the historical CSV.
