# Controlled native Chrome checklist (not yet executed)

The automated integration uses the popup's actual client code in Node, a real loopback API and mocked DOM rendering. It does not exercise Chrome extension loading, activeTab grants, native CORS or local-network policies.

1. Train and export a record with README commands. Load unpacked `extension/` in Chrome Developer mode and note its ID.
2. Start `python app.py --extension-id YOUR_ID`. In a second terminal run `python -m http.server 8000 --bind 127.0.0.1 --directory tests/fixtures`.
3. Visit `http://127.0.0.1:8000/controlled.html`. Open popup, click Observe. Expect Unable to evaluate, real observations, unavailable score and frozen offline threshold. No live model verdict is allowed.
4. Choose `artifacts/record.json`. Expect offline score, threshold and either suspicious or below threshold; changing the active URL must not change this record's prediction.
5. Stop API, repeat: expect Unable to evaluate. Restart and confirm recovery.
6. Try chrome://extensions and an invalid record JSON. Expect Unsupported page or request error.
7. Inspect popup DevTools network: requests only to http://127.0.0.1:5000, originating from extension popup; no content-script requests, arbitrary URL fetches or third-party services. Confirm Origin matches the exact configured ID and any needed preflight succeeds.
8. If a browser local-network prompt occurs, record Chrome version/policy and allow/deny results. Denied access must remain Unable to evaluate. Do not disable protections. Test on the target professor's browser separately before asserting compatibility.
9. Record Chrome version, OS, extension ID, backend command and actual outcomes in reports/VALIDATION.md. No live phishing site is needed.
