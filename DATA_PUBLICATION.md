# Product data publication

Product search uses the explicit `PUBLIC_ITEM_FIELDS` list in `app.py`. Only
those columns enter the search index or product responses, including suggestions,
comparisons and router results. Newly added source columns are excluded by default.
The list retains the existing customer catalogue names, descriptions, specifications
and document links. Review values as well as column names before publishing an export:
an allowlisted description can still contain inappropriate text.

Catalogue administration metadata, raw ERP fields and competitor cross-references
are excluded. If competitor lookup is an intended customer feature, approve the
specific mappings and add their fields explicitly; do not restore automatic exposure
of every source column. ERP pack quantities and units may be useful to customers,
but should be checked and deliberately approved before adding them. Existing field
names and values are otherwise preserved.

## Deployment choice

Set one of the following before deploying this change:

* For a deliberately public catalogue API, set `ALLOW_PUBLIC_API=true` and leave
  `API_KEY` unset.
* For a restricted integration, set a non-empty `API_KEY` in the hosting platform's
  secret settings and send it in the `X-API-Key` header over HTTPS. A configured key
  takes precedence even when `ALLOW_PUBLIC_API=true`.

With neither setting, product endpoints return 401. This intentionally changes the
old unauthenticated default. `/healthz`, `/help` and the standalone `/calc` remain
public; they do not return source product rows. Health errors no longer include
source paths or parser messages. Do not put the API key in client-side JavaScript.
Wildcard CORS is retained for public catalogue clients; CORS is not authentication.

## Repository data is a separate publication channel

This code change does not sanitize the committed CSVs, workbooks, JSON or Git
history. An API key cannot restrict files in a public repository. Before treating
the repository as a public data source, replace both `Items.csv` and
`data/Items.csv` with reviewed public exports, or move raw source data to private
storage and deploy it separately using `ITEMS_CSV_PATH`. Remove redundant exports
and unused workbooks/ranking files from the public tree when no consumer needs them.
Review archived workbook sheets too.

Deleting or replacing the current files does not remove earlier commits. If the
owner decides that historical content must be removed, plan history cleanup and
hosting/cache remediation separately. Do not force-push a rewritten history as part
of this application patch. Previously downloaded copies cannot be recalled.

Run the focused regression checks with:

```sh
python -m unittest discover -s tests -v
```
