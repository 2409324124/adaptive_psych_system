# Tencent homepage override

The production container keeps the CAT-Psych application image and mounts
`site/home.html` over `/app/web/home.html`, `site/home-assets` over
`/app/web/home-assets`, and the generated `site/robots.txt` and
`site/sitemap.xml` crawler files over their matching `/app/web` paths. This
keeps the React desktop homepage independently deployable while `/cat-psych`
and its persistent JSON session data remain unchanged. The production image
currently supports the `memory` and `json` session backends; keep
`CAT_PSYCH_SESSION_BACKEND=json` until the image and its session store are
deliberately upgraded together.

Before replacing the production Compose file, back up the existing file,
`site/home.html`, `site/home-assets`, `site/robots.txt`, and `site/sitemap.xml`
when present. Copy all four outputs from the same homepage build. Always run
`docker compose config` before recreating only the `app` service. The homepage
build appends content hashes to its CSS and JavaScript URLs so a release does
not depend on purging a previously cached static response.
