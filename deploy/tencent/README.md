# Tencent homepage override

The production container keeps the CAT-Psych application image and mounts
`site/home.html` over `/app/web/home.html`. This makes the personal homepage a
small, independently deployable file while `/cat-psych` and its persistent
JSON session data remain unchanged. The production image currently supports
the `memory` and `json` session backends; keep `CAT_PSYCH_SESSION_BACKEND=json`
until the image and its session store are deliberately upgraded together.

Before replacing the production Compose file, back up both the existing file
and the currently served `home.html`. Always run `docker compose config` before
recreating the `app` service.
