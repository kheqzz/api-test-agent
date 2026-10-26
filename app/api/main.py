"""Agent HTTP API.

Endpoints:
    GET  /                   — product landing page
    GET  /dashboard          — launch analyses, browse past runs
    POST /analyze            — start an analysis (spec file/URL + target URL)
    GET  /reports            — list past runs
    GET  /reports/{id}       — report as JSON
    GET  /reports/{id}/html  — report as a readable HTML page
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.api import storage
from app.api.home import render_dashboard, render_landing
from app.api.html_report import render_html, render_pending
from app.config import settings
from app.graph.workflow import run_workflow
from app.models.report import Report
from app.models.state import AgentState

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    storage.init_db()
    yield


app = FastAPI(
    title="API Test Agent",
    description=(
        "AI agent that reads an OpenAPI spec, generates realistic test "
        "scenarios with an LLM, executes them against a running API, and "
        "reports coverage and anomalies."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)

# Bundle the deliberately-buggy demo API in-process so a single deployed service
# is self-contained: its spec lives at /demo/openapi.json and its base URL is
# /demo. Disable with MOUNT_DEMO=0 (e.g. when running the demo as its own server).
if settings.mount_demo:
    from app.demo_api.main import app as demo_app

    app.mount("/demo", demo_app)


def _ensure_target_allowed(request: Request, url: str, field: str) -> None:
    """Guard against the public demo being used as an open proxy.

    When AGENT_RESTRICT_TARGETS is on, /analyze may only reach the app's own
    host (the bundled /demo API) or explicitly allowlisted hosts — otherwise
    anyone could point it at an arbitrary address or burn the Groq quota.
    Off by default, so local use stays unrestricted.
    """
    if not settings.restrict_targets:
        return
    host = (urlparse(url).hostname or "").lower()
    allowed = {
        (request.url.hostname or "").lower(),
        "127.0.0.1",
        "localhost",
        *settings.allowed_target_hosts,
    }
    if host not in allowed:
        raise HTTPException(
            403,
            f"{field} host {host!r} is not allowed on this public demo; "
            "it only analyzes the bundled demo API at /demo.",
        )


def _first(*values: str | None) -> str | None:
    """Return the first non-empty value among the candidates."""
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _extract_token(payload: Any, field: str) -> str | None:
    """Pull a JWT string out of a login response, supporting dotted paths."""
    cur: Any = payload
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            cur = None
            break
    if isinstance(cur, str) and cur.strip():
        return cur.strip()
    # Fall back to common token field names when the requested one is absent.
    if isinstance(payload, dict):
        for key in ("access_token", "token", "jwt", "jwt_token", "accessToken", "id_token"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for nested_key in ("data", "result", "response"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                found = _extract_token(nested, field)
                if found:
                    return found
    return None


def _call_token_endpoint(
    request: Request,
    token_url: str,
    login_token: str,
    field: str,
) -> str:
    """Call a JWT/token endpoint using the login token and return the JWT."""
    _ensure_target_allowed(request, token_url, "token_url")
    field = (field or "").strip() or "access_token"
    strategies: list[tuple[str, dict[str, str], dict[str, str] | None]] = [
        ("POST", {"Authorization": f"Bearer {login_token}"}, None),
        ("POST", {"Authorization": f"Bearer {login_token}"}, {"token": login_token}),
        ("POST", {"Authorization": f"Bearer {login_token}"}, {"access_token": login_token}),
        ("POST", {}, {"token": login_token}),
        ("POST", {}, {"access_token": login_token}),
        ("POST", {}, {"jwt": login_token}),
        ("GET", {"Authorization": f"Bearer {login_token}"}, None),
        ("GET", {}, None),
    ]
    last_error: str | None = None
    for method, headers, body in strategies:
        try:
            with httpx.Client(timeout=settings.request_timeout, follow_redirects=True) as client:
                if method == "POST":
                    if body is None:
                        response = client.post(token_url, headers=headers)
                    else:
                        response = client.post(token_url, headers=headers, json=body)
                else:
                    response = client.get(token_url, headers=headers)
        except httpx.HTTPError as exc:
            last_error = f"request failed: {exc}"
            continue
        if response.status_code >= 400:
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            continue
        # Try JSON first.
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            text = response.text.strip()
            if text:
                return text
            continue
        if isinstance(payload, str) and payload.strip():
            return payload.strip()
        token = _extract_token(payload, field)
        if token:
            return token
        # Some endpoints return the token under a non-standard key; try the whole
        # payload serialized? Fall back to scanning string values.
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, str) and value.strip() and value != login_token:
                    # heuristic: a JWT-like or longer token
                    if len(value.strip()) >= 8:
                        return value.strip()
        last_error = f"no token found in response: {str(payload)[:200]}"
    raise HTTPException(502, f"token endpoint failed: {last_error or 'unknown error'}")


def _login_for_token(
    request: Request,
    *,
    email: str | None,
    password: str | None,
    login_url: str | None,
    token_field: str,
    email_field: str,
    password_field: str,
    login_token_field: str = "token",
    token_url: str | None = None,
    jwt_token_field: str | None = None,
) -> dict[str, str]:
    """Call the configured login (+ optional JWT) endpoint and return an Authorization header."""
    if not (email and password):
        return {}
    # Backward compat: if no separate login_url was given but a url was, use it as login.
    if not login_url and token_url:
        login_url = token_url
        token_url = None
    if not login_url:
        return {}
    _ensure_target_allowed(request, login_url, "login_url")
    try:
        with httpx.Client(timeout=settings.request_timeout, follow_redirects=True) as client:
            response = client.post(
                login_url, json={email_field: email, password_field: password}
            )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"login request failed: {exc}") from exc
    if response.status_code >= 400:
        raise HTTPException(
            401 if response.status_code in (401, 403) else 422,
            f"login failed ({response.status_code}): {response.text[:300]}",
        )
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(502, f"login response is not JSON: {exc}") from exc
    intermediate_field = (login_token_field or "").strip() or "token"
    login_token = _extract_token(payload, intermediate_field)
    if not login_token:
        # Fall back to the configured token_field for single-step setups.
        login_token = _extract_token(payload, token_field)
    if not login_token:
        raise HTTPException(
            502,
            f"no token found under field {intermediate_field!r} in login response: "
            f"{str(payload)[:300]}",
        )

    # If a separate token/JWT endpoint is configured, exchange the login token for a JWT.
    if token_url:
        final_field = (jwt_token_field or "").strip() or (token_field or "").strip() or "access_token"
        jwt = _call_token_endpoint(request, token_url, login_token, final_field)
        return {"Authorization": f"Bearer {jwt}"}
    return {"Authorization": f"Bearer {login_token}"}


def _run_analysis(state: AgentState) -> None:
    """Background task: run the LangGraph workflow and persist the result."""
    try:
        final = run_workflow(state)
        report = final.report
        if report is None:  # defensive: both terminal nodes set a report
            raise RuntimeError("workflow finished without producing a report")
        storage.save_report(state.report_id, report)
    except Exception as exc:  # noqa: BLE001 - persist any crash for the client
        logger.exception("Analysis %s crashed", state.report_id)
        storage.mark_crashed(state.report_id, str(exc))


@app.get("/", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    """Product landing page."""
    return HTMLResponse(render_landing())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Dashboard: analysis launcher + list of past runs."""
    return HTMLResponse(render_dashboard(storage.list_reports()))


@app.post("/analyze", status_code=202)
def analyze(
    request: Request,
    background_tasks: BackgroundTasks,
    target_url: str = Form(..., description="Base URL of the running API under test"),
    spec_url: str | None = Form(None, description="URL of the OpenAPI spec"),
    spec_file: UploadFile | None = File(None, description="OpenAPI spec file (JSON/YAML)"),
    auth_headers: str | None = Form(
        None,
        description='Optional JSON object of headers, e.g. {"Authorization": "Bearer ..."}',
    ),
    # Login flow (optional): obtain a JWT by calling a login endpoint.
    email: str | None = Form(None, description="Login email/username"),
    password: str | None = Form(None, description="Login password"),
    username: str | None = Form(None, description="Alias for email"),
    auth_email: str | None = Form(None, description="Alias for email"),
    auth_password: str | None = Form(None, description="Alias for password"),
    login_url: str | None = Form(None, description="URL of the login (sign-in) endpoint"),
    login_endpoint: str | None = Form(None, description="Alias for login_url"),
    login_api: str | None = Form(None, description="Alias for login_url"),
    auth_url: str | None = Form(None, description="Alias for login_url"),
    sign_in_url: str | None = Form(None, description="Alias for login_url"),
    signin_url: str | None = Form(None, description="Alias for login_url"),
    auth_login_url: str | None = Form(None, description="Alias for login_url"),
    login_api_url: str | None = Form(None, description="Alias for login_url"),
    # Separate JWT/token endpoint (optional two-step auth).
    token_url: str | None = Form(None, description="URL of the JWT/token endpoint"),
    jwt_url: str | None = Form(None, description="Alias for token_url"),
    jwt_endpoint: str | None = Form(None, description="Alias for token_url"),
    token_endpoint: str | None = Form(None, description="Alias for token_url"),
    jwt_token_url: str | None = Form(None, description="Alias for token_url"),
    access_token_url: str | None = Form(None, description="Alias for token_url"),
    auth_token_url: str | None = Form(None, description="Alias for token_url"),
    jwt_api: str | None = Form(None, description="Alias for token_url"),
    token_api: str | None = Form(None, description="Alias for token_url"),
    # Token field in the login response.
    login_token_field: str | None = Form(None, description="Field in login response holding the intermediate token"),
    login_token: str | None = Form(None, description="Alias for login_token_field"),
    session_token_field: str | None = Form(None, description="Alias for login_token_field"),
    session_token: str | None = Form(None, description="Alias for login_token_field"),
    login_response_token_field: str | None = Form(None, description="Alias for login_token_field"),
    signin_token_field: str | None = Form(None, description="Alias for login_token_field"),
    # Token field in the JWT/token response (or single-step login response).
    token_field: str | None = Form(
        None, description="Response field holding the JWT (default access_token)"
    ),
    token_key: str | None = Form(None, description="Alias for token_field"),
    token_path: str | None = Form(None, description="Alias for token_field"),
    access_token_field: str | None = Form(None, description="Alias for token_field"),
    jwt_field: str | None = Form(None, description="Alias for token_field"),
    jwt_token_field: str | None = Form(None, description="Alias for token_field"),
    jwt_token: str | None = Form(None, description="Alias for token_field"),
    response_token_field: str | None = Form(None, description="Alias for token_field"),
    jwt_token_key: str | None = Form(None, description="Alias for token_field"),
    email_field: str | None = Form(None, description="Request body field name for email"),
    password_field: str | None = Form(None, description="Request body field name for password"),
) -> dict[str, str]:
    """Start an analysis run; returns a report id to poll."""
    # Browsers submit an empty file part when the input is left blank —
    # only treat the upload as a spec if it has a filename and content.
    spec_url = (spec_url or "").strip() or None
    raw_spec: str | None = None
    if spec_file is not None and (spec_file.filename or "").strip():
        content = spec_file.file.read().decode("utf-8", errors="replace")
        raw_spec = content if content.strip() else None
    if not spec_url and raw_spec is None:
        raise HTTPException(422, "provide either spec_url or spec_file")

    # Both target_url and spec_url are fetched server-side, so both are checked.
    _ensure_target_allowed(request, target_url, "target_url")
    if spec_url:
        _ensure_target_allowed(request, spec_url, "spec_url")

    parsed_headers: dict[str, str] = {}
    if auth_headers:
        try:
            loaded = json.loads(auth_headers)
            if not isinstance(loaded, dict):
                raise ValueError("must be a JSON object")
            parsed_headers = {str(k): str(v) for k, v in loaded.items()}
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(422, f"auth_headers must be a JSON object: {exc}") from exc

    # Resolve login-flow aliases.
    login_email = _first(email, auth_email, username)
    login_password = _first(password, auth_password)
    explicit_login_url = _first(
        login_url,
        login_endpoint,
        login_api,
        auth_url,
        sign_in_url,
        signin_url,
        auth_login_url,
        login_api_url,
    )
    jwt_endpoint_url = _first(
        token_url,
        jwt_url,
        jwt_endpoint,
        token_endpoint,
        jwt_token_url,
        access_token_url,
        auth_token_url,
        jwt_api,
        token_api,
    )
    # Backward compat: if no explicit login URL but a url was given, treat it as
    # the (single-step) login URL.
    if explicit_login_url:
        login_url_value = explicit_login_url
        token_url_value = jwt_endpoint_url
    else:
        login_url_value = jwt_endpoint_url
        token_url_value = None

    login_token_field_value = _first(
        login_token_field,
        login_token,
        session_token_field,
        session_token,
        login_response_token_field,
        signin_token_field,
    ) or "token"
    token_field_value = _first(
        token_field,
        token_key,
        token_path,
        access_token_field,
        jwt_field,
        jwt_token_field,
        jwt_token,
        response_token_field,
        jwt_token_key,
    ) or "access_token"
    email_field_value = (email_field or "").strip() or "email"
    password_field_value = (password_field or "").strip() or "password"

    # If a login flow is requested, obtain a JWT and turn it into a Bearer header.
    # Decide which field name applies to the intermediate (login) response vs the
    # final JWT response. In two-step mode, jwt-specific field names win for the
    # final response; otherwise token_field is used for the single login response.
    if token_url_value:
        # Two-step: keep token_field for the final JWT response, and use the
        # login token field for the intermediate response.
        intermediate_field_value = login_token_field_value
        final_field_value = token_field_value
    else:
        # Single-step: the user may have supplied either field name; prefer an
        # explicitly provided login token field, else token_field.
        if login_token_field is not None or login_token is not None or session_token_field is not None or session_token is not None or login_response_token_field is not None or signin_token_field is not None:
            intermediate_field_value = login_token_field_value
            final_field_value = login_token_field_value
        else:
            intermediate_field_value = token_field_value
            final_field_value = token_field_value

    login_headers = _login_for_token(
        request,
        email=login_email,
        password=login_password,
        login_url=login_url_value,
        token_field=final_field_value,
        email_field=email_field_value,
        password_field=password_field_value,
        login_token_field=intermediate_field_value,
        token_url=token_url_value,
        jwt_token_field=final_field_value,
    )
    # Login-derived headers first; explicit auth_headers win on conflict.
    merged_headers = {**login_headers, **parsed_headers}

    report_id = uuid.uuid4().hex[:12]
    storage.create_report(report_id, target_url)
    state = AgentState(
        report_id=report_id,
        target_base_url=target_url,
        raw_spec=raw_spec,
        spec_url=spec_url,
        auth_headers=merged_headers,
    )
    background_tasks.add_task(_run_analysis, state)
    return {
        "report_id": report_id,
        "status": "running",
        "report_url": f"/reports/{report_id}",
        "html_url": f"/reports/{report_id}/html",
    }


@app.get("/reports")
def list_reports() -> list[dict[str, Any]]:
    """Summaries of all analysis runs, newest first."""
    return storage.list_reports()


@app.get("/reports/{report_id}")
def get_report(report_id: str) -> dict[str, Any]:
    """Full report as JSON (``report`` is null while the run is in progress)."""
    row = storage.get_report(report_id)
    if row is None:
        raise HTTPException(404, f"no report with id {report_id}")
    return row


@app.get("/reports/{report_id}/html", response_class=HTMLResponse)
def get_report_html(report_id: str) -> HTMLResponse:
    """Readable HTML version of the report."""
    row = storage.get_report(report_id)
    if row is None:
        raise HTTPException(404, f"no report with id {report_id}")
    if row["report"] is None:
        detail = row["error"] or "the agent is parsing, planning, generating and executing tests"
        return HTMLResponse(render_pending(report_id, row["status"], detail))
    return HTMLResponse(render_html(Report.model_validate(row["report"])))


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
