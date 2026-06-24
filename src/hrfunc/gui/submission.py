"""HRF submission panel — desktop counterpart to hrfunc-web's /hrf_upload.

Researchers who've estimated HRFs in the HRfunc library can submit them
to the HRtree via the web form at https://www.hrfunc.org/hrf_upload.
This module ships the same flow inside the desktop GUI so users don't
have to context-switch to a browser to share their work:

- :class:`SubmissionMetadata` mirrors the web form's field names so
  the backend's expectations don't fork between the two clients.
- :func:`submit_payload` POSTs the payload JSON + metadata to
  hrfunc-web's ``/upload_json`` endpoint, which then validates, rate-
  limits, forwards to the canonical backend, and sends the
  confirmation email.
- :func:`check_hrserv_health` polls HRServ's ``/healthz`` so the panel
  can surface an "Accepting HRF Submissions" pill (mirrors the web
  form's JS-driven status pill).
- :func:`render_submission_panel` renders the NiceGUI form. Caller
  drops it on the Export tab (always) and on the HRFs tab (after a
  successful estimation) so the submission flow is reachable from
  the two places users naturally finish work.

The upload endpoint is overridable via the ``HRFUNC_UPLOAD_URL``
environment variable -- matches hrfunc-web's own override pattern so
the same variable means the same thing in both clients. Default
target is the production deployment at hrfunc-web. The health
endpoint similarly overridable via ``HRFUNC_HEALTH_URL``; defaults to
``https://api.hrfunc.org/healthz`` (HRServ's public health route).

What lives where:

- Form rendering / event wiring: :func:`render_submission_panel` (UI)
- Pure data shape: :class:`SubmissionMetadata` (testable without UI)
- HTTP I/O: :func:`submit_payload`, :func:`check_hrserv_health`
  (testable with mocked ``requests``)

No state on AppState -- the form's transient values are held in a
plain dataclass within the panel closure, same pattern the HRFs /
Activity panels use for their per-render options snapshots.
"""

from __future__ import annotations

import enum
import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


# Default hrfunc-web upload endpoint. Overridable via env var to point
# at a staging deployment without code changes -- mirrors hrfunc-web's
# own ``HRFUNC_UPLOAD_URL`` convention (though that variable on the
# server points to the *backend*, here it points to hrfunc-web itself).
DEFAULT_UPLOAD_URL = "https://www.hrfunc.org/upload_json"

# HRServ's public health-check route. The web form polls the same
# URL from JS; rendering the same pill in the desktop GUI keeps the
# two clients consistent so a HRServ outage looks identical from
# either entry point.
DEFAULT_HEALTH_URL = "https://api.hrfunc.org/healthz"

# Base URL of hrfunc-web. The submission flow's context-field labels
# link into its ``/experimental_contexts`` page so users can read
# pre-existing entries before deciding what to type. The anchor
# fragments below match the section IDs hrfunc-web renders in that
# page (see hrfunc-web/templates/experimental_contexts.html). When the
# upload URL is overridden, we derive this base from it so a local
# hrfunc-web instance has the help links pointing at the local copy
# rather than production.
HRFUNC_WEB_BASE_URL = "https://www.hrfunc.org"

# Per-field anchor fragments on /experimental_contexts. Keyed by the
# ``SubmissionMetadata`` attribute name so :func:`_text_input` can
# look them up by attr. Mirrors the ``href`` attributes the web
# form's label tags carry.
EXPERIMENTAL_CONTEXT_ANCHORS: Dict[str, str] = {
    "task": "context-tasks",
    "conditions": "context-conditions",
    "stimuli": "context-stimuli",
    "medium": "context-medium",
    "protocol": "context-protocols",
    "demographics": "context-demographics",
    "health_status": "context-health-status",
}


def _experimental_context_url(anchor: str) -> str:
    """Build a full URL into hrfunc-web's experimental-contexts page.

    Derives the base from ``HRFUNC_UPLOAD_URL`` when it's set to a
    non-production target, so pointing the desktop at a local
    hrfunc-web instance (``HRFUNC_UPLOAD_URL=http://localhost:8000/upload_json``)
    sends the help links to the same instance rather than to
    production. Falls back to :data:`HRFUNC_WEB_BASE_URL` when no
    override is in effect.
    """
    override = os.environ.get("HRFUNC_UPLOAD_URL")
    if override:
        # The upload URL has a path; trim it back to scheme+host so
        # we can append ``/experimental_contexts#anchor``.
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(override)
        base = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    else:
        base = HRFUNC_WEB_BASE_URL
    return f"{base}/experimental_contexts#{anchor}"


# How often the panel re-polls the health endpoint while it's open.
# Matches hrfunc-web's 60-second JS interval -- frequent enough that
# a state change becomes visible within a minute, infrequent enough
# that an open panel doesn't hammer the endpoint.
HEALTH_POLL_INTERVAL_S = 60.0

# Default timeout for the health GET. Should be much shorter than
# the submission timeout because a slow-responding healthz makes the
# pill look stuck in "Checking…" -- better to fail fast and render
# "down" so the user knows the backend can't be reached.
HEALTH_TIMEOUT_S = 5.0


class HealthState(str, enum.Enum):
    """Three discriminable states the health pill can render in.

    String values so the panel can pass them straight to NiceGUI's
    class-string attribute without an explicit conversion.

    - ``CHECKING``: poll in flight (or panel just mounted, no poll
      result yet). Grey pill with neutral copy.
    - ``OK``: HRServ returned HTTP 200. Green pill, "Accepting HRF
      Submissions" copy.
    - ``DOWN``: HRServ returned a non-200, or the request raised a
      transport exception. Red pill, "Submission system down" copy.
    """

    CHECKING = "checking"
    OK = "ok"
    DOWN = "down"


@dataclass
class SubmissionMetadata:
    """The form-field bundle the user fills out for each submission.

    Field names match the web form (``templates/hrf_upload.html``) so
    when this dataclass is serialised into multipart form-data it
    produces the same wire format the backend already understands.
    The two clients (web + desktop) talk to the same API; one
    canonical schema means no divergence to maintain.

    Fields with a dash in the wire name (``area-codes``, ``health-
    status``) use underscores here for valid Python identifiers; the
    serialiser converts them back to dashes in
    :meth:`to_form_dict`. The web form's two conditional sections
    (``dataset_permission`` only when not the owner; ``hrfunc_extension``
    only when not standard library) are not enforced as dataclass
    invariants -- the panel hides those inputs when they don't apply,
    and the backend tolerates missing values.
    """

    # ── Researcher contact
    name: str = ""
    email: str = ""
    phone: str = ""

    # ── Study identity
    study: str = ""
    area_codes: str = ""  # wire: area-codes
    doi: str = ""

    # ── Dataset rights
    # "yes" / "no". Empty string before the user picks.
    dataset_ownership: str = ""
    dataset_permission: str = ""  # only when ownership == "no"
    dataset_owner: str = ""       # only when permission == "yes"
    dataset_contact: str = ""     # only when permission == "yes"

    # ── HRfunc usage
    hrfunc_standard: str = ""       # "yes" / "no"
    hrfunc_extension: str = ""      # only when hrfunc_standard == "no"

    # ── Dataset scope
    dataset_subset: str = ""        # "yes" / "no"

    # ── Experimental context (all required by the web form)
    task: str = ""
    conditions: str = ""
    stimuli: str = ""
    medium: str = ""
    intensity: str = ""
    protocol: str = ""
    age: str = ""
    demographics: str = ""
    health_status: str = ""  # wire: health-status

    # ── Optional notes
    comment: str = ""

    # Mapping from this dataclass's Python attribute name to the wire
    # name expected by the backend. Most are identical; the dashes-vs-
    # underscores cases are the only divergences. Kept as a module-
    # level constant on the dataclass so :meth:`to_form_dict` and
    # tests share one source of truth.
    _WIRE_OVERRIDES = {
        "area_codes": "area-codes",
        "health_status": "health-status",
    }

    def to_form_dict(self) -> Dict[str, str]:
        """Serialise to the multipart form-data dict the backend wants.

        Skips the empty-string entries so the backend's
        ``request.form.to_dict()`` mirror doesn't get clutter. Wire-
        name overrides apply (``area_codes`` -> ``area-codes``, etc.).
        """
        out: Dict[str, str] = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if value in ("", None):
                continue
            wire = self._WIRE_OVERRIDES.get(f.name, f.name)
            out[wire] = str(value)
        return out

    def missing_required(self) -> list:
        """Return field labels the user still needs to fill in.

        The web form's ``required`` attribute drives this list. The
        conditional fields (dataset_permission / _owner / _contact /
        hrfunc_extension) are only required in the right branch -- the
        same conditionals the form encodes -- so they're checked
        contextually here, not unconditionally.
        """
        missing = []
        for attr, label in (
            ("name", "Your Name"),
            ("email", "Email"),
            ("phone", "Phone Number"),
            ("study", "Study Name"),
            ("area_codes", "Area Codes"),
            ("doi", "DOI"),
            ("dataset_ownership", "Dataset Ownership"),
            ("hrfunc_standard", "Used Unaltered HRfunc"),
            ("dataset_subset", "Dataset Subset"),
            ("task", "Task"),
            ("conditions", "Conditions"),
            ("stimuli", "Stimulus"),
            ("medium", "Stimulus Medium"),
            ("intensity", "Stimuli Intensity"),
            ("protocol", "Protocol"),
            ("age", "Age Range"),
            ("demographics", "Demographics"),
            ("health_status", "Health Status"),
        ):
            if not getattr(self, attr).strip():
                missing.append(label)

        # Conditional: permission only required when user doesn't own
        # the dataset.
        if self.dataset_ownership == "no" and not self.dataset_permission.strip():
            missing.append("Dataset Permission")
        # Conditional: owner + contact only required when the user does NOT
        # own the dataset AND has permission. The ownership clause guards
        # against a stale dataset_permission=="yes" left over from a prior
        # answer (the form hides these inputs once ownership flips back to
        # "yes"); without it, missing_required would demand owner/contact for
        # fields that aren't rendered, stranding the Submit button. The panel
        # also clears these children on the parent change, so this is
        # defence in depth.
        if self.dataset_ownership == "no" and self.dataset_permission == "yes":
            if not self.dataset_owner.strip():
                missing.append("Dataset Owner Name")
            if not self.dataset_contact.strip():
                missing.append("Dataset Owner Email")
        # Conditional: hrfunc_extension required when not standard.
        if self.hrfunc_standard == "no" and not self.hrfunc_extension.strip():
            missing.append("HRfunc Modifications")

        return missing


# Module-level so :func:`submit_payload` and tests can derive the
# default without duplicating the os.environ check.
def upload_url() -> str:
    """Return the hrfunc-web ``/upload_json`` URL to POST submissions to.

    Honors the ``HRFUNC_UPLOAD_URL`` env var; falls back to the
    production endpoint. Matches the override semantics hrfunc-web
    uses for its own backend-forwarding URL so the same variable does
    the same thing in both clients (desktop -> hrfunc-web vs
    hrfunc-web -> backend).
    """
    return os.environ.get("HRFUNC_UPLOAD_URL", DEFAULT_UPLOAD_URL)


def health_url() -> str:
    """Return the HRServ ``/healthz`` URL the pill polls.

    Honors the ``HRFUNC_HEALTH_URL`` env var; defaults to the
    production endpoint at ``https://api.hrfunc.org/healthz``.
    Override is useful when pointing the desktop at a staging
    HRServ during integration testing.
    """
    return os.environ.get("HRFUNC_HEALTH_URL", DEFAULT_HEALTH_URL)


def check_hrserv_health(
    *,
    target_url: Optional[str] = None,
    timeout_s: float = HEALTH_TIMEOUT_S,
) -> HealthState:
    """Probe HRServ's healthz endpoint and map the response to a state.

    Mirrors what the hrfunc-web JS pill does: GET, treat HTTP 200 as
    ``OK``, anything else (non-200 response, transport exception) as
    ``DOWN``. No retries -- the panel timer re-fires on the same
    interval (60 s by default) so a transient blip auto-recovers
    without us building a backoff ladder here.

    ``target_url`` defaults to :func:`health_url` (env-var override
    aware). Tests pass an explicit URL to point at a mock server.

    HEAD would be more efficient (HRServ's route accepts it) but
    requests' default ``redirect`` behavior strips the body on GET
    when the response is HEAD-style anyway, and using GET keeps the
    code identical to the web form's ``fetch(URL)``. Symmetry between
    the two clients matters more than the tiny bandwidth win.
    """
    import requests

    url = target_url or health_url()
    try:
        response = requests.get(url, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 -- mirror JS fetch's
        logger.debug("check_hrserv_health: transport error: %s", exc)
        return HealthState.DOWN
    return HealthState.OK if response.status_code == 200 else HealthState.DOWN


@dataclass
class SubmissionResult:
    """Outcome of a single submission attempt.

    Returned by :func:`submit_payload`. Three buckets:

    - ``ok=True``: HTTP 200, the backend accepted the file. The user
      will get a confirmation email at the address they entered.
    - ``ok=False, status_code is not None``: server returned an error;
      ``message`` carries the response body (truncated).
    - ``ok=False, status_code is None``: transport failure (DNS,
      connection refused, timeout); ``message`` is the exception text.
    """

    ok: bool
    status_code: Optional[int]
    message: str


def _iter_hrf_entries(data: Any, key: Optional[str] = None):
    """Yield ``(name, entry)`` for every HRF block in a payload.

    Handles both saved shapes: a dict keyed by channel name whose values are
    HRF entries (``montage.save``), and a list of ROI entries
    (``build_roi_entry``), plus any wrapper nesting. An HRF entry is any dict
    carrying an ``hrf_mean`` key.
    """
    if isinstance(data, dict):
        if "hrf_mean" in data:
            yield (key or data.get("name") or data.get("roi_anchor_key")
                   or "HRF", data)
            return
        for k, v in data.items():
            yield from _iter_hrf_entries(v, k)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            yield from _iter_hrf_entries(v, key=f"#{i}")


def inspect_payload_quality(payload_path: Path) -> list[str]:
    """Best-effort scientific sanity check of an HRF payload before submit.

    Returns a list of human-readable WARNINGS (empty = looks fine). Never
    raises -- on an unreadable / unexpected structure it returns either an
    empty list (JSON validity is checked separately by :func:`submit_payload`)
    or a single soft note, so the caller can still let the user proceed.

    Flags, per HRF entry: a flat/degenerate mean trace (zero or no finite
    variation), a missing/non-positive sampling frequency, and fewer than 2
    contributing subject estimates (a single-estimate HRF has no
    between-subject variability and is weak evidence for the shared library).
    """
    import numpy as np

    try:
        data = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- JSON validity reported by submit_payload
        return []

    entries = list(_iter_hrf_entries(data))
    if not entries:
        return [
            "Could not find any HRF traces in this file — double-check it is "
            "an HRF / montage JSON before submitting."
        ]

    flat: list[str] = []
    no_sfreq: list[str] = []
    single: list[str] = []
    for name, entry in entries:
        mean = entry.get("hrf_mean")
        arr = (
            np.asarray(mean, dtype=float)
            if mean is not None else np.array([], dtype=float)
        )
        if (
            arr.size == 0
            or not np.any(np.isfinite(arr))
            or float(np.nanstd(arr)) == 0.0
        ):
            flat.append(str(name))
        sfreq = entry.get("sfreq")
        try:
            if sfreq is None or float(sfreq) <= 0:
                no_sfreq.append(str(name))
        except (TypeError, ValueError):
            no_sfreq.append(str(name))
        n_est = len(entry.get("estimate_sources") or entry.get("estimates") or [])
        if n_est < 2:
            single.append(str(name))

    warnings: list[str] = []

    def _summ(items: list[str], label: str) -> None:
        if not items:
            return
        shown = ", ".join(items[:3]) + ("…" if len(items) > 3 else "")
        warnings.append(f"{len(items)} HRF(s) {label} ({shown}).")

    _summ(flat, "have a flat / degenerate mean trace")
    _summ(no_sfreq, "are missing a valid sampling frequency")
    _summ(
        single,
        "are built from fewer than 2 subject estimates "
        "(no between-subject variability)",
    )
    return warnings


def submit_payload(
    *,
    payload_path: Path,
    metadata: SubmissionMetadata,
    target_url: Optional[str] = None,
    timeout_s: float = 30.0,
) -> SubmissionResult:
    """POST ``payload_path`` + ``metadata`` to hrfunc-web's upload endpoint.

    ``payload_path`` must point to a readable JSON file. The file is
    sent as ``jsonFile`` multipart-form-data alongside the metadata
    form fields -- byte-identical to what the web form's ``<form
    enctype="multipart/form-data">`` produces, so the backend doesn't
    need a separate code path for desktop submissions.

    ``target_url`` defaults to :func:`upload_url` (env-var override
    aware). Tests pass an explicit URL to point at a mock server.

    Raises no exceptions on transport failure -- the failure is
    captured in the returned :class:`SubmissionResult` so the panel
    can surface it via ``ui.notify`` without try/except gymnastics.
    """
    import requests

    if not payload_path.is_file():
        return SubmissionResult(
            ok=False,
            status_code=None,
            message=f"Payload not found: {payload_path}",
        )

    # Validate the file parses as JSON before shipping it. The web
    # form does the same check; doing it client-side too means a
    # malformed file fails fast without a round-trip.
    try:
        json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return SubmissionResult(
            ok=False,
            status_code=None,
            message=f"Payload is not valid JSON: {exc}",
        )

    url = target_url or upload_url()
    form_data = metadata.to_form_dict()

    try:
        with payload_path.open("rb") as fh:
            response = requests.post(
                url,
                data=form_data,
                files={
                    "jsonFile": (
                        payload_path.name,
                        fh,
                        "application/json",
                    ),
                },
                timeout=timeout_s,
            )
    except Exception as exc:  # noqa: BLE001 -- transport surface
        logger.exception("submit_payload: transport error: %s", exc)
        return SubmissionResult(
            ok=False,
            status_code=None,
            message=f"{type(exc).__name__}: {exc}",
        )

    body = (response.text or "")[:512]
    if response.status_code == 200:
        return SubmissionResult(
            ok=True,
            status_code=200,
            message=body or "Submission accepted.",
        )
    return SubmissionResult(
        ok=False,
        status_code=response.status_code,
        message=body or f"HTTP {response.status_code}",
    )


# ---------------------------------------------------------------------------
# UI: render_submission_panel
# ---------------------------------------------------------------------------


# Yes/No select options shared by the three conditional questions.
_YES_NO_OPTIONS = {"": "Select...", "yes": "Yes", "no": "No"}


def render_submission_panel(state, *, default_path: Optional[Path] = None) -> None:
    """Render the HRF submission form inside the current NiceGUI context.

    The caller (Export tab body, HRFs tab body) wraps the call in
    whatever container they want -- this function emits the form
    rows directly so it composes cleanly inside an existing
    ``ui.card`` or ``ui.column``.

    ``default_path``: optional starting value for the file-path
    input. The Export tab passes ``state.last_saved_roi_path`` so the
    most recently saved montage is pre-filled.

    The form mirrors the web form's required-field rules. Submission
    runs ``metadata.missing_required()`` first and surfaces a notify
    listing what's missing rather than letting the user discover
    incomplete fields by waiting on the backend round-trip.
    """
    from nicegui import background_tasks, ui

    metadata = SubmissionMetadata()
    file_state: Dict[str, Optional[Path]] = {
        "path": default_path,
    }
    # Closure-held health state. Starts in CHECKING so the very first
    # render shows a neutral pill -- the ``ui.timer`` below fires on
    # mount (the ``active=True, immediate=True`` form) and replaces
    # this with the real state within the timeout window.
    health_state: Dict[str, HealthState] = {
        "value": HealthState.CHECKING,
    }

    @ui.refreshable
    def _body() -> None:
        ui.label("Submit HRFs to the HRtree").classes(
            "text-lg font-semibold"
        )
        ui.label(
            "Share your estimated HRFs with the community. Submissions "
            "are reviewed before they appear in the HRtree library. "
            "You'll receive a confirmation email at the address below."
        ).classes("text-xs opacity-70")
        ui.label(
            "All fields are required. For an experimental context or area "
            "code that doesn't apply to your study, enter N/A."
        ).classes("text-xs opacity-60 italic")

        # --- HRServ health pill ---------------------------------------
        # Mirrors the hrfunc-web JS pill at the top of /hrf_upload.
        # Three states: checking / ok / down. Polls every
        # ``HEALTH_POLL_INTERVAL_S`` (default 60 s) so a state change
        # surfaces within a minute without us hammering the endpoint.
        _render_health_pill(health_state["value"])

        # --- File selector --------------------------------------------
        ui.label("HRF JSON file").classes(
            "text-xs uppercase opacity-60 tracking-wide mt-3"
        )
        with ui.row().classes("w-full items-center gap-2"):
            current = file_state["path"]
            ui.label(
                str(current) if current else "(no file selected)"
            ).classes("text-xs font-mono opacity-80 break-all flex-1")

            async def _on_pick() -> None:
                from .components.dataset_picker import pick_file

                path = await pick_file(
                    file_types=[
                        "HRF montage (*.json)",
                        "All files (*.*)",
                    ],
                )
                if path is not None:
                    file_state["path"] = path
                    _body.refresh()

            ui.button(
                "Pick file", icon="folder_open", on_click=_on_pick,
            ).props("flat dense")

        # --- Researcher contact ---------------------------------------
        _section_label("Researcher")
        _text_input(metadata, "name", "Your name")
        _text_input(metadata, "email", "Email")
        _text_input(metadata, "phone", "Phone number")

        # --- Study identity -------------------------------------------
        _section_label("Study")
        _text_input(metadata, "study", "Study name")
        _text_input(metadata, "area_codes", "Area codes (comma-separated)")
        _text_input(metadata, "doi", "DOI of the paper detailing data collection")

        # --- Dataset rights -------------------------------------------
        # Conditional sub-fields are stamped onto the shared ``metadata`` by
        # their inputs and are NOT auto-cleared when the parent select hides
        # them. Without the resets below, flipping a parent back (e.g.
        # ownership "no" -> "yes") would leave a stale dataset_permission /
        # owner / contact on metadata: to_form_dict would then POST
        # contradictory rights metadata, and a stale permission could even
        # strand Submit on now-hidden required fields. Clear children on the
        # parent change before refreshing.
        def _on_ownership_change() -> None:
            if metadata.dataset_ownership != "no":
                metadata.dataset_permission = ""
                metadata.dataset_owner = ""
                metadata.dataset_contact = ""
            _body.refresh()

        def _on_permission_change() -> None:
            if metadata.dataset_permission != "yes":
                metadata.dataset_owner = ""
                metadata.dataset_contact = ""
            _body.refresh()

        _section_label("Dataset rights")
        _select_input(
            metadata, "dataset_ownership",
            "Do you own the dataset these HRFs were estimated from?",
            on_change=_on_ownership_change,
        )
        if metadata.dataset_ownership == "no":
            _select_input(
                metadata, "dataset_permission",
                "Do you have the owner's permission to add these HRFs to the HRtree?",
                on_change=_on_permission_change,
            )
            if metadata.dataset_permission == "no":
                with ui.row().classes("items-start gap-2"):
                    ui.icon("warning").classes("text-amber-500")
                    ui.label(
                        "Without explicit permission from the dataset owner "
                        "we can't add HRFs to the HRtree. Please reach out "
                        "to the owner before submitting."
                    ).classes("text-xs opacity-80")
            if metadata.dataset_permission == "yes":
                _text_input(metadata, "dataset_owner", "Dataset owner name")
                _text_input(metadata, "dataset_contact", "Dataset owner email")

        # --- HRfunc usage ---------------------------------------------
        # Same stale-child concern as Dataset rights: clear hrfunc_extension
        # when the user flips back to "yes" so a prior extension summary
        # isn't silently POSTed for a now-standard submission.
        def _on_hrfunc_standard_change() -> None:
            if metadata.hrfunc_standard != "no":
                metadata.hrfunc_extension = ""
            _body.refresh()

        _section_label("HRfunc usage")
        _select_input(
            metadata, "hrfunc_standard",
            "Did you estimate these HRFs using the unaltered HRfunc library?",
            on_change=_on_hrfunc_standard_change,
        )
        if metadata.hrfunc_standard == "no":
            ui.label(
                "Extensions are welcome, but the DOI above must detail "
                "how you deviated from the standard library so HRtree "
                "viewers can trace HRFs back to their origin."
            ).classes("text-xs opacity-60")
            _textarea_input(
                metadata, "hrfunc_extension",
                "Summarise the changes you made to the HRfunc library.",
            )

        # --- Dataset scope --------------------------------------------
        _section_label("Dataset scope")
        _select_input(
            metadata, "dataset_subset",
            "Are these HRFs estimated from a subset of your dataset?",
        )

        # --- Experimental context -------------------------------------
        _section_label("Experimental context")
        _text_input(metadata, "task", "Task (e.g. flanker, stroop, n-back)")
        _text_input(metadata, "conditions", "Conditions (e.g. congruent,incongruent)")
        _text_input(metadata, "stimuli", "Stimulus (e.g. arrows, colors, faces)")
        _text_input(metadata, "medium", "Stimulus medium (e.g. monitor, cards)")
        _text_input(
            metadata, "intensity",
            'Stimuli intensity (set to "1.0" if not modulated)',
        )
        _text_input(metadata, "protocol", 'Protocol (set to "default" if standard)')
        _text_input(metadata, "age", "Age range (e.g. (18, 65))")
        _text_input(
            metadata, "demographics",
            'Demographics (set to "all" if pool represents whole population)',
        )
        _text_input(
            metadata, "health_status",
            'Health status (set to "untested" if not surveyed)',
        )

        # --- Notes ----------------------------------------------------
        _section_label("Notes (optional)")
        _textarea_input(metadata, "comment", "Any additional details.")

        # --- Submit ---------------------------------------------------
        ui.separator()
        with ui.row().classes("w-full items-center gap-2 mt-2"):
            async def _on_submit() -> None:
                # Pre-flight: surface missing required fields before
                # we round-trip to the server. The web form does the
                # same check via the HTML ``required`` attribute --
                # we do it explicitly because NiceGUI inputs don't
                # carry HTML5 validation semantics.
                if file_state["path"] is None:
                    ui.notify(
                        "Pick a HRF JSON file before submitting.",
                        type="warning",
                    )
                    return
                missing = metadata.missing_required()
                if missing:
                    ui.notify(
                        "Missing required fields: " + ", ".join(missing),
                        type="warning",
                    )
                    return
                # Hard block on the "no permission" branch -- matches
                # the web form's banner.
                if (
                    metadata.dataset_ownership == "no"
                    and metadata.dataset_permission == "no"
                ):
                    ui.notify(
                        "Cannot submit without dataset-owner permission. "
                        "Reach out to the owner first.",
                        type="negative",
                    )
                    return

                # Scientific pre-flight: warn (don't hard-block) when the
                # payload looks degenerate / under-powered so a weak HRF
                # doesn't silently enter the shared library. The user can
                # still proceed deliberately.
                quality_warnings = inspect_payload_quality(file_state["path"])
                if quality_warnings:
                    with ui.dialog() as _qdlg, ui.card():
                        ui.label("Check before submitting").classes(
                            "text-lg font-bold"
                        )
                        ui.label(
                            "This HRF file has potential quality issues:"
                        ).classes("text-sm")
                        for _w in quality_warnings:
                            ui.label(f"• {_w}").classes(
                                "text-sm text-amber-800"
                            )
                        ui.label(
                            "Submit it to the shared library anyway?"
                        ).classes("text-sm q-mt-sm")
                        with ui.row().classes("justify-end w-full"):
                            ui.button(
                                "Cancel", on_click=lambda: _qdlg.submit(False)
                            ).props("flat")
                            ui.button(
                                "Submit anyway",
                                on_click=lambda: _qdlg.submit(True),
                            ).props("color=warning")
                    if not await _qdlg:
                        return

                ui.notify("Uploading…", type="info")

                # The HTTP call is sync (requests). Wrap in a thread
                # so the event loop stays responsive.
                import asyncio

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: submit_payload(
                        payload_path=file_state["path"],
                        metadata=metadata,
                    ),
                )
                if result.ok:
                    ui.notify(
                        "Submission accepted. Check your email for "
                        "confirmation.",
                        type="positive",
                    )
                else:
                    ui.notify(
                        f"Submission failed: {result.message}",
                        type="negative",
                    )

            ui.button(
                "Submit to HRtree",
                icon="cloud_upload",
                on_click=_on_submit,
            ).props("color=primary")
            ui.label(
                f"Endpoint: {upload_url()}"
            ).classes("text-xs font-mono opacity-50 break-all")

    _body()

    # Background poller -- runs on the NiceGUI event loop, dispatches
    # the blocking HTTP call to an executor thread so the UI stays
    # responsive. ``immediate=True`` triggers a first poll right after
    # mount instead of waiting a full interval to leave the pill
    # stuck in "Checking…".
    async def _poll_health() -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        new_state = await loop.run_in_executor(None, check_hrserv_health)
        if new_state == health_state["value"]:
            return  # No change -- skip the refresh to avoid render churn.
        health_state["value"] = new_state
        _body.refresh()

    ui.timer(HEALTH_POLL_INTERVAL_S, _poll_health, immediate=True)


# ---------------------------------------------------------------------------
# Private form helpers
# ---------------------------------------------------------------------------


def _render_health_pill(state: HealthState) -> None:
    """Render a coloured pill mirroring hrfunc-web's status indicator.

    Three visual states keyed off :class:`HealthState`:

    - ``CHECKING``: neutral grey, "Checking submission status…"
    - ``OK``: green, "Accepting HRF submissions"
    - ``DOWN``: red, "Submission system down"

    Class strings follow the same Tailwind palette as the web pill
    so the two clients are visually consistent. The web pill is
    just a coloured dot + status text; the desktop pill matches
    that exact pattern (no Material icon -- the dot's colour is
    the affordance, and the colour shift between gray / green / red
    is recognisable without a separate iconographic cue).
    """
    from nicegui import ui

    if state == HealthState.OK:
        bg = "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-200"
        dot = "bg-green-500"
        text = "Accepting HRF submissions"
    elif state == HealthState.DOWN:
        bg = "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-200"
        dot = "bg-red-500"
        text = (
            "Submission system down — please try again later or "
            "contact help@hrfunc.org."
        )
    else:  # CHECKING
        bg = (
            "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-200"
        )
        dot = "bg-gray-400"
        text = "Checking submission status…"

    with ui.row().classes(
        "items-center gap-2 px-3 py-1.5 rounded-full text-sm mt-2 mb-1 "
        + bg
    ).style("width: fit-content"):
        # Status dot. Sized + coloured to mirror the web pill's
        # 10px coloured circle.
        ui.element("div").classes(
            f"inline-block w-2.5 h-2.5 rounded-full {dot}"
        )
        ui.label(text)


def _section_label(text: str) -> None:
    from nicegui import ui

    ui.label(text).classes(
        "text-xs uppercase opacity-60 tracking-wide mt-3"
    )


def _text_input(
    metadata: SubmissionMetadata,
    attr: str,
    label_text: str,
) -> None:
    """One-line text input bound to ``metadata.<attr>``.

    When ``attr`` is one of the experimental-context fields (task,
    conditions, stimuli, etc.), a small "examples ↗" link is
    rendered below the input that opens hrfunc-web's
    ``/experimental_contexts`` page at the matching anchor. The
    link mirrors the corresponding ``<a target="_blank">`` element
    in hrfunc-web's hrf_upload form so users can browse existing
    entries before deciding what to type.
    """
    from nicegui import ui

    def _on_change(event) -> None:
        setattr(metadata, attr, str(event.value or ""))

    # Every field is required, but experimental-context fields and the area
    # codes may genuinely not apply to a study — so they accept an explicit
    # "N/A". Surface that as a placeholder so users fill it deliberately
    # instead of being stuck on a required field that doesn't apply.
    na_ok = attr in EXPERIMENTAL_CONTEXT_ANCHORS or attr == "area_codes"
    ui.input(
        label=label_text,
        value=getattr(metadata, attr),
        placeholder="value, or N/A if not applicable" if na_ok else None,
        on_change=_on_change,
    ).props("dense outlined").classes("w-full")

    anchor = EXPERIMENTAL_CONTEXT_ANCHORS.get(attr)
    if anchor is not None:
        url = _experimental_context_url(anchor)
        ui.link(
            "examples ↗", url, new_tab=True,
        ).classes(
            "text-xs text-indigo-500 hover:text-indigo-300 "
            "no-underline -mt-2"
        )


def _textarea_input(
    metadata: SubmissionMetadata,
    attr: str,
    label_text: str,
) -> None:
    """Multi-line textarea bound to ``metadata.<attr>``."""
    from nicegui import ui

    def _on_change(event) -> None:
        setattr(metadata, attr, str(event.value or ""))

    ui.textarea(
        label=label_text,
        value=getattr(metadata, attr),
        on_change=_on_change,
    ).props("dense outlined").classes("w-full")


def _select_input(
    metadata: SubmissionMetadata,
    attr: str,
    label_text: str,
    *,
    on_change=None,
) -> None:
    """Yes/no dropdown bound to ``metadata.<attr>``.

    ``on_change`` is called after the value is stamped onto metadata --
    used by conditional sections that need to re-render when the user
    picks yes or no.
    """
    from nicegui import ui

    def _handler(event) -> None:
        setattr(metadata, attr, str(event.value or ""))
        if on_change is not None:
            on_change()

    ui.label(label_text).classes("text-sm")
    ui.select(
        options=_YES_NO_OPTIONS,
        value=getattr(metadata, attr) or "",
        on_change=_handler,
    ).props("dense outlined").classes("w-full")
