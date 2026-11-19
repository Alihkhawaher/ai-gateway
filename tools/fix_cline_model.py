#!/usr/bin/env python3
"""
Point Cline at a gateway model with the right settings, through either provider:
LM Studio or OpenAI Compatible.  fix_cline_model.pyw is the same thing in a window.

Why the model sticks
--------------------
Cline (saoudrizwan.claude-dev 4.1.x) resolves the model it actually calls from
the LEGACY keys in  ~/.cline/data/globalState.json :

    <plan|act>Mode<Provider>ModelId    <- highest priority (e.g. actModeLmStudioModelId)
    <plan|act>ModeApiModelId           <- fallback, only if the above is empty

The provider settings you edit in the UI are written to
~/.cline/data/settings/providers.json, and that pick is not copied into the
legacy key, so every new session keeps the old model id.

Why the context shows 128k
--------------------------
The chat engine takes a model's context window from its record in
~/.cline/data/settings/models.json.  For an LM Studio model that record has no
contextWindow, and for an OpenAI Compatible model Cline wrote 128,000 when it
first set the provider up, so both fall back to 128,000.  This script writes the
window the gateway reports into the record (contextWindow and maxInputTokens).

Thinking
--------
Cline's settings page offers a reasoning-effort choice for OpenAI Compatible but
not for LM Studio, where the provider's saved setting is often "thinking off".
--thinking writes the choice into the provider's saved settings for either one:
off, low, medium, high, or "default" (send nothing: the model decides).

What this script does
---------------------
It asks which provider (or takes -p), lists the gateway's models with their
context window and vision support, and writes the model id, the window, vision,
the thinking choice and (with --activate) makes that provider the one Cline uses.
Backups are written next to each patched file (the newest 10 are kept) and
--revert restores the newest.

Usage
-----
    python fix_cline_model.py                       ask the provider, pick a model, write
    python fix_cline_model.py -p openai-compatible -m strata:swift-iq3_xxs --activate
    python fix_cline_model.py -p lmstudio --sync    use the model picked in Cline's settings page
    python fix_cline_model.py --thinking medium     with a model choice: also set the thinking level
    python fix_cline_model.py --check               what Cline really uses
    python fix_cline_model.py --list                models the gateway offers
    python fix_cline_model.py --revert              restore the newest backup

Restart VS Code afterwards: Cline reads these settings when it starts.

Exit codes: 0 done (or --check found nothing wrong), 1 nothing changed / --check found a mismatch, 2 files missing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

KEEP_BACKUPS = 10

# The two providers this tool sets up: id in providers.json/models.json -> label.
PROVIDERS = {"lmstudio": "LM Studio", "openai-compatible": "OpenAI Compatible"}

# provider id -> the value Cline keeps in globalState's <mode>ModeApiProvider
LEGACY_PROVIDER_ID = {"lmstudio": "lmstudio", "openai-compatible": "openai"}

# provider id -> legacy key suffix used by Cline ("actMode" + suffix)
PROVIDER_KEY_SUFFIX = {
    "openrouter": "OpenRouterModelId",
    "cline": "ClineModelId",
    "cline-pass": "ClinePassModelId",
    "openai": "OpenAiModelId",
    "openai-compatible": "OpenAiModelId",
    "ollama": "OllamaModelId",
    "lmstudio": "LmStudioModelId",
    "litellm": "LiteLlmModelId",
    "requesty": "RequestyModelId",
    "together": "TogetherModelId",
    "fireworks": "FireworksModelId",
    "sapaicore": "SapAiCoreModelId",
    "groq": "GroqModelId",
    "baseten": "BasetenModelId",
    "huggingface": "HuggingFaceModelId",
    "huawei-cloud-maas": "HuaweiCloudMaasModelId",
    "oca": "OcaModelId",
    "aihubmix": "AihubmixModelId",
    "hicap": "HiCapModelId",
    "vertex": "VertexModelId",
    "gemini": "GeminiModelId",
    "deepseek": "DeepSeekModelId",
    "bedrock": "AwsBedrockModelId",
}

# provider id -> globalState key holding its base URL
PROVIDER_BASE_URL_KEY = {
    "lmstudio": "lmStudioBaseUrl",
    "ollama": "ollamaBaseUrl",
    "litellm": "liteLlmBaseUrl",
    "openai": "openAiBaseUrl",
    "openai-compatible": "openAiBaseUrl",
    "requesty": "requestyBaseUrl",
    "together": "togetherBaseUrl",
    "groq": "groqBaseUrl",
    "gemini": "geminiBaseUrl",
    "openrouter": "openRouterBaseUrl",
}

STATIC_BASE_URL = {"openrouter": "https://openrouter.ai/api"}

# provider id -> legacy globalState key that holds a context size as a string
LEGACY_CONTEXT_KEY = {"lmstudio": "lmStudioMaxTokens"}

THINKING_CHOICES = ("default", "off", "low", "medium", "high")

DEFAULT_GATEWAY = "http://localhost:8090"


# ── files ────────────────────────────────────────────────────────────────
def read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data) -> None:
    """Write through a temp file, so a crash can't leave Cline a half-written settings file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def default_cline_dir() -> Path:
    return Path.home() / ".cline" / "data"


def cline_paths(cline_dir: Path):
    settings = cline_dir / "settings"
    return cline_dir / "globalState.json", settings / "providers.json", settings / "models.json"


class ClineFiles:
    """Cline's three settings files, read once and written back together."""

    def __init__(self, cline_dir: Path):
        self.dir = Path(cline_dir).expanduser()
        self.gs_path, self.pv_path, self.md_path = cline_paths(self.dir)
        missing = [str(p) for p in (self.gs_path, self.pv_path) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing file(s): " + ", ".join(missing))
        self.state = read_json(self.gs_path)
        self.providers = read_json(self.pv_path)
        self.models = read_json(self.md_path) if self.md_path.exists() else {"version": 1, "providers": {}}

    def current_provider(self) -> str:
        raw = self.state.get("actModeApiProvider") or self.providers.get("lastUsedProvider") or "lmstudio"
        return "openai-compatible" if raw == "openai" else raw

    def save(self) -> list[str]:
        """Back up all three files, then write them. Returns the backup paths."""
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backups = []
        for path in (self.gs_path, self.pv_path, self.md_path):
            if path.exists():
                backups.append(str(make_backup(path, stamp)))
        write_json(self.gs_path, self.state)
        write_json(self.pv_path, self.providers)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.md_path, self.models)
        return backups


def make_backup(path: Path, stamp: str) -> Path:
    dest = path.with_name(path.name + f".bak-{stamp}")
    with open(path, "rb") as src, open(dest, "wb") as out:
        out.write(src.read())
    for old in sorted(glob.glob(os.path.join(str(path.parent), path.name + ".bak-*")))[:-KEEP_BACKUPS]:
        try:
            os.remove(old)
        except OSError:
            pass
    return dest


def revert_latest(cline_dir: Path) -> list[str]:
    """Restore the newest backup of each file. Returns what was restored."""
    done = []
    for target in cline_paths(Path(cline_dir).expanduser()):
        found = sorted(glob.glob(os.path.join(str(target.parent), target.name + ".bak-*")))
        if not found:
            continue
        bak = Path(found[-1])
        with open(bak, "rb") as src, open(target, "wb") as out:
            out.write(src.read())
        done.append(f"restored {target} from {bak.name}")
    return done


# ── what Cline has now ───────────────────────────────────────────────────
def legacy_model_keys(provider: str, modes) -> list[str]:
    suffix = PROVIDER_KEY_SUFFIX.get(provider, "ApiModelId")
    return [f"{m}Mode{suffix}" for m in modes]


def provider_settings(providers, provider: str) -> dict:
    entry = (providers.get("providers") or {}).get(provider) or {}
    settings = entry.get("settings")
    return settings if isinstance(settings, dict) else entry


def model_record(models_json, provider: str, model: str) -> dict:
    return (((models_json.get("providers") or {}).get(provider) or {}).get("models") or {}).get(model) or {}


def server_root(base_url: str) -> str:
    """The server's address without a trailing /v1, so both 'http://host:8090' and 'http://host:8090/v1' work."""
    root = (base_url or "").strip().rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def base_url_for(provider: str, state, providers):
    # What the UI wrote wins: the legacy base-URL key can be as stale as the legacy model key.
    settings = provider_settings(providers, provider)
    if isinstance(settings.get("baseUrl"), str) and settings["baseUrl"].strip():
        return settings["baseUrl"].strip()
    key = PROVIDER_BASE_URL_KEY.get(provider)
    if key and isinstance(state.get(key), str) and state[key].strip():
        return state[key].strip()
    return STATIC_BASE_URL.get(provider)


def gateway_root_guess(files: ClineFiles) -> str:
    """The gateway address to offer: whichever of the two providers already points somewhere."""
    for provider in (files.current_provider(), "lmstudio", "openai-compatible"):
        url = base_url_for(provider, files.state, files.providers)
        if url:
            return server_root(url)
    return DEFAULT_GATEWAY


def provider_base_url(provider: str, root: str) -> str:
    """LM Studio's provider takes the server address; OpenAI Compatible takes it with /v1."""
    root = server_root(root)
    return root + "/v1" if provider == "openai-compatible" else root


def current_thinking(files: ClineFiles, provider: str) -> str:
    reasoning = provider_settings(files.providers, provider).get("reasoning")
    if not isinstance(reasoning, dict):
        return "default"
    if reasoning.get("enabled") is False:
        return "off"
    effort = reasoning.get("effort")
    return effort if effort in ("low", "medium", "high") else "default"


def describe(files: ClineFiles, provider: str, modes=("plan", "act")) -> list[str]:
    """What Cline uses now for this provider, one line each."""
    st, pv, md = files.state, files.providers, files.models
    settings = provider_settings(pv, provider)
    keys = legacy_model_keys(provider, modes)
    effective = st.get(keys[-1]) or settings.get("model") or ""
    record = model_record(md, provider, effective) if effective else {}
    lines = [
        f"Cline dir          : {files.dir}",
        f"provider in use    : {files.current_provider()}  (Cline: "
        + ", ".join(f"{m}={st.get(m + 'ModeApiProvider', '<unset>')}" for m in modes) + ")",
        f"this provider      : {provider} ({PROVIDERS.get(provider, provider)})",
        f"base url           : {settings.get('baseUrl') or st.get(PROVIDER_BASE_URL_KEY.get(provider, ''), '<unset>')}",
        "model (real)       : " + ", ".join(f"{k}={st.get(k, '<unset>')}" for k in keys),
        f"model (settings)   : {settings.get('model', '<unset>')}",
        f"context window     : {record.get('contextWindow', '<unset: Cline assumes 128000>')}",
        f"vision             : {_vision_of(record)}",
        f"thinking           : {current_thinking(files, provider)}",
    ]
    return lines


def _vision_of(record) -> str:
    if record.get("supportsVision") is not None:
        return "yes" if record["supportsVision"] else "no"
    caps = record.get("capabilities")
    if isinstance(caps, list):
        return "yes" if "images" in caps else "no"
    return "<unset>"


def mismatches(files: ClineFiles, provider: str, modes) -> list[str]:
    """Where what Cline will really use differs from what its settings page shows."""
    found = []
    picked = provider_settings(files.providers, provider).get("model")
    for key in legacy_model_keys(provider, modes):
        if picked and files.state.get(key) != picked:
            found.append(f"{key} = {files.state.get(key, '<unset>')!r}, but the settings page picked {picked!r}")
    effective = files.state.get(legacy_model_keys(provider, ["act"])[0]) or picked
    if effective:
        record = model_record(files.models, provider, effective)
        if not record.get("contextWindow") or record.get("contextWindow") == 128000:
            found.append(f"models.json has no real context window for {effective!r}: Cline will assume 128,000")
    return found


# ── the gateway's models ─────────────────────────────────────────────────
def _get_json(url: str, timeout: float, key: str = ""):
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _number(value):
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def fetch_models(base_url: str, timeout: float = 12.0, key: str = ""):
    """(address, models) from the gateway. The LM Studio list has the context window that is
    loaded now; the OpenAI list has vision. Both are read and merged by model id."""
    root = server_root(base_url)
    found, order, last_error, used = {}, [], None, []
    for url in (root + "/api/v0/models", root + "/v1/models"):
        try:
            payload = _get_json(url, timeout, key)
        except Exception as exc:  # noqa: BLE001 -- try the other list
            last_error = exc
            continue
        used.append(url)
        items = payload.get("data") or payload.get("models") or []
        for it in items if isinstance(items, list) else []:
            mid = it.get("id") or it.get("name")
            if not mid:
                continue
            m = found.get(mid)
            if m is None:
                m = found[mid] = {"id": mid, "name": "", "state": "", "tools": None, "context": None, "vision": None}
                order.append(mid)
            if not m["name"] and it.get("name") and it.get("name") != mid:
                m["name"] = it["name"]
            state = it.get("state") or ""
            if not state and isinstance(it.get("status"), dict):
                state = it["status"].get("value") or ""
            m["state"] = m["state"] or state
            caps = (it.get("meta") or {}).get("capabilities") or {}
            if m["tools"] is None:
                tools = caps.get("function_calling")
                if tools is None and isinstance(it.get("capabilities"), list):
                    tools = True if "tool_use" in it["capabilities"] else None
                m["tools"] = tools
            # the window that is loaded now; the model's maximum only when nothing else is known
            m["context"] = m["context"] or (
                _number(it.get("loaded_context_length")) or _number(it.get("context_length"))
                or _number(it.get("context_window")) or _number(it.get("max_context_length")))
            if m["vision"] is None:
                mods = it.get("modalities")
                if isinstance(mods, dict) and "vision" in mods:
                    m["vision"] = bool(mods["vision"])
                elif it.get("type") == "vlm":
                    m["vision"] = True
                elif "vision" in caps:
                    m["vision"] = bool(caps["vision"])
    if not order and last_error:
        raise last_error
    return (used[0] if used else root + "/v1/models"), [found[i] for i in order]


def model_line(m) -> str:
    flags = []
    if m.get("state"):
        flags.append(m["state"])
    if m.get("context"):
        flags.append(f"context {m['context']:,}")
    if m.get("vision"):
        flags.append("vision")
    if m.get("tools") is False:
        flags.append("NO tool calling")
    return "{:<44} {}".format(m["id"], ", ".join(flags))


def vscode_running() -> int:
    try:
        if os.name == "nt":
            res = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Code.exe", "/NH"],
                                 capture_output=True, text=True, timeout=20,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return res.stdout.lower().count("code.exe")
        res = subprocess.run(["pgrep", "-fc", "[Cc]ode"], capture_output=True, text=True, timeout=20)
        return int(res.stdout.strip() or 0)
    except Exception:  # noqa: BLE001
        return 0


# ── the change ───────────────────────────────────────────────────────────
def apply_choice(files: ClineFiles, provider: str, model: str, *, window=None, vision=None,
                 modes=("plan", "act"), gateway_root=None, activate=False, thinking=None,
                 generic=True) -> list[str]:
    """Write the choice into the loaded files (in memory). Returns what changed, one line each.
    Call files.save() to back up and write them."""
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}")
    model = (model or "").strip()
    if not model:
        raise ValueError("no model chosen")
    st, pv, md = files.state, files.providers, files.models
    changes = []

    def put(container, key, value, label):
        if container.get(key) != value:
            changes.append(f"{label}: {container.get(key, '<unset>')} -> {value}")
            container[key] = value

    # the model Cline really calls (legacy keys) and the generic fallback
    for key in legacy_model_keys(provider, modes):
        put(st, key, model, key)
    if generic:
        for m in modes:
            put(st, f"{m}ModeApiModelId", model, f"{m}ModeApiModelId")

    base = provider_base_url(provider, gateway_root) if gateway_root else base_url_for(provider, st, pv)
    if gateway_root:
        put(st, PROVIDER_BASE_URL_KEY[provider], base, PROVIDER_BASE_URL_KEY[provider])

    if activate:
        for m in modes:
            put(st, f"{m}ModeApiProvider", LEGACY_PROVIDER_ID[provider], f"{m}ModeApiProvider")
        put(pv, "lastUsedProvider", provider, "providers.json lastUsedProvider")

    # the settings page's copy (providers.json)
    entry = pv.setdefault("providers", {}).setdefault(provider, {})
    settings = entry.get("settings")
    if not isinstance(settings, dict):
        settings = entry["settings"] = {}
    before = json.dumps(settings, sort_keys=True)
    put(settings, "provider", provider, f"providers.json {provider}.provider")
    put(settings, "model", model, f"providers.json {provider}.model")
    if base:
        put(settings, "baseUrl", base, f"providers.json {provider}.baseUrl")
    if thinking and thinking in THINKING_CHOICES:
        old = settings.get("reasoning") if isinstance(settings.get("reasoning"), dict) else {}
        new = {k: v for k, v in old.items() if k not in ("enabled", "effort")}
        if thinking == "off":
            new["enabled"] = False
        elif thinking != "default":
            new["enabled"] = True
            new["effort"] = thinking
        if new != old:
            changes.append(f"providers.json {provider}.reasoning: {current_thinking(files, provider)} -> {thinking}")
            if new:
                settings["reasoning"] = new
            else:
                settings.pop("reasoning", None)
        if thinking in ("low", "medium", "high"):
            for m in modes:
                put(st, f"{m}ModeReasoningEffort", thinking, f"{m}ModeReasoningEffort")
    if json.dumps(settings, sort_keys=True) != before:
        entry["updatedAt"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    # the model's record (models.json): window, vision, and what the chat engine may use
    pentry = md.setdefault("providers", {}).setdefault(provider, {})
    pinfo = pentry.get("provider")
    if not isinstance(pinfo, dict):
        pinfo = pentry["provider"] = {"name": PROVIDERS[provider]}
    if base:
        # models.json keeps the OpenAI-style address (with /v1) for both providers
        api = server_root(base) + "/v1"
        put(pinfo, "baseUrl", api, f"models.json {provider}.provider.baseUrl")
        if "modelsSourceUrl" in pinfo:
            put(pinfo, "modelsSourceUrl", api + "/models", f"models.json {provider}.provider.modelsSourceUrl")
    put(pinfo, "defaultModelId", model, f"models.json {provider}.defaultModelId")
    record = pentry.setdefault("models", {}).setdefault(model, {"id": model, "name": model})
    if window:
        for field in ("contextWindow", "maxInputTokens"):
            put(record, field, int(window), f"models.json {provider}.{model}.{field}")
        legacy = LEGACY_CONTEXT_KEY.get(provider)
        if legacy:
            put(st, legacy, str(int(window)), legacy)
    if provider == "openai-compatible":
        caps = list(record.get("capabilities") or ["streaming", "tools"])
        for c in ("streaming", "tools"):
            if c not in caps:
                caps.append(c)
        if vision is True and "images" not in caps:
            caps.append("images")
        if vision is False and "images" in caps:
            caps.remove("images")
        put(record, "capabilities", caps, f"models.json {provider}.{model}.capabilities")
        # The settings page's "model info" for OpenAI Compatible (globalState), kept in step.
        for m in modes:
            info = st.get(f"{m}ModeOpenAiModelInfo")
            info = dict(info) if isinstance(info, dict) else {}
            new_info = dict(info)
            if window:
                new_info["contextWindow"] = int(window)
            if vision is not None:
                new_info["supportsImages"] = bool(vision)
            if new_info != info:
                changes.append(f"{m}ModeOpenAiModelInfo: context {info.get('contextWindow', '<unset>')} -> "
                               f"{new_info.get('contextWindow', '<unset>')}, images {info.get('supportsImages', '<unset>')} -> "
                               f"{new_info.get('supportsImages', '<unset>')}")
                st[f"{m}ModeOpenAiModelInfo"] = new_info
    else:
        if vision is not None:
            put(record, "supportsVision", bool(vision), f"models.json {provider}.{model}.supportsVision")
        # Cline offers a thinking choice only for a model marked as reasoning (LM Studio's records say false).
        put(record, "supportsReasoning", True, f"models.json {provider}.{model}.supportsReasoning")
    return changes


# ── command line ─────────────────────────────────────────────────────────
def ask(prompt: str) -> str:
    if not sys.stdin.isatty():
        return ""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def ask_provider(current: str) -> str | None:
    print("\n  which provider will Cline use?")
    ids = list(PROVIDERS)
    for i, p in enumerate(ids, 1):
        print(f"    {i}. {PROVIDERS[p]}{'   (in use now)' if p == current else ''}")
    raw = ask("  1 or 2 (Enter = the one in use, q = quit): ").lower()
    if raw in {"q", "quit", "exit"}:
        return None
    if raw in {"1", "2"}:
        return ids[int(raw) - 1]
    if raw in {"lm", "lmstudio"}:
        return "lmstudio"
    if raw in {"oi", "openai", "openai-compatible"}:
        return "openai-compatible"
    return current if current in PROVIDERS else "openai-compatible"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Point Cline at a gateway model, with its real context window, "
                                             "through LM Studio or OpenAI Compatible.")
    ap.add_argument("-p", "--provider", choices=list(PROVIDERS) + ["lm", "oi"],
                    help="lmstudio (lm) or openai-compatible (oi); asked when omitted")
    ap.add_argument("-m", "--model", help="model id to use; omit to choose from the gateway's list")
    ap.add_argument("--sync", action="store_true", help="use the model picked in Cline's settings page")
    ap.add_argument("--gateway", help=f"gateway address (default: what Cline has, else {DEFAULT_GATEWAY})")
    ap.add_argument("--key", default=os.environ.get("GATEWAY_KEY", ""),
                    help="gateway access key, only to list its models (or GATEWAY_KEY)")
    ap.add_argument("--activate", action="store_true", help="also make this the provider Cline uses")
    ap.add_argument("--thinking", choices=THINKING_CHOICES, help="thinking level to save for this provider")
    ap.add_argument("--mode", choices=["plan", "act", "both"], default="both")
    ap.add_argument("--context", type=int, help="context window to write (default: what the gateway reports)")
    ap.add_argument("--no-context", action="store_true", help="leave the context window alone")
    ap.add_argument("--cline-dir", default=str(default_cline_dir()))
    ap.add_argument("--check", action="store_true", help="only report (exit 1 on a mismatch)")
    ap.add_argument("--list", action="store_true", help="list the gateway's models, then exit")
    ap.add_argument("--revert", action="store_true", help="restore the newest backup of each file")
    ap.add_argument("--no-generic", action="store_true", help="do not also align <mode>ModeApiModelId")
    args = ap.parse_args(argv)

    if args.revert:
        done = revert_latest(Path(args.cline_dir))
        print("\n".join(done) if done else "no backups found")
        return 0 if done else 1
    try:
        files = ClineFiles(Path(args.cline_dir))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    provider = {"lm": "lmstudio", "oi": "openai-compatible"}.get(args.provider or "", args.provider or "")
    if not provider:
        if args.check or args.list or args.sync or args.model or not sys.stdin.isatty():
            provider = files.current_provider()
            if provider not in PROVIDERS:
                provider = "openai-compatible"
        else:
            provider = ask_provider(files.current_provider())
            if not provider:
                print("aborted - nothing changed.")
                return 1
    modes = ["plan", "act"] if args.mode == "both" else [args.mode]

    print()
    print("\n".join(describe(files, provider, modes)))
    wrong = mismatches(files, provider, modes)
    if wrong:
        print("\n  what is wrong now:")
        for line in wrong:
            print("    - " + line)
    if args.check:
        print("\n--check: nothing written.")
        return 1 if wrong else 0

    root = server_root(args.gateway) if args.gateway else gateway_root_guess(files)
    models, listed = [], None
    try:
        listed, models = fetch_models(root, key=args.key)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  could not list models at {root} : {exc}")
    if args.list:
        print(f"\n  {len(models)} model(s) at {root}:")
        for i, m in enumerate(models, 1):
            print(f"    {i:>2}. {model_line(m)}")
        return 0

    model = (args.model or "").strip()
    if not model and args.sync:
        model = str(provider_settings(files.providers, provider).get("model") or "").strip()
        if not model:
            print("--sync: the settings page has no model picked for this provider - nothing changed.")
            return 1
    if not model:
        print(f"\n  models at {root}:")
        for i, m in enumerate(models, 1):
            print(f"    {i:>2}. {model_line(m)}")
        raw = ask("  number, model id, or 'q' to quit: ")
        if not raw or raw.lower() in {"q", "quit", "exit"}:
            print("aborted - nothing changed.")
            return 1
        model = models[int(raw) - 1]["id"] if raw.isdigit() and 1 <= int(raw) <= len(models) else raw

    known = next((m for m in models if m["id"] == model), None) or \
        next((m for m in models if m["id"].lower() == model.lower()), None)
    if known:
        model = known["id"]
    else:
        print(f"  WARNING: {model!r} is not listed by the gateway; writing it anyway (--revert undoes it).")
    window = None if args.no_context else (args.context or (known or {}).get("context"))
    if not args.no_context and not window:
        print("  NOTE: the gateway gave no context window for this model; pass --context to set one.")
    thinking = args.thinking
    if thinking is None and sys.stdin.isatty() and not args.model and not args.sync:
        raw = ask(f"  thinking ({'/'.join(THINKING_CHOICES)}; Enter = keep {current_thinking(files, provider)}): ").lower()
        thinking = raw if raw in THINKING_CHOICES else None
    activate = args.activate
    if not activate and files.current_provider() != provider and sys.stdin.isatty() and not args.model:
        activate = ask(f"  make {PROVIDERS[provider]} the provider Cline uses? [Y/n]: ").lower() in {"", "y", "yes"}

    changes = apply_choice(files, provider, model, window=window, vision=(known or {}).get("vision"),
                           modes=modes, gateway_root=root if (args.gateway or listed) else None,
                           activate=activate, thinking=thinking, generic=not args.no_generic)
    if not changes:
        print("\nnothing to change: Cline already has these settings.")
        return 0
    for b in files.save():
        print("  backup  :", b)
    print("\napplied:")
    for line in changes:
        print("  " + line)
    running = vscode_running()
    if running:
        print(f"\nVS Code is running: restart it, or Cline may keep (and rewrite) the old settings.")
    else:
        print("\nStart VS Code and open a NEW chat; the header shows the model and context in use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
