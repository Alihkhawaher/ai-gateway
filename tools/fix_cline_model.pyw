"""Cline model setup, in a window. Double-click to run (pythonw).

Pick LM Studio or OpenAI Compatible, the gateway's model, its context window and the
thinking level; Apply backs up Cline's three settings files and writes them.

This file stands alone: it needs no other file. fix_cline_model.py is the same tool for
the command line and carries its own copy of the logic below; a fix to one belongs in both.
(Why the settings live where they do is explained at the top of fix_cline_model.py.)
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

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


# ── the window ───────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cline model setup")
        self.minsize(720, 560)
        self.models = []
        self.files = None

        self.cline_dir = tk.StringVar(value=str(default_cline_dir()))
        self.provider = tk.StringVar(value="openai-compatible")
        self.gateway = tk.StringVar(value=DEFAULT_GATEWAY)
        self.key = tk.StringVar()
        self.context = tk.StringVar()
        self.vision = tk.StringVar(value="as reported")
        self.thinking = tk.StringVar(value="keep")
        self.mode = tk.StringVar(value="both")
        self.activate = tk.BooleanVar(value=True)

        pad = {"padx": 8, "pady": 4}
        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Cline settings folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.cline_dir).grid(row=0, column=1, columnspan=3, sticky="ew")

        ttk.Label(top, text="Provider").grid(row=1, column=0, sticky="w")
        pf = ttk.Frame(top)
        pf.grid(row=1, column=1, columnspan=3, sticky="w")
        for pid, label in PROVIDERS.items():
            ttk.Radiobutton(pf, text=label, value=pid, variable=self.provider,
                            command=self.show_current).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(pf, text="make it the provider Cline uses", variable=self.activate).pack(side="left")

        ttk.Label(top, text="Gateway").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.gateway).grid(row=2, column=1, sticky="ew")
        ttk.Label(top, text="Access key").grid(row=2, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.key, show="•", width=18).grid(row=2, column=3, sticky="w")

        bar = ttk.Frame(self)
        bar.pack(fill="x", **pad)
        ttk.Button(bar, text="Load models", command=self.load_models).pack(side="left")
        ttk.Label(bar, text="  (the access key is only used to list models, and is not saved)").pack(side="left")

        lf = ttk.LabelFrame(self, text="Gateway models")
        lf.pack(fill="both", expand=True, **pad)
        cols = ("model", "context", "vision", "state")
        self.table = ttk.Treeview(lf, columns=cols, show="headings", height=8, selectmode="browse")
        for c, w in zip(cols, (330, 110, 70, 90)):
            self.table.heading(c, text=c.title())
            self.table.column(c, width=w, anchor="w" if c == "model" else "center")
        self.table.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.table.yview)
        sb.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=sb.set)
        self.table.bind("<<TreeviewSelect>>", self.on_pick)

        opts = ttk.Frame(self)
        opts.pack(fill="x", **pad)
        ttk.Label(opts, text="Context window").pack(side="left")
        ttk.Entry(opts, textvariable=self.context, width=10).pack(side="left", padx=(4, 16))
        ttk.Label(opts, text="Vision").pack(side="left")
        ttk.Combobox(opts, textvariable=self.vision, width=11, state="readonly",
                     values=("as reported", "yes", "no")).pack(side="left", padx=(4, 16))
        ttk.Label(opts, text="Thinking").pack(side="left")
        ttk.Combobox(opts, textvariable=self.thinking, width=9, state="readonly",
                     values=("keep",) + THINKING_CHOICES).pack(side="left", padx=(4, 16))
        ttk.Label(opts, text="Modes").pack(side="left")
        ttk.Combobox(opts, textvariable=self.mode, width=6, state="readonly",
                     values=("both", "act", "plan")).pack(side="left", padx=4)

        acts = ttk.Frame(self)
        acts.pack(fill="x", **pad)
        ttk.Button(acts, text="Apply", command=self.apply).pack(side="left")
        ttk.Button(acts, text="Show current", command=self.show_current).pack(side="left", padx=6)
        ttk.Button(acts, text="Undo last apply", command=self.revert).pack(side="left")

        self.log = tk.Text(self, height=12, wrap="word", font=("Consolas", 9))
        self.log.pack(fill="both", expand=False, **pad)

        self.after(50, self.startup)

    # ── helpers ──────────────────────────────────────────────────────
    def say(self, text: str = "", clear: bool = False):
        if clear:
            self.log.delete("1.0", "end")
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def modes(self):
        return ["plan", "act"] if self.mode.get() == "both" else [self.mode.get()]

    def read_files(self):
        try:
            self.files = ClineFiles(Path(self.cline_dir.get()))
            return True
        except FileNotFoundError as exc:
            self.say(str(exc), clear=True)
            self.files = None
            return False
        except Exception as exc:  # noqa: BLE001
            self.say(f"could not read Cline's settings: {exc}", clear=True)
            self.files = None
            return False

    def startup(self):
        if not self.read_files():
            return
        cur = self.files.current_provider()
        if cur in PROVIDERS:
            self.provider.set(cur)
        self.gateway.set(gateway_root_guess(self.files))
        self.show_current()
        self.load_models()

    def show_current(self):
        if not self.read_files():
            return
        p = self.provider.get()
        self.say("\n".join(describe(self.files, p, self.modes())), clear=True)
        for line in mismatches(self.files, p, self.modes()):
            self.say("  ! " + line)

    # ── models ───────────────────────────────────────────────────────
    def load_models(self):
        root = self.gateway.get().strip()
        key = self.key.get().strip()
        self.say(f"\nloading models from {root} …")

        results = queue.Queue()

        def work():  # never touches the window: tkinter is not thread-safe
            try:
                results.put((fetch_models(root, key=key)[1], None))
            except Exception as exc:  # noqa: BLE001
                results.put(([], exc))

        def poll():
            try:
                models, error = results.get_nowait()
            except queue.Empty:
                self.after(100, poll)
                return
            self.fill_models(models, error)

        threading.Thread(target=work, daemon=True).start()
        self.after(100, poll)

    def fill_models(self, models, error):
        self.table.delete(*self.table.get_children())
        self.models = models
        if error:
            self.say(f"could not list models: {error}")
            return
        chosen = ""
        if self.files:
            p = self.provider.get()
            chosen = (self.files.state.get(legacy_model_keys(p, ["act"])[0])
                      or provider_settings(self.files.providers, p).get("model") or "")
        for m in models:
            iid = self.table.insert("", "end", values=(
                m["id"], f"{m['context']:,}" if m.get("context") else "?",
                {True: "yes", False: "no"}.get(m.get("vision"), "?"), m.get("state") or ""))
            if m["id"] == chosen:
                self.table.selection_set(iid)
                self.table.see(iid)
        self.say(f"{len(models)} model(s) listed.")

    def picked(self):
        sel = self.table.selection()
        if not sel:
            return None
        mid = self.table.item(sel[0], "values")[0]
        return next((m for m in self.models if m["id"] == mid), None)

    def on_pick(self, _event=None):
        m = self.picked()
        if m:
            self.context.set(str(m["context"]) if m.get("context") else "")

    # ── actions ──────────────────────────────────────────────────────
    def apply(self):
        m = self.picked()
        if not m:
            messagebox.showinfo("Cline model setup", "Pick a model first (Load models fills the list).")
            return
        raw = self.context.get().replace(",", "").strip()
        if raw and not raw.isdigit():
            messagebox.showerror("Cline model setup", "The context window must be a number of tokens.")
            return
        if not self.read_files():
            return
        vision = {"yes": True, "no": False}.get(self.vision.get(), m.get("vision"))
        thinking = None if self.thinking.get() == "keep" else self.thinking.get()
        p = self.provider.get()
        try:
            changes = apply_choice(self.files, p, m["id"], window=int(raw) if raw else None,
                                        vision=vision, modes=self.modes(), gateway_root=self.gateway.get(),
                                        activate=self.activate.get(), thinking=thinking)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cline model setup", str(exc))
            return
        if not changes:
            self.say("\nnothing to change: Cline already has these settings.")
            return
        try:
            backups = self.files.save()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cline model setup", f"could not write: {exc}")
            return
        self.say(f"\napplied ({PROVIDERS[p]}, {m['id']}):", clear=True)
        for line in changes:
            self.say("  " + line)
        self.say(f"backups: {len(backups)} file(s), '.bak-' next to each (Undo last apply restores them)")
        restart = "VS Code is running: restart it so Cline reads the new settings." if vscode_running() \
            else "Start VS Code and open a new chat."
        self.say(restart)
        messagebox.showinfo("Cline model setup", f"Saved.\n\n{restart}")

    def revert(self):
        if not messagebox.askyesno("Cline model setup", "Put back the newest backup of Cline's settings files?"):
            return
        done = revert_latest(Path(self.cline_dir.get()))
        self.say("\n".join(done) if done else "no backups found", clear=True)


if __name__ == "__main__":
    App().mainloop()
