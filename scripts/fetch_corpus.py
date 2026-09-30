#!/usr/bin/env python3
"""Phase 0 — pull a curated set of Kubernetes documentation pages across
several release versions from the kubernetes/website GitHub repo.

This is a *curated* manifest, not a full doc crawl, and that's deliberate:
every page in it was picked because it has a real, citable difference across
the chosen versions (an API group promotion, a resource removal, a default
change). That's what makes version-sensitive questions in the golden eval set
possible to write with a known ground truth. A full crawl would be easy to
add later (Phase 2+ if retrieval quality on the curated set looks solid and
more coverage is needed) but would dilute the corpus with pages that carry no
version signal at all.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k8s_rag.config import RunConfig, REPO_ROOT  # noqa: E402

RAW_BASE = "https://raw.githubusercontent.com/{repo}/{branch}/content/en/docs/{page}.md"

# (page path relative to content/en/docs/, why it's here)
CURATED_PAGES = [
    ("concepts/services-networking/ingress",
     "Ingress API moved extensions/v1beta1 -> networking.k8s.io/v1beta1 -> networking.k8s.io/v1 (required from 1.22)"),
    ("concepts/workloads/controllers/deployment",
     "Deployment API group history (extensions/v1beta1 -> apps/v1)"),
    ("concepts/workloads/controllers/cron-jobs",
     "CronJob: batch/v1beta1 default pre-1.21, batch/v1 default from 1.21, v1beta1 removed in 1.25"),
    ("concepts/security/pod-security-policy",
     "PodSecurityPolicy: deprecated in 1.21, REMOVED in 1.25 (page itself disappears/redirects post-1.25)"),
    ("concepts/security/pod-security-standards",
     "Pod Security Admission: the 1.25+ replacement for PodSecurityPolicy"),
    ("concepts/workloads/pods/disruptions",
     "PodDisruptionBudget: policy/v1beta1 -> policy/v1 (v1 since 1.21, v1beta1 removed in 1.25)"),
    ("tasks/run-application/horizontal-pod-autoscale",
     "HorizontalPodAutoscaler: autoscaling/v2beta2 -> autoscaling/v2 (GA in 1.23)"),
    ("setup/production-environment/container-runtimes",
     "Dockershim removed in 1.24 -- container runtime setup instructions changed materially"),
    ("concepts/services-networking/dual-stack",
     "IPv4/IPv6 dual-stack: alpha pre-1.21, beta 1.21-1.22, stable from 1.23"),
    ("reference/using-api/deprecation-guide",
     "The canonical deprecation-guide page itself -- its content changes every release"),
]


def fetch_page(repo: str, branch: str, page: str, retries: int = 4) -> str | None:
    url = RAW_BASE.format(repo=repo, branch=branch, page=page)
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                return None
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        except requests.exceptions.RequestException as exc:
            last_exc = exc
        time.sleep(1.5 * (attempt + 1))
    print(f"    [warn] giving up on {page} after {retries} retries: {last_exc}")
    return None


def main() -> None:
    cfg = RunConfig.load()
    repo = cfg.get("corpus", "source_repo")
    branch_template = cfg.get("corpus", "branch_template")
    versions = cfg.get("corpus", "versions")

    corpus_dir = REPO_ROOT / "data" / "corpus"
    manifest: list[dict] = []

    for version in versions:
        branch = branch_template.format(version=version)
        out_dir = corpus_dir / version
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n== version {version} (branch {branch}) ==")
        for page, reason in CURATED_PAGES:
            text = fetch_page(repo, branch, page)
            time.sleep(0.2)  # be polite to raw.githubusercontent.com
            slug = page.replace("/", "__")
            if text is None:
                print(f"  [skip] {page} (not present at this version -- itself a signal, logged)")
                manifest.append({
                    "version": version, "page": page, "reason": reason,
                    "status": "missing", "file": None,
                })
                continue
            out_path = out_dir / f"{slug}.md"
            front_matter = (
                f"---\nk8s_version: \"{version}\"\n"
                f"source_page: \"{page}\"\n"
                f"source_url: \"{RAW_BASE.format(repo=repo, branch=branch, page=page)}\"\n"
                f"why_curated: \"{reason}\"\n---\n\n"
            )
            out_path.write_text(front_matter + text)
            print(f"  [ok]   {page} -> {out_path.relative_to(REPO_ROOT)} ({len(text)} chars)")
            manifest.append({
                "version": version, "page": page, "reason": reason,
                "status": "ok", "file": str(out_path.relative_to(REPO_ROOT)),
            })

    manifest_path = corpus_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"\nFetched {ok}/{len(manifest)} page-versions. Manifest -> {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
