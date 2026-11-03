#!/usr/bin/env python3
"""Upload and round-trip verify the historical Algernon sandbox archive.

Requires: pip install huggingface_hub
Credentials: hf auth login OR HF_TOKEN in environment, with write permission.
Never prints token; refuses public datasets.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

EXPECTED_SHA256 = "88d8e23581f0445aca9ac7b7b9a9c7f8d7df34f00bbc5126fced84511053f7b4"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path, help="Path to algernon_claude_handoff.zip")
    p.add_argument("--repo", default="AlephFunk/algernon-hm-sandbox-v1")
    p.add_argument("--revision", default="main")
    p.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    args = p.parse_args()
    if not args.archive.is_file():
        p.error(f"archive missing: {args.archive}")
    digest = sha256(args.archive)
    if digest != args.expected_sha256:
        p.error(f"SHA-256 mismatch: got {digest}, expected {args.expected_sha256}")
    api = HfApi()
    info = None
    try:
        info = api.repo_info(repo_id=args.repo, repo_type="dataset")
    except Exception as e:
        # Let create_repo check rights or whether the repo already exists.
        print(f"Dataset lookup did not return metadata: {type(e).__name__}")
    if info is not None and not getattr(info, "private", False):
        p.error("Existing target dataset is public; refusing upload")
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id=args.repo, repo_type="dataset")
    if not getattr(info, "private", False):
        p.error("Target dataset could not be verified private")
    path_in_repo = "archives/algernon_claude_handoff.zip"
    commit = api.upload_file(
        path_or_fileobj=str(args.archive),
        path_in_repo=path_in_repo,
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        commit_message="Archive reproducible H-M0/H-M1 CPU experiments",
    )
    verified = hf_hub_download(
        repo_id=args.repo,
        filename=path_in_repo,
        repo_type="dataset",
        revision=commit.oid,
        force_download=True,
    )
    downloaded_digest = sha256(Path(verified))
    if downloaded_digest != digest:
        raise RuntimeError(f"Hub round-trip hash mismatch: {downloaded_digest}")
    print(f"Verified private dataset: https://huggingface.co/datasets/{args.repo}/tree/{commit.oid}")
    print(f"Archived SHA-256: {digest}")
    print("H-M2 is NOT contained in this archive and is not independently reproducible.")

if __name__ == "__main__":
    main()
