# H-M0/H-M1 sandbox handoff and evidence status

This branch exists to bridge the independently executed ChatGPT CPU sandbox experiments into the Algernon research workflow. Based on Algernon PR #33; **no H-M2 results are verified**.

## Artifact custody

The complete archive is named `algernon_claude_handoff.zip`, SHA-256 `88d8e23581f0445aca9ac7b7b9a9c7f8d7df34f00bbc5126fced84511053f7b4`. It contains the original `algernon_hm1_revision_sandbox.zip` payload, whose SHA-256 is `68954e95d8b4c50ac480e22c0ab085e8deaa6f12f959ebc4fc24158df860ac22`, plus a handoff note. Source-only reproducibility archive `algernon_hm_source.tar.gz` has SHA-256 `432b45e513d00eef3b83d41ef1407074d763aa35b373daa5555298b1cc685578`.

**Current GitHub status:** This commit contains the custody manifest and reproduction instructions. The original ZIP and its binaries/raw records have **not** been uploaded to GitHub. They are attached to the ChatGPT conversation as sandbox downloads. Do not claim the data are hosted here until a separate upload has been hash-verified.

## Verified scope

- H-M0 baseline, terminal-utility, and stopping-comparison code and records exist in the original ZIP.
- H-M1 bounded grammar expansion code, trained-policy checkpoint, held-out and confirmation records, metrics, reports, and tests exist in the ZIP.
- The extracted archive's registry validator passed; 5 registry tests and 27 experimental tests passed on CPU.
- These checks establish executable contract tests, not an independent retraining or numerical reproduction.
- H-M2 was described in conversation, but no H-M2 code, manifest, checkpoint or records are available. Its reported numerical scores are **unverified**.
- There is no execution receipt for the actual Qwenputer-4B compiled processor or the Qwen3.5-0.8B recurrent graft.

## Download-to-checkout procedure

Download the attached `algernon_claude_handoff.zip` from the ChatGPT conversation, verify its SHA-256 above, extract the inner archive, and copy the `experiments/` directory plus provenance reports into a local scratch branch. Keep the raw records and model weights in an ignored or release-artifact location rather than the main Git tree.

On Windows PowerShell:

```powershell
$archive = Join-Path $HOME 'Documents\Codex\algernon_claude_handoff.zip'
(Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $HOME 'Documents\Codex\algernon_handoff') -Force
# Inspect README within extracted archive for the inner ZIP and expand it.
cd (Join-Path $HOME 'Documents\Codex\algernon_handoff\algernon_sandbox')
python -m research.algernon.validate_register
python -m unittest discover -s research/algernon/tests -v
python -m unittest discover -s experiments/tests -v
```

Compare hashes, do not mix experiments with the four unrun H-C/H-R/H-W/H-J hypotheses, and retain held-out splits. Import source into the research tree only after checks pass.

## Next publication decision

Place the experimental implementation and small receipts in `research/algernon/metacognition/` under a dedicated PR, with raw larger records in a hash-addressed artifact store. A university exploratory paper can cite these after independent reruns, full source/seed inspection, and protocol review. H-M2 must be rerun under a newly frozen specification and named as a new experiment.
