# DISTILL-12A / free Inky longitudinal sampling calibration

Original 4096-generation / 0.6-effort runs failed 8/8 trajectories due to
unclean termination. Preserve their journals and strict results. Fresh
calibration uses a paired factorial of efforts .15/.25/.4 and generation
caps 4096/8192, with a fresh legacy .6/4096 control. Four frontier program
roots (two ordinary, two shifted), 28 arms. Each root shares a private world
draw across all seven profiles. No feedback or evaluator labels are exposed
during solving. This is development calibration, not confirmation.

    python -m Distill.inky_program_calibration prepare --out runs/inky-calibration12
    python -m Distill.program_runner run --campaign runs/inky-calibration12/shared_control-0/g4096-e15 --confirm-free-tier --limit 1

Use the actual free provider and confirm identity/billing before further
sampling. Then run each registered arm (limit <=32 per invocation) until
terminal. Unclean, invalid or pending responses must remain in their
original journals; do not retry an arm or silently repair its response.
The host must reconcile ambiguous pending calls. A clean but wrong JSON
action is a failed arm. No paid fallback or context expansion. Both profiles
retain the 12,288 rendered context cap and 16,384 hard cap, including the
reserved generation ceiling; an 8192-generation profile can fail context
preflight if its input grows beyond 4096 tokens.

    python -m Distill.inky_program_calibration report --campaign runs/inky-calibration12

The report withholds utility and oracle labels until all arms complete or
fail with a durable invalid receipt. It counts all recorded output tokens,
including failures, and reports accepted decisions per 1000 output tokens
when receipt coverage is complete. Incomplete realized utility is censored,
not replaced with zero. Conditional utility among completed arms and
accepted-prefix regret are descriptive; the primary endpoint is completion
over matched roots. Inspect failures and model efficiency before choosing
a profile. Freeze a second fresh seed for confirmation after selection.

Non-default registered profiles use distill.program_run.v2. The legacy
profile keeps its v1 manifest and replay semantics. Fixtures use the same
journal and never invoke the provider. The operator must verify actual
provider bills and keep all raw journals private. All outputs remain
reference-only.
