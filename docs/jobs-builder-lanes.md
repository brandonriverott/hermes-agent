# Jobs Builder Lanes

This guide provisions the non-secret directory boundary for the twelve-seat
Jobs builder fleet. A declared or provisioned lane is not ready capacity:
health must still prove an isolated login, protected signer, supported
executor, resources, Git worktree support, and no active lease.

## Roots and policy

Configure roots in `config.yaml` when `--root` is not supplied:

```yaml
jobs:
  lanes:
    roots:
      mac: /Users/brandon/jobs/lanes
      pc: /home/brandon/jobs/lanes
```

The versioned `jobs-lanes.v1` policy declares three Claude and three Codex
seats on each host. It authorizes a matching Mac lane only when PC capacity is
unavailable for an eligible reason. Executor and model never change during
that fallback.

## Provision the Mac structure

Run a dry run and review all six exact lane IDs before applying:

```bash
scripts/jobs_provision_lanes.py \
  --host mac \
  --root /Users/brandon/jobs/lanes \
  --dry-run
```

The root must be an existing operator-owned directory. Apply only when the
dry-run list is exact:

```bash
scripts/jobs_provision_lanes.py \
  --host mac \
  --root /Users/brandon/jobs/lanes \
  --apply
```

Provisioning creates private lane directories, an isolated `auth/` home, a
distinct receipt-signing key, and non-secret lane config. It does not create
Claude or Codex credentials and never replaces an existing signing key.

## Mandatory interactive login checkpoint

Run these six login flows one at a time. Each command must complete its own
interactive browser or device authorization:

```bash
CLAUDE_CONFIG_DIR=/Users/brandon/jobs/lanes/claude-mac-1/auth claude auth login
CLAUDE_CONFIG_DIR=/Users/brandon/jobs/lanes/claude-mac-2/auth claude auth login
CLAUDE_CONFIG_DIR=/Users/brandon/jobs/lanes/claude-mac-3/auth claude auth login
CODEX_HOME=/Users/brandon/jobs/lanes/codex-mac-1/auth codex login --device-auth
CODEX_HOME=/Users/brandon/jobs/lanes/codex-mac-2/auth codex login --device-auth
CODEX_HOME=/Users/brandon/jobs/lanes/codex-mac-3/auth codex login --device-auth
```

Never copy an auth directory, OAuth token, cookie, session file, API key, or
credential-store entry between lanes. Until each interactive flow succeeds,
that lane remains `AUTH_REQUIRED` and contributes no capacity.

## Health and canary gate

Run sanitized health after the login checkpoint:

```bash
scripts/jobs_lane_health.py \
  --host mac \
  --root /Users/brandon/jobs/lanes \
  --json
```

Exit `0` means every requested lane is fresh `PASS`; exit `2` means at least
one lane is truthfully blocked, unprovisioned, active, failed, or awaiting
auth. Private executor output is never printed.

Run a canary only on a fresh `PASS/IDLE` lane. Its bounded goal is:

```text
Create canary.txt in the assigned temporary worktree containing exactly:
hermes-lane-canary-20260809
Read the file back, report its SHA-256, and make no network or external-system changes.
```

Require artifact readback, a valid signed receipt, legal graph transitions,
registered-worktree cleanup, return to `IDLE`, and one claim/release cycle.
No provider, production repository, live Jobs DB, or auth directory belongs in
this canary.

## PC checkpoint

Do not provision `/home/brandon/jobs/lanes` remotely while SSH authorization
fails. Tailscale reachability alone is insufficient. Record all six PC seats
as `BLOCKED/AUTH_INFRA/SSH_AUTH_FAILED`, repair SSH through an operator-reviewed
channel, then repeat dry run, apply, isolated logins, health, and canaries on
the PC itself.

## Rollback inventory

Rollback starts with a fresh read-only inventory:

```bash
scripts/jobs_provision_lanes.py \
  --host mac \
  --root /Users/brandon/jobs/lanes \
  --mode rollback \
  --dry-run
```

The command removes nothing. It lists only existing lane directories whose
identity, policy digest, public key, protected private key, and layout validate.
After confirming no active lease or worktree, remove or quarantine only the
exact listed directories:

```text
/Users/brandon/jobs/lanes/claude-mac-1
/Users/brandon/jobs/lanes/claude-mac-2
/Users/brandon/jobs/lanes/claude-mac-3
/Users/brandon/jobs/lanes/codex-mac-1
/Users/brandon/jobs/lanes/codex-mac-2
/Users/brandon/jobs/lanes/codex-mac-3
```

Never use the lane root, a home directory, a glob, a symlink, or an unresolved
environment variable as a rollback target. Attempt cleanup must never remove
`auth/`; whole-lane rollback is a separate operator action after custody is
empty.
