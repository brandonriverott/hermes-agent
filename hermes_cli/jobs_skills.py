"""Skill attachment for Jobs — what reference material a builder gets, and how.

A Job may *declare* the skills its builder needs; when it declares none, a small
deterministic rules table picks some from the Job's own words. Either way the
selected ``SKILL.md`` files are copied into the attempt's isolated worktree as a
read-only ``.cc-skill-attachments/`` directory, and taken out again before the
attempt's diff is read — so the builder can open them and no reviewer ever has
to look at them.

The module is fail-closed in one direction and forgiving in the other, and the
asymmetry is the whole design:

- A **declared** skill is an instruction. A name that traverses, that lands
  outside every store this machine keeps skills in, that names no file, or that
  blows a cap raises :class:`SkillAttachmentError` — before a worktree exists and
  before a provider is paid — because running the work *without* the material
  somebody asked for is not the work they asked for.
- A **default** skill is a guess this module made. One that is not installed on
  this machine is skipped and recorded, never raised: nobody asked for it, so it
  has no business failing a run.

Three properties the rest of Jobs relies on:

- **Nothing is read that the name did not name.** Every candidate is opened with
  ``O_NOFOLLOW``, and its real path re-checked against the root it was found
  under plus the shared stores that root's links are allowed to point into — so
  a symlinked ``SKILL.md`` is refused outright, and a symlinked parent directory
  can only land somewhere skills are kept, never elsewhere on the host.
- **What is staged is what was read.** Each file is hashed at the source and the
  staged copy is hashed again and compared, so a truncated or half-written
  attachment is a refusal rather than reference material with a hole in it.
- **Attachments are never deliverables.** :func:`strip` removes them before the
  adapter reads the diff, and :func:`is_attachment_path` lets the adapter
  discount any that a builder committed anyway.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from hermes_constants import get_optional_skills_dir, get_skills_dir

# The directory attachments live in *inside the worktree*. Named for the old
# Kanban pipeline's directory so an operator who has seen one recognises the
# other, and dot-prefixed so it stays out of casual listings.
ATTACHMENT_DIR = ".cc-skill-attachments"

# The environment variable the worker is told the attachment directory through.
# The goal text is stored verbatim and must never be rewritten to mention it, and
# the worker's argv is a fixed contract, so an environment variable is the one
# channel left that adds a fact without changing either.
SKILLS_ENV = "HERMES_JOB_SKILLS"

# Where a skills root's symlinks are allowed to land — see
# :func:`shared_skill_stores`. The default is where the shared-skills installer
# keeps its one canonical copy of each skill; ``os.pathsep``-separated for a
# machine with more than one store, so a different layout is configuration
# rather than a patch.
SHARED_SKILLS_ENV = "HERMES_SHARED_SKILLS_DIR"
_SHARED_SKILL_STORES: Tuple[str, ...] = (
    "~/.local/share/shared-skills/canonical",
)

# Ceilings. A declaration over any of them is refused rather than trimmed: a
# builder that silently got four of the six skills somebody asked for looks like
# it got all six.
MAX_SKILLS = 6
MAX_SKILL_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 256 * 1024
MAX_NAME_CHARS = 120
MAX_NAME_SEGMENTS = 4

# Defaults are a guess, so they get a tighter budget than a declaration: three
# files is enough to change how a builder approaches a work type, and past that
# the attachment starts competing with the goal for the builder's attention.
MAX_DEFAULT_SKILLS = 3

# A path segment: starts alphanumeric (so ``.``/``..`` cannot be one), then the
# characters real skill directories actually use.
_SEGMENT_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")

_FILENAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

_WORD_RE = re.compile(r"[^a-z0-9]+")


class SkillAttachmentError(ValueError):
    """A declared skill cannot be honoured, so the run must not start.

    A ``ValueError``, so the store and the CLI already handle it on the paths
    that validate a declaration at the write seam.
    """


# ---------------------------------------------------------------------------
# Declared names
# ---------------------------------------------------------------------------


def _validate_name(name: str) -> str:
    """Return ``name`` unchanged, or raise. The only gate on what may be looked up.

    Rejects everything that could make a name mean a path rather than a skill:
    an absolute path, a parent-directory hop, a backslash, whitespace, a NUL, an
    empty segment, or a name so long or so deep that it is describing a
    filesystem rather than naming a skill.
    """
    if not isinstance(name, str):
        raise SkillAttachmentError(
            f"skill names must be strings, got {type(name).__name__}"
        )
    if len(name) > MAX_NAME_CHARS:
        raise SkillAttachmentError(
            f"skill name is {len(name)} characters, over the {MAX_NAME_CHARS} cap"
        )
    if name.startswith("/"):
        raise SkillAttachmentError(
            f"skill name {name!r} is an absolute path; name a skill, not a file"
        )
    segments = name.split("/")
    if len(segments) > MAX_NAME_SEGMENTS:
        raise SkillAttachmentError(
            f"skill name {name!r} is more than {MAX_NAME_SEGMENTS} segments deep"
        )
    for segment in segments:
        if not _SEGMENT_RE.match(segment):
            raise SkillAttachmentError(
                f"skill name {name!r} has an unusable path segment {segment!r}"
            )
    return name


def parse_declared_skills(raw) -> Tuple[str, ...]:
    """Normalize a declaration into an ordered, de-duplicated tuple of names.

    Accepts what the surfaces actually produce: ``None``, one comma-separated
    string (the spelling ``hermes --skills`` has always taken), or a list of
    either. Order is the declaration's own — a builder reads the first skill
    first — and a repeat is dropped rather than attached twice.

    Every name is validated here, at the *declaration*, not at the read: that is
    what makes ``hermes jobs create`` refuse a traversal attempt seconds after a
    human typed it instead of minutes into a run nobody is watching.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: List = [raw]
    else:
        try:
            items = list(raw)
        except TypeError:
            raise SkillAttachmentError(
                "declared skills must be a name, a list of names, or None"
            )

    names: List[str] = []
    for item in items:
        if item is None:
            continue
        if not isinstance(item, str):
            raise SkillAttachmentError(
                f"skill names must be strings, got {type(item).__name__}"
            )
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            _validate_name(part)
            if part not in names:
                names.append(part)
    if len(names) > MAX_SKILLS:
        raise SkillAttachmentError(
            f"{len(names)} skills declared, over the {MAX_SKILLS} cap"
        )
    return tuple(names)


# ---------------------------------------------------------------------------
# Default selection
# ---------------------------------------------------------------------------

# Keyword -> skills, in priority order. Deliberately small, deliberately boring,
# and deliberately a table rather than a model call: the same Job text has to
# select the same skills on every machine, forever, or the attachment stops being
# evidence about the run. Keywords are matched as whole words against a
# punctuation-flattened copy of the text, so ``latest`` does not read as ``test``
# and ``scroll-driven`` does read as ``scroll driven``.
SKILL_DEFAULT_RULES: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (
        ("security", "vulnerability", "vulnerabilities", "exploit", "pentest",
         "cve", "xss", "csrf", "sql injection", "threat model", "hardening"),
        ("security/web-pentest", "security/oss-forensics"),
    ),
    (
        ("review", "reviewer", "code review", "pull request"),
        ("github/github-code-review",
         "software-development/requesting-code-review"),
    ),
    (
        ("test", "tests", "testing", "pytest", "regression", "regressions",
         "coverage", "tdd", "flaky"),
        ("software-development/test-driven-development",),
    ),
    (
        ("bug", "bugs", "debug", "debugging", "traceback", "stack trace",
         "crash", "repro"),
        ("software-development/systematic-debugging",),
    ),
    (
        ("gsap", "motion", "animation", "animations", "animate",
         "scroll driven", "keyframe", "keyframes", "easing"),
        ("creative/hyperframes", "creative/pretext"),
    ),
    (
        ("design", "visual", "ui", "ux", "frontend", "css", "tailwind",
         "layout", "typography"),
        ("creative/claude-design", "creative/design-md"),
    ),
)


def _haystack(text) -> str:
    """``text`` lowercased, punctuation flattened to spaces, space-delimited.

    The leading and trailing space are what make ``f" {keyword} " in haystack``
    a whole-word match without a regex per keyword.
    """
    flattened = _WORD_RE.sub(" ", str(text or "").lower()).strip()
    return f" {flattened} "


def select_default_skills(text) -> Tuple[str, ...]:
    """The skills this Job's own words ask for. Pure, total, deterministic.

    Every matching rule contributes, in table order, up to
    :data:`MAX_DEFAULT_SKILLS`. Nothing here checks whether a skill exists — that
    is :func:`resolve`'s job, and it is lenient about defaults on purpose.
    """
    haystack = _haystack(text)
    chosen: List[str] = []
    for keywords, skills in SKILL_DEFAULT_RULES:
        if not any(f" {keyword} " in haystack for keyword in keywords):
            continue
        for name in skills:
            if name not in chosen:
                chosen.append(name)
    return tuple(chosen[:MAX_DEFAULT_SKILLS])


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def bundled_roots() -> Tuple[Path, ...]:
    """The skill trees that ship with this checkout, in lookup order."""
    repo = Path(__file__).resolve().parent.parent
    return (repo / "skills", get_optional_skills_dir(repo / "optional-skills"))


def default_roots() -> Tuple[Path, ...]:
    """Where a skill name is looked up: the profile's installed skills, then ours.

    Installed skills come first so an operator who has customised a skill gets
    their copy rather than the one this checkout shipped.
    """
    roots: List[Path] = [get_skills_dir()]
    for root in bundled_roots():
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def shared_skill_stores() -> Tuple[Path, ...]:
    """The stores an installed skill's *real* files are allowed to live in.

    A shared skill is not copied into ``~/.hermes/skills``; it is *linked* there,
    one canonical copy behind however many profiles point at it. So the root a
    name was found under is routinely not the directory the bytes are in, and a
    containment check that knows only that root refuses every shared skill on the
    machine — which is the whole feature, refusing itself.

    Naming the stores up front keeps the check a check: a link out of a skills
    root may land in another place skills are kept, and nowhere else. Anywhere
    nobody named is still just somewhere on the host, and still refused.
    """
    override = os.getenv(SHARED_SKILLS_ENV, "").strip()
    raw = override.split(os.pathsep) if override else _SHARED_SKILL_STORES
    stores: List[Path] = []
    for entry in raw:
        entry = entry.strip()
        if not entry:
            continue
        try:
            store = Path(entry).expanduser()
        except RuntimeError:
            # No home to expand ``~`` against. One unusable store is not a
            # reason to stop honouring the others.
            continue
        # An unanchored store, or the filesystem root itself, would make
        # "somewhere skills are kept" mean "anywhere", so neither is a store.
        if not store.is_absolute() or store.parent == store:
            continue
        if store not in stores:
            stores.append(store)
    return tuple(stores)


@dataclass(frozen=True)
class SkillAttachment:
    """One resolved ``SKILL.md``: where it came from and exactly what it said."""

    name: str
    source: str
    sha256: str
    # repr=False: a debug log or a traceback holding a plan should show which
    # skills it holds, not several hundred kilobytes of their contents.
    data: bytes = field(repr=False)

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class SkillPlan:
    """What this attempt will attach, where it came from, and what it could not.

    ``source`` is ``"declared"`` (somebody named these), ``"default"`` (the rules
    table did), or ``"none"``. ``unavailable`` only ever holds default names —
    a declared name that cannot be resolved raises instead.
    """

    source: str
    requested: Tuple[str, ...] = ()
    attachments: Tuple[SkillAttachment, ...] = ()
    unavailable: Tuple[str, ...] = ()


def _contained(path: Path, roots: Sequence[Path]) -> bool:
    """True when ``path`` really resolves inside one of ``roots``. Symlinks included.

    The check is on the *fully resolved* path, so a link inside a store that
    points back out of every store is refused the same as one in a skills root:
    every hop is followed, and only where it finally lands counts.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            base = root.resolve()
        except OSError:
            continue
        if resolved == base or base in resolved.parents:
            return True
    return False


def _load(candidate: Path, root: Path) -> Optional[bytes]:
    """The bytes of ``candidate`` when it is a ``SKILL.md`` ``root`` may hand out.

    ``None`` means "nothing here, keep looking". Anything that *is* here but must
    not be read — a symlink, a directory, a device, a file that resolves outside
    both the root and every :func:`shared_skill_stores` entry, a file over the
    cap — raises instead, because quietly walking past it would let a planted
    name fall through and resolve to a different root's file under a name nobody
    chose.

    ``O_NOFOLLOW`` refuses the symlink at open time and makes the rest race-free:
    the descriptor being ``fstat``-ed is the descriptor being read. It applies to
    the ``SKILL.md`` itself only — a *directory* on the way there may be a link,
    which is how a shared skill is installed, so where it lands is what the
    containment check below is for.
    """
    try:
        fd = os.open(str(candidate), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise SkillAttachmentError(
                f"skill file {candidate} is a symlink; refusing to follow it"
            )
        raise SkillAttachmentError(f"cannot read skill file {candidate}: {exc}")
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SkillAttachmentError(
                f"skill file {candidate} is not a regular file"
            )
        if st.st_size > MAX_SKILL_BYTES:
            raise SkillAttachmentError(
                f"skill file {candidate} is {st.st_size} bytes, over the "
                f"{MAX_SKILL_BYTES}-byte cap"
            )
        allowed = (root,) + shared_skill_stores()
        if not _contained(candidate, allowed):
            raise SkillAttachmentError(
                f"skill file {candidate} resolves outside "
                + " and ".join(str(r) for r in allowed)
            )
        chunks: List[bytes] = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks)


def _candidates(name: str, root: Path) -> Iterable[Path]:
    """Every place ``name`` could name a skill under ``root``, in priority order.

    The exact path first (``security/web-pentest``), then one category level
    (``web-pentest`` under any category directory) so the short name an operator
    actually says still resolves. Categories are walked in sorted order, so two
    machines with the same tree resolve the same file.
    """
    yield root / name / "SKILL.md"
    if "/" in name or not root.is_dir():
        return
    try:
        categories = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return
    for category in categories:
        yield category / name / "SKILL.md"


def _locate(name: str, roots: Sequence[Path]) -> Optional[Tuple[Path, bytes]]:
    for root in roots:
        for candidate in _candidates(name, root):
            data = _load(candidate, root)
            if data is not None:
                return (candidate, data)
    return None


def resolve(
    names: Sequence[str],
    *,
    roots: Optional[Sequence[Path]] = None,
    required: bool = True,
) -> Tuple[Tuple[SkillAttachment, ...], Tuple[str, ...]]:
    """Read every named skill. Returns ``(attachments, unavailable)``.

    ``required=True`` is the declared case: a name that resolves to nothing, or
    to something that must not be read, raises. ``required=False`` is the
    defaults case: the same name is recorded as unavailable and the run goes on.
    """
    search = tuple(default_roots()) if roots is None else tuple(Path(r) for r in roots)
    attachments: List[SkillAttachment] = []
    unavailable: List[str] = []
    sources = set()
    total = 0
    for name in names:
        try:
            found = _locate(_validate_name(name), search)
        except SkillAttachmentError:
            if required:
                raise
            unavailable.append(name)
            continue
        if found is None:
            if required:
                raise SkillAttachmentError(
                    f"no skill named {name!r} under "
                    + ", ".join(str(r) for r in search)
                )
            unavailable.append(name)
            continue
        path, data = found
        if str(path) in sources:
            # Two names for one file (a short name and its category path). The
            # builder wanted the material, and it is already going to get it.
            continue
        if total + len(data) > MAX_TOTAL_BYTES:
            if required:
                raise SkillAttachmentError(
                    f"attaching {name!r} would take the attachment over the "
                    f"{MAX_TOTAL_BYTES}-byte total cap"
                )
            unavailable.append(name)
            continue
        sources.add(str(path))
        total += len(data)
        attachments.append(
            SkillAttachment(
                name=name,
                source=str(path),
                sha256=hashlib.sha256(data).hexdigest(),
                data=data,
            )
        )
    return (tuple(attachments), tuple(unavailable))


def plan(
    declared,
    *,
    text: str = "",
    roots: Optional[Sequence[Path]] = None,
) -> SkillPlan:
    """Decide this attempt's attachments from what the Job declared, or its words.

    ``declared=None`` is *unset* — nobody has an opinion, so the rules table gets
    to have one. An explicitly empty declaration is an opinion: it means no
    attachments, and no default is allowed to override it.
    """
    names = parse_declared_skills(declared)
    if names:
        attachments, unavailable = resolve(names, roots=roots, required=True)
        return SkillPlan(
            source="declared",
            requested=names,
            attachments=attachments,
            unavailable=unavailable,
        )
    if declared is not None:
        return SkillPlan(source="none")
    names = select_default_skills(text)
    if not names:
        return SkillPlan(source="none")
    attachments, unavailable = resolve(names, roots=roots, required=False)
    return SkillPlan(
        source="default" if attachments else "none",
        requested=names,
        attachments=attachments,
        unavailable=unavailable,
    )


# ---------------------------------------------------------------------------
# Staging and stripping
# ---------------------------------------------------------------------------


def attachment_dir(worktree) -> Path:
    """Where attachments live for a worktree. Pure; touches nothing."""
    return Path(worktree) / ATTACHMENT_DIR


def is_attachment_path(path: str) -> bool:
    """True for a repository-relative path that is (or is inside) the directory."""
    return path == ATTACHMENT_DIR or path.startswith(ATTACHMENT_DIR + "/")


def _staged_filenames(attachments: Sequence[SkillAttachment]) -> Tuple[str, ...]:
    """One readable, collision-free filename per attachment, in order."""
    used = set()
    out: List[str] = []
    for attachment in attachments:
        stem = _FILENAME_UNSAFE_RE.sub("-", attachment.name).strip("-.") or "skill"
        candidate = f"{stem}.md"
        suffix = 2
        while candidate in used:
            candidate = f"{stem}-{suffix}.md"
            suffix += 1
        used.add(candidate)
        out.append(candidate)
    return tuple(out)


def stage(worktree, attachments: Sequence[SkillAttachment]) -> Optional[Path]:
    """Write ``attachments`` into the worktree read-only. Returns the directory.

    ``None`` when there is nothing to attach, and in that case nothing is
    created — a Job with no skills leaves a worktree that is byte-identical to
    one from before this feature existed.

    Every copy is hashed after it is written and compared against the hash taken
    at the source, so a short write or a full disk is a refusal rather than
    reference material with a hole in it. A failure removes whatever this call
    already wrote: a half-staged attachment is worse than none, because the
    builder cannot tell the difference.
    """
    if not attachments:
        return None
    target = attachment_dir(worktree)
    try:
        # Never ``exist_ok``: a directory already sitting there is content from
        # the approved base commit, and overwriting it would put attachments in
        # the diff *and* destroy whatever they replaced.
        target.mkdir(mode=0o700)
    except FileExistsError:
        raise SkillAttachmentError(
            f"{target} already exists in the worktree; refusing to overwrite it"
        )
    except OSError as exc:
        raise SkillAttachmentError(f"cannot create {target}: {exc}")

    try:
        # mkdir's mode is masked by the umask; chmod is not. Staging needs the
        # directory writable, and the last thing this does is take that away.
        os.chmod(target, 0o700)
        for attachment, filename in zip(attachments, _staged_filenames(attachments)):
            path = target / filename
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            try:
                os.write(fd, attachment.data)
            finally:
                os.close(fd)
            written = path.read_bytes()
            if hashlib.sha256(written).hexdigest() != attachment.sha256:
                raise SkillAttachmentError(
                    f"staged copy of {attachment.name!r} does not match the "
                    "bytes read from its source"
                )
            os.chmod(path, 0o444)
        # Read and traversable, not writable: the attachments are reference
        # material, and a builder that edits them is editing nothing anybody
        # will ever read again.
        os.chmod(target, 0o500)
    except (OSError, SkillAttachmentError) as exc:
        strip(worktree)
        if isinstance(exc, SkillAttachmentError):
            raise
        raise SkillAttachmentError(f"cannot stage skill attachments: {exc}")
    return target


def strip(worktree) -> bool:
    """Remove the attachment directory from ``worktree``. True when it is gone.

    Called before any evidence is read, on every path — success, failure, and
    timeout alike — because reference material that survives into the diff stops
    being reference material and becomes work nobody did.

    Never follows a symlink: a directory replaced by one is unlinked as the link
    it is, not walked through.
    """
    target = attachment_dir(worktree)
    try:
        if target.is_symlink():
            os.unlink(str(target))
            return True
        if not target.exists():
            return True
        os.chmod(target, 0o700)
        for entry in os.scandir(str(target)):
            if entry.is_dir(follow_symlinks=False):
                continue
            os.unlink(entry.path)
        os.rmdir(str(target))
    except OSError:
        return False
    return True
