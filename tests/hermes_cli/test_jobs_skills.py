"""Tests for Jobs skill attachment (hermes_cli/jobs_skills).

Real files on a real filesystem — the traversal, symlink, and containment
refusals under test are the ones the OS actually enforces, so nothing here is
mocked. The one thing these pin that no other test can is the asymmetry: a
*declared* skill that cannot be honoured stops the run, and a *default* one that
is simply not installed does not.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from hermes_cli import jobs_skills as jskills


# ---------------------------------------------------------------------------
# A small, real skill tree
# ---------------------------------------------------------------------------


def _skill(root: Path, name: str, body: str = "# skill\n") -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def roots(tmp_path):
    """One skill root holding ``security/audit`` and ``creative/motion``."""
    root = tmp_path / "skills"
    _skill(root, "security/audit", "# audit\nLook for the sharp edges.\n")
    _skill(root, "creative/motion", "# motion\nEase in, ease out.\n")
    return (root,)


def _worktree(tmp_path) -> Path:
    wt = tmp_path / "worktree"
    wt.mkdir()
    return wt


# ---------------------------------------------------------------------------
# Declared names — parsing and the caps
# ---------------------------------------------------------------------------


def test_unset_is_not_the_same_as_empty():
    assert jskills.parse_declared_skills(None) == ()
    assert jskills.parse_declared_skills([]) == ()
    # The tuple is identical; what differs is what plan() does with it, which is
    # the distinction the whole feature hangs off.
    assert jskills.plan(None, text="no keywords here at all").source == "none"
    assert jskills.plan([], text="review the security of the tests").source == "none"


def test_a_declaration_may_be_repeated_or_comma_separated():
    assert jskills.parse_declared_skills("a,b") == ("a", "b")
    assert jskills.parse_declared_skills(["a", "b"]) == ("a", "b")
    assert jskills.parse_declared_skills(["a, b", "c"]) == ("a", "b", "c")
    assert jskills.parse_declared_skills("  a  ,  ,  b ") == ("a", "b")


def test_declaration_order_is_kept_and_repeats_are_dropped():
    assert jskills.parse_declared_skills(["b", "a", "b"]) == ("b", "a")


@pytest.mark.parametrize(
    "name",
    [
        "../etc/passwd",
        "security/../../etc/passwd",
        "/etc/passwd",
        "..",
        "./relative",
        "back\\slash",
        "has space",
        "a/b/c/d/e",
        "x" * 200,
        "trailing/",
    ],
)
def test_an_unusable_name_is_refused_at_the_declaration(name):
    with pytest.raises(jskills.SkillAttachmentError):
        jskills.parse_declared_skills(name)


def test_more_skills_than_the_cap_is_refused():
    names = [f"s{i}" for i in range(jskills.MAX_SKILLS + 1)]
    with pytest.raises(jskills.SkillAttachmentError):
        jskills.parse_declared_skills(names)
    # One under the cap is fine.
    assert len(jskills.parse_declared_skills(names[:-1])) == jskills.MAX_SKILLS


def test_a_non_string_declaration_is_refused():
    with pytest.raises(jskills.SkillAttachmentError):
        jskills.parse_declared_skills([{"name": "a"}])
    with pytest.raises(jskills.SkillAttachmentError):
        jskills.parse_declared_skills(7)


# ---------------------------------------------------------------------------
# Resolution — declared names fail closed
# ---------------------------------------------------------------------------


def test_a_declared_skill_resolves_by_full_path_or_short_name(roots):
    by_path = jskills.plan(["security/audit"], roots=roots)
    by_name = jskills.plan(["audit"], roots=roots)
    assert by_path.source == by_name.source == "declared"
    assert [a.source for a in by_path.attachments] == [
        a.source for a in by_name.attachments
    ]
    assert by_path.attachments[0].data.startswith(b"# audit")


def test_the_hash_is_of_the_bytes_that_were_actually_read(roots):
    plan = jskills.plan(["security/audit"], roots=roots)
    attachment = plan.attachments[0]
    on_disk = Path(attachment.source).read_bytes()
    assert attachment.sha256 == hashlib.sha256(on_disk).hexdigest()
    assert attachment.data == on_disk


def test_a_declared_skill_that_does_not_exist_fails_closed(roots):
    with pytest.raises(jskills.SkillAttachmentError):
        jskills.plan(["no-such-skill"], roots=roots)


def test_a_symlinked_skill_file_is_refused(tmp_path, roots):
    (roots[0] / "sneaky").mkdir()
    secret = tmp_path / "outside.md"
    secret.write_text("not yours\n")
    os.symlink(str(secret), str(roots[0] / "sneaky" / "SKILL.md"))
    with pytest.raises(jskills.SkillAttachmentError) as exc:
        jskills.plan(["sneaky"], roots=roots)
    assert "symlink" in str(exc.value)


def test_a_skill_reached_through_a_symlinked_directory_is_refused(tmp_path, roots):
    outside = tmp_path / "outside"
    (outside / "escapee").mkdir(parents=True)
    (outside / "escapee" / "SKILL.md").write_text("not yours\n")
    os.symlink(str(outside), str(roots[0] / "linked"))
    with pytest.raises(jskills.SkillAttachmentError) as exc:
        jskills.plan(["linked/escapee"], roots=roots)
    assert "outside" in str(exc.value)


def test_a_skill_over_the_size_cap_is_refused(roots):
    _skill(roots[0], "huge", "x" * (jskills.MAX_SKILL_BYTES + 1))
    with pytest.raises(jskills.SkillAttachmentError) as exc:
        jskills.plan(["huge"], roots=roots)
    assert "cap" in str(exc.value)


def test_a_directory_named_skill_md_is_refused_not_skipped(roots):
    (roots[0] / "impostor" / "SKILL.md").mkdir(parents=True)
    with pytest.raises(jskills.SkillAttachmentError):
        jskills.plan(["impostor"], roots=roots)


def test_two_names_for_one_file_attach_it_once(roots):
    plan = jskills.plan(["audit", "security/audit"], roots=roots)
    assert len(plan.attachments) == 1
    assert plan.requested == ("audit", "security/audit")


# ---------------------------------------------------------------------------
# Shared stores — an installed skill is a symlink, and the guard has to know it
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_store(tmp_path, monkeypatch):
    """A shared-skills store, standing in for the real canonical directory."""
    store = tmp_path / "shared-skills" / "canonical"
    store.mkdir(parents=True)
    monkeypatch.setenv(jskills.SHARED_SKILLS_ENV, str(store))
    return store


def test_the_default_store_is_the_shared_skills_canonical_directory(monkeypatch):
    """Unconfigured, the allowed store is the one skills are really installed from."""
    monkeypatch.delenv(jskills.SHARED_SKILLS_ENV, raising=False)
    assert jskills.shared_skill_stores() == (
        Path("~/.local/share/shared-skills/canonical").expanduser(),
    )


def test_stores_are_read_from_the_environment_pathsep_separated(tmp_path, monkeypatch):
    first, second = tmp_path / "store-a", tmp_path / "store-b"
    monkeypatch.setenv(
        jskills.SHARED_SKILLS_ENV, os.pathsep.join([str(first), str(second)])
    )
    assert jskills.shared_skill_stores() == (first, second)


@pytest.mark.parametrize("configured", ["/", "relative/store"])
def test_a_store_that_would_allow_everything_is_not_a_store(monkeypatch, configured):
    """``/`` contains every path there is, so it cannot be what "a store" means."""
    monkeypatch.setenv(jskills.SHARED_SKILLS_ENV, configured)
    assert jskills.shared_skill_stores() == ()


def test_a_skill_installed_as_a_symlink_into_the_store_attaches(roots, shared_store):
    """The shape every shared skill has on a real machine: the root holds a link."""
    _skill(shared_store, "tdd", "# tdd\nRed, green, refactor.\n")
    os.symlink(str(shared_store / "tdd"), str(roots[0] / "tdd"))

    plan = jskills.plan(["tdd"], roots=roots)

    assert plan.source == "declared"
    assert [a.name for a in plan.attachments] == ["tdd"]
    assert plan.attachments[0].data.startswith(b"# tdd")


def test_a_category_nested_symlinked_skill_attaches_by_either_name(roots, shared_store):
    _skill(shared_store, "test-driven-development", "# tdd\nWrite the test first.\n")
    (roots[0] / "software-development").mkdir()
    os.symlink(
        str(shared_store / "test-driven-development"),
        str(roots[0] / "software-development" / "test-driven-development"),
    )

    by_path = jskills.plan(
        ["software-development/test-driven-development"], roots=roots
    )
    by_name = jskills.plan(["test-driven-development"], roots=roots)

    assert by_name.attachments[0].data.startswith(b"# tdd")
    assert by_path.attachments[0].data == by_name.attachments[0].data


@pytest.mark.parametrize("outside", ["/etc/passwd", "documents"])
def test_a_configured_store_does_not_open_the_rest_of_the_host(
    tmp_path, roots, shared_store, outside
):
    """What the fix must not trade away: a path under no store is still refused."""
    if outside == "documents":
        target = tmp_path / "Documents" / "private-notes.md"
        target.parent.mkdir(parents=True)
        target.write_text("not yours\n")
    else:
        target = Path(outside)
    (roots[0] / "sneaky").mkdir()
    os.symlink(str(target), str(roots[0] / "sneaky" / "SKILL.md"))

    with pytest.raises(jskills.SkillAttachmentError):
        jskills.plan(["sneaky"], roots=roots)


def test_a_link_that_hops_through_the_store_and_back_out_is_refused(
    tmp_path, roots, shared_store
):
    """Every hop is followed: where a link lands is what is checked, not its shape."""
    elsewhere = tmp_path / "Documents" / "not-a-skill"
    elsewhere.mkdir(parents=True)
    (elsewhere / "SKILL.md").write_text("not yours\n")
    os.symlink(str(elsewhere), str(shared_store / "laundered"))
    os.symlink(str(shared_store / "laundered"), str(roots[0] / "laundered"))

    with pytest.raises(jskills.SkillAttachmentError) as exc:
        jskills.plan(["laundered"], roots=roots)
    assert "outside" in str(exc.value)


def test_a_store_nobody_configured_is_not_an_allowed_root(tmp_path, roots, monkeypatch):
    """The allowance comes from the named store, not from the link being a link."""
    unnamed = tmp_path / "unnamed-store" / "canonical"
    _skill(unnamed, "tdd", "# tdd\n")
    os.symlink(str(unnamed / "tdd"), str(roots[0] / "tdd"))
    monkeypatch.setenv(jskills.SHARED_SKILLS_ENV, str(tmp_path / "some-other-store"))

    with pytest.raises(jskills.SkillAttachmentError) as exc:
        jskills.plan(["tdd"], roots=roots)
    assert "outside" in str(exc.value)


# ---------------------------------------------------------------------------
# Defaults — a guess, so a missing one is not an error
# ---------------------------------------------------------------------------


def test_defaults_fire_on_known_keywords():
    assert "security/web-pentest" in jskills.select_default_skills(
        "Audit the login flow for a security vulnerability"
    )
    assert "software-development/test-driven-development" in (
        jskills.select_default_skills("Add regression tests for the parser")
    )
    assert "github/github-code-review" in jskills.select_default_skills(
        "Do a code review of the reviewer fix"
    )
    assert "creative/hyperframes" in jskills.select_default_skills(
        "Rebuild the hero with a scroll-driven GSAP animation"
    )
    assert "creative/claude-design" in jskills.select_default_skills(
        "Redesign the settings page layout and typography"
    )


def test_default_matching_is_by_whole_word_not_substring():
    # "latest" must not read as "test", nor "designation" as "design".
    assert jskills.select_default_skills("Ship the latest designation") == ()


def test_defaults_are_deterministic_and_bounded():
    text = "security review of the animation tests after a crash in the design"
    first = jskills.select_default_skills(text)
    assert first == jskills.select_default_skills(text)
    assert len(first) <= jskills.MAX_DEFAULT_SKILLS


def test_a_job_with_no_keywords_selects_nothing():
    assert jskills.select_default_skills("Rename a variable in the helper") == ()
    assert jskills.select_default_skills("") == ()
    assert jskills.select_default_skills(None) == ()


def test_defaults_use_the_jobs_words_when_nothing_is_declared(roots, monkeypatch):
    monkeypatch.setattr(jskills, "SKILL_DEFAULT_RULES",
                        ((("motion", "animation"), ("creative/motion",)),))
    plan = jskills.plan(None, text="Fix the hero animation", roots=roots)
    assert plan.source == "default"
    assert [a.name for a in plan.attachments] == ["creative/motion"]


def test_a_default_this_machine_does_not_have_is_skipped_not_raised(roots, monkeypatch):
    monkeypatch.setattr(
        jskills, "SKILL_DEFAULT_RULES",
        ((("motion",), ("creative/motion", "creative/not-installed")),),
    )
    plan = jskills.plan(None, text="motion work", roots=roots)
    assert [a.name for a in plan.attachments] == ["creative/motion"]
    assert plan.unavailable == ("creative/not-installed",)


def test_every_default_rule_names_a_skill_this_checkout_ships(monkeypatch):
    """A stale rules table would silently attach nothing. Catch it here."""
    monkeypatch.delenv("HERMES_OPTIONAL_SKILLS", raising=False)
    roots = jskills.bundled_roots()
    assert all(root.is_dir() for root in roots), roots
    for _keywords, names in jskills.SKILL_DEFAULT_RULES:
        for name in names:
            attachments, unavailable = jskills.resolve(
                [name], roots=roots, required=False
            )
            assert not unavailable, f"{name} no longer resolves under {roots}"
            assert attachments[0].data, name


# ---------------------------------------------------------------------------
# Staging and stripping
# ---------------------------------------------------------------------------


def test_staging_writes_read_only_files_the_builder_can_read(tmp_path, roots):
    wt = _worktree(tmp_path)
    plan = jskills.plan(["security/audit", "creative/motion"], roots=roots)
    target = jskills.stage(wt, plan.attachments)

    assert target == wt / jskills.ATTACHMENT_DIR
    staged = sorted(p.name for p in target.iterdir())
    assert staged == ["creative-motion.md", "security-audit.md"]
    for path in target.iterdir():
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert not (mode & 0o222), f"{path} is writable ({oct(mode)})"
        assert mode & 0o400, f"{path} is not readable ({oct(mode)})"
        # And the bytes really are the skill's bytes.
        assert path.read_bytes() in {a.data for a in plan.attachments}
    dir_mode = stat.S_IMODE(os.stat(target).st_mode)
    assert not (dir_mode & 0o222), oct(dir_mode)


def test_staging_nothing_creates_nothing(tmp_path):
    wt = _worktree(tmp_path)
    assert jskills.stage(wt, ()) is None
    assert list(wt.iterdir()) == []


def test_staging_refuses_to_overwrite_an_existing_directory(tmp_path, roots):
    wt = _worktree(tmp_path)
    (wt / jskills.ATTACHMENT_DIR).mkdir()
    plan = jskills.plan(["security/audit"], roots=roots)
    with pytest.raises(jskills.SkillAttachmentError):
        jskills.stage(wt, plan.attachments)


def _mismatched(attachment):
    """The same skill carrying a hash its bytes cannot produce."""
    return jskills.SkillAttachment(
        name=attachment.name,
        source=attachment.source,
        sha256="0" * 64,
        data=attachment.data,
    )


def test_a_staged_copy_that_does_not_match_its_source_is_refused(tmp_path, roots):
    wt = _worktree(tmp_path)
    plan = jskills.plan(["security/audit"], roots=roots)
    with pytest.raises(jskills.SkillAttachmentError) as exc:
        jskills.stage(wt, (_mismatched(plan.attachments[0]),))
    assert "does not match" in str(exc.value)
    assert not (wt / jskills.ATTACHMENT_DIR).exists()


def test_a_failed_staging_takes_the_files_it_already_wrote_with_it(tmp_path, roots):
    """A half-staged attachment is worse than none: the builder cannot tell."""
    wt = _worktree(tmp_path)
    plan = jskills.plan(["security/audit", "creative/motion"], roots=roots)
    good, doomed = plan.attachments[0], _mismatched(plan.attachments[1])

    with pytest.raises(jskills.SkillAttachmentError):
        jskills.stage(wt, (good, doomed))
    assert not (wt / jskills.ATTACHMENT_DIR).exists()
    assert list(wt.iterdir()) == []


def test_stripping_removes_read_only_attachments(tmp_path, roots):
    wt = _worktree(tmp_path)
    plan = jskills.plan(["security/audit", "creative/motion"], roots=roots)
    jskills.stage(wt, plan.attachments)

    assert jskills.strip(wt) is True
    assert not (wt / jskills.ATTACHMENT_DIR).exists()
    assert list(wt.iterdir()) == []


def test_stripping_is_a_no_op_when_nothing_was_attached(tmp_path):
    wt = _worktree(tmp_path)
    assert jskills.strip(wt) is True
    assert jskills.strip(wt) is True


def test_stripping_unlinks_a_symlink_rather_than_following_it(tmp_path):
    wt = _worktree(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "keepme.txt").write_text("still here\n")
    os.symlink(str(outside), str(wt / jskills.ATTACHMENT_DIR))

    assert jskills.strip(wt) is True
    assert not (wt / jskills.ATTACHMENT_DIR).exists()
    assert (outside / "keepme.txt").read_text() == "still here\n"


def test_attachment_paths_are_recognisable_in_a_diff():
    assert jskills.is_attachment_path(jskills.ATTACHMENT_DIR)
    assert jskills.is_attachment_path(f"{jskills.ATTACHMENT_DIR}/security-audit.md")
    assert not jskills.is_attachment_path("src/main.py")
    assert not jskills.is_attachment_path(f"src/{jskills.ATTACHMENT_DIR}/x.md")
