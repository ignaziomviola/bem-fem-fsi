"""Integrity of the vendored fluid solver (standard-library unittest).

Run:          python3 test_vendored.py
Regenerate:   python3 test_vendored.py --write

The panel method here is a verbatim copy of
ignaziomviola/time-dependent-free-wake-panel-method at the commit recorded in
docs/FLUID.md. This test recomputes the SHA-256 of every vendored file against
`fluid_manifest.txt` and fails if one has been edited.

It checks LOCAL integrity, not agreement with upstream: nothing here reaches
the network. Local integrity is the property that matters, because the failure
mode worth preventing is a quiet fix to the fluid code made while chasing a
coupling result, which would make every number in docs/COUPLING.md
irreproducible against the repository it claims to have come from. A genuine
fluid change belongs upstream, followed by a re-vendor and a new manifest in
the same commit.
"""

import hashlib
import pathlib
import sys
import unittest

MANIFEST = pathlib.Path(__file__).with_name("fluid_manifest.txt")
ROOT = pathlib.Path(__file__).parent

VENDORED = ("thick_panel_wing.py", "unsteady_wing.py", "freewake_kernels.py",
            "make_thick_sample_inputs.py", "test_thick_panel_wing.py",
            "test_unsteady_wing.py", "verify_analytic.py", "verify_unsteady.py")

UPSTREAM = ("https://github.com/ignaziomviola/"
            "time-dependent-free-wake-panel-method")
COMMIT = "d9d1f657e9191cf57959558685a4f06b3b46923c"


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_manifest():
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, _, name = line.partition("  ")
        entries[name.strip()] = sha.strip()
    return entries


def write_manifest():
    lines = [f"# SHA-256 of the files vendored verbatim from",
             f"# {UPSTREAM}",
             f"# commit {COMMIT}",
             f"# Regenerate with: python3 test_vendored.py --write"]
    lines += [f"{digest(ROOT / name)}  {name}" for name in VENDORED]
    MANIFEST.write_text("\n".join(lines) + "\n")
    return MANIFEST


class TestVendoredFluid(unittest.TestCase):

    def test_manifest_covers_every_vendored_file(self):
        entries = read_manifest()
        self.assertEqual(sorted(entries), sorted(VENDORED))

    def test_every_vendored_file_is_unmodified(self):
        for name, sha in read_manifest().items():
            path = ROOT / name
            self.assertTrue(path.exists(), f"{name} is missing")
            self.assertEqual(
                digest(path), sha,
                f"{name} has been modified since it was vendored from "
                f"{UPSTREAM} at {COMMIT}. The fluid solver is not maintained "
                f"here: change it upstream, re-vendor, and regenerate the "
                f"manifest with 'python3 test_vendored.py --write' in the same "
                f"commit.")

    def test_the_fluid_modules_import_without_side_effects(self):
        """No prompt, no print, no figure and no file at import."""
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            import thick_panel_wing            # noqa: F401
            import unsteady_wing               # noqa: F401
            import freewake_kernels            # noqa: F401
        self.assertEqual(buffer.getvalue(), "")
        self.assertNotIn("matplotlib", sys.modules,
                         "importing the solver pulled in a figure backend")


if __name__ == "__main__":
    if "--write" in sys.argv:
        print(f"wrote {write_manifest()}")
    else:
        unittest.main(verbosity=2)
