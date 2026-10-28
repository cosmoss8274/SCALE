"""Installation recovery checks; no downloads, models or predictions are executed."""

import ast
from contextlib import ExitStack, redirect_stdout
from importlib import metadata
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class ColabSetupTests(unittest.TestCase):
    def run_setup(
        self,
        root,
        backend_python=(3, 12, 12),
        fail_install=False,
        backend_gpu_available=True,
    ):
        notebook = (
            Path(__file__).resolve().parents[1] / "notebooks/SCALE_demo_colab.ipynb"
        )
        cells = json.loads(notebook.read_text())["cells"]
        source = next(
            "".join(c["source"])
            for c in cells
            if c["cell_type"] == "code" and "BOLTZ_ENV = Path" in "".join(c["source"])
        )
        paths = {
            "/content/SCALE": str(root / "checkout"),
            "/content/scale_run": str(root / "run"),
            "/content/.scale-boltz2-py312": str(root / "backend"),
        }

        class Locations(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value in paths:
                    return ast.copy_location(ast.Constant(paths[node.value]), node)
                return node

        tree = ast.fix_missing_locations(Locations().visit(ast.parse(source)))
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[3:4] == ["venv"]:
                folder = root / "backend/bin"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "python").touch()
                (folder / "boltz").touch()
            if fail_install and command[3:5] == ["pip", "install"]:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)

        def check_output(command, **kwargs):
            if command[0] == "git":
                return "test-code-revision\n"
            if "sys.version_info" in command[-1]:
                return json.dumps(list(backend_python))
            if "torch.cuda.is_available()" in command[-1]:
                if not backend_gpu_available:
                    raise subprocess.CalledProcessError(1, command)
                return json.dumps({"name": "test-GPU", "cuda": "test"})
            return json.dumps(
                {"boltz": "2.2.1", "torch": "test-backend", "numpy": "test"}
            )

        with ExitStack() as stack:
            stack.enter_context(patch.object(subprocess, "run", side_effect=run))
            stack.enter_context(
                patch.object(subprocess, "check_output", side_effect=check_output)
            )
            stack.enter_context(patch.object(importlib, "import_module"))
            stack.enter_context(
                patch.object(metadata, "version", return_value="test-kernel")
            )
            stack.enter_context(patch.dict(os.environ, {"PATH": "/original/bin"}))
            stack.enter_context(patch.object(sys, "path", list(sys.path)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            exec(compile(tree, str(notebook) + ":setup", "exec"), {})
        return commands

    def test_existing_checkout_recovers_after_failed_package_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkout/.git").mkdir(parents=True)
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_setup(root, fail_install=True)
            self.assertFalse((root / "run/environment.json").exists())

            commands = self.run_setup(root)
            self.assertTrue(any(c[3:5] == ["pip", "install"] for c in commands))
            self.assertFalse(any(c[:2] == ["git", "clone"] for c in commands))
            kernel_installs = [
                c for c in commands if c[1:4] == ["-m", "pip", "install"]
            ]
            self.assertTrue(kernel_installs)
            self.assertTrue(all("torch" not in c for c in kernel_installs))
            result = json.loads((root / "run/environment.json").read_text())
            self.assertEqual(result["backend_python"], "3.12.12")
            self.assertEqual(result["backend_packages"]["boltz"], "2.2.1")

    def test_backend_cuda_failure_does_not_write_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkout/.git").mkdir(parents=True)
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_setup(root, backend_gpu_available=False)
            self.assertFalse((root / "run/environment.json").exists())

    def test_wrong_backend_python_stops_before_package_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkout/.git").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "Expected Python 3.12"):
                self.run_setup(root, backend_python=(3, 13, 16))
            self.assertFalse((root / "run/environment.json").exists())


if __name__ == "__main__":
    unittest.main()
