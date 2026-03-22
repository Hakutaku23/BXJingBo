from __future__ import annotations

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent
MODULE_NAMES = [
    "window_encoder",
    "casebase",
    "runtime_config",
    "online_recommender",
]


extensions = [
    Extension(
        name=f"core.{module_name}",
        sources=[str(ROOT / f"{module_name}.py")],
    )
    for module_name in MODULE_NAMES
]


setup(
    name="t90-core",
    version="0.1.0",
    description="Cython build for the T90 core package",
    packages=["core"],
    package_dir={"core": "."},
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        annotate=False,
    ),
    zip_safe=False,
)
