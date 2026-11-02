from setuptools import find_packages, setup

setup(
    name="ddg_train",
    version="0.1.0",
    description="Training and evaluation harness for protein stability "
                "prediction (MultimodalDDG-compatible).",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "tqdm",
    ],
)
