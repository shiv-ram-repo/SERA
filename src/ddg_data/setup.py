from setuptools import find_packages, setup

setup(
    name="ddg_data",
    version="0.1.0",
    description="Standalone data loading for protein stability prediction "
                "(MultimodalDDG-compatible).",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "numpy",
        "pandas",
        "tqdm",
        "biopython",
        "fair-esm",
    ],
    extras_require={
        "lmdb": ["lmdb", "atom3d"],
    },
)
