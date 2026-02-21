from setuptools import find_packages, setup


setup(
    name="lunarlander-distill",
    version="0.1.0",
    description="Baseline PPO vs teacher-distilled PPO (KL) on LunarLander-v3.",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
