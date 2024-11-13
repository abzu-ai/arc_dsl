from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def read_readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="arc_dsl",
    url="https://github.com/abzu-ai/arc_dsl",
    version="0.1.0",
    author="Tom Jelen, Jaan Kasak",
    author_email="tom.jelen@abzu.ai, jaan.kasak@abzu.ai",
    description="Program synthesis for ARC puzzles using a custom DSL and transformer models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    keywords="ARC, program synthesis, DSL, transformer models, artificial intelligence, lark",
    packages=find_packages(),
    package_data={
        "arc_dsl": [
            "*.lark",
            "trained_models/**/*.keras",
            "data/**/*.json",
        ]
    },
    install_requires=read_requirements(),
    license="MIT",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Interpreters",
        "Framework :: TensorFlow",
        "Framework :: Keras",
        "Framework :: NumPy",
        "Environment :: GPU",
    ],
)
