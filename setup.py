import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements_txt = f.read().splitlines()

setuptools.setup(
    name="t5s",
    version="0.2",
    author="Jan Švec",
    author_email="honzas@ntis.zcu.cz",
    description="T5 simple (text-to-text transfer transformer made simple)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/honzas83/t5s",
    package_dir={'': 'src'},
    py_modules=["t5s"],
    scripts=["src/eval_tsv.py",
             "src/t5_convert_checkpoint.py",
             "src/t5_fine_tune.py",
             "src/t5_predict.py",
             "src/t5_evaluate.py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements_txt,
)
