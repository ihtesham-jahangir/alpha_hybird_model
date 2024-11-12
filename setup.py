import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alpha_hybird_model",
    version="0.1.0",
    author="ihtesham jahangir",
    author_email="ihteshamjahangir21@gmail.com",
    description="Alpha Hybrid CNN model for classification tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ihtesham-jahangir/alpha_hybird_model",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "tensorflow>=2.0",
    ],
    entry_points={
        'console_scripts': [
            'alpha_hybird_train=scripts.train_model:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
