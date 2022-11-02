import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prdc_cli",
    version="0.0.1",
    author="NAVER Corp., Mahmood Hussain",
    description="Compute precision, recall, density, and coverage metrics for custom datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahmood-hussain/prdc",
    packages=setuptools.find_packages(),
        entry_points ={
            'console_scripts': [
                'prdc_cli = prdc.prdc_cli:main'
            ]
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy',
        'joblib',
        'torch',
        'torchvision',
        'Pillow',
        'tqdm',
    ],
)
