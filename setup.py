from setuptools import setup, find_packages

setup(
    name='LibX',
    version='0.1',
    packages=find_packages(),  # Trouve automatiquement tous les packages
    package_dir={'': '.'},  # Le package root est le dossier actuel
)