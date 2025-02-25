from setuptools import setup, find_packages

setup(
    name='LibX',
    version='0.1.0',
    description='Module LibX regroupant CryptoDB, DataViz, Gan, TradingSimulator et TradingTools',
    author='Ton Nom',
    author_email='ton.email@example.com',
    url='https://github.com/ton-compte/LibX',  # Remplace par l'URL de ton repo GitHub
    packages=find_packages(),  # Recherche automatiquement tous les packages (dossiers avec __init__.py)
    install_requires=[
        # Liste les dépendances nécessaires, par exemple :
        # 'pandas',
        # 'numpy',
        # etc.
    ],
    classifiers=[
         'Programming Language :: Python :: 3',
         'Operating System :: OS Independent',
    ],
)
