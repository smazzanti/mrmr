from setuptools import setup

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

setup(
    name='mrmr_selection',
    version='0.2.9',
    description='minimum-Redundancy-Maximum-Relevance algorithm for feature selection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/smazzanti/mrmr',
    author='Samuele Mazzanti',
    author_email='mazzanti.sam@gmail.com',
    license='GNU General Public License v3.0',
    packages=['mrmr'],
    install_requires=[
        'category_encoders',
        'jinja2',
        'tqdm',
        'joblib',
        'pandas>=1.0.3',
        'numpy>=1.18.1',
        'scikit-learn',
        'scipy',
    ],
    extras_require={
        'polars': ['polars>=0.12.5'],
        'pyspark': ['pyspark>=3.4.1'],
        'bigquery': ['google.cloud.bigquery']
    },
    zip_safe=False
)
