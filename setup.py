from setuptools import setup

with open("README.md", encoding="utf8") as f:
    long_description = f.read()

setup(
    name='mrmr_selection',
    version='0.2.1',
    description='minimum-Redundancy-Maximum-Relevance algorithm for feature selection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/smazzanti/mrmr',
    author='Samuele Mazzanti',
    author_email='mazzanti.sam@gmail.com',
    license='MIT',
    packages=['mrmr'],
    install_requires=[
        'jinja2',
        'tqdm',
        'joblib',
        'pandas>=1.0.3',
        'numpy>=1.18.1',
        'sklearn',
        'category_encoders'
    ],
    zip_safe=False
)
