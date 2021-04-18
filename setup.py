from setuptools import setup

setup(
    name='mrmr',
    version='0.1',
    description='Maximum-Relevance-Minimum-Redundancy algorithm for feature selection',
    url='https://github.com/smazzanti/mrmr',
    author='Samuele Mazzanti',
    author_email='mazzanti.sam@gmail.com',
    license='MIT',
    packages=['mrmr'],
    install_requires=[
        'joblib',
        'multiprocessing',
        'pandas>=1.0.3',
        'numpy>=1.18.1',
        'sklearn',
        'rdc'
    ],
    zip_safe=False
)
