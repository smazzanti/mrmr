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
        'tqdm',
        'joblib',
        'pandas>=1.0.3',
        'numpy>=1.18.1',
        'sklearn',
        'category_encoders'
    ],
    zip_safe=False
)
