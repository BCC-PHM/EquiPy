from setuptools import setup, find_packages

setup(
    name='EquiPy', 
    version='0.1',  
    author='David Ellis',
    author_email='david.ellis@birmingham.gov.uk',
    description='Library to visualise differences in rates and outcomes across ethnicity and index of multiple deprivation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BCC-PHM/EquiPy',
    packages=find_packages(),
    install_requires=[  
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy'
    ],
    classifiers=[  
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License :: Open Government License v3 (OGL-3.0)'
    ],
)