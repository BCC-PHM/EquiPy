from setuptools import setup, find_packages

setup(
    name='inequality-vis', 
    version='0.1',  
    author='David Ellis',
    author_email='david.ellis@birmingham.gov.uk',
    description='Library to visualise differences in rates and outcomes across ethnicity and index of multiple deprivation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/David-Ellis/inequality-vis',
    packages=find_packages(),
    install_requires=[  
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn'
    ],
    classifiers=[  
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License :: Open Government License v3 (OGL-3.0)'
    ],
)