from setuptools import setup, find_namespace_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# The file is PEP 440 compliant:
# https://www.python.org/dev/peps/pep-0440/

dependency_links = [
    'git+https://github.com/caniko2/read_exdir.git'
]

install_requires =  [
    'read_exdir @ git+https://github.com/caniko2/read_exdir',
    'openpyxl',
    'tables',
    'pandas',
    'numpy',
    'vg'
]

setup(
    name='DeepCage',
    version='0.1',
    description='Package used for extending DeepLabCut to support several',
    long_description=long_description,
    long_description_content_type='markdown',
    url='https://github.com/caniko2/DeepCage',
    author='Can Hicabi Tartanoglu',
    author_email='canhtart@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Experimentalists :: Polymaths',
        'Topic :: Biology',
        'License :: LGPL-3.0',
        'Programming Language :: Python :: 3',
    ],
    packages=find_namespace_packages(),
    python_requires='>=3.6',
    install_requires=install_requires,
    dependency_links=dependency_links,
    project_urls={
        'Source': 'https://github.com/caniko2/DeepCage',
    },
)
