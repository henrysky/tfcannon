import os

from setuptools import setup, find_packages

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tfcannon',
    version='0.1.dev',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'astropy',
        'h5py',
        'matplotlib',
        'tqdm',
        'packaging'],
    extras_require={
        "tensorflow": ["tensorflow>=1.14.0"],
        "tensorflow-gpu": ["tensorflow-gpu>=1.14.0"],
        "tensorflow-probability": ["tensorflow-probability>=0.7.0"]},
    url='https://github.com/henrysky/tfcannon',
    project_urls={
        "Bug Tracker": "https://github.com/henrysky/tfcannon/issues",
        "Documentation": "https://github.com/henrysky/tfcannon",
        "Source Code": "https://github.com/henrysky/tfcannon",
    },
    license='MIT',
    author='Henry Leung',
    author_email='henrysky.leung@mail.utoronto.ca',
    description='Tensorflow implementation of the Cannon',
    long_description=long_description
)
