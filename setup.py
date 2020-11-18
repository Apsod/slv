#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='SLV-Kundo stuff',
      version='0.0.1',
      description='Helpers to get data form kundo',

      author='Amaru Cuba Gyllensten',
      author_email='amaru.cuba.gyllensten@ri.se',
      
      install_requires=[
          'transformers',
          'tqdm',
          'torch',
      ],
      packages=find_packages(),
      entry_points={
          'console_scripts': ['slv=slv.__main__:__main__'],
          },
      )
