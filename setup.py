#!/usr/bin/env python3
"""
Setup for the coach project
"""
from setuptools import setup, find_packages


setup(name='coach',
      version='0.7',
      description='Reinforcement Learning Coach',
      author='Caspi, Itai and Leibovich, Gal and Novik, Gal',
      author_email='gal.novik@intel.com',
      url='https://github.com/NervanaSystems/coach',
      packages=find_packages(),
      download_url='https://github.com/NervanaSystems/coach',
      keywords=['reinforcement', 'machine', 'learning'],
      install_requires=['annoy', 'Pillow', 'matplotlib', 'numpy', 'pandas',
                        'pygame', 'PyOpenGL', 'scipy', 'scikit-image',
                        'tensorflow', 'gym', 'PyYAML'],
      scripts=['scripts/coach'],
      classifiers=['Programming Language :: Python :: 3',
                   'Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: End Users/Desktop',
                   'License :: OSI Approved :: Apache Software License',
                   'Operating System :: OS Independent',
                   'Topic :: Scientific/Engineering :: Artificial '
                   'Intelligence'])
