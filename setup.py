import sys

from setuptools import setup

setup(name='saro',
      version='0.1.0', # defined in the __init__ module
      description='Statistical Analysis of Robust Observables. A companion package for xara for the reduction and interpretation of high resolution astronomical images and interferometric data.',
      url='http://github.com/rlaugier/saro',
      author='Romain Laugier',
      author_email='romain.laugier@oca.eu',
      license='',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Professional Astronomers',
          'Topic :: High Angular Resolution Astronomy :: Interferometry',
          'Programming Language :: Python :: 3.7'
      ],
      packages=['saro'],
      install_requires=[
          'numpy', 'scipy', 'matplotlib', 'astropy','tqdm','lmfit','xara'
      ],
      data_files = None,
      include_package_data=False,
      zip_safe=False)

