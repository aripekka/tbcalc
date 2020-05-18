__author__ = 'Ari-Pekka Honkanen'
__license__ = 'MIT'
__date__ = '2020-04-30'

from setuptools import setup
#
# memorandum (for pypi)
#
# python setup.py sdist upload


setup(name='tbcalc',
      version='1.0',
      description='Calculate X-ray diffraction curves of toroidally bent crystal analysers.',
      author='Ari-Pekka Honkanen',
      author_email='honkanen.ap@gmail.com',
      url='https://github.com/aripekka/tbcalc/',
      packages=[
                'tbcalc',
               ],
      install_requires=[
                        'numpy>=1.16.6',
                        'matplotlib>=2.2.3',
                        'pyTTE>=1.0',
                       ],
      include_package_data=True,
)
