from setuptools import setup, find_packages


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name='diffusers-interpret',
    version='0.0.0',
    description='', # TODO
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/JoaoLages/diffusers-interpret',
    author='Joao Lages',
    author_email='joaop.glages@gmail.com',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=required
)