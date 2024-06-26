from setuptools import setup, find_packages


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name='diffusers-interpret',
    version='0.5.1',
    description='diffusers-interpret: model explainability for 🤗 Diffusers',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ZacBi/diffusers-interpret.git',
    author='Zac Bi',
    author_email='zaci@outlook.com',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=required
)