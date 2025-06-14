#!/usr/bin/env python

import setuptools

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text(encoding='utf-8')

gui = [
    'PyQt6',
    'PyQt6-WebEngine',
    'pyside6',
    'pynput',
    'screeninfo',
    'latex2sympy2',
]
api = [
    'streamlit>=1.8.1',
    'fastapi>=0.75.2',
    'uvicorn[standard]',
    'python-multipart',
    'st_img_pastebutton>=0.0.3',
]
train = [
    'python-Levenshtein>=0.12.2',
    'torchtext>=0.6.0',
    'imagesize>=1.2.0',
]
highlight = ['pygments']

setuptools.setup(
    name='pix2tex',
    version='0.1.4',
    description='pix2tex: Using a ViT to convert images of equations into LaTeX code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lukas Blecher',
    author_email='luk.blecher@gmail.com',
    url='https://github.com/lukas-blecher/LaTeX-OCR/',
    license='MIT',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'image to text'
    ],
    packages=setuptools.find_packages(),
    package_data={
        'pix2tex': [
            'resources/*',
            'model/settings/*.yaml',
            'model/dataset/*.json',
        ]
    },
    extras_require={
        'all': gui+api+train+highlight,
        'gui': gui,
        'api': api,
        'train': train,
        'highlight': highlight,
    },
    entry_points={
        'console_scripts': [
            'pix2tex_gui = pix2tex.__main__:main',
            'pix2tex_cli = pix2tex.__main__:main',
            'latexocr = pix2tex.__main__:main',
            'pix2tex = pix2tex.__main__:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
)
