from setuptools import setup

setup(
    name='pyexample',
    version='0.1.0',    
    description='A minimal, functional interface to the semiempirical extended tight-binding (xtb) program',
    url='https://github.com/Bayer-Group/xtbf',
    author='Jan Wollschläger',
    author_email='janmwoll@gmail.com',
    license='BSD 3-clause',
    packages=['xtbf'],
    install_requires=[
        'joblib', 'tqdm','numpy', 'pandas',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python :: 3',
    ],
)