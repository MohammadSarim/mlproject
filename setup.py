from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'
def get_requirements(file_path : str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open('requirements.txt', 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements        


setup(
name='mlproject',
version='0.0.1',
author='Mohmmad Sarim',
author_email='mohdsarim8@gmail.com',
packages=find_packages(),
install_requires = get_requirements('requirements.txt'),
)

