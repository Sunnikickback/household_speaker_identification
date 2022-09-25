from setuptools import find_packages, setup


def main():
    console_scripts = [
        "train=household_speaker_identification.train:main",
    ]

    with open("requirements.txt") as fin:
        install_requires = fin.read()

    setup(
        name="household_speaker_identification",
        version="0.1",
        author="Sunni|kickback",
        description="",
        packages=find_packages("src"),
        package_dir={"": "src"},
        install_requires=install_requires,
        entry_points={
            "console_scripts": console_scripts,
        }
    )


if __name__ == "__main__":
    main()
