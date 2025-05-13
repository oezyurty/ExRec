from setuptools import setup, find_packages

setup(name="exercise_recommender",
      version="0.1.0",
      description="Exercise Recommender: A framework to train RL models on Knowledge Tracing problems.",
      author="SeKTER Team",
      author_email="",
      packages=find_packages(),
      python_requires='>=3.7',
      url="",
      install_requires=["numpy", "torch", "gymnasium"],
      extras_require={'demo': ["tensorflow==1.15.0", "stable-baselines[mpi]"]},
      tests_requires=["pytest"])