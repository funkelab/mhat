import git


def get_experiment_metadata():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return {"git_hash": sha}
