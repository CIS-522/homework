"""
This script is responsible for uploading your assignments to the gradebook.

You do not need to change this file.

"""

import os
import json
import sys
import requests
import glob

BASE_URL = "https://leaderboard.cis522.com"
TOKEN = os.getenv("CIS522_TOKEN", None)

if TOKEN is None:
    raise ValueError(
        "Please set the CIS522_TOKEN environment variable in your GitHub secrets."
    )


def get_current_git_commit_hash():
    """
    Get the current git commit hash.

    Returns:
        str: The current git commit hash.
    """
    return os.popen("git rev-parse HEAD").read().strip()


def get_git_remote_deep_url_for_hash(hash: str = None):
    """
    Get the deep URL for a git commit hash.

    Arguments:
        hash (str): The git commit hash.

    Returns:
        str: The deep URL for the git commit hash.
    """
    remote = os.popen("git config --get remote.origin.url").read().strip()
    if hash is None:
        return remote
    return f"{remote}/tree/{hash}"


def submit_tarball(assignment_headers: dict, tarball_path: str):
    """
    Submit a tarball to the gradebook.

    Arguments:
        assignment_headers (dict): A dictionary of headers to be sent with the
            request. The headers should include the assignment name, submission
            hash, and team name.
        tarball_path (str): The path to the tarball to be submitted.

    Returns:
        requests.Response: The response from the gradebook.
    """
    url = f"{BASE_URL}/api/submissions/new"
    files = {
        "file": open(tarball_path, "rb"),
    }
    response = requests.post(url, headers=assignment_headers, files=files)
    return response


def get_open_assignments():
    """
    Get a list of all the assignments that are currently open.

    Returns:
        list: A list of open assignments.
    """
    url = f"{BASE_URL}/api/assignments"
    headers = {}
    response = requests.get(url, headers=headers)
    return response.json().get("assignments", [])


def load_submission_json(path: str):
    """
    Load the submission.json file.

    Arguments:
        path (str): The path to the submission.json file.

    Returns:
        dict: The contents of the submission.json file.
    """
    with open(path, "r") as f:
        meta = json.load(f, strict=False)
        assert "assignment_name" in meta, "Missing assignment_name in submission.json"
        assert "contributors" in meta, "Missing contributors in submission.json"
        assert "sources" in meta, "Missing sources in submission.json"
        assert isinstance(
            meta["assignment_name"], str
        ), "assignment_name must be a string"
        assert isinstance(meta["contributors"], list), "contributors must be a list"
        for contributor in meta["contributors"]:
            assert isinstance(
                contributor["name"], str
            ), "contributor[].name must be a list of strings"
            assert isinstance(
                contributor["pennkey"], str
            ), "contributor[].pennkey must be a list of strings"
        assert isinstance(meta["sources"], list), "sources must be a list"
        for source in meta["sources"]:
            assert isinstance(source, str), "sources must be a list of strings"
        return meta


def get_latest_submission_path():
    """
    Get the path to the latest submission.

    Returns:
        str: The path to the latest submission.
    """
    assignments = glob.glob("assignments/week*")
    assignments.sort()
    return assignments[-1]


def submit_all_open_assignments():
    # Get the list of open assignments
    assignments = get_open_assignments()
    assignment_names = [assignment["name"] for assignment in assignments]

    # Get all week#/submission.json files
    submission_jsons = glob.glob("assignments/week*/submission.json")

    # For each submission.json file, check if the assignment is open
    for submission_json in submission_jsons:
        # Load the submission.json file
        meta = load_submission_json(submission_json)

        # Check if the assignment is open
        if meta["assignment_name"] not in assignment_names:
            continue

        print(f"Submitting {meta['assignment_name']}...")

        # Get the latest submission
        latest_submission_path = os.path.dirname(submission_json)

        # Create a tarball of the assignment
        tarball_path = os.path.join(latest_submission_path, "assignment.tar.gz")
        os.system(f"tar -czf ./{tarball_path} -C {latest_submission_path} .")

        # Create the assignment headers
        assignment_headers = {
            "Assignment": meta["assignment_name"],
            "Team": ",".join(
                [contributor["pennkey"] for contributor in meta["contributors"]]
            ),
            # JSON-stringify the sources
            "Sources": json.dumps(meta["sources"]),
            "Submission-Hash": get_current_git_commit_hash(),
            "Authorization": f"Token {TOKEN}",
            "Submission-Remote": get_git_remote_deep_url_for_hash(),
        }

        # Upload the assignment
        response = submit_tarball(assignment_headers, tarball_path)
        response.raise_for_status()
        if "error" in response.json():
            print(response.json()["error"])
            sys.exit(1)
        print(response.json())


def main():
    submit_all_open_assignments()


if __name__ == "__main__":
    main()
